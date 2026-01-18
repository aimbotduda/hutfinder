import argparse
import math
import mmap
import os
import re
import sys
import threading
import time
import tempfile
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import numpy as np
from scipy.spatial import cKDTree

# Try to import psutil for memory monitoring (optional but recommended)
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Try to import numba for JIT compilation (optional but faster)
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============== MEMORY MANAGEMENT ==============

def get_available_memory_gb():
    """Get available system memory in GB."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    else:
        # Fallback: try to read from /proc/meminfo on Linux/Mac
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemAvailable' in line:
                        return int(line.split()[1]) / (1024 ** 2)  # KB to GB
        except:
            pass
        # Default conservative estimate
        return 8.0


def get_total_memory_gb():
    """Get total system memory in GB."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        return mem.total / (1024 ** 3)
    else:
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if 'MemTotal' in line:
                        return int(line.split()[1]) / (1024 ** 2)
        except:
            pass
        return 16.0


def estimate_kdtree_memory_gb(num_points, leafsize):
    """
    Estimate memory usage for a cKDTree in GB.
    
    A cKDTree uses approximately:
    - Internal nodes: ~40 bytes each (for 2D)
    - The number of internal nodes is roughly num_points / leafsize
    - Plus overhead for the data reference
    
    This is a rough estimate - actual usage varies.
    """
    # Data array reference (the memmap is shared, but tree keeps reference)
    data_overhead_bytes = num_points * 2 * 4  # 2D int32, but tree may convert to float64
    
    # Tree structure overhead
    num_nodes = max(1, num_points // leafsize) * 2  # binary tree nodes
    node_overhead_bytes = num_nodes * 64  # estimated bytes per node
    
    # Additional overhead for indices and other structures
    index_overhead_bytes = num_points * 8  # indices as int64
    
    total_bytes = data_overhead_bytes + node_overhead_bytes + index_overhead_bytes
    return total_bytes / (1024 ** 3)


def compute_memory_safe_workers(num_places, leafsize, max_memory_gb=None, memory_reserve_gb=4.0):
    """
    Calculate the maximum number of workers that can safely run without exhausting memory.
    
    Each worker builds its own KDTree, which is the main memory consumer.
    
    Args:
        num_places: Number of data points
        leafsize: KDTree leafsize
        max_memory_gb: Optional cap on total memory usage
        memory_reserve_gb: Amount of memory to keep free for OS/other processes
    
    Returns:
        Tuple of (safe_worker_count, estimated_memory_per_worker_gb)
    """
    available_gb = get_available_memory_gb()
    total_gb = get_total_memory_gb()
    
    # Estimate memory per worker (KDTree + working memory)
    tree_memory_gb = estimate_kdtree_memory_gb(num_places, leafsize)
    working_memory_gb = 0.5  # Buffer for results, temporary data
    per_worker_gb = tree_memory_gb + working_memory_gb
    
    # Calculate usable memory
    if max_memory_gb is not None:
        usable_gb = min(max_memory_gb, available_gb) - memory_reserve_gb
    else:
        # Use available memory minus reserve, but cap at 80% of total
        usable_gb = min(available_gb - memory_reserve_gb, total_gb * 0.8)
    
    usable_gb = max(usable_gb, per_worker_gb)  # At least enough for one worker
    
    # Calculate safe worker count
    safe_workers = max(1, int(usable_gb / per_worker_gb))
    
    return safe_workers, per_worker_gb


def print_memory_status(prefix=""):
    """Print current memory usage status."""
    if PSUTIL_AVAILABLE:
        mem = psutil.virtual_memory()
        print(f"{prefix}Memory: {mem.used / (1024**3):.1f}GB used / "
              f"{mem.total / (1024**3):.1f}GB total "
              f"({mem.percent}% used, {mem.available / (1024**3):.1f}GB available)")
    else:
        print(f"{prefix}Memory monitoring unavailable (install psutil for detailed stats)")


# ============== OUTPUT FORMATTING & CLEANING ==============

def euclidean_distance(a, b):
    """Compute Euclidean distance between two 2D points."""
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    return math.hypot(ax - bx, ay - by)


def compute_pairwise_distance(coords):
    """Compute sum of all pairwise distances between points in group."""
    pair_sum = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            pair_sum += euclidean_distance(coords[i], coords[j])
    return pair_sum


def compute_spawn_distance(coords):
    """Compute sum of distances from origin (spawn) for all points."""
    return sum(math.hypot(float(c[0]), float(c[1])) for c in coords)


def canonicalize_coords(coords):
    """Sort coordinates for consistent ordering and deduplication."""
    return sorted(coords, key=lambda c: (int(c[0]), int(c[1])))


def format_group(coords, spawn_dist, pairwise_dist):
    """Format a group result as a clean string."""
    coords = canonicalize_coords(coords)
    coord_str = ' '.join([f'({int(c[0])}, {int(c[1])})' for c in coords])
    return f'{coord_str} spawn:{spawn_dist:.2f} spread:{pairwise_dist:.2f}'


def get_group_signature(coords):
    """Get a hashable signature for deduplication."""
    coords = canonicalize_coords(coords)
    return tuple((int(c[0]), int(c[1])) for c in coords)


def get_output_paths(base_path):
    """Generate spawn and spread output paths from a base path.
    
    e.g., 'output3Mon.txt' -> ('output3Mon_spawn.txt', 'output3Mon_spread.txt')
    """
    if '.' in base_path:
        name, ext = base_path.rsplit('.', 1)
        return f"{name}_spawn.{ext}", f"{name}_spread.{ext}"
    else:
        return f"{base_path}_spawn", f"{base_path}_spread"


def dedup_and_write_results(all_results, base_output_path, group_type="groups"):
    """Deduplicate results and write two sorted files (spawn and spread).
    
    Args:
        all_results: List of (coords, spawn_dist, pairwise_dist) tuples
        base_output_path: Base path for output files (will generate _spawn and _spread variants)
        group_type: Label for print messages ("triplets" or "quads")
    """
    # Deduplication - always performed
    print("Deduplicating results...")
    seen = {}
    for coords, spawn_dist, pairwise_dist in all_results:
        sig = get_group_signature(coords)
        if sig not in seen:
            seen[sig] = (coords, spawn_dist, pairwise_dist)
        # For duplicates, we already have this signature, skip
    all_results = list(seen.values())
    print(f"After dedup: {len(all_results)} unique {group_type}")
    
    # Get output paths
    spawn_path, spread_path = get_output_paths(base_output_path)
    
    # Write spawn-sorted file
    print(f"Writing {spawn_path} (sorted by distance from spawn)...")
    sorted_by_spawn = sorted(all_results, key=lambda x: x[1])
    with open(spawn_path, 'w') as f:
        for coords, spawn_dist, pairwise_dist in sorted_by_spawn:
            f.write(format_group(coords, spawn_dist, pairwise_dist) + '\n')
    
    # Write spread-sorted file
    print(f"Writing {spread_path} (sorted by group spread - tighter groups first)...")
    sorted_by_spread = sorted(all_results, key=lambda x: x[2])
    with open(spread_path, 'w') as f:
        for coords, spawn_dist, pairwise_dist in sorted_by_spread:
            f.write(format_group(coords, spawn_dist, pairwise_dist) + '\n')
    
    print(f"Completed: Found {len(all_results)} {group_type} total")
    return len(all_results)


def check_group(coords, r2):
    """Check if all points in group are within radius of each other."""
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dz = coords[i][1] - coords[j][1]
            if (dx*dx + dz*dz) > r2:
                return False
    return True


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def check_group_numba(coords, r2):
        """Numba-accelerated group check."""
        n = coords.shape[0]
        for i in range(n):
            for j in range(i+1, n):
                dx = coords[i, 0] - coords[j, 0]
                dz = coords[i, 1] - coords[j, 1]
                if (dx*dx + dz*dz) > r2:
                    return False
        return True
    
    @njit(cache=True)
    def compute_dist(x, z):
        """Compute distance from origin."""
        return math.sqrt(x*x + z*z)


def find_center(coords):
    """Find center coordinates of a group."""
    x = sum(c[0] for c in coords) / len(coords)
    z = sum(c[1] for c in coords) / len(coords)
    print(x, z)
    return (x, z)


def _estimate_record_count(file_path, sample_size=10_000_000):
    """Estimate total record count by sampling the file."""
    file_size = os.path.getsize(file_path)
    
    # Read a sample to estimate records per byte
    with open(file_path, 'rb') as f:
        sample = f.read(min(sample_size, file_size))
    
    sample_records = sample.count(b'->(')
    if sample_records == 0:
        return 0
    
    # Extrapolate
    bytes_per_record = len(sample) / sample_records
    estimated_total = int(file_size / bytes_per_record * 1.05)  # 5% buffer
    return estimated_total


def _parse_chunk(chunk_data):
    """Parse a chunk of bytes and return coordinates as numpy array."""
    # Pre-compiled regex for speed
    pattern = re.compile(rb'->\((-?\d+),(-?\d+)\)')
    matches = pattern.findall(chunk_data)
    
    if not matches:
        return np.empty((0, 2), dtype=np.int32)
    
    # Convert to numpy array in one operation
    coords = np.array([(int(x), int(z)) for x, z in matches], dtype=np.int32)
    return coords


def _parse_chunk_for_pool(args):
    """Worker function for parallel parsing."""
    chunk_data, chunk_id = args
    return _parse_chunk(chunk_data), chunk_id


def parse_to_memmap(input_path, memmap_path, num_parse_workers=None):
    """
    Fast file parsing with multiple optimizations:
    - Single pass over the file (estimates count from sample)
    - Memory-mapped file reading for speed
    - Regex-based batch extraction
    - Optional parallel chunk parsing
    - Batched writes to output memmap
    """
    print("Reading and parsing file of all huts/monuments (fast mode)")
    start_time = time.time()
    
    file_size = os.path.getsize(input_path)
    print(f"File size: {file_size / (1024**3):.2f} GB")
    
    # Estimate record count
    print("Estimating record count...")
    estimated_records = _estimate_record_count(input_path)
    print(f"Estimated records: ~{estimated_records:,}")
    
    if estimated_records == 0:
        print("No records found in file!")
        return np.empty((0, 2), dtype=np.int32)
    
    # Determine parsing strategy based on file size and available cores
    if num_parse_workers is None:
        num_parse_workers = max(1, mp.cpu_count() - 1)
    
    # For very large files, use parallel parsing
    use_parallel = file_size > 100_000_000 and num_parse_workers > 1  # >100MB
    
    # Pre-compile regex
    pattern = re.compile(rb'->\((-?\d+),(-?\d+)\)')
    
    # Chunk size for reading (64MB chunks work well for I/O)
    chunk_size = 64 * 1024 * 1024
    
    # We'll collect all coordinates and write to memmap at the end
    # This is actually faster than writing during parsing for large files
    all_coords = []
    total_found = 0
    bytes_read = 0
    last_progress = 0
    
    print(f"Parsing with chunk size: {chunk_size // (1024*1024)}MB" + 
          (f", {num_parse_workers} workers" if use_parallel else ", single-threaded"))
    
    with open(input_path, 'rb') as f:
        # Memory-map the file for faster reading
        try:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            use_mmap = True
        except:
            # Fallback if mmap fails
            use_mmap = False
            mm = f
        
        leftover = b''
        
        if use_parallel and use_mmap:
            # Parallel parsing: read chunks and parse in parallel
            chunks_to_parse = []
            
            while True:
                if use_mmap:
                    chunk = mm.read(chunk_size)
                else:
                    chunk = f.read(chunk_size)
                
                if not chunk:
                    # Process any leftover
                    if leftover:
                        coords = _parse_chunk(leftover)
                        if len(coords) > 0:
                            all_coords.append(coords)
                            total_found += len(coords)
                    break
                
                bytes_read += len(chunk)
                
                # Combine with leftover from previous chunk
                data = leftover + chunk
                
                # Find last newline to avoid splitting records
                last_newline = data.rfind(b'\n')
                if last_newline == -1:
                    leftover = data
                    continue
                
                # Split at last newline
                to_parse = data[:last_newline + 1]
                leftover = data[last_newline + 1:]
                
                chunks_to_parse.append((to_parse, len(chunks_to_parse)))
                
                # Parse in batches to limit memory
                if len(chunks_to_parse) >= num_parse_workers * 2:
                    with ProcessPoolExecutor(max_workers=num_parse_workers) as executor:
                        results = list(executor.map(_parse_chunk_for_pool, chunks_to_parse))
                    
                    for coords, _ in sorted(results, key=lambda x: x[1]):
                        if len(coords) > 0:
                            all_coords.append(coords)
                            total_found += len(coords)
                    
                    chunks_to_parse = []
                
                # Progress update
                progress = int(bytes_read / file_size * 100)
                if progress >= last_progress + 5:
                    elapsed = time.time() - start_time
                    speed_mb = bytes_read / (1024**2) / max(elapsed, 0.001)
                    print(f"  {progress}% read ({total_found:,} records found, {speed_mb:.1f} MB/s)")
                    last_progress = progress
            
            # Process remaining chunks
            if chunks_to_parse:
                with ProcessPoolExecutor(max_workers=num_parse_workers) as executor:
                    results = list(executor.map(_parse_chunk_for_pool, chunks_to_parse))
                
                for coords, _ in sorted(results, key=lambda x: x[1]):
                    if len(coords) > 0:
                        all_coords.append(coords)
                        total_found += len(coords)
        
        else:
            # Single-threaded parsing (simpler, still fast)
            while True:
                if use_mmap:
                    chunk = mm.read(chunk_size)
                else:
                    chunk = f.read(chunk_size)
                
                if not chunk:
                    if leftover:
                        coords = _parse_chunk(leftover)
                        if len(coords) > 0:
                            all_coords.append(coords)
                            total_found += len(coords)
                    break
                
                bytes_read += len(chunk)
                data = leftover + chunk
                
                last_newline = data.rfind(b'\n')
                if last_newline == -1:
                    leftover = data
                    continue
                
                to_parse = data[:last_newline + 1]
                leftover = data[last_newline + 1:]
                
                coords = _parse_chunk(to_parse)
                if len(coords) > 0:
                    all_coords.append(coords)
                    total_found += len(coords)
                
                progress = int(bytes_read / file_size * 100)
                if progress >= last_progress + 5:
                    elapsed = time.time() - start_time
                    speed_mb = bytes_read / (1024**2) / max(elapsed, 0.001)
                    print(f"  {progress}% read ({total_found:,} records found, {speed_mb:.1f} MB/s)")
                    last_progress = progress
        
        if use_mmap:
            mm.close()
    
    parse_time = time.time() - start_time
    print(f"Parsing complete: {total_found:,} records in {parse_time:.2f}s")
    
    if total_found == 0:
        print("No records found!")
        return np.empty((0, 2), dtype=np.int32)
    
    # Concatenate all coordinate arrays
    print("Concatenating arrays...")
    concat_start = time.time()
    all_places = np.concatenate(all_coords, axis=0)
    del all_coords  # Free memory
    print(f"Concatenation: {time.time() - concat_start:.2f}s")
    
    # Write to memmap
    print(f"Writing {len(all_places):,} records to memmap...")
    write_start = time.time()
    places_mm = np.memmap(memmap_path, dtype=np.int32, mode='w+', shape=all_places.shape)
    places_mm[:] = all_places
    places_mm.flush()
    del places_mm
    del all_places
    print(f"Write complete: {time.time() - write_start:.2f}s")
    
    # Reopen as read-only
    places = np.memmap(memmap_path, dtype=np.int32, mode='r', shape=(total_found, 2))
    
    total_time = time.time() - start_time
    print(f"Total parsing time: {total_time:.2f}s ({file_size / (1024**3) / total_time:.2f} GB/s)")
    print(f"Found {len(places):,} places")
    
    return places


def build_tree(places, leafsize):
    print("Building tree... this may take a while")
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    print("Built tree")
    return tree


def compute_auto_leafsize(num_places, available_memory_gb=None):
    """
    Calculate optimal KDTree leafsize based on dataset size and available memory.
    
    Guidelines:
    - Smaller leafsize = faster queries but more memory (more tree nodes)
    - Larger leafsize = less memory but slower queries (more brute-force at leaves)
    
    The sweet spot depends on dataset size:
    - Small datasets (<100K): leafsize 16-32, fast queries matter more
    - Medium datasets (100K-10M): leafsize 32-64, balanced
    - Large datasets (10M-100M): leafsize 64-128, memory matters more
    - Very large (>100M): leafsize 256-512, memory critical
    - Extreme (>500M): leafsize 512-1024, maximize memory efficiency
    """
    if available_memory_gb is None:
        available_memory_gb = get_available_memory_gb()
    
    # Base leafsize from dataset size
    if num_places < 100_000:
        base_leafsize = 16
    elif num_places < 1_000_000:
        base_leafsize = 32
    elif num_places < 10_000_000:
        base_leafsize = 48
    elif num_places < 50_000_000:
        base_leafsize = 64
    elif num_places < 100_000_000:
        base_leafsize = 128
    elif num_places < 200_000_000:
        base_leafsize = 192
    elif num_places < 500_000_000:
        base_leafsize = 384
    elif num_places < 1_000_000_000:
        base_leafsize = 512
    else:
        base_leafsize = 768
    
    # Increase leafsize if memory is constrained
    # Estimate memory per tree at this leafsize
    estimated_tree_gb = estimate_kdtree_memory_gb(num_places, base_leafsize)
    
    # If we'd use more than 25% of available memory for ONE tree, increase leafsize
    while estimated_tree_gb > available_memory_gb * 0.25 and base_leafsize < 2048:
        base_leafsize = int(base_leafsize * 1.5)
        estimated_tree_gb = estimate_kdtree_memory_gb(num_places, base_leafsize)
    
    return base_leafsize


# ============== PARALLEL WORKER FUNCTIONS ==============

# Global variables for worker processes (initialized by pool initializer)
_worker_progress_counter = None
_worker_found_counter = None
_worker_temp_dir = None


def _init_worker(progress_counter, found_counter, temp_dir=None):
    """Initialize worker process with shared counters."""
    global _worker_progress_counter, _worker_found_counter, _worker_temp_dir
    _worker_progress_counter = progress_counter
    _worker_found_counter = found_counter
    _worker_temp_dir = temp_dir


def _serialize_result_to_line(coords, spawn_dist, pairwise_dist):
    """Serialize a result to a single line for disk storage."""
    # Format: x1,z1;x2,z2;x3,z3[;x4,z4]|spawn_dist|pairwise_dist
    coord_str = ';'.join(f"{c[0]},{c[1]}" for c in coords)
    return f"{coord_str}|{spawn_dist:.6f}|{pairwise_dist:.6f}\n"


def _deserialize_result_from_line(line):
    """Deserialize a result from a line."""
    parts = line.strip().split('|')
    coord_strs = parts[0].split(';')
    coords = [tuple(map(int, c.split(','))) for c in coord_strs]
    spawn_dist = float(parts[1])
    pairwise_dist = float(parts[2])
    return coords, spawn_dist, pairwise_dist


def _worker_find_groups_3(args):
    """Worker function to find triplets in a chunk of indices.
    
    Results are written to a temporary file to minimize memory usage.
    """
    global _worker_progress_counter, _worker_found_counter, _worker_temp_dir
    chunk_start, chunk_end, places_path, places_shape, radius, leafsize, worker_id = args
    
    # Load the memmap in this worker
    places = np.memmap(places_path, dtype=np.int32, mode='r', shape=places_shape)
    
    # Rebuild tree in worker (required for multiprocessing - can't pickle cKDTree)
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    
    r2 = radius * radius
    local_count = 0
    found_count = 0
    
    # Write results to temporary file to avoid memory accumulation
    temp_file = None
    temp_path = None
    if _worker_temp_dir:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix=f'w{worker_id}_', dir=_worker_temp_dir)
        temp_file = os.fdopen(temp_fd, 'w')
    
    results_buffer = []
    buffer_flush_size = 10000  # Flush to disk every N results
    
    try:
        for i in range(chunk_start, chunk_end):
            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = sorted([idx for idx in neighbors if idx > i])
            
            for a_idx in range(len(neigh_indices)):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, len(neigh_indices)):
                    k = neigh_indices[b_idx]
                    coords = [(int(places[i][0]), int(places[i][1])),
                              (int(places[j][0]), int(places[j][1])),
                              (int(places[k][0]), int(places[k][1]))]
                    if check_group(coords, r2):
                        spawn_dist = compute_spawn_distance(coords)
                        pairwise_dist = compute_pairwise_distance(coords)
                        found_count += 1
                        
                        if temp_file:
                            results_buffer.append(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                            if len(results_buffer) >= buffer_flush_size:
                                temp_file.writelines(results_buffer)
                                results_buffer.clear()
                        else:
                            results_buffer.append((coords, spawn_dist, pairwise_dist))
            
            # Update shared progress counter periodically
            local_count += 1
            if local_count % 50000 == 0 and _worker_progress_counter is not None:
                with _worker_progress_counter.get_lock():
                    _worker_progress_counter.value += 50000
        
        # Flush remaining results
        if temp_file and results_buffer:
            temp_file.writelines(results_buffer)
            results_buffer.clear()
    finally:
        if temp_file:
            temp_file.close()
    
    # Clean up tree to free memory before returning
    del tree
    
    # Update remaining progress
    remaining = local_count % 50000
    if remaining > 0 and _worker_progress_counter is not None:
        with _worker_progress_counter.get_lock():
            _worker_progress_counter.value += remaining
    
    # Update found counter with final count
    if _worker_found_counter is not None:
        with _worker_found_counter.get_lock():
            _worker_found_counter.value += found_count
    
    # Return either temp file path or in-memory results (for backward compat)
    if temp_path:
        return ('file', temp_path, found_count), chunk_end - chunk_start, worker_id
    else:
        return ('memory', results_buffer, found_count), chunk_end - chunk_start, worker_id


def _worker_find_groups_4(args):
    """Worker function to find quads in a chunk of indices.
    
    Results are written to a temporary file to minimize memory usage.
    """
    global _worker_progress_counter, _worker_found_counter, _worker_temp_dir
    chunk_start, chunk_end, places_path, places_shape, radius, leafsize, worker_id = args
    
    places = np.memmap(places_path, dtype=np.int32, mode='r', shape=places_shape)
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    
    r2 = radius * radius
    local_count = 0
    found_count = 0
    
    # Write results to temporary file
    temp_file = None
    temp_path = None
    if _worker_temp_dir:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix=f'w{worker_id}_', dir=_worker_temp_dir)
        temp_file = os.fdopen(temp_fd, 'w')
    
    results_buffer = []
    buffer_flush_size = 10000
    
    try:
        for i in range(chunk_start, chunk_end):
            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = sorted([idx for idx in neighbors if idx > i])
            L = len(neigh_indices)
            
            for a_idx in range(L):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, L):
                    k = neigh_indices[b_idx]
                    for c_idx in range(b_idx + 1, L):
                        m = neigh_indices[c_idx]
                        coords = [(int(places[i][0]), int(places[i][1])),
                                  (int(places[j][0]), int(places[j][1])),
                                  (int(places[k][0]), int(places[k][1])),
                                  (int(places[m][0]), int(places[m][1]))]
                        if check_group(coords, r2):
                            spawn_dist = compute_spawn_distance(coords)
                            pairwise_dist = compute_pairwise_distance(coords)
                            found_count += 1
                            
                            if temp_file:
                                results_buffer.append(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                                if len(results_buffer) >= buffer_flush_size:
                                    temp_file.writelines(results_buffer)
                                    results_buffer.clear()
                            else:
                                results_buffer.append((coords, spawn_dist, pairwise_dist))
            
            local_count += 1
            if local_count % 50000 == 0 and _worker_progress_counter is not None:
                with _worker_progress_counter.get_lock():
                    _worker_progress_counter.value += 50000
        
        if temp_file and results_buffer:
            temp_file.writelines(results_buffer)
            results_buffer.clear()
    finally:
        if temp_file:
            temp_file.close()
    
    del tree
    
    remaining = local_count % 50000
    if remaining > 0 and _worker_progress_counter is not None:
        with _worker_progress_counter.get_lock():
            _worker_progress_counter.value += remaining
    
    if _worker_found_counter is not None:
        with _worker_found_counter.get_lock():
            _worker_found_counter.value += found_count
    
    if temp_path:
        return ('file', temp_path, found_count), chunk_end - chunk_start, worker_id
    else:
        return ('memory', results_buffer, found_count), chunk_end - chunk_start, worker_id


def _progress_monitor(progress_counter, found_counter, total, stop_event, label="groups"):
    """Background thread to print progress updates."""
    last_progress = 0
    last_found = 0
    while not stop_event.is_set():
        current_progress = progress_counter.value
        current_found = found_counter.value
        if current_progress != last_progress or current_found != last_found:
            percentage = (current_progress / total) * 100
            print(f"\r{percentage:.2f}% searched - Found {current_found} {label}    ", end="", flush=True)
            last_progress = current_progress
            last_found = current_found
        stop_event.wait(0.5)  # Update every 0.5 seconds
    # Final update
    print(f"\r{100.00:.2f}% searched - Found {found_counter.value} {label}    ")


def _merge_temp_files_and_dedup(temp_files, output_path, group_type="groups"):
    """
    Merge temporary result files, deduplicate, and write final sorted outputs.
    
    This processes files in a memory-efficient streaming manner.
    """
    print(f"Merging {len(temp_files)} result files...")
    
    # First pass: collect all unique results with deduplication
    seen = {}
    total_read = 0
    
    for temp_path in temp_files:
        if not temp_path or not os.path.exists(temp_path):
            continue
        try:
            with open(temp_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    coords, spawn_dist, pairwise_dist = _deserialize_result_from_line(line)
                    sig = get_group_signature(coords)
                    if sig not in seen:
                        seen[sig] = (coords, spawn_dist, pairwise_dist)
                    total_read += 1
            # Delete temp file after reading
            os.remove(temp_path)
        except Exception as e:
            print(f"Warning: Error processing temp file {temp_path}: {e}")
    
    print(f"Read {total_read} results, {len(seen)} unique after dedup")
    
    all_results = list(seen.values())
    del seen  # Free memory
    
    # Get output paths
    spawn_path, spread_path = get_output_paths(output_path)
    
    # Write spawn-sorted file
    print(f"Writing {spawn_path} (sorted by distance from spawn)...")
    sorted_by_spawn = sorted(all_results, key=lambda x: x[1])
    with open(spawn_path, 'w') as f:
        for coords, spawn_dist, pairwise_dist in sorted_by_spawn:
            f.write(format_group(coords, spawn_dist, pairwise_dist) + '\n')
    del sorted_by_spawn
    
    # Write spread-sorted file
    print(f"Writing {spread_path} (sorted by group spread - tighter groups first)...")
    sorted_by_spread = sorted(all_results, key=lambda x: x[2])
    with open(spread_path, 'w') as f:
        for coords, spawn_dist, pairwise_dist in sorted_by_spread:
            f.write(format_group(coords, spawn_dist, pairwise_dist) + '\n')
    
    print(f"Completed: Found {len(all_results)} {group_type} total")
    return len(all_results)


def find_groups_3_global_parallel(places, memmap_path, radius, output_path, leafsize, num_workers=None, max_memory_gb=None):
    """Parallel version of find_groups_3_global using multiple processes.
    
    Memory-aware: automatically adjusts workers based on available memory.
    Results are written to temp files to minimize memory usage.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    total_places = len(places)
    
    # Calculate memory-safe number of workers
    safe_workers, mem_per_worker = compute_memory_safe_workers(total_places, leafsize, max_memory_gb)
    
    if num_workers is None:
        num_workers = min(safe_workers, max(1, mp.cpu_count() - 1))
    else:
        if num_workers > safe_workers:
            print(f"WARNING: Requested {num_workers} workers but only {safe_workers} are memory-safe")
            print(f"         Each worker needs ~{mem_per_worker:.1f}GB for the KDTree")
            user_confirm = input(f"Continue with {num_workers} workers anyway? [y/N]: ").strip().lower()
            if user_confirm not in ('y', 'yes'):
                num_workers = safe_workers
                print(f"Reduced to {num_workers} workers")
    
    print_memory_status("Before search: ")
    print(f"Estimated memory per worker: {mem_per_worker:.1f}GB")
    
    chunk_size = max(1, total_places // (num_workers * 4))  # More chunks for better load balancing
    
    # Create list of work chunks
    chunks = []
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, memmap_path, places.shape, radius, leafsize, len(chunks)))
    
    print(f"Starting parallel search for triplets with {num_workers} workers, {len(chunks)} chunks")
    
    # Create temp directory for result files
    temp_dir = tempfile.mkdtemp(prefix='finder_triplets_')
    print(f"Using temp directory: {temp_dir}")
    
    # Create shared counters for progress tracking
    progress_counter = mp.Value('q', 0)  # 'q' = signed long long (64-bit)
    found_counter = mp.Value('q', 0)
    
    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(progress_counter, found_counter, total_places, stop_event, "triplets")
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    temp_files = []
    
    try:
        # Use maxtasksperchild to recycle workers and free memory periodically
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(progress_counter, found_counter, temp_dir),
            maxtasksperchild=max(1, len(chunks) // (num_workers * 2))  # Recycle workers periodically
        ) as pool:
            for result_info, count, worker_id in pool.imap_unordered(_worker_find_groups_3, chunks):
                result_type, data, found = result_info
                if result_type == 'file':
                    temp_files.append(data)
                else:
                    # In-memory results (fallback) - write to temp file
                    if data:
                        fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
                        with os.fdopen(fd, 'w') as f:
                            for coords, spawn_dist, pairwise_dist in data:
                                f.write(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                        temp_files.append(temp_path)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    print_memory_status("After search: ")
    
    # Merge temp files and write final output
    _merge_temp_files_and_dedup(temp_files, output_path, "triplets")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass


def find_groups_4_global_parallel(places, memmap_path, radius, output_path, leafsize, num_workers=None, max_memory_gb=None):
    """Parallel version of find_groups_4_global using multiple processes.
    
    Memory-aware: automatically adjusts workers based on available memory.
    Results are written to temp files to minimize memory usage.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    total_places = len(places)
    
    # Calculate memory-safe number of workers
    safe_workers, mem_per_worker = compute_memory_safe_workers(total_places, leafsize, max_memory_gb)
    
    if num_workers is None:
        num_workers = min(safe_workers, max(1, mp.cpu_count() - 1))
    else:
        if num_workers > safe_workers:
            print(f"WARNING: Requested {num_workers} workers but only {safe_workers} are memory-safe")
            print(f"         Each worker needs ~{mem_per_worker:.1f}GB for the KDTree")
            user_confirm = input(f"Continue with {num_workers} workers anyway? [y/N]: ").strip().lower()
            if user_confirm not in ('y', 'yes'):
                num_workers = safe_workers
                print(f"Reduced to {num_workers} workers")
    
    print_memory_status("Before search: ")
    print(f"Estimated memory per worker: {mem_per_worker:.1f}GB")
    
    chunk_size = max(1, total_places // (num_workers * 4))
    
    chunks = []
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, memmap_path, places.shape, radius, leafsize, len(chunks)))
    
    print(f"Starting parallel search for quads with {num_workers} workers, {len(chunks)} chunks")
    
    # Create temp directory for result files
    temp_dir = tempfile.mkdtemp(prefix='finder_quads_')
    print(f"Using temp directory: {temp_dir}")
    
    # Create shared counters for progress tracking
    progress_counter = mp.Value('q', 0)
    found_counter = mp.Value('q', 0)
    
    # Start progress monitor thread
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(progress_counter, found_counter, total_places, stop_event, "quads")
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    temp_files = []
    
    try:
        with mp.Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(progress_counter, found_counter, temp_dir),
            maxtasksperchild=max(1, len(chunks) // (num_workers * 2))
        ) as pool:
            for result_info, count, worker_id in pool.imap_unordered(_worker_find_groups_4, chunks):
                result_type, data, found = result_info
                if result_type == 'file':
                    temp_files.append(data)
                else:
                    if data:
                        fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
                        with os.fdopen(fd, 'w') as f:
                            for coords, spawn_dist, pairwise_dist in data:
                                f.write(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                        temp_files.append(temp_path)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    print_memory_status("After search: ")
    
    # Merge temp files and write final output
    _merge_temp_files_and_dedup(temp_files, output_path, "quads")
    
    # Clean up temp directory
    try:
        os.rmdir(temp_dir)
    except:
        pass


# ============== SINGLE-THREADED FUNCTIONS (for --single-threaded option) ==============

def find_groups_3_global(places, tree, radius, output_path):
    """Single-threaded version for finding triplets.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    r2 = radius * radius
    totalPlaces = len(places)
    all_results = []
    
    for i in range(totalPlaces):
        if i % 100000 == 0 and i > 0:
            percentage = (i / totalPlaces) * 100
            print(f"{percentage:.2f}% searched - Found {len(all_results)} groups")

        neighbors = tree.query_ball_point(places[i], r=radius)
        neigh_indices = [idx for idx in neighbors if idx > i]
        neigh_indices.sort()
        for a_idx in range(len(neigh_indices)):
            j = neigh_indices[a_idx]
            for b_idx in range(a_idx + 1, len(neigh_indices)):
                k = neigh_indices[b_idx]
                coords = [(int(places[i][0]), int(places[i][1])),
                          (int(places[j][0]), int(places[j][1])),
                          (int(places[k][0]), int(places[k][1]))]
                if check_group(coords, r2):
                    spawn_dist = compute_spawn_distance(coords)
                    pairwise_dist = compute_pairwise_distance(coords)
                    all_results.append((coords, spawn_dist, pairwise_dist))
    
    # Deduplicate and write both sorted files
    dedup_and_write_results(all_results, output_path, "triplets")


def find_groups_4_global(places, tree, radius, output_path):
    """Single-threaded version for finding quads.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    r2 = radius * radius
    totalPlaces = len(places)
    all_results = []
    
    for i in range(totalPlaces):
        if i % 100000 == 0 and i > 0:
            percentage = (i / totalPlaces) * 100
            print(f"{percentage:.2f}% searched - Found {len(all_results)} groups")

        neighbors = tree.query_ball_point(places[i], r=radius)
        neigh_indices = [idx for idx in neighbors if idx > i]
        neigh_indices.sort()
        L = len(neigh_indices)
        for a_idx in range(L):
            j = neigh_indices[a_idx]
            for b_idx in range(a_idx + 1, L):
                k = neigh_indices[b_idx]
                for c_idx in range(b_idx + 1, L):
                    m = neigh_indices[c_idx]
                    coords = [(int(places[i][0]), int(places[i][1])),
                              (int(places[j][0]), int(places[j][1])),
                              (int(places[k][0]), int(places[k][1])),
                              (int(places[m][0]), int(places[m][1]))]
                    if check_group(coords, r2):
                        spawn_dist = compute_spawn_distance(coords)
                        pairwise_dist = compute_pairwise_distance(coords)
                        all_results.append((coords, spawn_dist, pairwise_dist))
    
    # Deduplicate and write both sorted files
    dedup_and_write_results(all_results, output_path, "quads")


# ============== THREADED FUNCTIONS (shared tree, low memory) ==============

def _thread_worker_find_groups_3(args):
    """Thread worker for finding triplets using a shared tree.
    
    Unlike multiprocessing workers, threads share the same tree object,
    so we only need ONE tree in memory instead of one per worker.
    """
    chunk_start, chunk_end, places, tree, radius, temp_dir, worker_id, progress_counter, found_counter = args
    
    r2 = radius * radius
    local_count = 0
    found_count = 0
    
    # Write results to temporary file
    temp_path = None
    temp_file = None
    if temp_dir:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix=f't{worker_id}_', dir=temp_dir)
        temp_file = os.fdopen(temp_fd, 'w')
    
    results_buffer = []
    buffer_flush_size = 10000
    
    try:
        for i in range(chunk_start, chunk_end):
            # query_ball_point releases the GIL, allowing true parallelism
            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = sorted([idx for idx in neighbors if idx > i])
            
            for a_idx in range(len(neigh_indices)):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, len(neigh_indices)):
                    k = neigh_indices[b_idx]
                    coords = [(int(places[i][0]), int(places[i][1])),
                              (int(places[j][0]), int(places[j][1])),
                              (int(places[k][0]), int(places[k][1]))]
                    if check_group(coords, r2):
                        spawn_dist = compute_spawn_distance(coords)
                        pairwise_dist = compute_pairwise_distance(coords)
                        found_count += 1
                        
                        if temp_file:
                            results_buffer.append(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                            if len(results_buffer) >= buffer_flush_size:
                                temp_file.writelines(results_buffer)
                                results_buffer.clear()
                        else:
                            results_buffer.append((coords, spawn_dist, pairwise_dist))
            
            local_count += 1
            if local_count % 50000 == 0 and progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 50000
        
        if temp_file and results_buffer:
            temp_file.writelines(results_buffer)
            results_buffer.clear()
    finally:
        if temp_file:
            temp_file.close()
    
    # Update remaining progress
    remaining = local_count % 50000
    if remaining > 0 and progress_counter is not None:
        with progress_counter.get_lock():
            progress_counter.value += remaining
    
    if found_counter is not None:
        with found_counter.get_lock():
            found_counter.value += found_count
    
    if temp_path:
        return ('file', temp_path, found_count)
    else:
        return ('memory', results_buffer, found_count)


def _thread_worker_find_groups_4(args):
    """Thread worker for finding quads using a shared tree."""
    chunk_start, chunk_end, places, tree, radius, temp_dir, worker_id, progress_counter, found_counter = args
    
    r2 = radius * radius
    local_count = 0
    found_count = 0
    
    temp_path = None
    temp_file = None
    if temp_dir:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', prefix=f't{worker_id}_', dir=temp_dir)
        temp_file = os.fdopen(temp_fd, 'w')
    
    results_buffer = []
    buffer_flush_size = 10000
    
    try:
        for i in range(chunk_start, chunk_end):
            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = sorted([idx for idx in neighbors if idx > i])
            L = len(neigh_indices)
            
            for a_idx in range(L):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, L):
                    k = neigh_indices[b_idx]
                    for c_idx in range(b_idx + 1, L):
                        m = neigh_indices[c_idx]
                        coords = [(int(places[i][0]), int(places[i][1])),
                                  (int(places[j][0]), int(places[j][1])),
                                  (int(places[k][0]), int(places[k][1])),
                                  (int(places[m][0]), int(places[m][1]))]
                        if check_group(coords, r2):
                            spawn_dist = compute_spawn_distance(coords)
                            pairwise_dist = compute_pairwise_distance(coords)
                            found_count += 1
                            
                            if temp_file:
                                results_buffer.append(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                                if len(results_buffer) >= buffer_flush_size:
                                    temp_file.writelines(results_buffer)
                                    results_buffer.clear()
                            else:
                                results_buffer.append((coords, spawn_dist, pairwise_dist))
            
            local_count += 1
            if local_count % 50000 == 0 and progress_counter is not None:
                with progress_counter.get_lock():
                    progress_counter.value += 50000
        
        if temp_file and results_buffer:
            temp_file.writelines(results_buffer)
            results_buffer.clear()
    finally:
        if temp_file:
            temp_file.close()
    
    remaining = local_count % 50000
    if remaining > 0 and progress_counter is not None:
        with progress_counter.get_lock():
            progress_counter.value += remaining
    
    if found_counter is not None:
        with found_counter.get_lock():
            found_counter.value += found_count
    
    if temp_path:
        return ('file', temp_path, found_count)
    else:
        return ('memory', results_buffer, found_count)


def find_groups_3_threaded(places, tree, radius, output_path, num_threads=None):
    """
    Threaded version using a SINGLE SHARED TREE.
    
    This is much more memory-efficient than the multiprocessing version because:
    - Only ONE tree exists in memory (not one per worker)
    - cKDTree.query_ball_point releases the GIL, allowing true parallelism
    
    Use this mode when memory is limited but you still want parallelism.
    """
    if num_threads is None:
        num_threads = max(1, mp.cpu_count() - 1)
    
    total_places = len(places)
    chunk_size = max(1, total_places // (num_threads * 4))
    
    # Create work chunks - note we pass the SHARED tree object
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix='finder_triplets_threaded_')
    
    # Use multiprocessing Values for thread-safe counters
    progress_counter = mp.Value('q', 0)
    found_counter = mp.Value('q', 0)
    
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, places, tree, radius, temp_dir, len(chunks), progress_counter, found_counter))
    
    print(f"Starting THREADED search for triplets with {num_threads} threads, {len(chunks)} chunks")
    print(f"  (Using SINGLE SHARED TREE - memory efficient mode)")
    print(f"Using temp directory: {temp_dir}")
    print_memory_status("Before search: ")
    
    # Start progress monitor
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(progress_counter, found_counter, total_places, stop_event, "triplets")
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    temp_files = []
    
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_thread_worker_find_groups_3, chunk) for chunk in chunks]
            for future in futures:
                result_info = future.result()
                result_type, data, found = result_info
                if result_type == 'file':
                    temp_files.append(data)
                else:
                    if data:
                        fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
                        with os.fdopen(fd, 'w') as f:
                            for coords, spawn_dist, pairwise_dist in data:
                                f.write(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                        temp_files.append(temp_path)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    print_memory_status("After search: ")
    
    # Merge and deduplicate
    _merge_temp_files_and_dedup(temp_files, output_path, "triplets")
    
    try:
        os.rmdir(temp_dir)
    except:
        pass


def find_groups_4_threaded(places, tree, radius, output_path, num_threads=None):
    """
    Threaded version for quads using a SINGLE SHARED TREE.
    
    Memory efficient: only ONE tree in memory instead of one per worker.
    """
    if num_threads is None:
        num_threads = max(1, mp.cpu_count() - 1)
    
    total_places = len(places)
    chunk_size = max(1, total_places // (num_threads * 4))
    
    chunks = []
    temp_dir = tempfile.mkdtemp(prefix='finder_quads_threaded_')
    
    progress_counter = mp.Value('q', 0)
    found_counter = mp.Value('q', 0)
    
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, places, tree, radius, temp_dir, len(chunks), progress_counter, found_counter))
    
    print(f"Starting THREADED search for quads with {num_threads} threads, {len(chunks)} chunks")
    print(f"  (Using SINGLE SHARED TREE - memory efficient mode)")
    print(f"Using temp directory: {temp_dir}")
    print_memory_status("Before search: ")
    
    stop_event = threading.Event()
    monitor_thread = threading.Thread(
        target=_progress_monitor,
        args=(progress_counter, found_counter, total_places, stop_event, "quads")
    )
    monitor_thread.daemon = True
    monitor_thread.start()
    
    temp_files = []
    
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_thread_worker_find_groups_4, chunk) for chunk in chunks]
            for future in futures:
                result_info = future.result()
                result_type, data, found = result_info
                if result_type == 'file':
                    temp_files.append(data)
                else:
                    if data:
                        fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=temp_dir)
                        with os.fdopen(fd, 'w') as f:
                            for coords, spawn_dist, pairwise_dist in data:
                                f.write(_serialize_result_to_line(coords, spawn_dist, pairwise_dist))
                        temp_files.append(temp_path)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    print_memory_status("After search: ")
    
    _merge_temp_files_and_dedup(temp_files, output_path, "quads")
    
    try:
        os.rmdir(temp_dir)
    except:
        pass


# ============== INTERACTIVE CONFIG ==============

def prompt_input(prompt, default, type_converter=str):
    """Prompt user for input with a default value. Press Enter to use default."""
    user_input = input(f"{prompt} [{default}]: ").strip()
    if user_input == "":
        return default
    try:
        return type_converter(user_input)
    except ValueError:
        print(f"Invalid input, using default: {default}")
        return default


def prompt_yes_no(prompt, default=False):
    """Prompt user for yes/no with a default value."""
    default_str = "Y/n" if default else "y/N"
    user_input = input(f"{prompt} [{default_str}]: ").strip().lower()
    if user_input == "":
        return default
    return user_input in ("y", "yes", "1", "true")


def interactive_config():
    """Interactively configure settings with defaults."""
    print("=" * 60)
    print("  Hut/Monument Group Finder - Configuration")
    print("  Press Enter to accept default values shown in [brackets]")
    print("=" * 60)
    print()
    
    # Show memory status
    print_memory_status("System ")
    print()
    
    # Basic settings
    print("--- Input/Output Settings ---")
    input_file = prompt_input("Input file path", "allhuts.txt")
    memmap_file = prompt_input("Memmap cache file path", "places.memmap")
    radius = prompt_input("Search radius", 200, int)
    
    print()
    print("--- Search Options ---")
    find_triplets = prompt_yes_no("Find groups of 3 (triplets)?", True)
    find_quads = prompt_yes_no("Find groups of 4 (quads)?", True)
    
    out3 = None
    out4 = None
    if find_triplets:
        out3 = prompt_input("Output base name for triplets", "output3Mon.txt")
        spawn_path, spread_path = get_output_paths(out3)
        print(f"    -> Will create: {spawn_path} and {spread_path}")
    if find_quads:
        out4 = prompt_input("Output base name for quads", "output4Mon.txt")
        spawn_path, spread_path = get_output_paths(out4)
        print(f"    -> Will create: {spawn_path} and {spread_path}")
    
    print()
    print("--- Performance Settings ---")
    cpu_count = mp.cpu_count()
    total_mem_gb = get_total_memory_gb()
    available_mem_gb = get_available_memory_gb()
    default_workers = max(1, cpu_count - 1)
    print(f"CPU cores available: {cpu_count}")
    print(f"Memory: {available_mem_gb:.1f}GB available / {total_mem_gb:.1f}GB total")
    
    use_parallel = prompt_yes_no("Use parallel processing (multi-core)?", True)
    
    num_workers = default_workers
    max_memory = None
    use_shared_tree = False
    
    if use_parallel:
        print()
        print("  Parallel mode options:")
        print("    - MULTIPROCESS: Each worker builds its own tree (faster, uses more memory)")
        print("    - SHARED TREE:  All threads share ONE tree (slower, uses much less memory)")
        print()
        use_shared_tree = prompt_yes_no("Use shared tree mode (RECOMMENDED for large datasets)?", True)
        
        if use_shared_tree:
            num_workers = prompt_input(f"Number of threads", default_workers, int)
            num_workers = max(1, min(num_workers, cpu_count * 2))
            print("  Note: Shared tree mode uses 1 tree total, regardless of thread count")
        else:
            num_workers = prompt_input(f"Number of worker processes (0 = auto based on memory)", 0, int)
            if num_workers == 0:
                num_workers = None  # Will be auto-calculated
            else:
                num_workers = max(1, min(num_workers, cpu_count * 2))
            
            print()
            print("  Memory limit controls how much RAM the program will use.")
            print("  Set to 0 for auto (uses available memory minus safety reserve).")
            print(f"  Current available: {available_mem_gb:.1f}GB")
            max_memory = prompt_input("Max memory usage in GB (0 = auto)", 0, float)
            if max_memory <= 0:
                max_memory = None
    
    print("  (Leafsize 0 = auto-calculate based on dataset size and memory)")
    leafsize = prompt_input("KDTree leafsize (0 = auto, higher = less memory)", 0, int)
    
    print()
    print("=" * 60)
    print("  Configuration Summary:")
    print(f"    Input file:      {input_file}")
    print(f"    Radius:          {radius}")
    if find_triplets:
        s, p = get_output_paths(out3)
        print(f"    Triplets:        {s}, {p}")
    else:
        print(f"    Triplets:        skipped")
    if find_quads:
        s, p = get_output_paths(out4)
        print(f"    Quads:           {s}, {p}")
    else:
        print(f"    Quads:           skipped")
    
    if use_parallel:
        if use_shared_tree:
            print(f"    Mode:            SHARED TREE (1 tree, {num_workers} threads)")
        else:
            workers_str = "auto (memory-safe)" if num_workers is None else str(num_workers)
            print(f"    Mode:            MULTIPROCESS ({workers_str} workers, 1 tree each)")
            print(f"    Max memory:      {'auto' if max_memory is None else f'{max_memory:.1f}GB'}")
    else:
        print(f"    Mode:            Single-threaded")
    print(f"    Leafsize:        {'auto' if leafsize == 0 else leafsize}")
    print("  Note: Results are always deduplicated and output to two files")
    print("        (one sorted by spawn distance, one by group spread)")
    print("=" * 60)
    print()
    
    confirm = prompt_yes_no("Proceed with these settings?", True)
    if not confirm:
        print("Aborted by user.")
        sys.exit(0)
    
    return {
        'input': input_file,
        'memmap': memmap_file,
        'radius': radius,
        'out3': out3,
        'out4': out4,
        'use_parallel': use_parallel,
        'use_shared_tree': use_shared_tree,
        'num_workers': num_workers,
        'leafsize': leafsize,
        'max_memory': max_memory,
    }


def format_duration(seconds):
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.2f} seconds"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.2f}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = seconds % 60
        return f"{hours}h {minutes}m {secs:.2f}s"


def main():
    start_time = time.time()
    
    parser = argparse.ArgumentParser(
        description="Find groups of huts/monuments within a radius. "
                    "Results are always deduplicated and output to two files per group type "
                    "(one sorted by spawn distance, one by group spread)."
    )
    parser.add_argument("--input", default=None, help="Path to input file (text)")
    parser.add_argument("--memmap", default=None, help="Path to memmap file (will be created)")
    parser.add_argument("--radius", type=int, default=None, help="Search radius")
    parser.add_argument("--leafsize", type=int, default=None, help="KDTree leafsize (0 = auto based on dataset size)")
    parser.add_argument("--out3", default=None, help="Base output name for groups of 3 (creates _spawn and _spread files)")
    parser.add_argument("--out4", default=None, help="Base output name for groups of 4 (creates _spawn and _spread files)")
    parser.add_argument("--skip3", action="store_true", help="Skip groups of 3")
    parser.add_argument("--skip4", action="store_true", help="Skip groups of 4")
    parser.add_argument("--workers", type=int, default=None, help="Number of workers/threads (0 = auto)")
    parser.add_argument("--max-memory", type=float, default=None, 
                        help="Maximum memory to use in GB (default: auto, uses available minus reserve)")
    parser.add_argument("--shared-tree", action="store_true", 
                        help="Use shared tree mode: 1 tree shared by all threads (MUCH lower memory, recommended for large datasets)")
    parser.add_argument("--single-threaded", action="store_true", help="Disable parallelism, use single-threaded code")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive prompts, use defaults")
    args = parser.parse_args()
    
    # Check if any arguments were provided via command line
    has_cli_args = any([
        args.input is not None,
        args.memmap is not None,
        args.radius is not None,
        args.leafsize is not None,
        args.out3 is not None,
        args.out4 is not None,
        args.skip3,
        args.skip4,
        args.workers is not None,
        args.max_memory is not None,
        args.shared_tree,
        args.single_threaded,
        args.no_interactive,
    ])
    
    # Interactive mode if no CLI args provided
    if not has_cli_args and not args.no_interactive:
        config = interactive_config()
        input_file = config['input']
        memmap_file = config['memmap']
        radius = config['radius']
        out3_path = config['out3']
        out4_path = config['out4']
        use_parallel = config['use_parallel']
        use_shared_tree = config['use_shared_tree']
        num_workers = config['num_workers']
        leafsize = config['leafsize']
        max_memory_gb = config['max_memory']
    else:
        # CLI mode - use provided args or defaults
        input_file = args.input or "allhuts.txt"
        memmap_file = args.memmap or "places.memmap"
        radius = args.radius or 200
        leafsize = args.leafsize if args.leafsize is not None else 0  # 0 = auto
        out3_path = None if args.skip3 else (args.out3 or "output3Mon.txt")
        out4_path = None if args.skip4 else (args.out4 or "output4Mon.txt")
        use_parallel = not args.single_threaded
        use_shared_tree = args.shared_tree
        max_memory_gb = args.max_memory if args.max_memory and args.max_memory > 0 else None
        
        # Workers: None means auto-calculate based on memory (for multiprocess) or cpu count (for shared tree)
        if args.workers is not None and args.workers > 0:
            num_workers = args.workers
        else:
            num_workers = None  # Will be auto-calculated
        
        print_memory_status("System ")
        print(f"CPU cores available: {mp.cpu_count()}")
        if use_parallel:
            if use_shared_tree:
                print("Mode: SHARED TREE (memory efficient - single tree, multiple threads)")
                if num_workers:
                    print(f"Using {num_workers} threads")
                else:
                    print("Threads will be set to CPU count - 1")
            else:
                print("Mode: MULTIPROCESS (each worker builds its own tree)")
                if num_workers:
                    print(f"Requested {num_workers} worker processes")
                else:
                    print("Workers will be auto-calculated based on available memory")
                if max_memory_gb:
                    print(f"Max memory limit: {max_memory_gb:.1f}GB")
        else:
            print("Running in single-threaded mode")
        
        # Show output files that will be created
        if out3_path:
            s, p = get_output_paths(out3_path)
            print(f"Triplet outputs: {s}, {p}")
        if out4_path:
            s, p = get_output_paths(out4_path)
            print(f"Quad outputs: {s}, {p}")

    places = parse_to_memmap(input_file, memmap_file)
    
    # Auto-calculate leafsize if set to 0
    if leafsize <= 0:
        leafsize = compute_auto_leafsize(len(places))
        print(f"Auto leafsize: {leafsize} (based on {len(places):,} places)")
    
    # Calculate memory estimates and worker counts
    if use_parallel:
        if use_shared_tree:
            # Shared tree mode: only 1 tree, so we can use more threads
            tree_mem_gb = estimate_kdtree_memory_gb(len(places), leafsize)
            print(f"Estimated tree memory: {tree_mem_gb:.1f}GB (shared by all threads)")
            if num_workers is None:
                num_workers = max(1, mp.cpu_count() - 1)
            print(f"Using {num_workers} threads with shared tree")
        else:
            # Multiprocess mode: calculate safe workers
            safe_workers, mem_per_worker = compute_memory_safe_workers(len(places), leafsize, max_memory_gb)
            print(f"Memory-safe workers: {safe_workers} (estimated {mem_per_worker:.1f}GB per worker)")
            if num_workers is None:
                num_workers = min(safe_workers, max(1, mp.cpu_count() - 1))
                print(f"Using {num_workers} workers")

    # Clean up old output files
    if out3_path:
        spawn_path, spread_path = get_output_paths(out3_path)
        if os.path.exists(spawn_path):
            os.remove(spawn_path)
        if os.path.exists(spread_path):
            os.remove(spread_path)
    if out4_path:
        spawn_path, spread_path = get_output_paths(out4_path)
        if os.path.exists(spawn_path):
            os.remove(spawn_path)
        if os.path.exists(spread_path):
            os.remove(spread_path)

    if not use_parallel:
        # Single-threaded mode
        tree = build_tree(places, leafsize)
        if out3_path:
            find_groups_3_global(places, tree, radius, out3_path)
        if out4_path:
            find_groups_4_global(places, tree, radius, out4_path)
    elif use_shared_tree:
        # Shared tree mode (threaded) - build ONE tree, share among threads
        tree = build_tree(places, leafsize)
        print_memory_status("After tree build: ")
        if out3_path:
            find_groups_3_threaded(places, tree, radius, out3_path, num_workers)
        if out4_path:
            find_groups_4_threaded(places, tree, radius, out4_path, num_workers)
    else:
        # Multiprocess mode - each worker builds its own tree
        if out3_path:
            find_groups_3_global_parallel(places, memmap_file, radius, out3_path, leafsize, num_workers, max_memory_gb)
        if out4_path:
            find_groups_4_global_parallel(places, memmap_file, radius, out4_path, leafsize, num_workers, max_memory_gb)

    # Print total execution time
    elapsed_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Completed in {format_duration(elapsed_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
