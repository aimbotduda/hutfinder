import argparse
import math
import os
import sys
import threading
import time
import multiprocessing as mp
import numpy as np
from scipy.spatial import cKDTree

# Try to import numba for JIT compilation (optional but faster)
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


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


def parse_to_memmap(input_path, memmap_path):
    print("Reading and parsing file of all huts/monuments")
    total_lines = 0
    total_records = 0
    with open(input_path) as f:
        for line in f:
            total_lines += 1
            if "->(" in line:
                total_records += 1
    print(f"Lines: {total_lines}, records: {total_records}")

    places_mm = np.memmap(memmap_path, dtype=np.int32, mode='w+', shape=(total_records, 2))

    written = 0
    processed = 0
    with open(input_path) as f:
        for line in f:
            processed += 1
            if "->(" in line:
                coords = line.split("->(")[1].split(")reg")[0]
                x_str, z_str = coords.split(",")
                places_mm[written, 0] = int(x_str)
                places_mm[written, 1] = int(z_str)
                written += 1
                if written % 5_000_000 == 0:
                    percentage = (processed / max(total_lines, 1)) * 100
                    print(f"{percentage:.2f}% read - {written} records")

    places_mm.flush()
    del places_mm
    places = np.memmap(memmap_path, dtype=np.int32, mode='r', shape=(total_records, 2))
    print("Parsed file")
    print(f"Found {len(places)} places")
    return places


def build_tree(places, leafsize):
    print("Building tree... this may take a while")
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    print("Built tree")
    return tree


def compute_auto_leafsize(num_places):
    """
    Calculate optimal KDTree leafsize based on dataset size.
    
    Guidelines:
    - Smaller leafsize = faster queries but more memory (more tree nodes)
    - Larger leafsize = less memory but slower queries (more brute-force at leaves)
    
    The sweet spot depends on dataset size:
    - Small datasets (<100K): leafsize 16-32, fast queries matter more
    - Medium datasets (100K-10M): leafsize 32-64, balanced
    - Large datasets (10M-100M): leafsize 64-128, memory matters more
    - Very large (>100M): leafsize 128-256, memory critical
    """
    if num_places < 100_000:
        return 16
    elif num_places < 1_000_000:
        return 32
    elif num_places < 10_000_000:
        return 48
    elif num_places < 50_000_000:
        return 64
    elif num_places < 100_000_000:
        return 96
    elif num_places < 200_000_000:
        return 128
    else:
        return 192


# ============== PARALLEL WORKER FUNCTIONS ==============

# Global variables for worker processes (initialized by pool initializer)
_worker_progress_counter = None
_worker_found_counter = None


def _init_worker(progress_counter, found_counter):
    """Initialize worker process with shared counters."""
    global _worker_progress_counter, _worker_found_counter
    _worker_progress_counter = progress_counter
    _worker_found_counter = found_counter


def _worker_find_groups_3(args):
    """Worker function to find triplets in a chunk of indices."""
    global _worker_progress_counter, _worker_found_counter
    chunk_start, chunk_end, places_path, places_shape, radius, leafsize, worker_id = args
    
    # Load the memmap in this worker
    places = np.memmap(places_path, dtype=np.int32, mode='r', shape=places_shape)
    
    # Rebuild tree in worker (required for multiprocessing - can't pickle cKDTree)
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    
    r2 = radius * radius
    results = []
    local_count = 0
    
    for i in range(chunk_start, chunk_end):
        neighbors = tree.query_ball_point(places[i], r=radius)
        neigh_indices = sorted([idx for idx in neighbors if idx > i])
        
        for a_idx in range(len(neigh_indices)):
            j = neigh_indices[a_idx]
            for b_idx in range(a_idx + 1, len(neigh_indices)):
                k = neigh_indices[b_idx]
                # Convert to plain tuples for clean output
                coords = [(int(places[i][0]), int(places[i][1])),
                          (int(places[j][0]), int(places[j][1])),
                          (int(places[k][0]), int(places[k][1]))]
                if check_group(coords, r2):
                    spawn_dist = compute_spawn_distance(coords)
                    pairwise_dist = compute_pairwise_distance(coords)
                    results.append((coords, spawn_dist, pairwise_dist))
        
        # Update shared progress counter periodically
        local_count += 1
        if local_count % 50000 == 0 and _worker_progress_counter is not None:
            with _worker_progress_counter.get_lock():
                _worker_progress_counter.value += 50000
    
    # Update remaining progress
    remaining = local_count % 50000
    if remaining > 0 and _worker_progress_counter is not None:
        with _worker_progress_counter.get_lock():
            _worker_progress_counter.value += remaining
    
    # Update found counter with final count
    if _worker_found_counter is not None:
        with _worker_found_counter.get_lock():
            _worker_found_counter.value += len(results)
    
    return results, chunk_end - chunk_start, worker_id


def _worker_find_groups_4(args):
    """Worker function to find quads in a chunk of indices."""
    global _worker_progress_counter, _worker_found_counter
    chunk_start, chunk_end, places_path, places_shape, radius, leafsize, worker_id = args
    
    places = np.memmap(places_path, dtype=np.int32, mode='r', shape=places_shape)
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    
    r2 = radius * radius
    results = []
    local_count = 0
    
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
                    # Convert to plain tuples for clean output
                    coords = [(int(places[i][0]), int(places[i][1])),
                              (int(places[j][0]), int(places[j][1])),
                              (int(places[k][0]), int(places[k][1])),
                              (int(places[m][0]), int(places[m][1]))]
                    if check_group(coords, r2):
                        spawn_dist = compute_spawn_distance(coords)
                        pairwise_dist = compute_pairwise_distance(coords)
                        results.append((coords, spawn_dist, pairwise_dist))
        
        # Update shared progress counter periodically
        local_count += 1
        if local_count % 50000 == 0 and _worker_progress_counter is not None:
            with _worker_progress_counter.get_lock():
                _worker_progress_counter.value += 50000
    
    # Update remaining progress
    remaining = local_count % 50000
    if remaining > 0 and _worker_progress_counter is not None:
        with _worker_progress_counter.get_lock():
            _worker_progress_counter.value += remaining
    
    # Update found counter with final count
    if _worker_found_counter is not None:
        with _worker_found_counter.get_lock():
            _worker_found_counter.value += len(results)
    
    return results, chunk_end - chunk_start, worker_id


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


def find_groups_3_global_parallel(places, memmap_path, radius, output_path, leafsize, num_workers=None):
    """Parallel version of find_groups_3_global using multiple processes.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    total_places = len(places)
    chunk_size = max(1, total_places // (num_workers * 4))  # More chunks for better load balancing
    
    # Create list of work chunks
    chunks = []
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, memmap_path, places.shape, radius, leafsize, len(chunks)))
    
    print(f"Starting parallel search for triplets with {num_workers} workers, {len(chunks)} chunks")
    
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
    
    all_results = []
    
    try:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(progress_counter, found_counter)) as pool:
            for results, count, worker_id in pool.imap_unordered(_worker_find_groups_3, chunks):
                all_results.extend(results)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    # Deduplicate and write both sorted files
    dedup_and_write_results(all_results, output_path, "triplets")


def find_groups_4_global_parallel(places, memmap_path, radius, output_path, leafsize, num_workers=None):
    """Parallel version of find_groups_4_global using multiple processes.
    
    Results are always deduplicated and written to two files:
    - {output_path}_spawn.txt (sorted by distance from origin)
    - {output_path}_spread.txt (sorted by pairwise distance, tighter groups first)
    """
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)
    
    total_places = len(places)
    chunk_size = max(1, total_places // (num_workers * 4))
    
    chunks = []
    for i in range(0, total_places, chunk_size):
        end = min(i + chunk_size, total_places)
        chunks.append((i, end, memmap_path, places.shape, radius, leafsize, len(chunks)))
    
    print(f"Starting parallel search for quads with {num_workers} workers, {len(chunks)} chunks")
    
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
    
    all_results = []
    
    try:
        with mp.Pool(processes=num_workers, initializer=_init_worker, initargs=(progress_counter, found_counter)) as pool:
            for results, count, worker_id in pool.imap_unordered(_worker_find_groups_4, chunks):
                all_results.extend(results)
    finally:
        stop_event.set()
        monitor_thread.join(timeout=1.0)
    
    # Deduplicate and write both sorted files
    dedup_and_write_results(all_results, output_path, "quads")


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
    default_workers = max(1, cpu_count - 1)
    print(f"CPU cores available: {cpu_count}")
    
    use_parallel = prompt_yes_no("Use parallel processing (multi-core)?", True)
    
    num_workers = default_workers
    if use_parallel:
        num_workers = prompt_input(f"Number of worker processes", default_workers, int)
        num_workers = max(1, min(num_workers, cpu_count * 2))  # Sanity bounds
    
    print("  (Leafsize 0 = auto-calculate based on dataset size)")
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
    print(f"    Parallel:        {use_parallel}" + (f" ({num_workers} workers)" if use_parallel else ""))
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
        'num_workers': num_workers,
        'leafsize': leafsize,
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
    parser.add_argument("--workers", type=int, default=None, help="Number of worker processes (0 = auto)")
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
        num_workers = config['num_workers']
        leafsize = config['leafsize']
    else:
        # CLI mode - use provided args or defaults
        input_file = args.input or "allhuts.txt"
        memmap_file = args.memmap or "places.memmap"
        radius = args.radius or 200
        leafsize = args.leafsize if args.leafsize is not None else 0  # 0 = auto
        out3_path = None if args.skip3 else (args.out3 or "output3Mon.txt")
        out4_path = None if args.skip4 else (args.out4 or "output4Mon.txt")
        use_parallel = not args.single_threaded
        num_workers = args.workers if args.workers and args.workers > 0 else max(1, mp.cpu_count() - 1)
        
        print(f"CPU cores available: {mp.cpu_count()}")
        if use_parallel:
            print(f"Using {num_workers} worker processes")
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
        tree = build_tree(places, leafsize)
        if out3_path:
            find_groups_3_global(places, tree, radius, out3_path)
        if out4_path:
            find_groups_4_global(places, tree, radius, out4_path)
    else:
        # Parallel mode - tree is built per worker
        if out3_path:
            find_groups_3_global_parallel(places, memmap_file, radius, out3_path, leafsize, num_workers)
        if out4_path:
            find_groups_4_global_parallel(places, memmap_file, radius, out4_path, leafsize, num_workers)

    # Print total execution time
    elapsed_time = time.time() - start_time
    print()
    print("=" * 60)
    print(f"  Completed in {format_duration(elapsed_time)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
