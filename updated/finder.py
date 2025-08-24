import argparse
import math
import os
import sys
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree

def check_group(coords, r2):
    for i in range(len(coords)):
        for j in range(i+1, len(coords)):
            dx = coords[i][0] - coords[j][0]
            dz = coords[i][1] - coords[j][1]
            if (dx*dx + dz*dz) > r2:
                return False
    return True

#function to find center coordinates of a group
def find_center(coords):
    x = 0
    z = 0
    for i in range(len(coords)):
        x += coords[i][0]
        z += coords[i][1]
    x /= len(coords)
    z /= len(coords)
    print (x, z)
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

    # Preallocate a memory-mapped array on disk to avoid large Python lists
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

    # Flush content to disk and reopen in read-only mode for downstream use
    places_mm.flush()
    del places_mm
    places = np.memmap(memmap_path, dtype=np.int32, mode='r', shape=(total_records, 2))
    print("Parsed file")
    print(f"Found {len(places)} places")
    return places



def build_tree(places, leafsize):
    # Build a kd-tree from the (x, z) coordinates of the places
    print("Building tree... this may take a while")
    # cKDTree may copy and cast to float64 internally; copy_data=False avoids an extra copy
    tree = cKDTree(places, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)
    print("Built tree")
    return tree


def compute_bounds(places):
    # places is memmap int32 Nx2
    min_x = int(np.min(places[:, 0]))
    max_x = int(np.max(places[:, 0]))
    min_z = int(np.min(places[:, 1]))
    max_z = int(np.max(places[:, 1]))
    return min_x, max_x, min_z, max_z


def tile_index_for_coord(x, z, min_x, min_z, tile_size):
    tx = (x - min_x) // tile_size
    tz = (z - min_z) // tile_size
    return int(tx), int(tz)


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


class FileHandleCache:
    def __init__(self, capacity=128):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, path):
        fh = self.cache.get(path)
        if fh is not None:
            self.cache.move_to_end(path)
            return fh
        # open new
        fh = open(path, 'a')
        self.cache[path] = fh
        if len(self.cache) > self.capacity:
            old_path, old_fh = self.cache.popitem(last=False)
            try:
                old_fh.close()
            except Exception:
                pass
        return fh

    def close_all(self):
        for _, fh in list(self.cache.items()):
            try:
                fh.close()
            except Exception:
                pass
        self.cache.clear()


def build_tile_files(places, tile_dir, tile_size):
    ensure_dir(tile_dir)
    # Clean existing
    for name in os.listdir(tile_dir):
        if name.startswith('tile_') and name.endswith('.txt'):
            os.remove(os.path.join(tile_dir, name))

    min_x, max_x, min_z, max_z = compute_bounds(places)
    width = (max_x - min_x) + 1
    height = (max_z - min_z) + 1
    tiles_x = max(1, (width + tile_size - 1) // tile_size)
    tiles_z = max(1, (height + tile_size - 1) // tile_size)
    print(f"Tiling: {tiles_x} x {tiles_z} tiles (tile_size={tile_size})")

    fh_cache = FileHandleCache(128)
    try:
        for idx in range(len(places)):
            x = int(places[idx, 0])
            z = int(places[idx, 1])
            tx, tz = tile_index_for_coord(x, z, min_x, min_z, tile_size)
            path = os.path.join(tile_dir, f"tile_{tx}_{tz}.txt")
            fh = fh_cache.get(path)
            # Write: global_index,x,z (CSV)
            fh.write(f"{idx},{x},{z}\n")
            if idx % 5_000_000 == 0 and idx > 0:
                print(f"Tiling progress: {idx}/{len(places)}")
    finally:
        fh_cache.close_all()

    return min_x, min_z, tiles_x, tiles_z


 


def find_groups_3_global(places, tree, radius, output_path):
    r2 = radius * radius
    found = 0
    totalPlaces = len(places)
    with open(output_path, "a") as f:
        for i in range(totalPlaces):
            if i % 100000 == 0 and i > 0:
                percentage = (i / totalPlaces) * 100
                print(f"{percentage:.2f}% searched - Found {found} groups")

            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = [idx for idx in neighbors if idx > i]
            neigh_indices.sort()
            for a_idx in range(len(neigh_indices)):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, len(neigh_indices)):
                    k = neigh_indices[b_idx]
                    if check_group([places[i], places[j], places[k]], r2):
                        dist = (
                            (places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                            (places[j][0] ** 2 + places[j][1] ** 2) ** 0.5 +
                            (places[k][0] ** 2 + places[k][1] ** 2) ** 0.5
                        )
                        f.write(f"{([places[i], places[j], places[k]], dist)}\n")
                        found += 1

def find_groups_4_global(places, tree, radius, output_path):
    r2 = radius * radius
    found = 0
    totalPlaces = len(places)
    with open(output_path, "a") as f:
        for i in range(totalPlaces):
            if i % 100000 == 0 and i > 0:
                percentage = (i / totalPlaces) * 100
                print(f"{percentage:.2f}% searched - Found {found} groups")

            neighbors = tree.query_ball_point(places[i], r=radius)
            neigh_indices = [idx for idx in neighbors if idx > i]
            neigh_indices.sort()
            L = len(neigh_indices)
            for a_idx in range(L):
                j = neigh_indices[a_idx]
                for b_idx in range(a_idx + 1, L):
                    k = neigh_indices[b_idx]
                    for c_idx in range(b_idx + 1, L):
                        l = neigh_indices[c_idx]
                        if check_group([places[i], places[j], places[k], places[l]], r2):
                            dist = (
                                (places[i][0] ** 2 + places[i][1] ** 2) ** 0.5 +
                                (places[j][0] ** 2 + places[j][1] ** 2) ** 0.5 +
                                (places[k][0] ** 2 + places[k][1] ** 2) ** 0.5 +
                                (places[l][0] ** 2 + places[l][1] ** 2) ** 0.5
                            )
                            f.write(f"{([places[i], places[j], places[k], places[l]], dist)}\n")
                            found += 1


def load_tile_file(path):
    idxs = []
    coords = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            gi_str, x_str, z_str = line.split(',')
            idxs.append(int(gi_str))
            coords.append((int(x_str), int(z_str)))
    if not idxs:
        return np.empty((0,), dtype=np.int64), np.empty((0, 2), dtype=np.int32)
    return np.array(idxs, dtype=np.int64), np.array(coords, dtype=np.int32)


def find_groups_tiled(places, tile_dir, min_x, min_z, tiles_x, tiles_z, tile_size, radius, out3, out4, leafsize):
    r2 = radius * radius
    radius_tiles = max(1, (radius + tile_size - 1) // tile_size)

    for tx in range(tiles_x):
        for tz in range(tiles_z):
            base_path = os.path.join(tile_dir, f"tile_{tx}_{tz}.txt")
            if not os.path.exists(base_path):
                continue
            base_idx, base_coords = load_tile_file(base_path)
            if base_idx.size == 0:
                continue

            # Load neighbor tiles within halo
            neighbor_idxs = []
            neighbor_coords = []
            # Read base first so their positions are [0, len(base))
            neighbor_idxs.append(base_idx)
            neighbor_coords.append(base_coords)

            for ntx in range(max(0, tx - radius_tiles), min(tiles_x, tx + radius_tiles + 1)):
                for ntz in range(max(0, tz - radius_tiles), min(tiles_z, tz + radius_tiles + 1)):
                    if ntx == tx and ntz == tz:
                        continue
                    npath = os.path.join(tile_dir, f"tile_{ntx}_{ntz}.txt")
                    if not os.path.exists(npath):
                        continue
                    gi, cr = load_tile_file(npath)
                    if gi.size:
                        neighbor_idxs.append(gi)
                        neighbor_coords.append(cr)

            all_idx = np.concatenate(neighbor_idxs, axis=0)
            all_coords = np.vstack(neighbor_coords).astype(np.int32, copy=False)

            # Build KDTree locally for neighbors
            tree = cKDTree(all_coords, leafsize=leafsize, compact_nodes=True, balanced_tree=False, copy_data=False)

            # Base positions are the first len(base_idx) rows
            base_count = base_idx.shape[0]

            # Triplets
            if out3 is not None:
                with open(out3, 'a') as f3:
                    for i_pos in range(base_count):
                        i_coord = all_coords[i_pos]
                        i_gid = all_idx[i_pos]
                        neigh = tree.query_ball_point(i_coord, r=radius)
                        # neighbors that are after i_gid to avoid duplicates
                        neigh = [p for p in neigh if p != i_pos and all_idx[p] > i_gid]
                        neigh.sort(key=lambda p: all_idx[p])
                        for a in range(len(neigh)):
                            j_pos = neigh[a]
                            for b in range(a + 1, len(neigh)):
                                k_pos = neigh[b]
                                if check_group([all_coords[i_pos], all_coords[j_pos], all_coords[k_pos]], r2):
                                    dist = (
                                        (all_coords[i_pos][0] ** 2 + all_coords[i_pos][1] ** 2) ** 0.5 +
                                        (all_coords[j_pos][0] ** 2 + all_coords[j_pos][1] ** 2) ** 0.5 +
                                        (all_coords[k_pos][0] ** 2 + all_coords[k_pos][1] ** 2) ** 0.5
                                    )
                                    f3.write(f"{([all_coords[i_pos], all_coords[j_pos], all_coords[k_pos]], dist)}\n")

            # Quads
            if out4 is not None:
                with open(out4, 'a') as f4:
                    for i_pos in range(base_count):
                        i_coord = all_coords[i_pos]
                        i_gid = all_idx[i_pos]
                        neigh = tree.query_ball_point(i_coord, r=radius)
                        neigh = [p for p in neigh if p != i_pos and all_idx[p] > i_gid]
                        neigh.sort(key=lambda p: all_idx[p])
                        L = len(neigh)
                        for a in range(L):
                            j_pos = neigh[a]
                            for b in range(a + 1, L):
                                k_pos = neigh[b]
                                for c in range(b + 1, L):
                                    l_pos = neigh[c]
                                    if check_group([all_coords[i_pos], all_coords[j_pos], all_coords[k_pos], all_coords[l_pos]], r2):
                                        dist = (
                                            (all_coords[i_pos][0] ** 2 + all_coords[i_pos][1] ** 2) ** 0.5 +
                                            (all_coords[j_pos][0] ** 2 + all_coords[j_pos][1] ** 2) ** 0.5 +
                                            (all_coords[k_pos][0] ** 2 + all_coords[k_pos][1] ** 2) ** 0.5 +
                                            (all_coords[l_pos][0] ** 2 + all_coords[l_pos][1] ** 2) ** 0.5
                                        )
                                        f4.write(f"{([all_coords[i_pos], all_coords[j_pos], all_coords[k_pos], all_coords[l_pos]], dist)}\n")


def main():
    parser = argparse.ArgumentParser(description="Find groups of huts/monuments within a radius")
    parser.add_argument("--input", default="allhuts.txt", help="Path to input file (text)")
    parser.add_argument("--memmap", default="places.memmap", help="Path to memmap file (will be created)")
    parser.add_argument("--radius", type=int, default=200, help="Search radius")
    parser.add_argument("--leafsize", type=int, default=64, help="KDTree leafsize (memory/time tradeoff)")
    parser.add_argument("--out3", default="output3Mon.txt", help="Output text for groups of 3")
    parser.add_argument("--out4", default="output4Mon.txt", help="Output text for groups of 4")
    parser.add_argument("--skip3", action="store_true", help="Skip groups of 3")
    parser.add_argument("--skip4", action="store_true", help="Skip groups of 4")
    parser.add_argument("--use_tiling", action="store_true", help="Enable spatial tiling to bound KDTree memory")
    parser.add_argument("--tile_size", type=int, default=0, help="Tile size in coordinate units (default: 2*radius)")
    parser.add_argument("--tile_dir", default="tiles", help="Directory for tile spill files")
    args = parser.parse_args()

    places = parse_to_memmap(args.input, args.memmap)

    # Truncate outputs if exist to avoid appending previous runs
    out3_path = None if args.skip3 else args.out3
    out4_path = None if args.skip4 else args.out4
    if out3_path and os.path.exists(out3_path):
        os.remove(out3_path)
    if out4_path and os.path.exists(out4_path):
        os.remove(out4_path)

    if args.use_tiling:
        tile_size = args.tile_size if args.tile_size > 0 else max(1, 2 * args.radius)
        min_x, min_z, tiles_x, tiles_z = build_tile_files(places, args.tile_dir, tile_size)
        find_groups_tiled(
            places=places,
            tile_dir=args.tile_dir,
            min_x=min_x,
            min_z=min_z,
            tiles_x=tiles_x,
            tiles_z=tiles_z,
            tile_size=tile_size,
            radius=args.radius,
            out3=out3_path,
            out4=out4_path,
            leafsize=args.leafsize,
        )
    else:
        # cKDTree requires float input internally; pass as-is and let it view/cast without extra copy if possible
        tree = build_tree(places, args.leafsize)
        if out3_path:
            find_groups_3_global(places, tree, args.radius, out3_path)
        if out4_path:
            find_groups_4_global(places, tree, args.radius, out4_path)


if __name__ == "__main__":
    main()