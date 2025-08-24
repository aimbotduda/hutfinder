import argparse
import re
import math


def parse_args():
    p = argparse.ArgumentParser(description='Clean and canonicalize finder outputs')
    p.add_argument('--input', '-i', default='output3Mon.txt', help='Path to input .txt (from finder)')
    p.add_argument('--output', '-o', default='cleaned_output.txt', help='Path to output cleaned .txt')
    p.add_argument('--dedup', action='store_true', help='Remove duplicates in-memory (may use RAM)')
    p.add_argument('--sort', action='store_true', help='Sort by pairwise distance sum ascending (requires in-memory)')
    return p.parse_args()


def euclidean_distance(a, b):
    ax, ay = int(a[0]), int(a[1])
    bx, by = int(b[0]), int(b[1])
    return math.hypot(ax - bx, ay - by)


def clean_line(line):
    # Extract all [x, y] occurrences regardless of 'array(' wrappers
    coords = re.findall(r'\[\s*(-?\d+),\s*(-?\d+)\s*\]', line)
    if len(coords) < 3:
        return None
    # Extract the last float as the origin distance sum provided by finder
    floats = re.findall(r'[-+]?\d*\.\d+', line)
    origin_sum = floats[-1] if floats else '0.0'

    # Canonicalize coordinate order
    coords = sorted(coords, key=lambda xy: (int(xy[0]), int(xy[1])))

    # Compute pairwise distance sum for N points
    pair_sum = 0.0
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            pair_sum += euclidean_distance(coords[i], coords[j])

    # Prepare cleaned text line
    cleaned_coordinates = ' '.join([f'({int(x)}, {int(y)})' for x, y in coords])
    cleaned_line = f'{cleaned_coordinates} {origin_sum} {pair_sum:.2f}'

    # Signature for dedup: tuple of canonical coords
    signature = tuple((int(x), int(y)) for x, y in coords)
    return cleaned_line, pair_sum, signature


def main():
    args = parse_args()

    if args.dedup or args.sort:
        # In-memory path: collect for dedup/sort
        unique = {}
        with open(args.input, 'r') as infile:
            for raw in infile:
                parsed = clean_line(raw)
                if not parsed:
                    continue
                cleaned, pair_sum, sig = parsed
                if args.dedup:
                    # Keep the smallest pair_sum per signature
                    prev = unique.get(sig)
                    if prev is None or pair_sum < prev[0]:
                        unique[sig] = (pair_sum, cleaned)
                else:
                    # No dedup but still want to sort: use unique dict with unique key
                    unique[(sig, len(unique))] = (pair_sum, cleaned)

        items = list(unique.values())
        if args.sort:
            items.sort(key=lambda x: x[0])
        with open(args.output, 'w') as out:
            for _, cleaned in items:
                out.write(cleaned + '\n')
    else:
        # Streaming: canonicalize each line, write through; no dedup, no sort
        with open(args.input, 'r') as infile, open(args.output, 'w') as out:
            for raw in infile:
                parsed = clean_line(raw)
                if not parsed:
                    continue
                cleaned, _, _ = parsed
                out.write(cleaned + '\n')

    print(f"Wrote cleaned output to {args.output}")


if __name__ == '__main__':
    main()



