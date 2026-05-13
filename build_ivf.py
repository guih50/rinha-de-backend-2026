#!/usr/bin/env python3
"""
Build a new IVF index with configurable nlist using sklearn MiniBatchKMeans.
Generates ivf.bin in the same binary format as build.rs.

Usage: python3 build_ivf.py [--nlist N] [--iters I] [--output path/to/ivf.bin]
"""
import sys, gzip, json, struct, time, argparse
import numpy as np

SCALE   = 16000.0
DIM     = 14
DIM_PAD = 16
MAGIC   = 0x524E484132303236

def quantize(x: float) -> int:
    return max(-32768, min(32767, round(x * SCALE)))

def load_references(gz_path: str):
    t0 = time.time()
    print(f"Loading {gz_path}...", flush=True)
    with gzip.open(gz_path, 'rt') as f:
        refs = json.load(f)
    n = len(refs)
    print(f"  {n:,} entries loaded in {time.time()-t0:.1f}s", flush=True)

    t1 = time.time()
    print("Quantizing to int16...", end="", flush=True)
    vectors = np.zeros((n, DIM_PAD), dtype=np.int16)
    labels  = np.zeros(n, dtype=np.uint8)
    for i, ref in enumerate(refs):
        for j, x in enumerate(ref['vector'][:DIM]):
            vectors[i, j] = quantize(x)
        labels[i] = 1 if ref['label'] == 'fraud' else 0
        if i % 500_000 == 0 and i > 0:
            print(f" {i//1000}k", end="", flush=True)
    print(f" done ({time.time()-t1:.1f}s)", flush=True)
    return vectors, labels

def kmeans_sklearn(vectors_i16: np.ndarray, k: int, iters: int):
    from sklearn.cluster import MiniBatchKMeans
    print(f"MiniBatchKMeans(k={k}, max_iter={iters})...", flush=True)
    t0 = time.time()
    km = MiniBatchKMeans(
        n_clusters=k,
        max_iter=iters,
        batch_size=min(50_000, len(vectors_i16)),
        n_init=3,
        random_state=42,
        verbose=0,
    )
    km.fit(vectors_i16.astype(np.float32))
    print(f"  done in {time.time()-t0:.1f}s", flush=True)

    # Quantize centroids to int16 (matching build.rs behavior)
    centroids_f = km.cluster_centers_   # (k, DIM_PAD) float32
    centroids   = np.zeros((k, DIM_PAD), dtype=np.int16)
    for i, c in enumerate(centroids_f):
        for j, x in enumerate(c):
            centroids[i, j] = max(-32768, min(32767, round(x)))

    # Get assignments for all vectors
    print("Assigning vectors to clusters...", end="", flush=True)
    t1 = time.time()
    assignments = km.predict(vectors_i16.astype(np.float32))   # (n,) int
    print(f" {time.time()-t1:.1f}s", flush=True)

    return centroids, assignments

def write_ivf(out_path: str, centroids_i16: np.ndarray, vectors_i16: np.ndarray,
              labels_u8: np.ndarray, assignments: np.ndarray):
    nlist, _ = centroids_i16.shape
    n        = len(vectors_i16)
    t0 = time.time()
    print(f"Writing {out_path} (nlist={nlist}, n={n:,})...", flush=True)

    # Sort vectors by cluster
    order = np.argsort(assignments, kind='stable')
    sorted_vecs   = vectors_i16[order]
    sorted_labels = labels_u8[order]
    sorted_asgn   = assignments[order]

    # Compute cluster offsets
    counts  = np.bincount(sorted_asgn, minlength=nlist).astype(np.uint32)
    offsets = np.zeros(nlist + 1, dtype=np.uint32)
    offsets[1:] = np.cumsum(counts)

    with open(out_path, 'wb') as f:
        # Header (32 bytes)
        f.write(struct.pack('<Q', MAGIC))     # magic  8B
        f.write(struct.pack('<I', nlist))     # nlist  4B
        f.write(struct.pack('<I', 50))        # unused 4B
        f.write(struct.pack('<I', n))         # n_vec  4B
        f.write(struct.pack('<I', DIM_PAD))   # dim    4B
        f.write(b'\x00' * 8)                 # pad    8B

        # Centroids
        f.write(centroids_i16.tobytes())

        # Offsets
        f.write(offsets.tobytes())

        # Sorted vectors
        f.write(sorted_vecs.tobytes())

        # Sorted labels
        f.write(sorted_labels.tobytes())

    size_mb = sum([
        32,
        nlist * 32,
        (nlist + 1) * 4,
        n * 32,
        n,
    ]) / 1e6
    print(f"  Written {size_mb:.1f} MB in {time.time()-t0:.1f}s", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nlist',  type=int, default=4096)
    parser.add_argument('--iters',  type=int, default=100)
    parser.add_argument('--input',  default='resources/references.json.gz')
    parser.add_argument('--output', default='ivf_4096.bin')
    args = parser.parse_args()

    t_total = time.time()
    vectors, labels = load_references(args.input)
    centroids, assignments = kmeans_sklearn(vectors, args.nlist, args.iters)
    write_ivf(args.output, centroids, vectors, labels, assignments)

    total = time.time() - t_total
    print(f"\nTotal: {total:.1f}s  →  {args.output}")
    print("Run recall_check.py with this index:")
    print(f"  python3 recall_check.py {args.output}")

if __name__ == '__main__':
    main()
