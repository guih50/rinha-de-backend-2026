#!/usr/bin/env python3
"""
Fast NPROBE sweep using numpy BLAS batch operations.
Vectorizes centroid and cluster distance steps across all queries at once.

Usage: python3 recall_check.py [ivf.bin] [test-data.json]
"""
import sys, json, struct, time
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────────────────
SCALE         = 16000.0
MAX_AMOUNT    = 10000.0
MAX_INSTALL   = 12.0
AMOUNT_RATIO  = 10.0
MAX_MINUTES   = 1440.0
MAX_KM        = 1000.0
MAX_TX        = 20.0
MAX_MERCH_AVG = 10000.0

def q(x: float) -> int:
    v = x * SCALE
    return max(-32768, min(32767, int(v)))

INSTALL_LUT = [q(min(i / MAX_INSTALL, 1.0)) for i in range(13)]
HOUR_LUT    = [q(i / 23.0) for i in range(24)]
DOW_LUT     = [q(i / 6.0) for i in range(7)]
TX_LUT      = [q(min(i / MAX_TX, 1.0)) for i in range(21)]

MCC_RISK = {
    "4511": 5600, "5999": 8000, "5411": 2400, "5944": 7200,
    "5311": 4000, "5812": 4800, "7802": 12000, "5912": 3200,
    "7995": 13600, "7801": 12800,
}

def parse_iso(ts: str):
    if not ts or len(ts) < 19:
        return 0, 0
    h = int(ts[11:13])
    y, m, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
    T = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4]
    yy = y - 1 if m < 3 else y
    dow_sun = (yy + yy//4 - yy//100 + yy//400 + T[m-1] + d) % 7
    dow = 6 if dow_sun == 0 else dow_sun - 1
    return h, dow

def iso_to_minutes(ts: str) -> float:
    if not ts or len(ts) < 19:
        return 0.0
    y, mo, d = int(ts[0:4]), int(ts[5:7]), int(ts[8:10])
    h, mi, s = int(ts[11:13]), int(ts[14:16]), int(ts[17:19])
    return y*525960.0 + mo*43800.0 + d*1440.0 + h*60.0 + mi + s/60.0

def vectorize(req: dict) -> np.ndarray:
    tx   = req.get("transaction", {})
    cust = req.get("customer", {})
    merch= req.get("merchant", {})
    term = req.get("terminal", {})
    last = req.get("last_transaction")

    amount      = float(tx.get("amount", 0))
    installments= int(tx.get("installments", 0))
    requested_at= tx.get("requested_at", "")
    hour, dow   = parse_iso(requested_at)

    cust_avg    = max(float(cust.get("avg_amount", 1.0)), 1e-9)
    tx_count    = int(cust.get("tx_count_24h", 0))
    known       = cust.get("known_merchants", [])
    merch_id    = merch.get("id", "")
    unknown_merch = merch_id not in known
    mcc         = merch.get("mcc", "")
    merch_avg   = float(merch.get("avg_amount", 0))
    is_online   = bool(term.get("is_online", False))
    card_present= bool(term.get("card_present", True))
    km_home     = float(term.get("km_from_home", 0))

    if last is None:
        minutes_since = None
        km_last = None
    else:
        last_ts = last.get("timestamp", "")
        minutes_since = max(iso_to_minutes(requested_at) - iso_to_minutes(last_ts), 0.0)
        km_last = last.get("km_from_current")
        if km_last is not None:
            km_last = float(km_last)

    v = np.zeros(16, dtype=np.int16)
    v[0]  = q(min(amount / MAX_AMOUNT, 1.0))
    v[1]  = INSTALL_LUT[min(installments, 12)]
    v[2]  = q(min((amount / cust_avg) / AMOUNT_RATIO, 1.0))
    v[3]  = HOUR_LUT[hour]
    v[4]  = DOW_LUT[dow]
    v[5]  = -int(SCALE) if minutes_since is None else q(min(minutes_since / MAX_MINUTES, 1.0))
    v[6]  = -int(SCALE) if km_last is None else q(min(km_last / MAX_KM, 1.0))
    v[7]  = q(min(km_home / MAX_KM, 1.0))
    v[8]  = TX_LUT[min(tx_count, 20)]
    v[9]  = int(SCALE) if is_online else 0
    v[10] = int(SCALE) if card_present else 0
    v[11] = int(SCALE) if unknown_merch else 0
    v[12] = MCC_RISK.get(mcc, 8000)
    v[13] = q(min(merch_avg / MAX_MERCH_AVG, 1.0))
    return v

def load_ivf(path: str):
    with open(path, "rb") as f:
        data = f.read()
    magic = struct.unpack_from("<Q", data, 0)[0]
    assert magic == 0x524E484132303236, f"bad magic: {hex(magic)}"
    nlist     = struct.unpack_from("<I", data, 8)[0]
    n_vectors = struct.unpack_from("<I", data, 16)[0]
    off = 32
    centroids = np.frombuffer(data, dtype=np.int16, count=nlist*16, offset=off).reshape(nlist, 16).copy()
    off += nlist * 32
    offsets = np.frombuffer(data, dtype=np.uint32, count=nlist+1, offset=off).copy()
    off += (nlist + 1) * 4
    vectors = np.frombuffer(data, dtype=np.int16, count=n_vectors*16, offset=off).reshape(n_vectors, 16).copy()
    off += n_vectors * 32
    labels = np.frombuffer(data, dtype=np.uint8, count=n_vectors, offset=off).copy()
    print(f"Index loaded: nlist={nlist}, n_vectors={n_vectors:,}")
    return centroids, offsets, vectors, labels

# ── Fast batched sweep ────────────────────────────────────────────────────────────────────

def batch_dists(a_i16: np.ndarray, b_i16: np.ndarray) -> np.ndarray:
    """Batch squared L2: a (N, 16) i16, b (M, 16) i16 → (N, M) i64."""
    af = a_i16.astype(np.float64)
    bf = b_i16.astype(np.float64)
    a_sq = (af * af).sum(axis=1)   # (N,)
    b_sq = (bf * bf).sum(axis=1)   # (M,)
    dot  = af @ bf.T               # (N, M) BLAS DGEMM
    return (a_sq[:, None] + b_sq[None, :] - 2.0 * dot).astype(np.int64)

def sweep_fast(queries_mat, exp_approved, centroids, offsets, vectors, labels, nprobe_values):
    N     = len(queries_mat)
    nlist = len(centroids)

    # Step 1: all centroid distances at once via BLAS
    t = time.time()
    print(f"  Centroid distances ({N}×{nlist})...", end="", flush=True)
    cent_dists = batch_dists(queries_mat, centroids)  # (N, nlist)
    print(f" {time.time()-t:.1f}s", flush=True)

    # Step 2: top-max_nprobe centroids per query (precomputed once)
    max_nprobe = min(max(nprobe_values), nlist)
    t = time.time()
    print(f"  Sorting top-{max_nprobe} centroids...", end="", flush=True)
    part = np.argpartition(cent_dists, max_nprobe, axis=1)[:, :max_nprobe]
    part_d = cent_dists[np.arange(N)[:, None], part]
    order = np.argsort(part_d, axis=1)
    top_sorted = part[np.arange(N)[:, None], order]   # (N, max_nprobe) sorted by dist
    print(f" {time.time()-t:.1f}s", flush=True)

    results = {}

    for nprobe in sorted(nprobe_values):
        t = time.time()
        probe_clusters = top_sorted[:, :nprobe]        # (N, nprobe)

        # Top-5 heap: dist (large sentinel) + label (0=not-fraud)
        heap_d = np.full((N, 5), np.iinfo(np.int64).max, dtype=np.int64)
        heap_l = np.zeros((N, 5), dtype=np.uint8)

        # Per-cluster vectorized update
        for c in range(nlist):
            mask = (probe_clusters == c).any(axis=1)
            if not mask.any():
                continue
            start, end = int(offsets[c]), int(offsets[c + 1])
            if start >= end:
                continue

            gidx  = np.where(mask)[0]
            Q_c   = len(gidx)
            vecs  = vectors[start:end]     # (M_c, 16)
            lbls  = labels[start:end]      # (M_c,)
            M_c   = end - start

            # Batch distances (Q_c × M_c)
            d_mat = batch_dists(queries_mat[gidx], vecs)  # (Q_c, M_c)

            # Merge current heap + new cluster dists → keep top-5
            merged_d = np.concatenate([heap_d[gidx], d_mat], axis=1)       # (Q_c, 5+M_c)
            merged_l = np.concatenate(
                [heap_l[gidx], np.tile(lbls, (Q_c, 1))], axis=1
            )

            # argpartition is O(k) per row, much faster than full sort
            k = merged_d.shape[1]
            if k > 5:
                top_k = np.argpartition(merged_d, 5, axis=1)[:, :5]
            else:
                top_k = np.tile(np.arange(k), (Q_c, 1))
            heap_d[gidx] = merged_d[np.arange(Q_c)[:, None], top_k]
            heap_l[gidx] = merged_l[np.arange(Q_c)[:, None], top_k]

        fraud_counts = heap_l.sum(axis=1)
        approved     = fraud_counts < 3
        fp = int(((~approved) &  exp_approved).sum())
        fn = int(( approved   & ~exp_approved).sum())
        e  = fp + fn * 3
        results[nprobe] = (fp, fn, e)
        print(f"  nprobe={nprobe:>3}: FP={fp} FN={fn} E={e}  ({time.time()-t:.1f}s)", flush=True)

    return results

# ── Main ────────────────────────────────────────────────────────────────────────────────

def main():
    # argv[1]: ivf path, argv[2]: nprobe comma-list OR test-data path, argv[3]: nprobe comma-list
    ivf_path  = sys.argv[1] if len(sys.argv) > 1 else \
        "target/x86_64-unknown-linux-musl/release/build/rinha-980b0ce36b48eb51/out/ivf.bin"

    nprobe_values = [6, 8, 10, 12, 14, 16, 20, 25, 27, 30, 38]
    data_path = "test/test-data.json"

    for arg in sys.argv[2:]:
        try:
            nprobe_values = [int(x) for x in arg.split(',')]
        except ValueError:
            data_path = arg

    print(f"Loading index: {ivf_path}")
    centroids, offsets, vectors, labels = load_ivf(ivf_path)

    print(f"Loading test data: {data_path}")
    with open(data_path) as f:
        root = json.load(f)
    entries = root["entries"]
    N = len(entries)
    print(f"Entries: {N:,}")

    t0 = time.time()
    print("Vectorizing...", end="", flush=True)
    queries_list = [vectorize(e["request"]) for e in entries]
    exp_approved = np.array([e["expected_approved"] for e in entries], dtype=bool)
    queries_mat  = np.stack(queries_list)
    print(f" done ({time.time()-t0:.1f}s)")

    print("\nSweeping NPROBE values:")
    results = sweep_fast(queries_mat, exp_approved, centroids, offsets, vectors, labels, nprobe_values)

    penalty_per_E = 270.93 / 7
    print(f"\n{'NPROBE':>8} {'FP':>5} {'FN':>5} {'E':>5}  {'det_score':>10}  notes")
    print("-" * 60)
    for nprobe in sorted(results.keys()):
        fp, fn, e = results[nprobe]
        det = max(0.0, 3000 - e * penalty_per_E)
        note = "← current" if nprobe == 27 else ""
        print(f"{nprobe:>8} {fp:>5} {fn:>5} {e:>5}  {det:>10.1f}  {note}")

    print("\nDone. Use E values to pick NPROBE, then measure p99 with docker-compose.local.yml.")

if __name__ == "__main__":
    main()
