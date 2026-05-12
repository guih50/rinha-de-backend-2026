// IVF index: load from binary blob, query with AVX2-accelerated distance.
// Binary layout (see build.rs for details):
//   [header 32B][centroids nlist×32B][offsets (nlist+1)×4B][vectors n×32B][labels n×1B]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub struct IvfIndex {
    nlist: usize,
    n_vectors: usize,
    centroids: *const [i16; 16],
    cluster_offsets: *const u32,
    vectors: *const [i16; 16],
    labels: *const u8,
}

// SAFETY: the data is &'static [u8] from include_bytes!, never mutated.
unsafe impl Send for IvfIndex {}
unsafe impl Sync for IvfIndex {}

impl IvfIndex {
    pub fn from_static_bytes(data: &'static [u8]) -> Self {
        assert!(data.len() >= 32, "ivf.bin too small");

        let magic = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert_eq!(magic, 0x52_4E_48_41_32_30_32_36, "bad magic in ivf.bin");

        let nlist = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
        let n_vectors = u32::from_le_bytes(data[16..20].try_into().unwrap()) as usize;

        if n_vectors == 0 {
            // Empty index (no dataset yet)
            return IvfIndex {
                nlist: 0,
                n_vectors: 0,
                centroids: std::ptr::null(),
                cluster_offsets: std::ptr::null(),
                vectors: std::ptr::null(),
                labels: std::ptr::null(),
            };
        }

        let mut offset = 32usize; // after header

        // Centroids
        let centroids_ptr = data[offset..].as_ptr() as *const [i16; 16];
        offset += nlist * 32;

        // Cluster offsets
        let cluster_offsets_ptr = data[offset..].as_ptr() as *const u32;
        offset += (nlist + 1) * 4;

        // Vectors
        let vectors_ptr = data[offset..].as_ptr() as *const [i16; 16];
        offset += n_vectors * 32;

        // Labels
        let labels_ptr = data[offset..].as_ptr();

        IvfIndex {
            nlist,
            n_vectors,
            centroids: centroids_ptr,
            cluster_offsets: cluster_offsets_ptr,
            vectors: vectors_ptr,
            labels: labels_ptr,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.n_vectors == 0
    }

    /// Touch every 4 KB page of vector and label data to pre-fault them into RAM.
    /// Call once at startup before accepting connections.
    pub fn warmup(&self) {
        if self.is_empty() {
            return;
        }
        let mut acc = 0i32;
        unsafe {
            // Touch centroid pages
            let centroids = std::slice::from_raw_parts(self.centroids, self.nlist);
            for c in centroids.iter() {
                acc = acc.wrapping_add(c[0] as i32);
            }
            // Touch every 4 KB page of vector data (128 vectors × 32 B = 4096 B)
            let vectors = std::slice::from_raw_parts(self.vectors, self.n_vectors);
            for i in (0..self.n_vectors).step_by(128) {
                acc = acc.wrapping_add(vectors[i][0] as i32);
            }
            let labels = std::slice::from_raw_parts(self.labels, self.n_vectors);
            for i in (0..self.n_vectors).step_by(4096) {
                acc = acc.wrapping_add(labels[i] as i32);
            }
        }
        std::hint::black_box(acc);
    }

    /// Returns the number of fraud labels among the 5 nearest neighbors (0..=5).
    pub fn query(&self, q: &[i16; 16], nprobe: usize) -> u8 {
        if self.is_empty() {
            return 0;
        }

        let nprobe = nprobe.min(self.nlist);

        // Step 1: find top-nprobe centroids
        let nprobe_capped = nprobe.min(256);
        let mut centroid_dists = [i64::MAX; 256];
        let mut centroid_ids = [0u32; 256];

        unsafe {
            let centroids = std::slice::from_raw_parts(self.centroids, self.nlist);
            let mut worst_in_top = i64::MAX;

            for (c, cent) in centroids.iter().enumerate() {
                let d = dist_sq_avx2(q, cent);
                if d < worst_in_top || c < nprobe_capped {
                    insert_top(
                        &mut centroid_dists[..nprobe_capped],
                        &mut centroid_ids[..nprobe_capped],
                        d,
                        c as u32,
                    );
                    worst_in_top = centroid_dists[nprobe_capped - 1];
                }
            }
        }

        // Step 2: scan selected clusters, maintain top-5
        let mut heap_dists = [i64::MAX; 5];
        let mut heap_fraud = [0u8; 5];
        let mut heap_worst = i64::MAX;

        unsafe {
            let offsets = std::slice::from_raw_parts(self.cluster_offsets, self.nlist + 1);
            let vectors = std::slice::from_raw_parts(self.vectors, self.n_vectors);
            let labels = std::slice::from_raw_parts(self.labels, self.n_vectors);

            for p in 0..nprobe_capped {
                let c = centroid_ids[p] as usize;
                let start = offsets[c] as usize;
                let end = offsets[c + 1] as usize;

                // Prefetch next cluster's opening vectors: hardware prefetcher can't
                // predict the cluster-to-cluster pointer jump.
                #[cfg(target_arch = "x86_64")]
                if p + 1 < nprobe_capped {
                    let nc = centroid_ids[p + 1] as usize;
                    let ns = offsets[nc] as usize;
                    let ne = (offsets[nc + 1] as usize).min(ns + 16);
                    let mut k = ns;
                    while k < ne {
                        _mm_prefetch(vectors.as_ptr().add(k) as *const i8, _MM_HINT_T1);
                        k += 2;
                    }
                }

                for idx in start..end {
                    let d = dist_sq_avx2(q, &vectors[idx]);
                    if d < heap_worst {
                        insert_heap5(
                            &mut heap_dists,
                            &mut heap_fraud,
                            d,
                            labels[idx],
                        );
                        heap_worst = heap_dists[4];
                    }
                }
            }
        }

        heap_fraud.iter().sum()
    }
}

// ── Distance functions ────────────────────────────────────────────────────────

#[inline(always)]
fn dist_sq(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    let mut s = 0i64;
    for i in 0..14 {
        let d = a[i] as i64 - b[i] as i64;
        s += d * d;
    }
    s
}

/// AVX2 distance squared (14 active dims + 2 padding zeros).
/// Uses i64 accumulation to handle SCALE=16000 (max diff=32000, MADD pair≤2.048B, i64 hsum safe).
/// Compiled with +avx2 in rustflags — no runtime detection needed.
#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn dist_sq_avx2(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    dist_sq_avx2_inner(a, b)
}

#[cfg(not(target_arch = "x86_64"))]
#[inline(always)]
unsafe fn dist_sq_avx2(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    dist_sq(a, b)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dist_sq_avx2_inner(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    let va = _mm256_loadu_si256(a.as_ptr() as *const __m256i);
    let vb = _mm256_loadu_si256(b.as_ptr() as *const __m256i);
    let diff = _mm256_sub_epi16(va, vb);
    // madd: diff[i]^2 summed in pairs → 8 × i32 (max pair=2.048B < i32::MAX for SCALE=16000)
    let sq = _mm256_madd_epi16(diff, diff);
    hsum_epi32_to_i64(sq)
}

/// Sum 8 × i32 from a __m256i into i64, avoiding intermediate i32 overflow.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_epi32_to_i64(v: __m256i) -> i64 {
    let lo128 = _mm256_castsi256_si128(v);
    let hi128 = _mm256_extracti128_si256(v, 1);
    let lo64 = _mm256_cvtepi32_epi64(lo128);
    let hi64 = _mm256_cvtepi32_epi64(hi128);
    let sum256 = _mm256_add_epi64(lo64, hi64);
    let sum_lo = _mm256_castsi256_si128(sum256);
    let sum_hi = _mm256_extracti128_si256(sum256, 1);
    let sum128 = _mm_add_epi64(sum_lo, sum_hi);
    let sum = _mm_add_epi64(sum128, _mm_srli_si128(sum128, 8));
    _mm_cvtsi128_si64(sum)
}

// ── Heap maintenance ──────────────────────────────────────────────────────────

#[inline]
fn insert_heap5(dists: &mut [i64; 5], frauds: &mut [u8; 5], d: i64, is_fraud: u8) {
    if d >= dists[4] {
        return;
    }
    let pos = dists.partition_point(|&x| x < d);
    dists[pos..].rotate_right(1);
    frauds[pos..].rotate_right(1);
    dists[pos] = d;
    frauds[pos] = is_fraud;
}

#[inline]
fn insert_top(dists: &mut [i64], ids: &mut [u32], d: i64, id: u32) {
    let k = dists.len();
    if d >= dists[k - 1] {
        return;
    }
    let pos = dists.partition_point(|&x| x < d);
    dists[pos..].rotate_right(1);
    ids[pos..].rotate_right(1);
    dists[pos] = d;
    ids[pos] = id;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_insert_heap5() {
        let mut dists = [i64::MAX; 5];
        let mut frauds = [0u8; 5];
        insert_heap5(&mut dists, &mut frauds, 100, 1);
        insert_heap5(&mut dists, &mut frauds, 50, 0);
        insert_heap5(&mut dists, &mut frauds, 200, 1);
        insert_heap5(&mut dists, &mut frauds, 10, 1);
        insert_heap5(&mut dists, &mut frauds, 75, 0);
        assert_eq!(dists, [10, 50, 75, 100, 200]);
        assert_eq!(frauds[0], 1);
    }

    #[test]
    fn test_dist_sq_consistency() {
        let a: [i16; 16] = [100, 200, 300, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b: [i16; 16] = [110, 190, 310, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let scalar = dist_sq(&a, &b);
        let avx2 = unsafe { dist_sq_avx2(&a, &b) };
        assert_eq!(scalar, avx2);
        let a3: [i16; 16] = [-16000, -16000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let b3: [i16; 16] = [16000, 16000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let scalar3 = dist_sq(&a3, &b3);
        let avx2_3 = unsafe { dist_sq_avx2(&a3, &b3) };
        assert_eq!(scalar3, avx2_3);
    }
}
