use flate2::read::GzDecoder;
use rayon::prelude::*;
use serde::de::Deserializer as _;
use serde::Deserialize;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

const NLIST: usize = 1024;
const KMEANS_ITERS: usize = 15;
const SCALE: f32 = 16000.0;
const DIM_PAD: usize = 16;
const DIM: usize = 14;

#[derive(Deserialize)]
struct RefEntry {
    vector: Vec<f32>,
    label: String,
}

fn quantize(x: f32) -> i16 {
    (x * SCALE).round().clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn distance_sq(a: &[i16; 16], b: &[i16; 16]) -> i64 {
    let mut s = 0i64;
    for i in 0..DIM {
        let d = a[i] as i64 - b[i] as i64;
        s += d * d;
    }
    s
}

fn main() {
    println!("cargo:rerun-if-changed=resources/references.json.gz");
    println!("cargo:rerun-if-changed=resources/mcc_risk.json");
    println!("cargo:rerun-if-env-changed=IVF_BIN");

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    generate_mcc_lut(&out_dir);

    // Allow skipping k-means by providing a pre-built index
    if let Ok(prebuilt) = std::env::var("IVF_BIN") {
        let ivf_path = out_dir.join("ivf.bin");
        eprintln!("build.rs: using pre-built IVF: {prebuilt}");
        std::fs::copy(&prebuilt, &ivf_path).expect("copy pre-built IVF");
        return;
    }

    let gz_path = PathBuf::from("resources/references.json.gz");
    if !gz_path.exists() {
        write_empty_index(&out_dir);
        eprintln!("cargo:warning=references.json.gz not found — empty index. Place it in resources/ and rebuild.");
        return;
    }

    let t0 = std::time::Instant::now();
    eprintln!("build.rs: loading references.json.gz …");

    let file = std::fs::File::open(&gz_path).expect("open gz");
    let reader = BufReader::with_capacity(1 << 23, GzDecoder::new(file));

    let mut raw_vectors: Vec<[i16; 16]> = Vec::with_capacity(3_200_000);
    let mut raw_labels: Vec<u8> = Vec::with_capacity(3_200_000);

    use serde::de::SeqAccess;
    struct V<'a> {
        vecs: &'a mut Vec<[i16; 16]>,
        lbls: &'a mut Vec<u8>,
    }
    impl<'de, 'a> serde::de::Visitor<'de> for V<'a> {
        type Value = ();
        fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            write!(f, "array")
        }
        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
            let mut n = 0usize;
            while let Some(e) = seq.next_element::<RefEntry>()? {
                let mut v = [0i16; 16];
                for (i, &x) in e.vector.iter().enumerate().take(DIM) {
                    v[i] = quantize(x);
                }
                self.vecs.push(v);
                self.lbls.push(if e.label == "fraud" { 1 } else { 0 });
                n += 1;
                if n % 500_000 == 0 {
                    eprintln!("build.rs:   loaded {}k…", n / 1000);
                }
            }
            Ok(())
        }
    }

    serde_json::Deserializer::from_reader(reader)
        .deserialize_seq(V { vecs: &mut raw_vectors, lbls: &mut raw_labels })
        .expect("parse json");

    let n = raw_vectors.len();
    eprintln!("build.rs: {} vectors loaded in {:.1}s", n, t0.elapsed().as_secs_f32());

    eprintln!("build.rs: k-means nlist={} iters={} (parallel)…", NLIST, KMEANS_ITERS);
    let t1 = std::time::Instant::now();
    let centroids = kmeans_parallel(&raw_vectors, NLIST, KMEANS_ITERS);
    eprintln!("build.rs: k-means done in {:.1}s", t1.elapsed().as_secs_f32());

    let assignments: Vec<u32> = raw_vectors
        .par_iter()
        .map(|vec| {
            let mut best_c = 0u32;
            let mut best_d = i64::MAX;
            for (c, cent) in centroids.iter().enumerate() {
                let d = distance_sq(vec, cent);
                if d < best_d {
                    best_d = d;
                    best_c = c as u32;
                }
            }
            best_c
        })
        .collect();

    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| assignments[i]);

    let mut cluster_offsets = vec![0u32; NLIST + 1];
    for &c in &assignments {
        cluster_offsets[c as usize + 1] += 1;
    }
    for i in 1..=NLIST {
        cluster_offsets[i] += cluster_offsets[i - 1];
    }

    eprintln!("build.rs: writing ivf.bin…");
    let ivf_path = out_dir.join("ivf.bin");
    let mut out = std::io::BufWriter::with_capacity(1 << 24,
        std::fs::File::create(&ivf_path).unwrap());

    out.write_all(&0x52_4E_48_41_32_30_32_36u64.to_le_bytes()).unwrap();
    out.write_all(&(NLIST as u32).to_le_bytes()).unwrap();
    out.write_all(&50u32.to_le_bytes()).unwrap();
    out.write_all(&(n as u32).to_le_bytes()).unwrap();
    out.write_all(&(DIM_PAD as u32).to_le_bytes()).unwrap();
    out.write_all(&[0u8; 8]).unwrap();

    for cent in &centroids {
        out.write_all(unsafe {
            std::slice::from_raw_parts(cent.as_ptr() as *const u8, 32)
        }).unwrap();
    }
    for &off in &cluster_offsets {
        out.write_all(&off.to_le_bytes()).unwrap();
    }
    for &idx in &order {
        out.write_all(unsafe {
            std::slice::from_raw_parts(raw_vectors[idx].as_ptr() as *const u8, 32)
        }).unwrap();
    }
    for &idx in &order {
        out.write_all(&[raw_labels[idx]]).unwrap();
    }
    out.flush().unwrap();

    let mb = std::fs::metadata(&ivf_path).unwrap().len() as f64 / 1e6;
    eprintln!("build.rs: done. ivf.bin = {:.1} MB, total = {:.1}s", mb, t0.elapsed().as_secs_f32());
}

fn write_empty_index(out_dir: &PathBuf) {
    let path = out_dir.join("ivf.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(&0x52_4E_48_41_32_30_32_36u64.to_le_bytes()).unwrap();
    f.write_all(&(NLIST as u32).to_le_bytes()).unwrap();
    f.write_all(&50u32.to_le_bytes()).unwrap();
    f.write_all(&0u32.to_le_bytes()).unwrap();
    f.write_all(&(DIM_PAD as u32).to_le_bytes()).unwrap();
    f.write_all(&[0u8; 8]).unwrap();
}

fn generate_mcc_lut(out_dir: &PathBuf) {
    let content = std::fs::read_to_string("resources/mcc_risk.json")
        .unwrap_or_else(|_| "{}".to_string());
    let map: std::collections::HashMap<String, f64> = serde_json::from_str(&content).unwrap();

    let mut f = std::fs::File::create(out_dir.join("mcc_lut.rs")).unwrap();
    writeln!(f, "pub fn mcc_risk(code: &[u8]) -> i16 {{").unwrap();
    writeln!(f, "    match code {{").unwrap();
    for (code, risk) in &map {
        writeln!(f, "        b\"{}\" => {},", code, (risk * SCALE as f64).round() as i16).unwrap();
    }
    writeln!(f, "        _ => {},", (SCALE / 2.0) as i16).unwrap();
    writeln!(f, "    }}").unwrap();
    writeln!(f, "}}").unwrap();
}

fn kmeans_parallel(vectors: &[[i16; 16]], k: usize, iters: usize) -> Vec<[i16; 16]> {
    let n = vectors.len();
    let mut rng: u64 = 0xdeadbeef_cafebabe;
    let mut centroids: Vec<[i16; 16]> = Vec::with_capacity(k);
    centroids.push(vectors[lcg(&mut rng) as usize % n]);

    let stride = n / k;
    for i in 1..k {
        let offset = (lcg(&mut rng) as usize % stride.max(1)) + i * stride;
        centroids.push(vectors[offset % n]);
    }

    let mut assignments: Vec<u32> = vec![0u32; n];
    let changes_counter = AtomicUsize::new(0);

    for iter in 0..iters {
        changes_counter.store(0, Ordering::Relaxed);
        let new_assignments: Vec<u32> = vectors
            .par_iter()
            .map(|vec| {
                let mut best_c = 0u32;
                let mut best_d = i64::MAX;
                for (c, cent) in centroids.iter().enumerate() {
                    let d = distance_sq(vec, cent);
                    if d < best_d {
                        best_d = d;
                        best_c = c as u32;
                    }
                }
                best_c
            })
            .collect();

        let changes = assignments.iter().zip(new_assignments.iter()).filter(|(a, b)| a != b).count();
        assignments = new_assignments;

        let num_cpus = rayon::current_num_threads();
        let chunk = (n + num_cpus - 1) / num_cpus;

        let partial: Vec<(Vec<[i64; 16]>, Vec<u32>)> = (0..num_cpus)
            .into_par_iter()
            .map(|t| {
                let start = t * chunk;
                let end = (start + chunk).min(n);
                let mut sums = vec![[0i64; 16]; k];
                let mut counts = vec![0u32; k];
                for i in start..end {
                    let c = assignments[i] as usize;
                    counts[c] += 1;
                    for d in 0..DIM {
                        sums[c][d] += vectors[i][d] as i64;
                    }
                }
                (sums, counts)
            })
            .collect();

        let mut sums = vec![[0i64; 16]; k];
        let mut counts = vec![0u32; k];
        for (ps, pc) in &partial {
            for c in 0..k {
                counts[c] += pc[c];
                for d in 0..DIM {
                    sums[c][d] += ps[c][d];
                }
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..DIM {
                    centroids[c][d] = (sums[c][d] / counts[c] as i64) as i16;
                }
            }
        }

        let pct = changes as f64 / n as f64 * 100.0;
        eprintln!("build.rs:   iter {}/{}: {:.2}% reassigned", iter + 1, iters, pct);
        if pct < 0.05 {
            eprintln!("build.rs:   converged early");
            break;
        }
    }

    centroids
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    *s
}
