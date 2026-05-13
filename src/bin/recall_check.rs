// recall_check: measure FP/FN/E for each NPROBE value using test-data.json
//
// Usage: recall_check [test-data.json]
// Reads the same IVF index embedded in the binary.

use std::path::PathBuf;

mod index {
    include!("../index.rs");
}
mod vectorize {
    include!("../vectorize.rs");
}
mod parse {
    include!("../parse.rs");
}

static INDEX_DATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ivf.bin"));

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("test/test-data.json"));

    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));

    let root: serde_json::Value = serde_json::from_str(&raw)
        .expect("invalid JSON in test-data.json");

    let entries = root["entries"].as_array().expect("entries missing");

    let idx = index::IvfIndex::from_static_bytes(INDEX_DATA);
    idx.warmup();

    let nprobe_values: &[usize] = &[6, 8, 10, 12, 15, 20, 25, 27, 30, 38];

    println!("{:<8} {:>4} {:>4} {:>6}  {}", "NPROBE", "FP", "FN", "E", "notes");
    println!("{}", "-".repeat(50));

    for &nprobe in nprobe_values {
        let mut fp = 0usize;
        let mut fn_ = 0usize;

        for entry in entries {
            let req_bytes = serde_json::to_vec(&entry["request"]).unwrap();
            let tx = match parse::parse_transaction(&req_bytes) {
                Ok(t) => t,
                Err(_) => continue,
            };
            let q = vectorize::vectorize(&tx);
            let fraud_count = idx.query(&q, nprobe, 0);
            let approved = fraud_count < 3;
            let expected_approved = entry["expected_approved"].as_bool().unwrap_or(true);

            if approved && !expected_approved {
                fn_ += 1; // said ok, was fraud
            } else if !approved && expected_approved {
                fp += 1;  // said fraud, was ok
            }
        }

        // E = FP×1 + FN×3  (matches observed: FP=1, FN=2 → E=7)
        let e = fp + fn_ * 3;
        let note = if nprobe == 27 { "← current" } else { "" };
        println!("{:<8} {:>4} {:>4} {:>6}  {}", nprobe, fp, fn_, e, note);
    }
}
