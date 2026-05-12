// Converts a parsed Transaction into a quantized [i16; 16] vector.
// Indices 0-13 are the 14 feature dimensions; 14-15 are padding zeros for AVX2.
// Scale: values in [0,1] map to [0, 8192]; sentinel -1.0 → -8192 (matches reference data encoding).

use crate::parse::Transaction;

include!(concat!(env!("OUT_DIR"), "/mcc_lut.rs"));

const SCALE: f32 = 16000.0;
const MAX_AMOUNT: f32 = 10000.0;
const MAX_INSTALLMENTS: f32 = 12.0;
const AMOUNT_VS_AVG_RATIO: f32 = 10.0;
const MAX_MINUTES: f32 = 1440.0;
const MAX_KM: f32 = 1000.0;
const MAX_TX_COUNT: f32 = 20.0;
const MAX_MERCHANT_AVG: f32 = 10000.0;

const INSTALLMENTS_LUT: [i16; 13] = build_lut_13(MAX_INSTALLMENTS);
const HOUR_LUT: [i16; 24] = build_lut_24();
const DOW_LUT: [i16; 7] = build_lut_7();
const TX_COUNT_LUT: [i16; 21] = build_lut_21(MAX_TX_COUNT);

const fn q(x: f32) -> i16 {
    let v = x * SCALE;
    let v = if v < i16::MIN as f32 { i16::MIN as f32 } else if v > i16::MAX as f32 { i16::MAX as f32 } else { v };
    v as i16
}

const fn build_lut_13(max: f32) -> [i16; 13] {
    let mut t = [0i16; 13];
    let mut i = 0;
    while i < 13 {
        let x = if i as f32 / max > 1.0 { 1.0 } else { i as f32 / max };
        t[i] = q(x);
        i += 1;
    }
    t
}

const fn build_lut_24() -> [i16; 24] {
    let mut t = [0i16; 24];
    let mut i = 0;
    while i < 24 {
        t[i] = q(i as f32 / 23.0);
        i += 1;
    }
    t
}

const fn build_lut_7() -> [i16; 7] {
    let mut t = [0i16; 7];
    let mut i = 0;
    while i < 7 {
        t[i] = q(i as f32 / 6.0);
        i += 1;
    }
    t
}

const fn build_lut_21(max: f32) -> [i16; 21] {
    let mut t = [0i16; 21];
    let mut i = 0;
    while i < 21 {
        let x = if i as f32 / max > 1.0 { 1.0 } else { i as f32 / max };
        t[i] = q(x);
        i += 1;
    }
    t
}

#[inline]
fn clamp_q(x: f32) -> i16 {
    q(x.clamp(0.0, 1.0))
}

pub fn vectorize(tx: &Transaction) -> [i16; 16] {
    let mut v = [0i16; 16];
    v[0] = clamp_q(tx.amount as f32 / MAX_AMOUNT);
    v[1] = INSTALLMENTS_LUT[tx.installments.min(12) as usize];
    v[2] = clamp_q((tx.amount as f32 / tx.customer_avg_amount as f32) / AMOUNT_VS_AVG_RATIO);
    v[3] = HOUR_LUT[tx.hour as usize];
    v[4] = DOW_LUT[tx.day_of_week as usize];
    v[5] = match tx.minutes_since_last {
        None => -(SCALE as i16),
        Some(m) => clamp_q(m as f32 / MAX_MINUTES),
    };
    v[6] = match tx.km_from_last {
        None => -(SCALE as i16),
        Some(km) => clamp_q(km as f32 / MAX_KM),
    };
    v[7] = clamp_q(tx.km_from_home as f32 / MAX_KM);
    v[8] = TX_COUNT_LUT[tx.tx_count_24h.min(20) as usize];
    v[9] = if tx.is_online { SCALE as i16 } else { 0 };
    v[10] = if tx.card_present { SCALE as i16 } else { 0 };
    v[11] = if tx.unknown_merchant { SCALE as i16 } else { 0 };
    v[12] = mcc_risk(&tx.mcc[..tx.mcc_len as usize]);
    v[13] = clamp_q(tx.merchant_avg_amount as f32 / MAX_MERCHANT_AVG);
    v
}
