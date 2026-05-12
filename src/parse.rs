// Manual JSON parser for /fraud-score payloads.
// Avoids serde allocations on the hot path.
// Returns Err only on truly malformed input; missing optional fields use sentinels.

use memchr::memmem;
use lexical_core;

#[derive(Default)]
pub struct Transaction {
    pub amount: f64,
    pub installments: u32,
    // hour 0-23 UTC
    pub hour: u8,
    // day of week: 0=Mon .. 6=Sun
    pub day_of_week: u8,
    pub customer_avg_amount: f64,
    pub tx_count_24h: u32,
    // true if merchant.id is NOT in known_merchants
    pub unknown_merchant: bool,
    pub mcc: [u8; 8],
    pub mcc_len: u8,
    pub merchant_avg_amount: f64,
    pub is_online: bool,
    pub card_present: bool,
    pub km_from_home: f64,
    // None when last_transaction is null
    pub minutes_since_last: Option<f64>,
    pub km_from_last: Option<f64>,
}

pub fn parse_transaction(buf: &[u8]) -> Result<Transaction, ()> {
    let mut tx = Transaction::default();

    tx.amount = find_number(buf, b"\"amount\"").unwrap_or(0.0);
    tx.installments = find_u32(buf, b"\"installments\"").unwrap_or(0);

    if let Some(ts) = find_string_value(buf, b"\"requested_at\"") {
        let (h, dow) = parse_iso8601_hour_dow(ts);
        tx.hour = h;
        tx.day_of_week = dow;
    }

    tx.customer_avg_amount = find_number(buf, b"\"avg_amount\"").unwrap_or(1.0).max(1e-9);
    tx.tx_count_24h = find_u32(buf, b"\"tx_count_24h\"").unwrap_or(0);

    if let Some(merchant_id) = find_string_value_nth(buf, b"\"id\"", 2) {
        if let Some(km_start) = find_array_start(buf, b"\"known_merchants\"") {
            tx.unknown_merchant = !slice_contains(km_start, merchant_id);
        } else {
            tx.unknown_merchant = true;
        }
    }

    if let Some(mcc) = find_string_value(buf, b"\"mcc\"") {
        let len = mcc.len().min(8);
        tx.mcc[..len].copy_from_slice(&mcc[..len]);
        tx.mcc_len = len as u8;
    }

    tx.merchant_avg_amount = find_number_nth(buf, b"\"avg_amount\"", 2).unwrap_or(0.0);
    tx.is_online = find_bool(buf, b"\"is_online\"").unwrap_or(false);
    tx.card_present = find_bool(buf, b"\"card_present\"").unwrap_or(true);
    tx.km_from_home = find_number(buf, b"\"km_from_home\"").unwrap_or(0.0);

    let is_null = memmem::find(buf, b"\"last_transaction\":null").is_some()
        || memmem::find(buf, b"\"last_transaction\": null").is_some();

    if !is_null {
        if let (Some(req_ts), Some(last_ts)) = (
            find_string_value(buf, b"\"requested_at\""),
            find_string_value_nth(buf, b"\"timestamp\"", 1),
        ) {
            tx.minutes_since_last = Some(minutes_between(last_ts, req_ts));
        }
        tx.km_from_last = find_number(buf, b"\"km_from_current\"");
    }

    Ok(tx)
}

fn find_number(buf: &[u8], key: &[u8]) -> Option<f64> {
    find_number_nth(buf, key, 1)
}

fn find_number_nth(buf: &[u8], key: &[u8], nth: usize) -> Option<f64> {
    let mut remaining = buf;
    let mut found = 0;
    loop {
        let pos = memmem::find(remaining, key)?;
        remaining = &remaining[pos + key.len()..];
        let after = skip_colon_ws(remaining)?;
        found += 1;
        if found == nth {
            return parse_f64_from_slice(after);
        }
    }
}

fn find_u32(buf: &[u8], key: &[u8]) -> Option<u32> {
    find_number(buf, key).map(|v| v as u32)
}

fn find_bool(buf: &[u8], key: &[u8]) -> Option<bool> {
    let pos = memmem::find(buf, key)?;
    let after = skip_colon_ws(&buf[pos + key.len()..])?;
    if after.starts_with(b"true") {
        Some(true)
    } else if after.starts_with(b"false") {
        Some(false)
    } else {
        None
    }
}

fn find_string_value<'a>(buf: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    find_string_value_nth(buf, key, 1)
}

fn find_string_value_nth<'a>(buf: &'a [u8], key: &[u8], nth: usize) -> Option<&'a [u8]> {
    let mut remaining = buf;
    let mut found = 0;
    loop {
        let pos = memmem::find(remaining, key)?;
        remaining = &remaining[pos + key.len()..];
        let after = skip_colon_ws(remaining)?;
        if after.first() == Some(&b'"') {
            found += 1;
            if found == nth {
                let inner = &after[1..];
                let end = memchr::memchr(b'"', inner)?;
                return Some(&inner[..end]);
            }
        }
    }
}

fn find_array_start<'a>(buf: &'a [u8], key: &[u8]) -> Option<&'a [u8]> {
    let pos = memmem::find(buf, key)?;
    let after = skip_colon_ws(&buf[pos + key.len()..])?;
    if after.first() == Some(&b'[') {
        let end = memchr::memchr(b']', after)?;
        Some(&after[1..end])
    } else {
        None
    }
}

fn slice_contains(haystack: &[u8], needle: &[u8]) -> bool {
    memmem::find(haystack, needle).is_some()
}

fn skip_colon_ws(buf: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i < buf.len() && matches!(buf[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    if i < buf.len() && buf[i] == b':' {
        i += 1;
    }
    while i < buf.len() && matches!(buf[i], b' ' | b'\t' | b'\n' | b'\r') {
        i += 1;
    }
    if i < buf.len() { Some(&buf[i..]) } else { None }
}

fn parse_f64_from_slice(buf: &[u8]) -> Option<f64> {
    let end = buf
        .iter()
        .position(|&b| matches!(b, b',' | b'}' | b']' | b' ' | b'\n' | b'\r' | b'\t'))
        .unwrap_or(buf.len());
    lexical_core::parse::<f64>(&buf[..end]).ok()
}

fn parse_iso8601_hour_dow(ts: &[u8]) -> (u8, u8) {
    if ts.len() < 19 {
        return (0, 0);
    }
    let hour = parse_decimal2(&ts[11..13]);
    let y = parse_decimal4(&ts[0..4]) as i32;
    let m = parse_decimal2(&ts[5..7]) as i32;
    let d = parse_decimal2(&ts[8..10]) as i32;
    let dow = day_of_week(y, m, d);
    (hour, dow)
}

fn minutes_between(from: &[u8], to: &[u8]) -> f64 {
    let from_min = iso_to_minutes(from);
    let to_min = iso_to_minutes(to);
    (to_min - from_min).max(0.0)
}

fn iso_to_minutes(ts: &[u8]) -> f64 {
    if ts.len() < 19 {
        return 0.0;
    }
    let y = parse_decimal4(&ts[0..4]) as f64;
    let mo = parse_decimal2(&ts[5..7]) as f64;
    let d = parse_decimal2(&ts[8..10]) as f64;
    let h = parse_decimal2(&ts[11..13]) as f64;
    let mi = parse_decimal2(&ts[14..16]) as f64;
    let s = parse_decimal2(&ts[17..19]) as f64;
    y * 525960.0 + mo * 43800.0 + d * 1440.0 + h * 60.0 + mi + s / 60.0
}

pub fn parse_decimal2(b: &[u8]) -> u8 {
    if b.len() < 2 { return 0; }
    (b[0] - b'0') * 10 + (b[1] - b'0')
}

fn parse_decimal4(b: &[u8]) -> u16 {
    if b.len() < 4 { return 0; }
    (b[0] - b'0') as u16 * 1000
        + (b[1] - b'0') as u16 * 100
        + (b[2] - b'0') as u16 * 10
        + (b[3] - b'0') as u16
}

fn day_of_week(y: i32, m: i32, d: i32) -> u8 {
    static T: [i32; 12] = [0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4];
    let y = if m < 3 { y - 1 } else { y };
    let dow_sun = ((y + y / 4 - y / 100 + y / 400 + T[(m - 1) as usize] + d) % 7) as u8;
    if dow_sun == 0 { 6 } else { dow_sun - 1 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_legit_example() {
        let body = br#"{
            "id": "tx-1329056812",
            "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
            "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
            "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
            "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.23},
            "last_transaction": null
        }"#;
        let tx = parse_transaction(body).unwrap();
        assert!((tx.amount - 41.12).abs() < 0.01);
        assert_eq!(tx.installments, 2);
        assert_eq!(tx.hour, 18);
        assert!(!tx.unknown_merchant);
        assert!(!tx.is_online);
        assert!(tx.card_present);
        assert!(tx.minutes_since_last.is_none());
    }

    #[test]
    fn test_day_of_week_wednesday() {
        let (_, dow) = parse_iso8601_hour_dow(b"2026-03-11T18:45:53Z");
        assert_eq!(dow, 2);
    }
}
