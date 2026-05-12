// Pre-computed HTTP/1.1 responses for all 6 possible fraud scores.
// fraud_count = 0..=5 → responses[fraud_count]
// Errors return responses[0] (approved:true, score:0.0) to avoid HTTP 500 weight=5 penalty.

pub const RESPONSE_BODY_OFFSET: usize = 95;

pub static RESPONSE_BODIES: [&[u8]; 6] = [
    b"{\"approved\":true,\"fraud_score\":0.0}",
    b"{\"approved\":true,\"fraud_score\":0.2}",
    b"{\"approved\":true,\"fraud_score\":0.4}",
    b"{\"approved\":false,\"fraud_score\":0.6}",
    b"{\"approved\":false,\"fraud_score\":0.8}",
    b"{\"approved\":false,\"fraud_score\":1.0}",
];

pub static RESPONSE_CONTENT_LENGTHS: [&str; 6] = ["35", "35", "35", "36", "36", "36"];

pub static RESPONSES: [&[u8]; 6] = [
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 33\r\nconnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.0}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 33\r\nconnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.2}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 33\r\nconnection: keep-alive\r\n\r\n{\"approved\":true,\"fraud_score\":0.4}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\nconnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.6}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\nconnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":0.8}",
    b"HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: 34\r\nconnection: keep-alive\r\n\r\n{\"approved\":false,\"fraud_score\":1.0}",
];

pub static READY_RESPONSE: &[u8] =
    b"HTTP/1.1 200 OK\r\ncontent-length: 2\r\nconnection: keep-alive\r\n\r\nOK";

pub static NOT_FOUND_RESPONSE: &[u8] =
    b"HTTP/1.1 404 Not Found\r\ncontent-length: 0\r\n\r\n";
