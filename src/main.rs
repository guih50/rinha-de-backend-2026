mod index;
mod parse;
mod responses;
mod vectorize;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::body::Incoming;
use hyper::server::conn::http1;
use hyper::service::service_fn;
use hyper::{Method, Request, Response, StatusCode};
use hyper_util::rt::TokioIo;
use std::os::unix::fs::PermissionsExt;
use std::sync::LazyLock;
use tokio::net::{TcpListener, UnixListener};

use index::IvfIndex;
use responses::{NOT_FOUND_RESPONSE, READY_RESPONSE, RESPONSE_BODIES, RESPONSE_CONTENT_LENGTHS};

// ── Static index loaded from embedded binary ──────────────────────────────────
static INDEX_DATA: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ivf.bin"));
static INDEX: LazyLock<IvfIndex> = LazyLock::new(|| IvfIndex::from_static_bytes(INDEX_DATA));

static NPROBE: LazyLock<usize> = LazyLock::new(|| {
    std::env::var("NPROBE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50)
});

// ── HTTP handler ──────────────────────────────────────────────────────────────

async fn handle(req: Request<Incoming>) -> Result<Response<Full<Bytes>>, hyper::Error> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/ready") => Ok(ready_resp()),
        (&Method::POST, "/fraud-score") => {
            let body = req.collect().await?.to_bytes();
            Ok(fraud_score_handler(body))
        }
        _ => Ok(not_found_resp()),
    }
}

fn fraud_score_handler(body: Bytes) -> Response<Full<Bytes>> {
    let fraud_count = match parse::parse_transaction(&body) {
        Ok(tx) => {
            let query = vectorize::vectorize(&tx);
            INDEX.query(&query, *NPROBE)
        }
        Err(_) => 0,
    };
    let idx = fraud_count.min(5) as usize;
    Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header("content-length", RESPONSE_CONTENT_LENGTHS[idx])
        .header("connection", "keep-alive")
        .body(Full::new(Bytes::from_static(RESPONSE_BODIES[idx])))
        .unwrap()
}

fn ready_resp() -> Response<Full<Bytes>> {
    let _ = READY_RESPONSE;
    Response::builder()
        .status(StatusCode::OK)
        .header("content-length", "2")
        .header("connection", "keep-alive")
        .body(Full::new(Bytes::from_static(b"OK")))
        .unwrap()
}

fn not_found_resp() -> Response<Full<Bytes>> {
    let _ = NOT_FOUND_RESPONSE;
    Response::builder()
        .status(StatusCode::NOT_FOUND)
        .header("content-length", "0")
        .body(Full::new(Bytes::from_static(b"")))
        .unwrap()
}

// ── Server entry point ────────────────────────────────────────────────────────

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    // Force index initialization and pre-fault all 96 MB of vector pages into RAM
    let _ = &*INDEX;
    INDEX.warmup();
    let nprobe = *NPROBE;

    let listen_addr = std::env::var("LISTEN_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:9999".to_string());

    eprintln!(
        "rinha: nprobe={}, listening on {}",
        nprobe, listen_addr
    );

    if listen_addr.starts_with('/') {
        serve_unix(&listen_addr).await;
    } else {
        serve_tcp(&listen_addr).await;
    }
}

async fn serve_tcp(addr: &str) {
    let listener = TcpListener::bind(addr).await.expect("bind TCP");
    loop {
        let (stream, _) = listener.accept().await.expect("accept");
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service_fn(handle))
                .await
            {
                let s = e.to_string();
                if !s.contains("closed") && !s.contains("reset") {
                    eprintln!("conn err: {s}");
                }
            }
        });
    }
}

async fn serve_unix(path: &str) {
    let _ = std::fs::remove_file(path);
    let listener = UnixListener::bind(path).expect("bind UDS");
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o666))
        .expect("chmod socket");

    loop {
        let (stream, _) = listener.accept().await.expect("accept UDS");
        let io = TokioIo::new(stream);
        tokio::spawn(async move {
            if let Err(e) = http1::Builder::new()
                .keep_alive(true)
                .serve_connection(io, service_fn(handle))
                .await
            {
                let s = e.to_string();
                if !s.contains("closed") && !s.contains("reset") {
                    eprintln!("uds err: {s}");
                }
            }
        });
    }
}
