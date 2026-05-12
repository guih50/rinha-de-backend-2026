# busybox:musl provides 'test' for the healthcheck (adds only ~1MB)
FROM busybox:musl
COPY target/x86_64-unknown-linux-musl/release/rinha /app
ENTRYPOINT ["/app"]
