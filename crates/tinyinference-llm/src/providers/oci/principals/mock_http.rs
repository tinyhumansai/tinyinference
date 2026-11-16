//! A tiny loopback HTTP server for exchange tests (no external network).

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};

/// A request captured by the mock server.
#[derive(Debug, Clone)]
pub(super) struct Recorded {
    pub(super) method: String,
    pub(super) path: String,
    pub(super) headers: Vec<(String, String)>,
    pub(super) body: String,
}

impl Recorded {
    pub(super) fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(key, _)| key.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

/// Starts a server answering each request with `handler(&recorded) ->
/// (status, body)`. Returns the base URL and the shared request log.
pub(super) fn start(
    handler: impl Fn(&Recorded) -> (u16, String) + Send + 'static,
) -> (String, Arc<Mutex<Vec<Recorded>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind loopback");
    let url = format!("http://{}", listener.local_addr().unwrap());
    let log = Arc::new(Mutex::new(Vec::new()));
    let thread_log = Arc::clone(&log);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { break };
            let mut data = Vec::new();
            let mut chunk = [0u8; 4096];
            let header_end = loop {
                let read = stream.read(&mut chunk).unwrap_or(0);
                if read == 0 {
                    break None;
                }
                data.extend_from_slice(&chunk[..read]);
                if let Some(pos) = data.windows(4).position(|w| w == b"\r\n\r\n") {
                    break Some(pos + 4);
                }
            };
            let Some(header_end) = header_end else {
                continue;
            };
            let head = String::from_utf8_lossy(&data[..header_end]).to_string();
            let mut lines = head.lines();
            let request_line = lines.next().unwrap_or_default().to_string();
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let path = parts.next().unwrap_or_default().to_string();
            let headers: Vec<(String, String)> = lines
                .filter_map(|line| line.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            let length = headers
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
                .and_then(|(_, v)| v.parse::<usize>().ok())
                .unwrap_or(0);
            while data.len() < header_end + length {
                let read = stream.read(&mut chunk).unwrap_or(0);
                if read == 0 {
                    break;
                }
                data.extend_from_slice(&chunk[..read]);
            }
            let body = String::from_utf8_lossy(&data[header_end..]).to_string();
            let recorded = Recorded {
                method,
                path,
                headers,
                body,
            };
            let (status, reply) = handler(&recorded);
            thread_log.lock().unwrap().push(recorded);
            let response = format!(
                "HTTP/1.1 {status} X\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{reply}",
                reply.len()
            );
            let _ = stream.write_all(response.as_bytes());
        }
    });
    (url, log)
}

/// Builds an unsigned JWT-shaped token with the given claims.
pub(super) fn jwt(iat: i64, exp: i64) -> String {
    use base64::Engine as _;
    use base64::engine::general_purpose::URL_SAFE_NO_PAD as B;
    format!(
        "{}.{}.sig",
        B.encode(r#"{"alg":"RS256"}"#),
        B.encode(format!(r#"{{"iat":{iat},"exp":{exp}}}"#))
    )
}
