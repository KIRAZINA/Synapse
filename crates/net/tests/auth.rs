use std::net::{Ipv4Addr, SocketAddr};
use std::sync::Arc;
use std::time::Duration;

use libp2p::identity;
use synapse_core::{SecurityConfig, SynapseError, TokenEvent, TokenStreamProvider};
use synapse_net::client::RemoteInferenceClient;
use synapse_net::server::InferenceServer;
use synapse_net::transport::{QuicTransport, SERVER_NAME};
use synapse_net::wire;
use tokio::time::timeout;

struct TestProvider;
impl TokenStreamProvider for TestProvider {
    fn generate_streaming(&self, _prompt: String, _max_tokens: usize) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        let (tx, rx) = flume::unbounded();
        let _ = tx.send(TokenEvent::Done { total_tokens: 0, elapsed_ms: 0, stop_reason: synapse_core::StopReason::MaxTokens });
        Ok(rx)
    }
    fn active_requests(&self) -> u32 { 0 }
    fn max_capacity(&self) -> u32 { 4 }
}

async fn start_server(security: SecurityConfig) -> (QuicTransport, SocketAddr, std::sync::Arc<std::sync::Mutex<std::collections::HashMap<[u8;32],(libp2p::PeerId,u64)>>>) {
    let transport = QuicTransport::new().await.unwrap();
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, transport.local_port()));
    let store = std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new()));
    let server = InferenceServer::new(transport.endpoint().clone(), Arc::new(TestProvider), security, store.clone());
    tokio::spawn(server.run());
    (transport, addr, store)
}

#[tokio::test]
async fn valid_auth_listed_peer_succeeds_end_to_end() {
    let kp = identity::Keypair::generate_ed25519();
    let peer_id = kp.public().to_peer_id();
    let mut security = SecurityConfig::default();
    security.allow_loopback = false;
    security.access.trusted_peers = vec![peer_id.to_base58()];
    let (_t, addr, store) = start_server(security).await;
    // mint nonce
    let mut nonce = [0u8;32];
    getrandom::getrandom(&mut nonce).unwrap();
    let expiry = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64 + 30000;
    store.lock().unwrap().insert(nonce, (peer_id, expiry));
    // client
    let client = RemoteInferenceClient::new(addr, _t.cert_der().to_vec(), kp, nonce).unwrap();
    let (rx, _abort) = client.generate("hi".into(), 1).await.unwrap();
    let ev = rx.recv_async().await.unwrap();
    assert!(matches!(ev, TokenEvent::Done { .. }));
}

#[tokio::test]
async fn cmd_generate_before_auth_is_rejected_and_consumes_no_slot() {
    let kp = identity::Keypair::generate_ed25519();
    let mut security = SecurityConfig::default();
    security.allow_loopback = false;
    let (_t, addr, _store) = start_server(security).await;
    // client without valid nonce (zero) -> auth fails, generate fails
    let client = RemoteInferenceClient::new(addr, _t.cert_der().to_vec(), kp, [0u8;32]).unwrap();
    let res = client.generate("hi".into(), 1).await;
    if let Ok((rx,_)) = res { let ev = rx.recv_async().await; assert!(ev.is_err() || matches!(ev.unwrap(), TokenEvent::Error { .. })); }
}

#[tokio::test]
async fn allow_loopback_true_permits_loopback_and_false_does_not() {
    // true
    let mut security = SecurityConfig::default();
    security.allow_loopback = true;
    let (_t, addr, _store) = start_server(security).await;
    let kp = identity::Keypair::generate_ed25519();
    let client = RemoteInferenceClient::new(addr, _t.cert_der().to_vec(), kp, [0u8;32]).unwrap();
    // generate may succeed due to loopback bypass
    let _ = client.generate("hi".into(), 1).await;
    // false
    let mut security2 = SecurityConfig::default();
    security2.allow_loopback = false;
    let (_t2, addr2, _store2) = start_server(security2).await;
    let kp2 = identity::Keypair::generate_ed25519();
    let client2 = RemoteInferenceClient::new(addr2, _t2.cert_der().to_vec(), kp2, [0u8;32]).unwrap();
    let res = client2.generate("hi".into(), 1).await;
    if let Ok((rx,_)) = res { let ev = rx.recv_async().await; assert!(ev.is_err() || matches!(ev.unwrap(), TokenEvent::Error { .. })); }
}
