//! The QUIC inference server.
//!
//! The server is the bridge between the data plane and the inference engine:
//! it reads a generation request from a QUIC stream, hands it to a
//! [`TokenStreamProvider`] and forwards every generated token back to the
//! client until the generation is done.
//!
//! The provider is a trait, so the data plane does not depend on the engine.
//! The engine implements the trait and the node wires both together.
//!
//! # Stopping a generation
//!
//! A generation has to stop as soon as nobody is waiting for its tokens, and
//! it has to stop without the client saying so. A client that is killed with
//! `SIGKILL`, that crashes or that loses connectivity never sends the stream
//! reset of a graceful abort, and a write to a dead peer does not fail over
//! UDP either. Three signals cover that, in the order the server sees them:
//!
//! * the other half of the stream ends or is reset, which is what an abort of
//!   a live client looks like,
//! * a write to the stream fails, which happens when the stream is reset but
//!   the connection is still usable,
//! * the connection closes, which the QUIC idle timeout reports for a peer
//!   that simply stopped answering.

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use quinn::{Connection, Endpoint, RecvStream, SendStream};
use synapse_core::{SecurityConfig, SynapseError, TokenEvent, TokenStreamProvider};
use tokio::sync::{oneshot, watch};
use tracing::{debug, info, warn};

type NonceStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<[u8; 32], (libp2p::PeerId, u64)>>>;

pub type PeerStateMap = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<libp2p::PeerId, PeerState>>>;

#[derive(Default)]
pub struct PeerState {
    inflight: u32,
    budget: std::collections::VecDeque<(std::time::Instant, u64)>,
}

use crate::wire;

/// How long the server waits for the client to close its side of a stream
/// after the terminal event of a generation was sent.
///
/// The wait is what keeps the terminal event from being discarded: a
/// connection that is closed immediately drops stream data that has not been
/// acknowledged yet. It only guards against a client that never closes.
const TERMINAL_EVENT_GRACE: Duration = Duration::from_secs(5);

/// The QUIC server that serves generation requests.
#[derive(Clone)]
pub struct InferenceServer {
    endpoint: Endpoint,
    provider: Arc<dyn TokenStreamProvider>,
    security: SecurityConfig,
    nonce_store: NonceStore,
    peer_state: PeerStateMap,
}

impl InferenceServer {
    pub fn new(
        endpoint: Endpoint,
        provider: Arc<dyn TokenStreamProvider>,
        security: SecurityConfig,
        nonce_store: NonceStore,
        peer_state: PeerStateMap,
    ) -> Self {
        Self { endpoint, provider, security, nonce_store, peer_state }
    }

    /// Accepts connections until the endpoint is closed.
    ///
    /// Every connection is handled in its own task, so a slow client never
    /// blocks the others.
    pub async fn run(self) {
        while let Some(incoming) = self.endpoint.accept().await {
            let provider = Arc::clone(&self.provider);
            let security = self.security.clone();
            let nonce_store = Arc::clone(&self.nonce_store);
            let peer_state = Arc::clone(&self.peer_state);

            tokio::spawn(async move {
                match incoming.await {
                    Ok(connection) => {
                        let remote = connection.remote_address();
                        info!(%remote, "quic connection accepted");
                        handle_connection(provider, connection, security, nonce_store, peer_state).await;
                    }
                    Err(err) => {
                        warn!("failed to establish the incoming quic connection: {err}");
                    }
                }
            });
        }
    }
}

/// Serves the streams of a single connection until the connection is closed.
///
/// The connection is the unit of cancellation: when it ends, for whatever
/// reason, every generation that is running on it is aborted. Each stream is
/// served in its own task, so one slow generation does not stop the
/// connection from accepting the next stream.
async fn handle_connection(
    provider: Arc<dyn TokenStreamProvider>,
    connection: Connection,
    security: SecurityConfig,
    nonce_store: NonceStore,
    peer_state: PeerStateMap,
) {
    let remote = connection.remote_address();
    let (cancel_sender, cancel_receiver) = watch::channel(false);
    let auth_state = Arc::new(tokio::sync::Mutex::new((false, None::<libp2p::PeerId>)));
    let allow_loopback = security.allow_loopback && remote.ip().is_loopback();

    // Auth timeout task.
    if !allow_loopback {
        let conn = connection.clone();
        let state = Arc::clone(&auth_state);
        let timeout = Duration::from_millis(security.auth_timeout_ms);
        tokio::spawn(async move {
            tokio::time::sleep(timeout).await;
            if !state.lock().await.0 {
                conn.close(quinn::VarInt::from_u32(1), b"auth timeout");
            }
        });
    } else {
        // loopback bypass: mark authenticated
        let mut s = auth_state.lock().await;
        s.0 = true;
    }

    loop {
        tokio::select! {
            biased;
            reason = connection.closed() => {
                info!(%remote, reason = %reason, "the quic connection closed");
                break;
            }
            stream = connection.accept_bi() => {
                let (send, recv) = match stream {
                    Ok(s) => s,
                    Err(err) => { info!(%remote, "failed to accept stream: {err}"); break; }
                };
                let provider = Arc::clone(&provider);
                let cancelled = cancel_receiver.clone();
                let auth_state = Arc::clone(&auth_state);
                let security = security.clone();
                let nonce_store = Arc::clone(&nonce_store);
                let peer_state = Arc::clone(&peer_state);
                tokio::spawn(async move {
                    if let Err(err) = handle_stream(provider, send, recv, cancelled, auth_state, security, allow_loopback, remote, nonce_store, peer_state).await {
                        warn!(%remote, "stream failed: {err}");
                    }
                });
            }
        }
    }
    let _ = cancel_sender.send(true);
}

/// Handles a single stream: dispatches the command and serves it.
#[allow(clippy::too_many_arguments, clippy::collapsible_if)]
async fn handle_stream(
    provider: Arc<dyn TokenStreamProvider>,
    mut send: SendStream,
    mut recv: RecvStream,
    cancelled: watch::Receiver<bool>,
    auth_state: Arc<tokio::sync::Mutex<(bool, Option<libp2p::PeerId>)>>,
    security: SecurityConfig,
    allow_loopback: bool,
    remote: SocketAddr,
    nonce_store: NonceStore,
    peer_state: PeerStateMap,
) -> Result<(), SynapseError> {
    // The command byte decides which protocol is spoken on the stream.
    let mut command = [0_u8; 1];
    recv.read_exact(&mut command)
        .await
        .map_err(|err| SynapseError::Network(format!("failed to read the command: {err}")))?;

    match command[0] {
        wire::CMD_AUTH => {
            if allow_loopback {
                let mut s = auth_state.lock().await;
                s.0 = true;
                let _ = send.finish();
                return Ok(());
            }
            let (pid_bytes, pk_bytes, nonce, sig) = wire::read_auth(&mut recv).await.map_err(|err| {
                SynapseError::Network(format!("failed to read auth: {err}"))
            })?;
            // verify
            let claimed = match libp2p::PeerId::from_bytes(&pid_bytes) {
                Ok(p) => p,
                Err(_) => {
                    warn!(%remote, "auth: bad peer id");
                    return Err(SynapseError::Network("bad peer id".into()));
                }
            };
            let pk = match libp2p::identity::PublicKey::try_decode_protobuf(&pk_bytes) {
                Ok(p) => p,
                Err(_) => {
                    warn!(%remote, "auth: bad public key");
                    return Err(SynapseError::Network("bad public key".into()));
                }
            };
            if pk.to_peer_id() != claimed {
                warn!(%remote, "auth: peer id mismatch");
                return Err(SynapseError::Network("peer id mismatch".into()));
            }
            // nonce lookup
            let issued = {
                let mut store = nonce_store.lock().unwrap_or_else(|e| e.into_inner());
                store.remove(&nonce)
            };
            let now_ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
            let mut ok = false;
            if let Some((issued_peer, expiry)) = issued {
                if expiry >= now_ms {
                    if issued_peer == claimed {
                        // verify signature
                        let mut msg = Vec::with_capacity(16 + 32 + pid_bytes.len());
                        msg.extend_from_slice(b"synapse-auth-v1");
                        msg.extend_from_slice(&nonce);
                        msg.extend_from_slice(&pid_bytes);
                        if pk.verify(&msg, &sig) {
                            ok = true;
                        }
                    }
                }
            }
            if !ok {
                warn!(%remote, peer = %claimed, "auth rejected");
                return Err(SynapseError::Network("auth rejected".into()));
            }
            // allowlist check
            if !allow_loopback {
                if security.mode == "allowlist" {
                    if !security.access.trusted_peers.iter().any(|p| p == &claimed.to_base58()) {
                        warn!(%remote, peer = %claimed, "peer not allowlisted");
                        return Err(SynapseError::Network("not allowlisted".into()));
                    }
                }
            }
            // bind
            {
                let mut s = auth_state.lock().await;
                s.0 = true;
                s.1 = Some(claimed);
            }
            let _ = send.finish();
            Ok(())
        }
        wire::CMD_GENERATE => {
            // require auth unless loopback bypass
            if !allow_loopback {
                let authed = auth_state.lock().await.0;
                if !authed {
                    warn!(%remote, "generate before auth");
                    return Err(SynapseError::Network("unauthenticated".into()));
                }
            }
            let (prompt, max_tokens) = wire::read_request_body(&mut recv).await.map_err(|err| {
                SynapseError::Network(format!("failed to read the request: {err}"))
            })?;

            // max_tokens ceiling
            if max_tokens as u64 > security.max_tokens_per_request {
                metrics::counter!("synapse_auth_rejected_total", "reason" => "max_tokens_exceeded").increment(1);
                return Err(SynapseError::Network("max_tokens_exceeded".into()));
            }

            // peer quotas (non-loopback only)
            if !allow_loopback {
                let peer = auth_state.lock().await.1;
                if let Some(peer) = peer {
                    // concurrency
                    {
                        let mut map = peer_state.lock().unwrap_or_else(|e| e.into_inner());
                        let st = map.entry(peer).or_default();
                        if st.inflight >= security.access.per_peer_concurrency as u32 {
                            metrics::counter!("synapse_peer_quota_rejected_total").increment(1);
                            metrics::counter!("synapse_auth_rejected_total", "reason" => "peer_quota").increment(1);
                            return Err(SynapseError::Network("peer_quota".into()));
                        }
                        st.inflight += 1;
                    }
                    // token budget
                    {
                        let now = std::time::Instant::now();
                        let mut map = peer_state.lock().unwrap_or_else(|e| e.into_inner());
                        let st = map.entry(peer).or_default();
                        while let Some((t, _)) = st.budget.front() {
                            if now.duration_since(*t).as_secs() >= 60 { st.budget.pop_front(); } else { break; }
                        }
                        let used: u64 = st.budget.iter().map(|(_, t)| t).sum();
                        if used + max_tokens as u64 > security.access.per_peer_tokens_per_minute {
                            if let Some(st) = map.get_mut(&peer) { st.inflight = st.inflight.saturating_sub(1); }
                            metrics::counter!("synapse_token_budget_rejected_total").increment(1);
                            metrics::counter!("synapse_auth_rejected_total", "reason" => "token_budget").increment(1);
                            return Err(SynapseError::Network("token_budget".into()));
                        }
                        st.budget.push_back((now, max_tokens as u64));
                    }
                }
            }

            let result = serve_generation(provider, send, recv, prompt, max_tokens, cancelled).await;

            // decrement inflight
            if !allow_loopback {
                let peer = auth_state.lock().await.1;
                if let Some(peer) = peer {
                    let mut map = peer_state.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(st) = map.get_mut(&peer) {
                        st.inflight = st.inflight.saturating_sub(1);
                    }
                }
            }

            result
        }
        wire::CMD_PING => {
            // The data plane smoke test of the control plane phase.
            let mut payload = [0_u8; 3];
            recv.read_exact(&mut payload)
                .await
                .map_err(|err| SynapseError::Network(format!("failed to read the ping: {err}")))?;

            send.write_all(b"PONG")
                .await
                .map_err(|err| SynapseError::Network(format!("failed to send the pong: {err}")))?;
            let _ = send.finish();

            // The stream is kept alive until the client has finished its side
            // of it, otherwise the answer could be lost when the stream is
            // torn down.
            let _ = tokio::time::timeout(TERMINAL_EVENT_GRACE, recv.read_to_end(0)).await;

            debug!("quic ping answered");
            Ok(())
        }
        other => Err(SynapseError::Network(format!(
            "unknown command {other:#04x}"
        ))),
    }
}

/// Serves a generation request on an open stream.
async fn serve_generation(
    provider: Arc<dyn TokenStreamProvider>,
    mut send: SendStream,
    mut recv: RecvStream,
    prompt: String,
    max_tokens: u32,
    mut cancelled: watch::Receiver<bool>,
) -> Result<(), SynapseError> {
    let events = match provider.generate_streaming(prompt, max_tokens as usize) {
        Ok(events) => events,
        Err(err) => {
            // A generation can be rejected before it starts. An at-capacity
            // node does exactly that, and it is an answer, not a dead
            // connection: reporting it as a protocol error event tells the
            // client why the request was not served, instead of leaving it
            // with a vague "the stream ended" message.
            warn!(%err, "the generation was rejected");

            let event = TokenEvent::Error {
                message: err.to_string(),
            };

            if let Err(write_err) = wire::write_event(&mut send, &event).await {
                warn!("failed to report the rejected generation: {write_err}");
            }

            // Keep the stream alive until the client has read the event and
            // closed its side, for the same reason the terminal event of a
            // normal generation is not followed by an immediate close.
            let _ = tokio::time::timeout(TERMINAL_EVENT_GRACE, recv.read_to_end(0)).await;
            let _ = send.finish();

            return Ok(());
        }
    };

    // The generation itself is instrumented by the engine: the gauge, the
    // token counter and the time-to-first-token histogram cover generations
    // that are served over QUIC as well as generations that run locally.

    // The other half of the stream is watched: a client that resets or closes
    // it is gone, and the generation has to stop.
    let (abort_sender, mut abort_receiver) = oneshot::channel();
    let watcher = tokio::spawn(async move {
        let _ = recv.read_to_end(0).await;
        let _ = abort_sender.send(());
    });

    let mut terminal_event_sent = false;

    loop {
        tokio::select! {
            event = events.recv_async() => {
                let Ok(event) = event else {
                    warn!("the generation event stream ended");
                    break;
                };

                let is_terminal = matches!(event, TokenEvent::Done { .. } | TokenEvent::Error { .. });

                if let Err(err) = wire::write_event(&mut send, &event).await {
                    // The client is gone. Dropping the receiver of the event
                    // stream makes the engine abort the generation: the token
                    // sender sees that nobody is listening any more.
                    warn!("the client disconnected, the generation is aborted: {err}");
                    break;
                }

                if is_terminal {
                    terminal_event_sent = true;
                    break;
                }
            }
            _ = &mut abort_receiver => {
                // The client closed or reset the stream: the generation is
                // aborted. Dropping the receiver stops the engine.
                warn!("the client disconnected, the generation is aborted");
                break;
            }
            _ = cancelled.changed() => {
                // The connection is closed, which is the only signal that
                // reaches the server when the client was killed or lost its
                // network: it cannot reset the stream any more. The channel
                // reports an error when the connection task is gone, which
                // means the same thing here.
                warn!("the quic connection is gone, the generation is aborted");
                break;
            }
        }
    }

    // Dropping the receiver tells the engine that nobody is listening any
    // more, so the generation stops immediately.
    drop(events);

    if !terminal_event_sent {
        info!("the generation was aborted before it completed");
    }

    // The last event of a generation is only delivered reliably while the
    // connection is still alive: closing it right away discards stream data
    // that has not been acknowledged yet, and the client would report a lost
    // connection instead of the terminal event. The watcher finishes as soon
    // as the client closes its side of the stream, which it does after it has
    // seen the terminal event; a client that is gone has already finished it
    // as well. The timeout only guards against a client that never closes.
    let _ = tokio::time::timeout(TERMINAL_EVENT_GRACE, watcher).await;

    let _ = send.finish();
    Ok(())
}
