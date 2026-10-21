//! The QUIC inference server.
//!
//! The server is the bridge between the data plane and the inference engine:
//! it reads a generation request from a QUIC stream, hands it to a
//! [`TokenStreamProvider`] and forwards every generated token back to the
//! client until the generation is done.
//!
//! The provider is a trait, so the data plane does not depend on the engine.
//! The engine implements the trait and the node wires both together.

use std::sync::Arc;

use quinn::{Connection, Endpoint, RecvStream, SendStream};
use synapse_core::{SynapseError, TokenEvent, TokenStreamProvider};
use tokio::sync::oneshot;
use tracing::{debug, info, warn};

use crate::wire;

/// The QUIC server that serves generation requests.
#[derive(Clone)]
pub struct InferenceServer {
    /// The endpoint the server accepts connections on.
    endpoint: Endpoint,
    /// The provider of the token streams.
    provider: Arc<dyn TokenStreamProvider>,
}

impl InferenceServer {
    /// Builds a server for the given endpoint and provider.
    pub fn new(endpoint: Endpoint, provider: Arc<dyn TokenStreamProvider>) -> Self {
        Self { endpoint, provider }
    }

    /// Accepts connections until the endpoint is closed.
    ///
    /// Every connection is handled in its own task, so a slow client never
    /// blocks the others.
    pub async fn run(self) {
        while let Some(incoming) = self.endpoint.accept().await {
            let provider = Arc::clone(&self.provider);

            tokio::spawn(async move {
                match incoming.await {
                    Ok(connection) => {
                        let remote = connection.remote_address();
                        info!(%remote, "quic connection accepted");
                        if let Err(err) = handle_connection(provider, connection).await {
                            warn!(%remote, "the quic connection failed: {err}");
                        }
                    }
                    Err(err) => {
                        warn!("failed to establish the incoming quic connection: {err}");
                    }
                }
            });
        }
    }
}

/// Handles a single connection: dispatches the command and serves it.
async fn handle_connection(
    provider: Arc<dyn TokenStreamProvider>,
    connection: Connection,
) -> Result<(), SynapseError> {
    let (mut send, mut recv) = connection
        .accept_bi()
        .await
        .map_err(|err| SynapseError::Network(format!("failed to accept a quic stream: {err}")))?;
    // The command byte decides which protocol is spoken on the stream.
    let mut command = [0_u8; 1];
    recv.read_exact(&mut command)
        .await
        .map_err(|err| SynapseError::Network(format!("failed to read the command: {err}")))?;

    match command[0] {
        wire::CMD_GENERATE => {
            let (prompt, max_tokens) = wire::read_request_body(&mut recv).await.map_err(|err| {
                SynapseError::Network(format!("failed to read the request: {err}"))
            })?;

            serve_generation(provider, send, recv, prompt, max_tokens).await
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

            // The connection is kept alive until the client has finished its
            // side of the stream, otherwise the answer could be lost when the
            // connection is torn down.
            let _ =
                tokio::time::timeout(std::time::Duration::from_secs(5), recv.read_to_end(0)).await;

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
) -> Result<(), SynapseError> {
    let events = provider
        .generate_streaming(prompt, max_tokens as usize)
        .map_err(|err| SynapseError::Network(format!("failed to start the generation: {err}")))?;

    // The other half of the stream is watched: a client that resets or closes
    // it is gone, and the generation has to stop. Without this the generation
    // would run into the token limit, because a write to a dead peer does not
    // fail over UDP.
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
        }
    }

    // Dropping the receiver tells the engine that nobody is listening any
    // more, so the generation stops immediately.
    drop(events);

    if !terminal_event_sent {
        info!("the generation was aborted before it completed");
    }

    watcher.abort();
    let _ = send.finish();
    Ok(())
}
