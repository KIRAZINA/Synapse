//! The QUIC client that requests a generation from a remote node.

use std::net::SocketAddr;

use quinn::{Endpoint, VarInt};
use synapse_core::{SynapseError, TokenEvent};
use tokio::sync::oneshot;
use tracing::{info, warn};

use crate::transport::{build_client_config, SERVER_NAME};
use crate::wire;

/// The error code that is used when a generation is aborted.
const ABORT_CODE: u32 = 1;

/// A handle that aborts a running remote generation.
#[derive(Debug)]
pub struct AbortHandle {
    sender: Option<oneshot::Sender<()>>,
}

impl AbortHandle {
    /// Aborts the generation.
    ///
    /// The stream is reset, so the remote node stops generating immediately
    /// instead of running into the token limit.
    pub fn abort(mut self) {
        if let Some(sender) = self.sender.take() {
            let _ = sender.send(());
        }
    }
}

/// A client that runs a generation on a remote Synapse node.
#[derive(Debug, Clone)]
pub struct RemoteInferenceClient {
    /// The local QUIC endpoint.
    endpoint: Endpoint,
    /// The QUIC address of the remote node.
    addr: SocketAddr,
    /// The DER encoded certificate of the remote node.
    cert_der: Vec<u8>,
}

impl RemoteInferenceClient {
    /// Builds a client for a remote node.
    ///
    /// The certificate is the one the remote node published over the control
    /// plane; it is pinned, so the connection is only accepted when the
    /// remote node owns the matching private key.
    pub fn new(addr: SocketAddr, cert_der: Vec<u8>) -> Result<Self, SynapseError> {
        let bind_addr: SocketAddr = ([0, 0, 0, 0], 0).into();
        let endpoint = Endpoint::client(bind_addr).map_err(|err| {
            SynapseError::Network(format!("failed to create the quic endpoint: {err}"))
        })?;

        Ok(Self {
            endpoint,
            addr,
            cert_der,
        })
    }

    /// The address of the remote node.
    pub fn remote_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Starts a generation on the remote node.
    ///
    /// The returned receiver yields the events of the generation. Dropping it
    /// closes the connection, which makes the remote node abort the
    /// generation; the returned abort handle does the same explicitly.
    pub async fn generate(
        &self,
        prompt: String,
        max_tokens: u32,
    ) -> Result<(flume::Receiver<TokenEvent>, AbortHandle), SynapseError> {
        let client_config = build_client_config(self.cert_der.clone())?;

        let connecting = self
            .endpoint
            .connect_with(client_config, self.addr, SERVER_NAME)
            .map_err(|err| {
                SynapseError::Network(format!("failed to start the quic connection: {err}"))
            })?;

        let connection = connecting.await.map_err(|err| {
            SynapseError::Network(format!(
                "the quic connection to {} failed: {err}",
                self.addr
            ))
        })?;
        info!(addr = %self.addr, "quic connection established");

        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .map_err(|err| SynapseError::Network(format!("failed to open a quic stream: {err}")))?;

        wire::write_request(&mut send, &prompt, max_tokens)
            .await
            .map_err(|err| SynapseError::Network(format!("failed to send the request: {err}")))?;

        let (event_sender, event_receiver) = flume::unbounded();
        let (abort_sender, mut abort_receiver) = oneshot::channel();

        tokio::spawn(async move {
            loop {
                tokio::select! {
                    result = wire::read_event(&mut recv) => {
                        match result {
                            Ok(event) => {
                                let is_terminal = matches!(
                                    event,
                                    TokenEvent::Done { .. } | TokenEvent::Error { .. }
                                );

                                if event_sender.send(event).is_err() {
                                    // The consumer is gone: reset the stream so
                                    // that the remote node stops generating.
                                    let _ = send.reset(VarInt::from_u32(ABORT_CODE));
                                    break;
                                }

                                if is_terminal {
                                    break;
                                }
                            }
                            Err(err) => {
                                // The stream ended without a terminal event.
                                warn!("the generation stream ended unexpectedly: {err}");
                                let _ = event_sender.send(TokenEvent::Error {
                                    message: format!(
                                        "the generation stream ended unexpectedly: {err}"
                                    ),
                                });
                                break;
                            }
                        }
                    }
                    _ = &mut abort_receiver => {
                        // The consumer aborted the generation: reset the
                        // stream, the remote node stops generating.
                        info!("the remote generation is aborted");
                        let _ = send.reset(VarInt::from_u32(ABORT_CODE));
                        break;
                    }
                }
            }

            let _ = send.finish();
            connection.close(VarInt::from_u32(0), b"done");
        });

        Ok((
            event_receiver,
            AbortHandle {
                sender: Some(abort_sender),
            },
        ))
    }
}
