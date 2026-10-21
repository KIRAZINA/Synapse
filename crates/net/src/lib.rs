//! The networking layer of Synapse.
//!
//! Synapse uses a split plane network:
//!
//! * the **control plane** ([`discovery`]) is a `libp2p` swarm. It is used to
//!   find peers (Kademlia), to learn their addresses (Identify) and to
//!   exchange the QUIC endpoint information of a peer (request/response).
//! * the **data plane** ([`transport`]) is a `quinn` endpoint that carries the
//!   inference traffic.
//!
//! The two planes are strictly decoupled: they only exchange the
//! [`transport::HandshakeInfo`] structure, so the QUIC stack can be replaced
//! without touching the swarm, and the swarm can run without a data plane.

pub mod client;
pub mod discovery;
pub mod server;
pub mod transport;
pub mod wire;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;

use futures_util::StreamExt;
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::{identity, noise, tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder};
use synapse_core::{SynapseError, TokenStreamProvider};
use tracing::{info, warn};

use crate::discovery::{
    HandshakeRequest, HandshakeResponse, SynapseBehaviour, SynapseBehaviourEvent,
};
use crate::server::InferenceServer;
use crate::transport::QuicTransport;

pub use libp2p::Multiaddr as PeerMultiaddr;

/// The multiaddr the control plane listens on.
const LISTEN_ADDRESS: &str = "/ip4/0.0.0.0/tcp/0";

/// A Synapse node that takes part in the peer to peer network.
pub struct NetworkNode {
    /// The control plane.
    swarm: Swarm<SynapseBehaviour>,
    /// The data plane.
    quic: QuicTransport,
    /// The provider that serves the generation requests of the peers.
    provider: Arc<dyn TokenStreamProvider>,
    /// The identity of this node.
    peer_id: PeerId,
    /// The node that is dialed on start up, if any.
    bootstrap: Option<Multiaddr>,
    /// The addresses of the peers that are connected over libp2p.
    peer_addresses: HashMap<PeerId, SocketAddr>,
}

impl NetworkNode {
    /// Creates a node with a fresh identity.
    pub async fn new(
        bootstrap: Option<Multiaddr>,
        provider: Arc<dyn TokenStreamProvider>,
    ) -> Result<Self, SynapseError> {
        let keypair = identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        let behaviour = SynapseBehaviour::new(&keypair)?;

        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                |key: &identity::Keypair| noise::Config::new(key),
                yamux::Config::default,
            )
            .map_err(|err| {
                SynapseError::Network(format!("failed to configure the tcp transport: {err}"))
            })?
            .with_behaviour(|_key| behaviour)
            .map_err(|never: std::convert::Infallible| match never {})?
            .with_swarm_config(|config| {
                // The control plane connection has to survive the handshake:
                // with the default idle timeout of zero libp2p closes a
                // connection as soon as it becomes idle, which can drop the
                // handshake response.
                config.with_idle_connection_timeout(std::time::Duration::from_secs(60))
            })
            .build();

        let quic = QuicTransport::new().await?;

        info!(%peer_id, "network node initialised");

        Ok(Self {
            swarm,
            quic,
            provider,
            peer_id,
            bootstrap,
            peer_addresses: HashMap::new(),
        })
    }

    /// The identity of this node.
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    /// The data plane of this node.
    pub fn quic(&self) -> &QuicTransport {
        &self.quic
    }

    /// The provider that serves the generation requests of the peers.
    pub fn provider(&self) -> &Arc<dyn TokenStreamProvider> {
        &self.provider
    }
    /// The addresses the control plane is listening on.
    pub fn listen_addresses(&self) -> Vec<Multiaddr> {
        self.swarm.listeners().cloned().collect()
    }

    /// Runs the node until the swarm or the data plane is closed.
    pub async fn run(&mut self) -> Result<(), SynapseError> {
        let listen_address: Multiaddr = LISTEN_ADDRESS
            .parse()
            .map_err(|err| SynapseError::Network(format!("invalid listen address: {err}")))?;
        self.swarm
            .listen_on(listen_address)
            .map_err(|err| SynapseError::Network(format!("failed to listen: {err}")))?;

        if let Some(bootstrap) = self.bootstrap.clone() {
            discovery::dial_bootstrap(&mut self.swarm, &bootstrap);
        }

        // The data plane runs in its own task: the swarm loop must never be
        // blocked by a QUIC operation. The server hands the generation
        // requests to the inference provider.
        let server = InferenceServer::new(self.quic.endpoint().clone(), Arc::clone(&self.provider));
        tokio::spawn(async move {
            server.run().await;
        });

        loop {
            let event = self.swarm.select_next_some().await;
            self.handle_event(event).await;
        }
    }

    /// Handles a single event of the control plane.
    async fn handle_event(&mut self, event: SwarmEvent<SynapseBehaviourEvent>) {
        match event {
            SwarmEvent::NewListenAddr { address, .. } => {
                discovery::log_listen_address(&address, self.peer_id);
            }
            SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                let remote = endpoint.get_remote_address();
                if let Some(ip) = discovery::ip_from_multiaddr(remote) {
                    self.peer_addresses.insert(peer_id, SocketAddr::new(ip, 0));
                }
                info!(%peer_id, %remote, "libp2p connection established");
                self.request_handshake(peer_id);
            }
            SwarmEvent::ConnectionClosed {
                peer_id,
                num_established,
                cause,
                ..
            } => {
                info!(%peer_id, num_established, ?cause, "libp2p connection closed");
                if num_established == 0 {
                    self.peer_addresses.remove(&peer_id);
                }
            }
            SwarmEvent::OutgoingConnectionError { error, .. } => {
                warn!(%error, "libp2p connection error");
            }
            SwarmEvent::Behaviour(event) => self.handle_behaviour_event(event),
            _ => {}
        }
    }

    /// Handles an event of one of the control plane behaviours.
    fn handle_behaviour_event(&mut self, event: SynapseBehaviourEvent) {
        match event {
            SynapseBehaviourEvent::RequestResponse(request_response::Event::Message {
                peer,
                message,
            }) => match message {
                request_response::Message::Request { channel, .. } => {
                    let response =
                        discovery::handshake_response(self.quic.local_port(), self.quic.cert_der());
                    if self
                        .swarm
                        .behaviour_mut()
                        .request_response
                        .send_response(channel, response)
                        .is_err()
                    {
                        warn!(%peer, "failed to answer the handshake request");
                    } else {
                        info!(%peer, "answered the handshake request");
                    }
                }
                request_response::Message::Response { response, .. } => {
                    info!(
                        %peer,
                        quic_port = response.quic_port,
                        "received the handshake response"
                    );
                    self.connect_quic(peer, response);
                }
            },
            SynapseBehaviourEvent::Kad(event) => discovery::log_routing_update(&event),
            SynapseBehaviourEvent::Identify(libp2p::identify::Event::Received {
                peer_id,
                info,
                ..
            }) => {
                info!(%peer_id, addresses = ?info.listen_addrs, "identified peer");
            }
            _ => {}
        }
    }

    /// Asks a peer for its QUIC endpoint information.
    fn request_handshake(&mut self, peer: PeerId) {
        self.swarm
            .behaviour_mut()
            .request_response
            .send_request(&peer, HandshakeRequest);
        info!(%peer, "sent the handshake request");
    }

    /// Starts the QUIC connection to a peer in a separate task.
    fn connect_quic(&self, peer: PeerId, response: HandshakeResponse) {
        let Some(control_plane_addr) = self.peer_addresses.get(&peer) else {
            warn!(%peer, "no known address for the peer, cannot start the quic connection");
            return;
        };

        let addr = SocketAddr::new(control_plane_addr.ip(), response.quic_port);
        let endpoint = self.quic.endpoint().clone();

        tokio::spawn(async move {
            if let Err(err) = transport::connect(endpoint, addr, response.quic_cert_der).await {
                warn!(%peer, %addr, "the quic connection failed: {err}");
            }
        });
    }
}
