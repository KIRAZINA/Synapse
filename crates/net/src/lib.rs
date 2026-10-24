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
pub mod orchestrator;
pub mod reputation;
pub mod routing;
pub mod server;
pub mod transport;
pub mod verify;
pub mod wire;

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

type NonceStore = std::sync::Arc<std::sync::Mutex<std::collections::HashMap<[u8; 32], (PeerId, u64)>>>;

use futures_util::StreamExt;
use libp2p::request_response;
use libp2p::swarm::SwarmEvent;
use libp2p::{noise, tcp, yamux, Multiaddr, PeerId, Swarm, SwarmBuilder};
use synapse_core::{
    NodeTelemetry, SecurityConfig, SynapseError, TokenStreamProvider, VerificationConfig,
};
use tracing::{debug, info, warn};

use crate::discovery::{
    telemetry_request_topic, telemetry_topic, HandshakeRequest, HandshakeResponse, SynapseBehaviour,
    SynapseBehaviourEvent, TELEMETRY_REQUEST_PAYLOAD,
};
use crate::routing::{encode_telemetry, now_ms, RoutingTable};
use crate::server::InferenceServer;
use crate::transport::{QuicTransport, TransportSettings};
use crate::verify::VerificationPolicy;



/// How often the telemetry of a node is broadcast.
pub const TELEMETRY_INTERVAL: Duration = Duration::from_secs(5);
pub use libp2p::Multiaddr as PeerMultiaddr;

/// The models a node has actually loaded.
///
/// The telemetry is built from this registry, never from the configuration: a
/// node that fails to load its model must not advertise it. The loader records
/// a model here only after the engine reported success.
#[derive(Debug, Default)]
pub struct ServedModels {
    /// The names of the loaded models. The engine keeps one model resident,
    /// so loading a model replaces the previous entry.
    names: std::sync::RwLock<Vec<String>>,
}

impl ServedModels {
    /// Creates an empty registry: the node serves no model yet.
    pub fn new() -> Self {
        Self::default()
    }

    /// Records the model that was just loaded.
    ///
    /// This replaces the previous entry, because the engine keeps exactly one
    /// model resident.
    pub fn set_loaded(&self, name: impl Into<String>) {
        let name = name.into();

        match self.names.write() {
            Ok(mut names) => {
                names.clear();
                names.push(name);
            }
            // A poisoned lock only affects the next telemetry broadcast; the
            // node itself keeps running.
            Err(err) => warn!("the served model registry is poisoned: {err}"),
        }
    }

    /// The names of the loaded models.
    pub fn names(&self) -> Vec<String> {
        match self.names.read() {
            Ok(names) => names.clone(),
            Err(err) => {
                warn!("the served model registry is poisoned: {err}");
                Vec::new()
            }
        }
    }
}

/// The settings a node hands to the networking layer.
#[derive(Debug, Clone)]
pub struct NetworkSettings {
    /// The multiaddr the control plane listens on.
    pub listen_addr: Multiaddr,
    /// The bootstrap peers that are dialed on start up.
    pub bootstrap: Vec<Multiaddr>,
    /// The QUIC timeouts of every connection.
    pub transport: TransportSettings,
    /// The models the node has loaded and may advertise.
    pub served_models: Arc<ServedModels>,
    /// The security policy of the node.
    pub security: SecurityConfig,
    /// The compute-integrity policy of the node.
    pub verification: VerificationConfig,
}

/// A Synapse node that takes part in the peer to peer network.
pub struct NetworkNode {
    /// The control plane.
    swarm: Swarm<SynapseBehaviour>,
    /// The data plane.
    quic: QuicTransport,
    /// The provider that serves the generation requests of the peers.
    provider: Arc<dyn TokenStreamProvider>,
    /// The telemetry of the known peers.
    routing: Arc<RoutingTable>,
    /// The identity of this node.
    peer_id: PeerId,
    /// The multiaddr the control plane listens on.
    listen_addr: Multiaddr,
    /// The peers that are dialed on start up.
    bootstrap: Vec<Multiaddr>,
    /// The models this node has loaded.
    served_models: Arc<ServedModels>,
    /// The addresses of the peers that are connected over libp2p.
    peer_addresses: HashMap<PeerId, SocketAddr>,
    /// Nonces issued for QUIC data-plane authentication.
    nonce_store: NonceStore,
    /// The security policy of the node.
    security: SecurityConfig,
    /// The compute-integrity policy of the node.
    verification: VerificationPolicy,
}

impl NetworkNode {
    /// Creates a node with a fresh identity.
    pub async fn new(
        settings: NetworkSettings,
        provider: Arc<dyn TokenStreamProvider>,
    ) -> Result<Self, SynapseError> {
        let keypair = libp2p::identity::Keypair::generate_ed25519();
        let peer_id = keypair.public().to_peer_id();

        let behaviour = SynapseBehaviour::new(&keypair)?;

        let swarm = SwarmBuilder::with_existing_identity(keypair)
            .with_tokio()
            .with_tcp(
                tcp::Config::default().nodelay(true),
                |key: &libp2p::identity::Keypair| noise::Config::new(key),
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

        let quic = QuicTransport::with_settings(settings.transport).await?;

        info!(%peer_id, "network node initialised");

        Ok(Self {
            swarm,
            quic,
            provider,
            routing: Arc::new(RoutingTable::with_policy(
                routing::DEFAULT_TTL,
                &settings.verification,
            )),
            peer_id,
            listen_addr: settings.listen_addr,
            bootstrap: settings.bootstrap,
            served_models: settings.served_models,
            peer_addresses: HashMap::new(),
            nonce_store: Arc::new(std::sync::Mutex::new(HashMap::new())),
            security: settings.security,
            verification: VerificationPolicy::new(settings.verification),
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

    /// The telemetry of the known peers.
    pub fn routing(&self) -> &Arc<RoutingTable> {
        &self.routing
    }

    /// The compute-integrity policy of this node.
    pub fn verification(&self) -> &VerificationPolicy {
        &self.verification
    }

    /// The addresses the control plane is listening on.
    pub fn listen_addresses(&self) -> Vec<Multiaddr> {
        self.swarm.listeners().cloned().collect()
    }

    /// Runs the node until the swarm or the data plane is closed.
    pub async fn run(&mut self) -> Result<(), SynapseError> {
        self.swarm
            .listen_on(self.listen_addr.clone())
            .map_err(|err| SynapseError::Network(format!("failed to listen: {err}")))?;

        if !self.bootstrap.is_empty() {
            for bootstrap in self.bootstrap.clone() {
                discovery::register_bootstrap(&mut self.swarm, &bootstrap);
                discovery::dial_bootstrap(&mut self.swarm, &bootstrap);
            }

            if let Err(err) = self.swarm.behaviour_mut().kad.bootstrap() {
                debug!("kademlia bootstrap is not possible yet: {err}");
            }

            let local_peer_id = *self.swarm.local_peer_id();
            let _ = self
                .swarm
                .behaviour_mut()
                .kad
                .get_closest_peers(local_peer_id);
        }

        // The node joins the telemetry topic, so that the clients can see how
        // busy it is, and the refresh topic, so that it can answer a client
        // that joins the mesh.
        for topic in [telemetry_topic(), telemetry_request_topic()] {
            if let Err(err) = self.swarm.behaviour_mut().gossipsub.subscribe(&topic) {
                warn!(%topic, "failed to subscribe to a telemetry topic: {err}");
            }
        }

        // The data plane runs in its own task: the swarm loop must never be
        // blocked by a QUIC operation. The server hands the generation
        // requests to the inference provider.
        let server = InferenceServer::new(
            self.quic.endpoint().clone(),
            Arc::clone(&self.provider),
            self.security.clone(),
            Arc::clone(&self.nonce_store),
            Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
        );
        tokio::spawn(async move {
            server.run().await;
        });

        let mut telemetry_tick = tokio::time::interval(TELEMETRY_INTERVAL);
        telemetry_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);

        loop {
            tokio::select! {
                event = self.swarm.select_next_some() => {
                    self.handle_event(event).await;
                }
                _ = telemetry_tick.tick() => {
                    self.publish_telemetry();
                    self.routing.remove_stale();
                    // The reputation of every peer is observable per node, not
                    // only as a mesh-wide aggregate, and the ledger is pruned
                    // on the same tick as the telemetry so neither structure
                    // can grow without bound.
                    verify::publish_reputation(self.routing.reputation_ledger());
                }
            }
        }
    }

    /// Broadcasts the telemetry of this node.
    fn publish_telemetry(&mut self) {
        // The listener is bound to `0.0.0.0`, which is not a destination.
        // Advertise the address of the interface that the default route uses.
        let addresses = self
            .swarm
            .listeners()
            .map(discovery::advertised_address)
            .map(|address| address.to_string())
            .collect();

        let telemetry = NodeTelemetry {
            peer_id: self.peer_id.to_base58(),
            quic_port: self.quic.local_port(),
            addresses,
            active_requests: self.provider.active_requests(),
            max_capacity: self.provider.max_capacity(),
            // The registry only ever contains a model that the engine loaded,
            // so a broken model file never turns into a false advertisement.
            supported_models: self.served_models.names(),
            timestamp_ms: now_ms(),
        };

        let payload = match encode_telemetry(&telemetry) {
            Ok(payload) => payload,
            Err(err) => {
                warn!("{err}");
                return;
            }
        };

        // `publish` only queues the message on the swarm; the event loop is
        // never blocked by the network.
        match self
            .swarm
            .behaviour_mut()
            .gossipsub
            .publish(telemetry_topic(), payload)
        {
            Ok(_) => {
                info!(
                    active_requests = telemetry.active_requests,
                    max_capacity = telemetry.max_capacity,
                    supported_models = ?telemetry.supported_models,
                    "telemetry published"
                );
            }
            Err(err) => {
                // A node without peers in the mesh has nobody to gossip to.
                debug!("failed to publish the telemetry: {err}");
            }
        }
    }

    /// Asks every node in the mesh to publish its telemetry right away.
    fn request_telemetry(&mut self) {
        match self.swarm.behaviour_mut().gossipsub.publish(
            telemetry_request_topic(),
            TELEMETRY_REQUEST_PAYLOAD.to_vec(),
        ) {
            Ok(_) => debug!("requested an immediate telemetry broadcast"),
            Err(err) => {
                // A node without peers in the mesh has nobody to ask.
                debug!("failed to request the telemetry: {err}");
            }
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
                let remote = endpoint.get_remote_address().clone();
                if let Some(ip) = discovery::ip_from_multiaddr(&remote) {
                    self.peer_addresses.insert(peer_id, SocketAddr::new(ip, 0));
                }

                // Kademlia learns the real address of the peer from the
                // connection, not only from the wildcard listen address.
                let address = discovery::strip_p2p(&remote);
                if !address.is_empty() {
                    self.swarm
                        .behaviour_mut()
                        .kad
                        .add_address(&peer_id, address);
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
                    // Issue a single-use nonce bound to this peer's identity.
                    let mut nonce = [0u8; 32];
                    let _ = getrandom::getrandom(&mut nonce);
                    let now_ms = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .map(|d| d.as_millis() as u64)
                        .unwrap_or(0);
                    let expiry = now_ms.saturating_add(self.security.nonce_ttl_ms);
                    if let Ok(mut store) = self.nonce_store.lock() {
                        // Bound the store: evict expired entries first.
                        store.retain(|_, (_, exp)| *exp > now_ms);
                        if store.len() < 1024 {
                            store.insert(nonce, (peer, expiry));
                        }
                    }
                    let response = discovery::handshake_response(
                        self.quic.local_port(),
                        self.quic.cert_der(),
                        nonce,
                        expiry,
                    );
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
            SynapseBehaviourEvent::Kad(event) => {
                match &event {
                    libp2p::kad::Event::RoutablePeer { peer, address }
                    | libp2p::kad::Event::PendingRoutablePeer { peer, address } => {
                        if let Some(ip) = discovery::ip_from_multiaddr(address) {
                            self.peer_addresses
                                .entry(*peer)
                                .or_insert(SocketAddr::new(ip, 0));
                        }
                    }
                    libp2p::kad::Event::RoutingUpdated { peer, addresses, .. } => {
                        if let Some(ip) =
                            addresses.iter().find_map(discovery::ip_from_multiaddr)
                        {
                            self.peer_addresses
                                .entry(*peer)
                                .or_insert(SocketAddr::new(ip, 0));
                        }
                    }
                    _ => {}
                }
                discovery::log_routing_update(&event);
            }
            SynapseBehaviourEvent::Gossipsub(event) => {
                match event {
                    libp2p::gossipsub::Event::Message { message, .. } => {
                        if message.topic == telemetry_request_topic().hash() {
                            if message.data.as_slice() == TELEMETRY_REQUEST_PAYLOAD {
                                // A client joined: answer immediately.
                                self.publish_telemetry();
                            } else {
                                debug!("ignoring a malformed telemetry refresh request");
                            }
                            return;
                        }

                        if message.topic != telemetry_topic().hash() {
                            return;
                        }

                        let Some(peer) = message.source else {
                            debug!("ignoring an unsigned telemetry message");
                            return;
                        };

                        match self.routing.ingest(peer, &message.data) {
                            Ok(telemetry) => {
                                info!(
                                    %peer,
                                    active_requests = telemetry.active_requests,
                                    max_capacity = telemetry.max_capacity,
                                    "received the telemetry of a peer"
                                );
                            }
                            Err(err) => {
                                warn!(%peer, "ignoring a malformed telemetry message: {err}");
                            }
                        }
                    }
                    libp2p::gossipsub::Event::Subscribed { peer_id, topic } => {
                        if topic != telemetry_topic().hash() {
                            debug!(%peer_id, %topic, "a peer joined another telemetry topic");
                            return;
                        }

                        info!(%peer_id, %topic, "a peer joined the telemetry topic");
                        // Publish right away and ask every other node to do
                        // the same, so that a client that just joined does not
                        // have to wait for the next five second tick of each
                        // node.
                        self.publish_telemetry();
                        self.request_telemetry();
                    }
                    _ => {}
                }
            }
            SynapseBehaviourEvent::Identify(libp2p::identify::Event::Received {
                peer_id,
                info,
                ..
            }) => {
                for address in info.listen_addrs {
                    let address = discovery::strip_p2p(&address);
                    if let Some(ip) = discovery::ip_from_multiaddr(&address) {
                        if !ip.is_unspecified() {
                            self.peer_addresses
                                .entry(peer_id)
                                .or_insert(SocketAddr::new(ip, 0));
                        }
                    }
                    if !address.is_empty() {
                        self.swarm
                            .behaviour_mut()
                            .kad
                            .add_address(&peer_id, address);
                    }
                }
                info!(%peer_id, "identified peer");
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


/// Generates a fresh libp2p keypair for data-plane authentication.
pub fn generate_keypair() -> libp2p::identity::Keypair {
    libp2p::identity::Keypair::generate_ed25519()
}
