//! The libp2p control plane of a Synapse node.
//!
//! The control plane is only responsible for discovery and for exchanging the
//! QUIC endpoint information of a peer. It never carries inference data: the
//! tokens of a prompt travel over the QUIC data plane, which is described in
//! [`crate::transport`].

use std::collections::{HashMap, HashSet};
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::sync::OnceLock;
use std::time::Duration;

use futures_util::StreamExt;
use libp2p::gossipsub;
use libp2p::gossipsub::IdentTopic;
use libp2p::identify;
use libp2p::kad::store::MemoryStore;
use libp2p::multiaddr::Protocol;
use libp2p::request_response::cbor::Behaviour as CborBehaviour;
use libp2p::request_response::ProtocolSupport;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{identity, kad, request_response, Multiaddr, PeerId, StreamProtocol};
use serde::{Deserialize, Serialize};
use synapse_core::{NodeTelemetry, SynapseError};
use tracing::{debug, info, warn};

use crate::routing::RoutingTable;

/// The protocol used to exchange the QUIC endpoint information.
pub const HANDSHAKE_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/handshake/1.0");

/// The Kademlia protocol used for peer discovery.
pub const KAD_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/kad/1.0");

/// The identify protocol used to learn the addresses of the other peers.
pub const IDENTIFY_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/identify/1.0");

/// The GossipSub topic the nodes broadcast their telemetry on.
pub const TELEMETRY_TOPIC: &str = "synapse/telemetry/1.0.0";

/// The topic a node broadcasts a telemetry refresh request on.
///
/// A client only listens for three seconds, but the nodes publish every five
/// seconds. When a client joins, the node it is connected to sends a refresh
/// request, so every node answers immediately instead of at its next tick.
pub const TELEMETRY_REQUEST_TOPIC: &str = "synapse/telemetry/request/1.0.0";

/// The payload of a telemetry refresh request.
pub const TELEMETRY_REQUEST_PAYLOAD: &[u8] = b"synapse/telemetry/request-v1";

/// The telemetry topic of the mesh.
pub fn telemetry_topic() -> IdentTopic {
    IdentTopic::new(TELEMETRY_TOPIC)
}

/// The telemetry refresh request topic of the mesh.
pub fn telemetry_request_topic() -> IdentTopic {
    IdentTopic::new(TELEMETRY_REQUEST_TOPIC)
}

/// Extracts the peer id from a `/p2p/<id>` component of a multiaddr.
pub fn peer_id_from_multiaddr(address: &Multiaddr) -> Option<PeerId> {
    address.iter().find_map(|protocol| match protocol {
        Protocol::P2p(peer) => Some(peer),
        _ => None,
    })
}

/// Returns the address without its trailing `/p2p/<id>` component.
///
/// Kademlia appends the peer id itself; giving it an address that already
/// contains a `/p2p` component makes it reject the address.
pub fn strip_p2p(address: &Multiaddr) -> Multiaddr {
    let mut stripped = Multiaddr::empty();
    for protocol in address.iter() {
        if !matches!(protocol, Protocol::P2p(_)) {
            stripped.push(protocol);
        }
    }
    stripped
}

/// Whether an address can be dialed from another host.
///
/// A wildcard address (`0.0.0.0` / `::`) only tells a peer to listen on every
/// interface; it is not a destination.
pub fn is_usable_address(address: &Multiaddr) -> bool {
    address.iter().all(|protocol| match protocol {
        Protocol::Ip4(ip) => !ip.is_unspecified(),
        Protocol::Ip6(ip) => !ip.is_unspecified(),
        _ => true,
    })
}

/// The address of this host on the default route.
///
/// Nodes listen on `0.0.0.0`, which must not be gossiped as a destination.
/// Connecting a UDP socket does not send a packet: the operating system only
/// selects the interface (and therefore the local address) that would be used.
/// If the host has no route, the loopback address is advertised instead.
pub fn advertised_ip() -> IpAddr {
    static ADVERTISED_IP: OnceLock<IpAddr> = OnceLock::new();

    *ADVERTISED_IP.get_or_init(|| {
        std::net::UdpSocket::bind((Ipv4Addr::UNSPECIFIED, 0))
            .and_then(|socket| {
                socket.connect((Ipv4Addr::new(8, 8, 8, 8), 80))?;
                socket.local_addr()
            })
            .map(|address| address.ip())
            .ok()
            .filter(|ip| !ip.is_unspecified())
            .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
    })
}

/// Replaces a wildcard IP of a listen address with an address a peer can dial.
pub fn advertised_address(address: &Multiaddr) -> Multiaddr {
    let advertised = advertised_ip();
    let mut result = Multiaddr::empty();

    for protocol in address.iter() {
        match protocol {
            Protocol::Ip4(ip) if ip.is_unspecified() => match advertised {
                IpAddr::V4(ip) => result.push(Protocol::Ip4(ip)),
                IpAddr::V6(_) => result.push(Protocol::Ip4(Ipv4Addr::LOCALHOST)),
            },
            Protocol::Ip6(ip) if ip.is_unspecified() => {
                result.push(Protocol::Ip6(Ipv6Addr::LOCALHOST));
            }
            other => result.push(other),
        }
    }

    result
}

/// Gives Kademlia an address for the bootstrap peer.
///
/// Returns the peer id of the bootstrap node, or `None` when the multiaddr has
/// no `/p2p` component.
pub fn register_bootstrap(
    swarm: &mut libp2p::Swarm<SynapseBehaviour>,
    bootstrap: &Multiaddr,
) -> Option<PeerId> {
    let peer = peer_id_from_multiaddr(bootstrap)?;
    let address = strip_p2p(bootstrap);

    if address.is_empty() {
        warn!(%bootstrap, "the bootstrap address has no dialable component");
        return None;
    }

    if let kad::RoutingUpdate::Failed = swarm.behaviour_mut().kad.add_address(&peer, address) {
        warn!(%bootstrap, "kademlia rejected the bootstrap address");
    }

    Some(peer)
}

/// The request of the handshake protocol.
///
/// The request carries no data: the peer answers with its QUIC endpoint.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct HandshakeRequest;

/// The answer of the handshake protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeResponse {
    /// The UDP port the QUIC endpoint of the peer listens on.
    pub quic_port: u16,
    /// The DER encoded certificate of the peer.
    pub quic_cert_der: Vec<u8>,
}

/// The combined behaviour of the control plane.
#[derive(NetworkBehaviour)]
pub struct SynapseBehaviour {
    /// Kademlia, used to route peers and to discover new ones.
    pub kad: kad::Behaviour<MemoryStore>,
    /// The handshake protocol, used to exchange the QUIC endpoint information.
    pub request_response: CborBehaviour<HandshakeRequest, HandshakeResponse>,
    /// Identify, used to learn the addresses of the other peers.
    pub identify: identify::Behaviour,
    /// GossipSub, used to broadcast the telemetry of the nodes.
    pub gossipsub: gossipsub::Behaviour,
}

impl SynapseBehaviour {
    /// Builds the behaviour for the given identity.
    pub fn new(keypair: &identity::Keypair) -> Result<Self, SynapseError> {
        let peer_id = keypair.public().to_peer_id();

        let kad = kad::Behaviour::with_config(
            peer_id,
            MemoryStore::new(peer_id),
            kad::Config::new(KAD_PROTOCOL),
        );

        let request_response = CborBehaviour::new(
            [(HANDSHAKE_PROTOCOL, ProtocolSupport::Full)],
            request_response::Config::default(),
        );

        let mut identify_config =
            identify::Config::new(IDENTIFY_PROTOCOL.to_string(), keypair.public());
        identify_config.push_listen_addr_updates = true;
        let identify = identify::Behaviour::new(identify_config);

        let gossipsub_params = gossipsub::ConfigBuilder::default()
            .build()
            .map_err(|err| SynapseError::Network(format!("invalid gossipsub parameters: {err}")))?;
        let gossipsub = gossipsub::Behaviour::new(
            gossipsub::MessageAuthenticity::Signed(keypair.clone()),
            gossipsub_params,
        )
        .map_err(|err| {
            SynapseError::Network(format!("failed to create the gossipsub behaviour: {err}"))
        })?;

        Ok(Self {
            kad,
            request_response,
            identify,
            gossipsub,
        })
    }
}

/// Handles an inbound handshake request by answering with the local QUIC
/// endpoint information.
pub fn handshake_response(quic_port: u16, quic_cert_der: &[u8]) -> HandshakeResponse {
    HandshakeResponse {
        quic_port,
        quic_cert_der: quic_cert_der.to_vec(),
    }
}

/// Logs the addresses a node is listening on.
pub fn log_listen_address(address: &Multiaddr, peer_id: PeerId) {
    let mut full = address.clone();
    full.push(libp2p::multiaddr::Protocol::P2p(peer_id));
    info!(%address, %full, "libp2p listening");
}

/// Parses a bootstrap multiaddr.
pub fn parse_bootstrap(address: &str) -> Result<Multiaddr, SynapseError> {
    address.parse::<Multiaddr>().map_err(|err| {
        SynapseError::Network(format!("invalid bootstrap multiaddr '{address}': {err}"))
    })
}

/// Dials a bootstrap node and reports the outcome.
pub fn dial_bootstrap(swarm: &mut libp2p::Swarm<SynapseBehaviour>, bootstrap: &Multiaddr) {
    match swarm.dial(bootstrap.clone()) {
        Ok(()) => {
            info!(%bootstrap, "dialing the bootstrap node");
        }
        Err(err) => {
            warn!(%bootstrap, "failed to dial the bootstrap node: {err}");
        }
    }
}

/// Reports a Kademlia routing update.
pub fn log_routing_update(event: &kad::Event) {
    match event {
        kad::Event::RoutingUpdated {
            peer, is_new_peer, ..
        } => {
            debug!(%peer, is_new_peer, "kademlia routing updated");
        }
        kad::Event::RoutablePeer { peer, address } => {
            debug!(%peer, %address, "kademlia peer is routable");
        }
        kad::Event::UnroutablePeer { peer } => {
            debug!(%peer, "kademlia peer is not routable");
        }
        _ => {}
    }
}

/// Extracts the IP address of a multiaddr.
pub fn ip_from_multiaddr(address: &Multiaddr) -> Option<IpAddr> {
    address.iter().find_map(|protocol| match protocol {
        libp2p::multiaddr::Protocol::Ip4(ip) => Some(IpAddr::V4(ip)),
        libp2p::multiaddr::Protocol::Ip6(ip) => Some(IpAddr::V6(ip)),
        _ => None,
    })
}

/// Dials a peer and returns the QUIC endpoint information of that peer.
///
/// The swarm that is built here is only used to reach the peer: the
/// generation itself travels over the QUIC data plane, which is completely
/// independent of the control plane.
pub async fn fetch_quic_info(target: &Multiaddr) -> Result<(SocketAddr, Vec<u8>), SynapseError> {
    let keypair = identity::Keypair::generate_ed25519();
    let behaviour = SynapseBehaviour::new(&keypair)?;

    let mut swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default().nodelay(true),
            |key: &identity::Keypair| libp2p::noise::Config::new(key),
            libp2p::yamux::Config::default,
        )
        .map_err(|err| {
            SynapseError::Network(format!("failed to configure the tcp transport: {err}"))
        })?
        .with_behaviour(|_key| behaviour)
        .map_err(|never: std::convert::Infallible| match never {})?
        .with_swarm_config(|config| {
            // The connection has to survive the handshake: with the default
            // idle timeout of zero libp2p closes a connection as soon as it
            // becomes idle, which can drop the handshake response.
            config.with_idle_connection_timeout(std::time::Duration::from_secs(60))
        })
        .build();

    swarm
        .dial(target.clone())
        .map_err(|err| SynapseError::Network(format!("failed to dial {target}: {err}")))?;

    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut peer_ip: Option<IpAddr> = None;

    loop {
        let event = tokio::time::timeout_at(deadline, swarm.select_next_some())
            .await
            .map_err(|_| {
                SynapseError::Network(format!(
                    "timed out while waiting for the handshake with {target}"
                ))
            })?;

        match event {
            libp2p::swarm::SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                peer_ip = ip_from_multiaddr(endpoint.get_remote_address());
                info!(%peer_id, %target, "libp2p connection established, requesting the handshake");
                swarm
                    .behaviour_mut()
                    .request_response
                    .send_request(&peer_id, HandshakeRequest);
            }
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::RequestResponse(
                request_response::Event::Message {
                    message: request_response::Message::Response { response, .. },
                    ..
                },
            )) => {
                let ip = peer_ip.ok_or_else(|| {
                    SynapseError::Network("the address of the peer is unknown".to_owned())
                })?;

                info!(
                    quic_port = response.quic_port,
                    "received the handshake response"
                );
                return Ok((
                    SocketAddr::new(ip, response.quic_port),
                    response.quic_cert_der,
                ));
            }
            _ => {}
        }
    }
}

/// The telemetry and the addresses that a client learned from the mesh.
#[derive(Debug, Default)]
pub struct DiscoverySnapshot {
    /// The load aware routing table.
    routing: RoutingTable,
    /// The last address that was observed for every peer.
    addresses: HashMap<PeerId, Multiaddr>,
}

impl DiscoverySnapshot {
    /// The load aware routing table.
    pub fn routing(&self) -> &RoutingTable {
        &self.routing
    }

    /// The number of peers in the routing table.
    pub fn len(&self) -> usize {
        self.routing.len()
    }

    /// Whether the routing table is empty.
    pub fn is_empty(&self) -> bool {
        self.routing.is_empty()
    }

    /// The peer with the lowest load that can serve `model`.
    ///
    /// When `model` is `None` any peer with free capacity is eligible; when it
    /// is `Some`, only peers that advertise that model are considered.
    pub fn best_peer(&self, model: Option<&str>) -> Option<NodeTelemetry> {
        self.routing.get_best_peer(model)
    }

    /// The models that the discovered peers serve, sorted and deduplicated.
    pub fn models(&self) -> Vec<String> {
        self.routing.models()
    }

    /// The address of a peer, with its `/p2p/<id>` component restored.
    ///
    /// The observed control-plane address is preferred over the address the
    /// node gossiped about itself, because the observed address is known to be
    /// reachable from this client.
    pub fn address_of(&self, telemetry: &NodeTelemetry) -> Option<Multiaddr> {
        let peer = telemetry
            .peer_id
            .parse::<PeerId>()
            .ok()
            .or_else(|| {
                self.addresses
                    .iter()
                    .find_map(|(peer, _)| (peer.to_base58() == telemetry.peer_id).then_some(*peer))
            })?;

        let address = self
            .addresses
            .get(&peer)
            .filter(|address| is_usable_address(address))
            .cloned()
            .or_else(|| {
                telemetry
                    .addresses
                    .iter()
                    .filter_map(|address| address.parse::<Multiaddr>().ok())
                    .find(is_usable_address)
            })?;

        address.with_p2p(peer).ok()
    }
}

/// Dials a peer that was discovered over Kademlia.
fn dial_discovered(
    swarm: &mut libp2p::Swarm<SynapseBehaviour>,
    peer: PeerId,
    address: &Multiaddr,
    connected: &HashSet<PeerId>,
    dialing: &mut HashSet<PeerId>,
) {
    if peer == *swarm.local_peer_id() || connected.contains(&peer) || dialing.contains(&peer) {
        return;
    }

    let address = strip_p2p(address);
    if address.is_empty() || !is_usable_address(&address) {
        return;
    }

    let Ok(target) = address.with_p2p(peer) else {
        return;
    };

    match swarm.dial(target.clone()) {
        Ok(()) => {
            dialing.insert(peer);
            debug!(%peer, %target, "dialing a discovered peer");
        }
        Err(err) => {
            warn!(%peer, %target, "failed to dial a discovered peer: {err}");
        }
    }
}

/// Applies the routing information of a Kademlia event.
fn handle_kad_event(
    event: kad::Event,
    snapshot: &mut DiscoverySnapshot,
    connected: &HashSet<PeerId>,
    dialing: &mut HashSet<PeerId>,
    swarm: &mut libp2p::Swarm<SynapseBehaviour>,
) {
    match event {
        kad::Event::RoutingUpdated {
            peer, addresses, ..
        } => {
            if let Some(address) = addresses.iter().find(|address| is_usable_address(address)) {
                let address = strip_p2p(address);
                snapshot.addresses.entry(peer).or_insert(address.clone());
                dial_discovered(swarm, peer, &address, connected, dialing);
            }
            debug!(%peer, "kademlia routing updated");
        }
        kad::Event::RoutablePeer { peer, address } => {
            let address = strip_p2p(&address);
            if is_usable_address(&address) {
                snapshot.addresses.entry(peer).or_insert(address.clone());
                dial_discovered(swarm, peer, &address, connected, dialing);
            }
        }
        kad::Event::OutboundQueryProgressed {
            result: kad::QueryResult::GetClosestPeers(Ok(result)),
            ..
        } => {
            for info in result.peers {
                if let Some(address) = info.addrs.iter().find(|address| is_usable_address(address)) {
                    let address = strip_p2p(address);
                    snapshot.addresses.entry(info.peer_id).or_insert(address.clone());
                    dial_discovered(swarm, info.peer_id, &address, connected, dialing);
                }
            }
        }
        other => log_routing_update(&other),
    }
}

/// Joins the telemetry mesh and collects the telemetry of the peers.
///
/// The swarm is only used to receive the gossiped telemetry; the generation
/// itself travels over the QUIC data plane.
pub async fn collect_telemetry(
    bootstrap: &Multiaddr,
    duration: Duration,
) -> Result<DiscoverySnapshot, SynapseError> {
    let keypair = identity::Keypair::generate_ed25519();
    let behaviour = SynapseBehaviour::new(&keypair)?;

    let mut swarm = libp2p::SwarmBuilder::with_existing_identity(keypair)
        .with_tokio()
        .with_tcp(
            libp2p::tcp::Config::default().nodelay(true),
            |key: &identity::Keypair| libp2p::noise::Config::new(key),
            libp2p::yamux::Config::default,
        )
        .map_err(|err| {
            SynapseError::Network(format!("failed to configure the tcp transport: {err}"))
        })?
        .with_behaviour(|_key| behaviour)
        .map_err(|never: std::convert::Infallible| match never {})?
        .with_swarm_config(|config| config.with_idle_connection_timeout(Duration::from_secs(60)))
        .build();

    // Subscribe before dialing: gossipsub announces the subscription to every
    // newly connected peer, and a node publishes its telemetry immediately
    // when it sees a new subscriber.
    if let Err(err) = swarm
        .behaviour_mut()
        .gossipsub
        .subscribe(&telemetry_topic())
    {
        warn!("failed to subscribe to the telemetry topic: {err}");
    }

    let mut snapshot = DiscoverySnapshot::default();
    let mut connected: HashSet<PeerId> = HashSet::new();
    let mut dialing: HashSet<PeerId> = HashSet::new();

    let bootstrap_peer = register_bootstrap(&mut swarm, bootstrap);
    if let Err(err) = swarm.behaviour_mut().kad.bootstrap() {
        debug!("kademlia bootstrap is not possible yet: {err}");
    }

    // Ask the bootstrap node for every peer it knows, so that the client can
    // dial the other nodes and receive their telemetry right away instead of
    // waiting for the next five second broadcast of each node.
    let local_peer_id = *swarm.local_peer_id();
    let _ = swarm.behaviour_mut().kad.get_closest_peers(local_peer_id);

    swarm
        .dial(bootstrap.clone())
        .map_err(|err| SynapseError::Network(format!("failed to dial {bootstrap}: {err}")))?;
    if let Some(peer) = bootstrap_peer {
        dialing.insert(peer);
    }

    let deadline = tokio::time::Instant::now() + duration;

    loop {
        let event = tokio::time::timeout_at(deadline, swarm.select_next_some()).await;
        let Ok(event) = event else {
            break;
        };

        match event {
            libp2p::swarm::SwarmEvent::ConnectionEstablished {
                peer_id, endpoint, ..
            } => {
                connected.insert(peer_id);
                dialing.remove(&peer_id);

                let address = strip_p2p(endpoint.get_remote_address());
                if is_usable_address(&address) {
                    snapshot
                        .addresses
                        .entry(peer_id)
                        .or_insert_with(|| address.clone());
                }
                if !address.is_empty() {
                    swarm
                        .behaviour_mut()
                        .kad
                        .add_address(&peer_id, address);
                }
            }
            libp2p::swarm::SwarmEvent::ConnectionClosed {
                peer_id,
                num_established,
                ..
            } => {
                if num_established == 0 {
                    connected.remove(&peer_id);
                    dialing.remove(&peer_id);
                }
            }
            libp2p::swarm::SwarmEvent::OutgoingConnectionError {
                peer_id, error, ..
            } => {
                if let Some(peer) = peer_id {
                    dialing.remove(&peer);
                }
                warn!(?error, "failed to dial a peer");
            }
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::Gossipsub(
                gossipsub::Event::Message { message, .. },
            )) => {
                let Some(peer) = message.source else {
                    continue;
                };

                match snapshot.routing.ingest(peer, &message.data) {
                    Ok(telemetry) => {
                        info!(
                            %peer,
                            active_requests = telemetry.active_requests,
                            max_capacity = telemetry.max_capacity,
                            "received the telemetry of a peer"
                        );

                        // Dial the peer as soon as its telemetry is known:
                        // the subscription makes it publish its own telemetry
                        // immediately instead of at the next tick.
                        if !connected.contains(&peer) {
                            if let Some(address) = telemetry
                                .addresses
                                .iter()
                                .filter_map(|address| address.parse::<Multiaddr>().ok())
                                .find(is_usable_address)
                            {
                                snapshot
                                    .addresses
                                    .entry(peer)
                                    .or_insert_with(|| strip_p2p(&address));
                                dial_discovered(
                                    &mut swarm,
                                    peer,
                                    &address,
                                    &connected,
                                    &mut dialing,
                                );
                            }
                        }
                    }
                    Err(err) => {
                        warn!(%peer, "ignoring a malformed telemetry message: {err}");
                    }
                }
            }
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::Identify(
                identify::Event::Received { peer_id, info, .. },
            )) => {
                for address in info.listen_addrs {
                    let address = strip_p2p(&address);
                    if address.is_empty() {
                        continue;
                    }
                    if is_usable_address(&address) {
                        snapshot
                            .addresses
                            .entry(peer_id)
                            .or_insert_with(|| address.clone());
                    }
                    swarm
                        .behaviour_mut()
                        .kad
                        .add_address(&peer_id, address);
                }
            }
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::Kad(event)) => {
                handle_kad_event(
                    event,
                    &mut snapshot,
                    &connected,
                    &mut dialing,
                    &mut swarm,
                );
            }
            _ => {}
        }
    }

    Ok(snapshot)
}

/// Runs the control plane on its own.
///
/// The swarm listens on a random TCP port, dials the bootstrap node when one
/// is given and answers the handshake requests of the other peers with the
/// local QUIC endpoint information.
pub async fn run_discovery(
    mut swarm: libp2p::Swarm<SynapseBehaviour>,
    quic_port: u16,
    quic_cert_der: Vec<u8>,
    bootstrap: Option<Multiaddr>,
) {
    let listen_address = match "/ip4/0.0.0.0/tcp/0".parse::<Multiaddr>() {
        Ok(address) => address,
        Err(err) => {
            warn!("invalid listen address: {err}");
            return;
        }
    };

    if let Err(err) = swarm.listen_on(listen_address) {
        warn!("failed to listen: {err}");
        return;
    }

    if let Some(bootstrap) = bootstrap {
        dial_bootstrap(&mut swarm, &bootstrap);
    }

    loop {
        let event = swarm.select_next_some().await;

        match event {
            libp2p::swarm::SwarmEvent::NewListenAddr { address, .. } => {
                log_listen_address(&address, *swarm.local_peer_id());
            }
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::RequestResponse(
                request_response::Event::Message {
                    peer,
                    message: request_response::Message::Request { channel, .. },
                },
            )) => {
                let response = handshake_response(quic_port, &quic_cert_der);
                if swarm
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
            libp2p::swarm::SwarmEvent::Behaviour(SynapseBehaviourEvent::Kad(event)) => {
                log_routing_update(&event);
            }
            _ => {}
        }
    }
}
