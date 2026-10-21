//! The libp2p control plane of a Synapse node.
//!
//! The control plane is only responsible for discovery and for exchanging the
//! QUIC endpoint information of a peer. It never carries inference data: the
//! tokens of a prompt travel over the QUIC data plane, which is described in
//! [`crate::transport`].

use futures_util::StreamExt;
use libp2p::identify;
use libp2p::kad::store::MemoryStore;
use libp2p::request_response::cbor::Behaviour as CborBehaviour;
use libp2p::request_response::ProtocolSupport;
use libp2p::swarm::NetworkBehaviour;
use libp2p::{identity, kad, request_response, Multiaddr, PeerId, StreamProtocol};
use serde::{Deserialize, Serialize};
use synapse_core::SynapseError;
use tracing::{debug, info, warn};

/// The protocol used to exchange the QUIC endpoint information.
pub const HANDSHAKE_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/handshake/1.0");

/// The Kademlia protocol used for peer discovery.
pub const KAD_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/kad/1.0");

/// The identify protocol used to learn the addresses of the other peers.
pub const IDENTIFY_PROTOCOL: StreamProtocol = StreamProtocol::new("/synapse/identify/1.0");

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

        Ok(Self {
            kad,
            request_response,
            identify,
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
