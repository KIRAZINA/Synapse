//! The load aware routing table of a client.
//!
//! The table is filled with the telemetry that the nodes gossip about
//! themselves. A peer that stops broadcasting is considered dead and is
//! dropped from the table.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

use dashmap::mapref::entry::Entry;
use dashmap::DashMap;
use libp2p::PeerId;
use synapse_core::NodeTelemetry;

/// How long a telemetry entry stays valid.
pub const DEFAULT_TTL: Duration = Duration::from_secs(15);

/// The telemetry of the known peers, keyed by their peer id.
#[derive(Debug)]
pub struct RoutingTable {
    /// The telemetry of the peers.
    peers: DashMap<PeerId, NodeTelemetry>,
    /// How long a telemetry entry stays valid.
    ttl: Duration,
}

impl Default for RoutingTable {
    fn default() -> Self {
        Self::new()
    }
}

impl RoutingTable {
    /// Builds a table with the default time to live.
    pub fn new() -> Self {
        Self::with_ttl(DEFAULT_TTL)
    }

    /// Builds a table with the given time to live.
    pub fn with_ttl(ttl: Duration) -> Self {
        Self {
            peers: DashMap::new(),
            ttl,
        }
    }

    /// Inserts or updates the telemetry of a peer.
    ///
    /// An older telemetry payload never replaces a newer one: a message that
    /// was delayed in the mesh must not make a node look fresher than it is.
    pub fn update(&self, peer: PeerId, telemetry: NodeTelemetry) {
        match self.peers.entry(peer) {
            Entry::Occupied(mut entry) => {
                if telemetry.timestamp_ms >= entry.get().timestamp_ms {
                    entry.insert(telemetry);
                }
            }
            Entry::Vacant(entry) => {
                entry.insert(telemetry);
            }
        }
    }

    /// Decodes and stores a gossiped telemetry payload.
    ///
    /// Malformed payloads, messages with trailing bytes and messages whose
    /// advertised peer id does not match the peer that sent them are rejected;
    /// the caller decides whether to log them.
    pub fn ingest(&self, source: PeerId, payload: &[u8]) -> Result<NodeTelemetry, String> {
        let telemetry = decode_telemetry(payload, source)?;
        self.update(source, telemetry.clone());
        Ok(telemetry)
    }

    /// The number of peers in the table.
    pub fn len(&self) -> usize {
        self.peers.len()
    }

    /// Whether the table is empty.
    pub fn is_empty(&self) -> bool {
        self.peers.is_empty()
    }

    /// The telemetry of every known peer.
    pub fn peers(&self) -> Vec<NodeTelemetry> {
        self.peers
            .iter()
            .filter(|entry| !self.is_stale(entry.value()))
            .map(|entry| entry.value().clone())
            .collect()
    }

    /// The peer with the lowest load.
    ///
    /// Peers that are at capacity are skipped: a request that is sent to them
    /// would only queue up. The peer with the fewest active requests wins; ties
    /// are broken by the peer id so that the choice is stable.
    pub fn get_best_peer(&self) -> Option<NodeTelemetry> {
        self.peers
            .iter()
            .filter(|entry| !self.is_stale(entry.value()))
            .map(|entry| entry.value().clone())
            .filter(|telemetry| telemetry.active_requests < telemetry.max_capacity)
            .min_by(|left, right| {
                left.active_requests
                    .cmp(&right.active_requests)
                    .then_with(|| left.peer_id.cmp(&right.peer_id))
            })
    }

    /// Removes the entries that are older than the time to live.
    ///
    /// Returns the number of removed entries.
    pub fn remove_stale(&self) -> usize {
        let now = now_ms();
        let stale: Vec<PeerId> = self
            .peers
            .iter()
            .filter(|entry| {
                now.saturating_sub(entry.value().timestamp_ms) > self.ttl.as_millis() as u64
            })
            .map(|entry| *entry.key())
            .collect();

        let removed = stale.len();
        for peer in stale {
            self.peers.remove(&peer);
        }
        removed
    }

    /// Whether the telemetry of a peer is older than the time to live.
    fn is_stale(&self, telemetry: &NodeTelemetry) -> bool {
        now_ms().saturating_sub(telemetry.timestamp_ms) > self.ttl.as_millis() as u64
    }
}

/// Decodes a telemetry payload and checks that it was sent by the peer it
/// claims to describe.
pub fn decode_telemetry(payload: &[u8], source: PeerId) -> Result<NodeTelemetry, String> {
    let (telemetry, consumed) = bincode::serde::decode_from_slice::<NodeTelemetry, _>(
        payload,
        bincode::config::standard(),
    )
    .map_err(|err| format!("invalid telemetry payload: {err}"))?;

    if consumed != payload.len() {
        return Err(format!(
            "the telemetry payload has {} trailing bytes",
            payload.len() - consumed
        ));
    }

    if telemetry.peer_id != source.to_base58() {
        return Err(format!(
            "the telemetry claims to be from '{}' but was sent by {source}",
            telemetry.peer_id
        ));
    }

    Ok(telemetry)
}

/// Encodes a telemetry payload for the gossip topic.
pub fn encode_telemetry(telemetry: &NodeTelemetry) -> Result<Vec<u8>, String> {
    bincode::serde::encode_to_vec(telemetry, bincode::config::standard())
        .map_err(|err| format!("failed to serialize the telemetry: {err}"))
}

/// The current unix time in milliseconds.
pub fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn peer() -> PeerId {
        PeerId::random()
    }

    fn telemetry(peer: &PeerId, active_requests: u32, max_capacity: u32) -> NodeTelemetry {
        NodeTelemetry {
            peer_id: peer.to_base58(),
            quic_port: 9000,
            addresses: vec!["/ip4/127.0.0.1/tcp/4001".to_owned()],
            active_requests,
            max_capacity,
            supported_models: vec!["tinyllama-1.1b".to_owned()],
            timestamp_ms: now_ms(),
        }
    }

    #[test]
    fn default_table_uses_the_default_ttl() {
        assert_eq!(RoutingTable::default().ttl, DEFAULT_TTL);
    }

    #[test]
    fn picks_the_least_loaded_peer() {
        let table = RoutingTable::new();
        let busy = peer();
        let idle = peer();

        table.update(busy, telemetry(&busy, 3, 4));
        table.update(idle, telemetry(&idle, 0, 1));

        let best = table.get_best_peer();
        assert_eq!(best.map(|telemetry| telemetry.peer_id), Some(idle.to_base58()));
    }

    #[test]
    fn skips_peers_that_are_at_capacity() {
        let table = RoutingTable::new();
        let full = peer();

        table.update(full, telemetry(&full, 1, 1));

        assert!(table.get_best_peer().is_none());
    }

    #[test]
    fn removes_entries_that_stopped_broadcasting() {
        let table = RoutingTable::with_ttl(Duration::from_secs(15));
        let dead = peer();
        let mut stale = telemetry(&dead, 0, 1);
        stale.timestamp_ms = now_ms().saturating_sub(16_000);
        table.update(dead, stale);

        assert!(table.get_best_peer().is_none());
        assert_eq!(table.remove_stale(), 1);
        assert!(table.is_empty());
    }

    #[test]
    fn ignores_older_telemetry() {
        let table = RoutingTable::new();
        let node = peer();
        let mut newer = telemetry(&node, 3, 4);
        newer.timestamp_ms = now_ms();
        table.update(node, newer);

        let mut older = telemetry(&node, 0, 4);
        older.timestamp_ms = now_ms().saturating_sub(10_000);
        table.update(node, older);

        let best = table.get_best_peer();
        assert_eq!(best.map(|telemetry| telemetry.active_requests), Some(3));
    }

    #[test]
    fn round_trips_and_rejects_malformed_payloads() {
        let node = peer();
        let payload = telemetry(&node, 1, 2);
        let encoded = match encode_telemetry(&payload) {
            Ok(encoded) => encoded,
            Err(err) => panic!("encoding failed: {err}"),
        };

        let decoded = match decode_telemetry(&encoded, node) {
            Ok(decoded) => decoded,
            Err(err) => panic!("decoding failed: {err}"),
        };
        assert_eq!(decoded, payload);

        let other = peer();
        assert!(decode_telemetry(&encoded, other).is_err());
        assert!(decode_telemetry(&encoded[..encoded.len() - 1], node).is_err());
    }
}
