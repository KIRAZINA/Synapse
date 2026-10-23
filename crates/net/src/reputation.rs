//! The reputation ledger: what a peer's answers have been worth.
//!
//! Authentication (Phase 11) proves *who* answered a request. It says nothing
//! about whether the answer was *correct*: an authenticated peer can return
//! garbage, a truncated generation, or the output of a cheaper model and the
//! transport will happily deliver it. The ledger is the memory that makes
//! redundant verification ([`crate::verify`]) worth running more than once:
//! every comparison that agrees or disagrees moves a peer's score, and the
//! routing table prefers peers that have a history of agreeing.
//!
//! # Bounds
//!
//! A ledger that only ever grows is a memory leak with extra steps, and a
//! ledger that only ever accumulates is a score that a peer can farm. Both are
//! handled here:
//!
//! * every score is clamped to `[min, max]`, so a long run of good answers
//!   cannot buy unlimited credit and a single bad answer cannot bury a peer
//!   forever;
//! * an entry that has not been touched for `entry_ttl_ms` is pruned, and the
//!   number of entries is capped at `max_entries` (the least recently updated
//!   entry is evicted first), so the ledger has a hard memory ceiling.
//!
//! Pruning is the same idea as the idle-peer pruning of the routing table: an
//! entry that nobody has vouched for in an hour is not evidence any more.

use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use synapse_core::VerificationConfig;

/// The reputation of a single peer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReputationEntry {
    /// The current score, always within `[min, max]`.
    pub score: f64,
    /// How many comparisons this peer has taken part in.
    pub observations: u64,
    /// How many of those it agreed with the majority.
    pub agreements: u64,
    /// When the entry was last updated.
    last_updated: Instant,
}

impl ReputationEntry {
    /// The fraction of comparisons this peer agreed with, or `None` when it
    /// has never been compared.
    ///
    /// This is the number an operator actually wants: a peer with a score of
    /// `+3` from three agreements is a different proposition from a peer with
    /// `+3` from three hundred.
    pub fn agreement_rate(&self) -> Option<f64> {
        if self.observations == 0 {
            return None;
        }
        Some(self.agreements as f64 / self.observations as f64)
    }
}

/// A bounded, prune-able ledger of per-peer reputation.
///
/// The ledger is cheap to share: it is wrapped in an [`Arc`](std::sync::Arc)
/// by the routing table and read on the routing hot path, so every operation
/// takes the lock for a single map access and never holds it across an await.
#[derive(Debug)]
pub struct ReputationLedger {
    entries: Mutex<HashMap<String, ReputationEntry>>,
    min: f64,
    max: f64,
    ttl: Duration,
    max_entries: usize,
}

impl ReputationLedger {
    /// Builds a ledger from the verification policy.
    pub fn from_config(config: &VerificationConfig) -> Self {
        Self::new(
            config.min_reputation,
            config.max_reputation,
            Duration::from_millis(config.entry_ttl_ms),
            config.max_entries,
        )
    }

    /// Builds a ledger with explicit bounds.
    pub fn new(min: f64, max: f64, ttl: Duration, max_entries: usize) -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
            min,
            max,
            ttl,
            // A ledger with a capacity of zero would silently forget every
            // observation, which is worse than a ledger of one.
            max_entries: max_entries.max(1),
        }
    }

    /// The score of a peer. An unknown peer scores `0.0`: neutral, not
    /// trusted and not condemned.
    pub fn score(&self, peer: &str) -> f64 {
        self.entry(peer).map_or(0.0, |entry| entry.score)
    }

    /// The full entry of a peer, if it has one.
    pub fn entry(&self, peer: &str) -> Option<ReputationEntry> {
        let entries = self.lock();
        entries.get(peer).copied()
    }

    /// Applies a delta to a peer's score and records the observation.
    ///
    /// `agreed` says whether the peer matched the majority; it is what the
    /// agreement rate is computed from. The score is clamped, so the ledger
    /// can never leave `[min, max]` no matter how many deltas arrive.
    pub fn record(&self, peer: &str, delta: f64, agreed: bool) {
        // A NaN delta would poison the score and make every comparison with it
        // false, so it is dropped instead of being stored.
        if !delta.is_finite() {
            return;
        }

        let mut entries = self.lock();

        if !entries.contains_key(peer) && entries.len() >= self.max_entries {
            // The ledger is full: evict the least recently updated entry. A
            // peer that has not been compared in a while is the cheapest thing
            // to forget, and it keeps the memory ceiling hard.
            if let Some(oldest) = entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_updated)
                .map(|(peer, _)| peer.clone())
            {
                entries.remove(&oldest);
            }
        }

        let entry = entries.entry(peer.to_owned()).or_insert(ReputationEntry {
            score: 0.0,
            observations: 0,
            agreements: 0,
            last_updated: Instant::now(),
        });

        entry.score = (entry.score + delta).clamp(self.min, self.max);
        entry.observations = entry.observations.saturating_add(1);
        if agreed {
            entry.agreements = entry.agreements.saturating_add(1);
        }
        entry.last_updated = Instant::now();
    }

    /// Removes the entries that have not been updated within the time to live.
    ///
    /// Returns the number of removed entries. This is called from the same
    /// tick that prunes stale telemetry, so the ledger cannot grow without
    /// bound on a node that has seen many peers over a long uptime.
    pub fn prune(&self) -> usize {
        let now = Instant::now();
        let mut entries = self.lock();
        let before = entries.len();

        entries.retain(|_, entry| now.duration_since(entry.last_updated) <= self.ttl);

        before - entries.len()
    }

    /// The number of peers in the ledger.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether the ledger is empty.
    pub fn is_empty(&self) -> bool {
        self.lock().is_empty()
    }

    /// A snapshot of every entry, sorted by peer id.
    ///
    /// This is what an operator endpoint renders: the scores are observable
    /// per node, not only as a mesh-wide aggregate.
    pub fn snapshot(&self) -> Vec<(String, ReputationEntry)> {
        let entries = self.lock();
        let mut snapshot: Vec<(String, ReputationEntry)> = entries
            .iter()
            .map(|(peer, entry)| (peer.clone(), *entry))
            .collect();
        snapshot.sort_by(|left, right| left.0.cmp(&right.0));
        snapshot
    }

    /// Locks the entries, recovering from a poisoned mutex.
    ///
    /// A panic in another thread must not take the ledger down: the worst case
    /// of continuing is a slightly stale score, which is strictly better than
    /// panicking on the routing hot path.
    fn lock(&self) -> std::sync::MutexGuard<'_, HashMap<String, ReputationEntry>> {
        self.entries.lock().unwrap_or_else(|err| err.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ledger() -> ReputationLedger {
        ReputationLedger::new(-10.0, 10.0, Duration::from_secs(60), 4)
    }

    #[test]
    fn an_unknown_peer_is_neutral() {
        let ledger = ledger();
        assert_eq!(ledger.score("nobody"), 0.0);
        assert!(ledger.is_empty());
    }

    #[test]
    fn a_score_is_clamped_to_its_bounds() {
        let ledger = ledger();
        for _ in 0..100 {
            ledger.record("peer", 1.0, true);
        }
        assert_eq!(ledger.score("peer"), 10.0);

        for _ in 0..100 {
            ledger.record("peer", -1.0, false);
        }
        assert_eq!(ledger.score("peer"), -10.0);
    }

    #[test]
    fn the_agreement_rate_is_tracked() {
        let ledger = ledger();
        ledger.record("peer", 1.0, true);
        ledger.record("peer", 1.0, true);
        ledger.record("peer", -1.0, false);

        let entry = ledger.entry("peer").unwrap_or_else(|| panic!("no entry"));
        assert_eq!(entry.observations, 3);
        assert_eq!(entry.agreements, 2);
        assert_eq!(entry.agreement_rate(), Some(2.0 / 3.0));
    }

    #[test]
    fn a_non_finite_delta_is_ignored() {
        let ledger = ledger();
        ledger.record("peer", f64::NAN, true);
        assert_eq!(ledger.score("peer"), 0.0);
        assert!(ledger.is_empty());
    }

    #[test]
    fn the_ledger_evicts_when_it_is_full() {
        let ledger = ledger();
        for index in 0..4 {
            ledger.record(&format!("peer{index}"), 1.0, true);
        }
        assert_eq!(ledger.len(), 4);

        // The fifth peer pushes the least recently updated one out.
        ledger.record("peer4", 1.0, true);
        assert_eq!(ledger.len(), 4);
    }

    #[test]
    fn prune_removes_entries_that_went_quiet() {
        let ledger = ReputationLedger::new(-10.0, 10.0, Duration::from_millis(0), 4);
        ledger.record("peer", 1.0, true);
        assert_eq!(ledger.prune(), 1);
        assert!(ledger.is_empty());
    }
}
