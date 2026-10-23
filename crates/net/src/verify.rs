//! Redundant verification: is the answer that came back actually correct?
//!
//! # The gap this closes
//!
//! Phase 11 authenticates a peer: the QUIC handshake proves that the node on
//! the other end owns the key it claims to own. That is *identity*, and it is
//! all it is. An authenticated node is free to return garbage, a truncated
//! generation, or the output of a smaller and cheaper model, and every layer
//! below this one will deliver it faithfully.
//!
//! Verification is the pragmatic answer: run the same request on several
//! nodes, compare the outputs, and let the majority decide. There is no zkML
//! here and this is not a proof of computation; it is a spot check that
//! catches gross misbehaviour at a bounded cost.
//!
//! # The cost, stated plainly
//!
//! A verified request costs `K` times the compute of a normal one, where `K`
//! is the redundancy. That is why verification is **off by default** and why
//! it is only ever reached in two bounded ways:
//!
//! * **sampled** — [`VerificationPolicy::should_sample`] verifies one request
//!   in every `N`, so the overhead is `1/N` of the traffic times `K`;
//! * **explicit** — a caller asks for a single request to be verified.
//!
//! It is never forced on all traffic. See [`synapse_core::VerificationConfig`].
//!
//! # The determinism constraint
//!
//! Comparing outputs is only valid when the participants were *supposed* to
//! produce the same output. That requires the same model, the same
//! quantization, and greedy decoding (temperature 0) with the same seed. Any
//! sampling at all makes the comparison meaningless: two honest nodes would
//! disagree and one of them would be punished for it.
//!
//! [`VerificationRequest::new`] therefore refuses a request whose
//! [`Determinism`] is not comparable, and [`verify`] refuses a set of
//! participants whose configurations are not identical. Refusing is the
//! correct behaviour: a false accusation against an honest node is worse than
//! not verifying at all.

use std::collections::HashMap;

use synapse_core::{Determinism, VerificationConfig};

/// Why a verification did not produce a verdict.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VerificationError {
    /// The decoding configuration cannot produce comparable outputs.
    ///
    /// This is the determinism gate: temperature above zero, or an
    /// unspecified model or quantization.
    NotDeterministic {
        /// A human readable reason, safe to log.
        reason: String,
    },
    /// The participants do not all run the same configuration.
    IncompatibleParticipants {
        /// A human readable reason, safe to log.
        reason: String,
    },
    /// Fewer participants than the policy requires.
    NotEnoughParticipants {
        /// How many were supplied.
        got: usize,
        /// How many are required.
        need: usize,
    },
}

impl std::fmt::Display for VerificationError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotDeterministic { reason } => {
                write!(formatter, "the request is not deterministic: {reason}")
            }
            Self::IncompatibleParticipants { reason } => {
                write!(formatter, "the participants are not comparable: {reason}")
            }
            Self::NotEnoughParticipants { got, need } => write!(
                formatter,
                "verification needs at least {need} participants, got {got}"
            ),
        }
    }
}

impl std::error::Error for VerificationError {}

/// The outcome of comparing the outputs of several nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Verdict {
    /// A strict majority produced the same output.
    Accepted {
        /// The output the majority agreed on.
        output: String,
        /// The peers that produced it.
        agreeing: Vec<String>,
        /// The peers that produced something else.
        outliers: Vec<String>,
    },
    /// No output reached a strict majority.
    ///
    /// With an even number of participants, or with three different answers,
    /// there is no way to tell which side is honest. The request is flagged
    /// rather than accepted, and no reputation is moved: punishing a peer on a
    /// tie would be punishing it for a coin flip.
    Rejected {
        /// The peers that took part.
        participants: Vec<String>,
        /// How many distinct outputs were seen.
        distinct_outputs: usize,
    },
}

impl Verdict {
    /// Whether the outputs were accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(self, Self::Accepted { .. })
    }

    /// The peers that agreed with the majority, if there was one.
    pub fn agreeing(&self) -> &[String] {
        match self {
            Self::Accepted { agreeing, .. } => agreeing,
            Self::Rejected { .. } => &[],
        }
    }

    /// The peers that disagreed with the majority, if there was one.
    pub fn outliers(&self) -> &[String] {
        match self {
            Self::Accepted { outliers, .. } => outliers,
            Self::Rejected { .. } => &[],
        }
    }
}

/// A request that may be verified.
///
/// Construction is the determinism gate: a request that cannot be compared is
/// rejected here, before any compute is spent on it.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationRequest {
    /// The prompt that was sent to every participant.
    pub prompt: String,
    /// The decoding configuration every participant has to share.
    pub determinism: Determinism,
    /// The token limit every participant has to share.
    pub max_tokens: u32,
}

impl VerificationRequest {
    /// Builds a verifiable request, or explains why it is not verifiable.
    ///
    /// This is the constraint from the design, enforced in code rather than in
    /// a comment: comparison is only valid with temperature 0 and an identical
    /// model and quantization.
    pub fn new(
        prompt: impl Into<String>,
        determinism: Determinism,
        max_tokens: u32,
    ) -> Result<Self, VerificationError> {
        if determinism.temperature != 0.0 {
            return Err(VerificationError::NotDeterministic {
                reason: format!(
                    "temperature is {} but a comparison requires 0.0: sampled outputs differ \
                     between honest nodes",
                    determinism.temperature
                ),
            });
        }

        if determinism.model.trim().is_empty() {
            return Err(VerificationError::NotDeterministic {
                reason: "the model is unspecified, so the outputs are not comparable".to_owned(),
            });
        }

        if determinism.quantization.trim().is_empty() {
            return Err(VerificationError::NotDeterministic {
                reason: "the quantization is unspecified, so the outputs are not comparable"
                    .to_owned(),
            });
        }

        Ok(Self {
            prompt: prompt.into(),
            determinism,
            max_tokens,
        })
    }
}

/// The verification policy of a node.
///
/// Wraps [`VerificationConfig`] with the sampling decision, so that the
/// "opt-in or sampled, never forced" rule lives in one place.
#[derive(Debug, Clone)]
pub struct VerificationPolicy {
    config: VerificationConfig,
}

impl VerificationPolicy {
    /// Builds a policy from the configuration.
    pub fn new(config: VerificationConfig) -> Self {
        Self { config }
    }

    /// The underlying configuration.
    pub fn config(&self) -> &VerificationConfig {
        &self.config
    }

    /// Whether verification is enabled at all.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// The redundancy a verified request uses.
    pub fn redundancy(&self) -> usize {
        self.config.redundancy
    }

    /// Whether the request with this sequence number is sampled.
    ///
    /// Sampling is deterministic in the counter, so a long run of requests is
    /// spread evenly instead of being verified in random clumps.
    pub fn should_sample(&self, request_seq: u64) -> bool {
        self.config.should_sample(request_seq)
    }

    /// Whether a request should be verified, given an explicit request flag.
    ///
    /// The explicit flag wins over sampling: a caller that asked for a check
    /// gets one even when sampling is off, as long as verification is enabled.
    pub fn should_verify(&self, request_seq: u64, explicitly_requested: bool) -> bool {
        if !self.config.enabled {
            return false;
        }
        explicitly_requested || self.should_sample(request_seq)
    }
}

/// Compares the outputs of several nodes and decides what happened.
///
/// `outputs` is the output of each participant, keyed by peer id. The
/// comparison is exact: with greedy decoding the token sequences are supposed
/// to be identical, so a fuzzy match would hide exactly the drift this is
/// meant to catch.
///
/// # Errors
///
/// Returns [`VerificationError`] when the participants cannot be compared:
/// too few of them, or configurations that differ. In that case **no** verdict
/// is produced and no reputation is moved, because a comparison that was never
/// valid must not cost anyone anything.
pub fn verify(
    request: &VerificationRequest,
    participants: &[(String, Determinism, String)],
) -> Result<Verdict, VerificationError> {
    if participants.len() < 2 {
        return Err(VerificationError::NotEnoughParticipants {
            got: participants.len(),
            need: 2,
        });
    }

    // Every participant has to run the same deterministic configuration as the
    // request. A node that reports a different model or quantization is not a
    // liar, it is simply not part of this comparison.
    for (peer, determinism, _) in participants {
        if !request.determinism.is_compatible_with(determinism) {
            return Err(VerificationError::IncompatibleParticipants {
                reason: format!(
                    "peer {peer} runs {} {} at temperature {} seed {}, but the request pinned \
                     {} {} at temperature {} seed {}",
                    determinism.model,
                    determinism.quantization,
                    determinism.temperature,
                    determinism.seed,
                    request.determinism.model,
                    request.determinism.quantization,
                    request.determinism.temperature,
                    request.determinism.seed
                ),
            });
        }
    }

    // Count the outputs. The order of `participants` is preserved in the
    // buckets so that the verdict is stable for a given input.
    let mut buckets: HashMap<&str, Vec<String>> = HashMap::new();
    for (peer, _, output) in participants {
        buckets
            .entry(output.as_str())
            .or_default()
            .push(peer.clone());
    }

    metrics::counter!("synapse_verification_attempts_total").increment(1);

    let total = participants.len();
    let mut ranked: Vec<(&str, Vec<String>)> = buckets.into_iter().collect();
    // Most agreed output first; ties broken by the output itself so the
    // verdict does not depend on hash iteration order.
    ranked.sort_by(|left, right| {
        right
            .1
            .len()
            .cmp(&left.1.len())
            .then_with(|| left.0.cmp(right.0))
    });

    let (winner_output, winner_peers) = match ranked.first() {
        Some((output, peers)) => (*output, peers.clone()),
        // Unreachable: `participants` is non-empty, so there is at least one
        // bucket. Handled without a panic anyway.
        None => {
            return Err(VerificationError::NotEnoughParticipants {
                got: participants.len(),
                need: 2,
            })
        }
    };

    // A strict majority is required. With two participants that means both of
    // them; with three it tolerates exactly one liar.
    let strict_majority = winner_peers.len() * 2 > total;

    if !strict_majority {
        metrics::counter!("synapse_verification_mismatches_total").increment(1);
        return Ok(Verdict::Rejected {
            participants: participants.iter().map(|(peer, _, _)| peer.clone()).collect(),
            distinct_outputs: ranked.len(),
        });
    }

    let outliers: Vec<String> = participants
        .iter()
        .map(|(peer, _, _)| peer.clone())
        .filter(|peer| !winner_peers.contains(peer))
        .collect();

    if outliers.is_empty() {
        metrics::counter!("synapse_verification_matches_total").increment(1);
    } else {
        // A majority agreed but somebody did not: this is a mismatch, and the
        // outlier is the one that gets penalized.
        metrics::counter!("synapse_verification_mismatches_total").increment(1);
    }

    Ok(Verdict::Accepted {
        output: winner_output.to_owned(),
        agreeing: winner_peers,
        outliers,
    })
}

/// Applies a verdict to the reputation ledger.
///
/// Only a verdict with a strict majority moves reputation. A tie moves
/// nothing: there is no evidence about who was wrong, and a ledger that
/// punishes on a coin flip is worse than no ledger.
///
/// Returns the number of peers whose score changed.
pub fn apply_verdict(
    verdict: &Verdict,
    ledger: &crate::reputation::ReputationLedger,
    config: &VerificationConfig,
) -> usize {
    match verdict {
        Verdict::Accepted {
            agreeing, outliers, ..
        } => {
            for peer in agreeing {
                ledger.record(peer, config.reward, true);
            }
            for peer in outliers {
                ledger.record(peer, config.penalty, false);
            }
            agreeing.len() + outliers.len()
        }
        Verdict::Rejected { .. } => 0,
    }
}

/// Publishes the per-node reputation as a gauge.
///
/// Called from the same tick that prunes the ledger, so an operator can graph
/// the score of every peer instead of only a mesh-wide aggregate.
pub fn publish_reputation(ledger: &crate::reputation::ReputationLedger) {
    let snapshot = ledger.snapshot();
    for (peer, entry) in &snapshot {
        metrics::gauge!("synapse_peer_reputation", "peer" => peer.clone()).set(entry.score);
    }
    metrics::gauge!("synapse_reputation_entries").set(snapshot.len() as f64);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic() -> Determinism {
        Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 1234)
    }

    fn request() -> VerificationRequest {
        match VerificationRequest::new("hello", deterministic(), 16) {
            Ok(request) => request,
            Err(err) => panic!("the request should be verifiable: {err}"),
        }
    }

    #[test]
    fn a_sampled_request_is_refused() {
        let mut determinism = deterministic();
        determinism.temperature = 0.7;

        let result = VerificationRequest::new("hello", determinism, 16);
        assert!(matches!(
            result,
            Err(VerificationError::NotDeterministic { .. })
        ));
    }

    #[test]
    fn an_unspecified_quantization_is_refused() {
        let determinism = Determinism::greedy("tinyllama-1.1b", "", 1234);
        let result = VerificationRequest::new("hello", determinism, 16);
        assert!(matches!(
            result,
            Err(VerificationError::NotDeterministic { .. })
        ));
    }

    #[test]
    fn identical_outputs_are_accepted() {
        let participants = vec![
            ("a".to_owned(), deterministic(), "same".to_owned()),
            ("b".to_owned(), deterministic(), "same".to_owned()),
            ("c".to_owned(), deterministic(), "same".to_owned()),
        ];

        let verdict = match verify(&request(), &participants) {
            Ok(verdict) => verdict,
            Err(err) => panic!("verification failed: {err}"),
        };

        assert!(verdict.is_accepted());
        assert_eq!(verdict.agreeing().len(), 3);
        assert!(verdict.outliers().is_empty());
    }

    #[test]
    fn a_single_outlier_is_flagged() {
        let participants = vec![
            ("a".to_owned(), deterministic(), "same".to_owned()),
            ("b".to_owned(), deterministic(), "same".to_owned()),
            ("c".to_owned(), deterministic(), "garbage".to_owned()),
        ];

        let verdict = match verify(&request(), &participants) {
            Ok(verdict) => verdict,
            Err(err) => panic!("verification failed: {err}"),
        };

        assert!(verdict.is_accepted());
        assert_eq!(verdict.outliers(), ["c".to_owned()]);
    }

    #[test]
    fn a_tie_is_rejected_without_blaming_anyone() {
        let participants = vec![
            ("a".to_owned(), deterministic(), "left".to_owned()),
            ("b".to_owned(), deterministic(), "right".to_owned()),
        ];

        let verdict = match verify(&request(), &participants) {
            Ok(verdict) => verdict,
            Err(err) => panic!("verification failed: {err}"),
        };

        assert!(!verdict.is_accepted());
        assert!(verdict.agreeing().is_empty());
        assert!(verdict.outliers().is_empty());
    }

    #[test]
    fn a_different_quantization_is_not_comparable() {
        let participants = vec![
            ("a".to_owned(), deterministic(), "same".to_owned()),
            (
                "b".to_owned(),
                Determinism::greedy("tinyllama-1.1b", "Q8_0", 1234),
                "same".to_owned(),
            ),
        ];

        let result = verify(&request(), &participants);
        assert!(matches!(
            result,
            Err(VerificationError::IncompatibleParticipants { .. })
        ));
    }

    #[test]
    fn a_different_seed_is_not_comparable() {
        let participants = vec![
            ("a".to_owned(), deterministic(), "same".to_owned()),
            (
                "b".to_owned(),
                Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 99),
                "same".to_owned(),
            ),
        ];

        let result = verify(&request(), &participants);
        assert!(matches!(
            result,
            Err(VerificationError::IncompatibleParticipants { .. })
        ));
    }

    #[test]
    fn a_single_participant_is_not_enough() {
        let participants = vec![("a".to_owned(), deterministic(), "same".to_owned())];
        let result = verify(&request(), &participants);
        assert!(matches!(
            result,
            Err(VerificationError::NotEnoughParticipants { got: 1, need: 2 })
        ));
    }

    #[test]
    fn sampling_is_deterministic_in_the_counter() {
        let config = VerificationConfig {
            enabled: true,
            sample_every: 4,
            ..VerificationConfig::default()
        };
        let policy = VerificationPolicy::new(config);

        assert!(policy.should_sample(0));
        assert!(!policy.should_sample(1));
        assert!(!policy.should_sample(3));
        assert!(policy.should_sample(4));
    }

    #[test]
    fn verification_is_off_by_default() {
        let policy = VerificationPolicy::new(VerificationConfig::default());
        assert!(!policy.is_enabled());
        // Even an explicit request is refused while verification is disabled.
        assert!(!policy.should_verify(0, true));
    }
}
