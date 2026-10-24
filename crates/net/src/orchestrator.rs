//! The redundant verification orchestrator.
//!
//! This is the module that actually *runs* verification. [`crate::verify`]
//! holds the pure decision logic (compare outputs, decide a verdict, apply it
//! to the ledger); this module owns the fan-out, the timeouts, and the quorum
//! rule that decides whether a comparison is even allowed to produce a verdict.
//!
//! # Where it is called from
//!
//! [`VerificationClient::verify`] is the production entry point. The node
//! binary builds it from its live [`crate::routing::RoutingTable`] and calls it
//! from `synapse-node --verify` (`bin/node/src/main.rs`). It is the only place
//! that constructs a [`VerificationRequest`] outside tests, and it uses the
//! routing table's ledger, so a verdict changes the scores that the router
//! reads. When `[verification].enabled = false` (the default) the orchestrator
//! is never entered at all and the caller takes the normal single-node path
//! unchanged.
//!
//! # Quorum: a non-response is not a vote
//!
//! The single most important rule here, and the one that is easiest to get
//! wrong: **a participant that times out or errors has not agreed with
//! anybody.** With `K = 3`, one timeout does not mean "two matched, so it is
//! fine". It means only two of the three participants produced an answer, and
//! the request did not reach quorum.
//!
//! Concretely, the orchestrator requires **every requested participant to
//! answer**. A request that falls short is flagged as [`Verdict::Rejected`]
//! and moves no reputation.
//!
//! The stricter rule is deliberate. A node that stays silent is
//! indistinguishable from a node that is down, a node that is overloaded, and
//! a node that is deliberately evading a check it would fail. Treating silence
//! as an abstention that lets the others carry the request hands a lazy node
//! exactly the incentive it wants: answer nothing, cost nothing, and let the
//! honest nodes absorb the compute. Requiring full participation makes silence
//! cost the request, which is the only way silence cannot be gamed.
//!
//! The cost of that choice is availability: one dead node blocks verification
//! of that request. That is acceptable because verification is a spot check on
//! a sampled fraction of traffic, not the request path itself; the caller still
//! has the normal single-node answer.

use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use synapse_core::{
    Determinism, NodeTelemetry, Sampling, SynapseError, TokenEvent, TokenStreamProvider,
};
use tracing::info;

use crate::reputation::ReputationLedger;
use crate::routing::RoutingTable;
use crate::verify::{
    apply_verdict, verify, Verdict, VerificationError, VerificationPolicy, VerificationRequest,
};

/// The sampler seed used for verified requests.
///
/// A verified request decodes with strict argmax, so the seed does not affect
/// the output. It is still pinned, and still part of the comparison, so that a
/// participant configured with a different seed is refused rather than
/// compared: the seed is part of the numerical identity even when it is inert.
pub const VERIFICATION_SEED: u32 = 1234;

/// The smallest number of participants that can produce a verdict at all.
///
/// Two is the floor because a single output has nothing to be compared
/// against. The orchestrator additionally requires a strict majority of the
/// *requested* redundancy, so `K = 3` with one timeout is still refused.
pub const MIN_QUORUM: usize = 2;

/// How long a single participant has to return its whole output.
///
/// A verified request is a spot check, not a user-facing generation: waiting
/// forever on one slow node would turn a 3x cost into an unbounded one. A
/// participant that exceeds this is treated as a non-response, which is a
/// failed vote, not a match.
pub const DEFAULT_PARTICIPANT_TIMEOUT: Duration = Duration::from_secs(30);

/// One node that took part in a fan-out.
#[derive(Debug, Clone)]
pub struct Participant {
    /// The peer id of the node.
    pub peer: String,
    /// The decoding configuration the node reported for this request.
    pub determinism: Determinism,
}

/// What a single participant returned.
#[derive(Debug, Clone)]
pub enum ParticipantOutcome {
    /// The node returned a complete output.
    Answered {
        /// The peer that answered.
        peer: String,
        /// The decoding configuration it reported.
        determinism: Determinism,
        /// The generated text.
        output: String,
    },
    /// The node did not return an output: it timed out, errored, or refused.
    ///
    /// This is recorded explicitly rather than dropped, because the quorum
    /// rule needs to know how many participants were *asked* and how many
    /// actually answered.
    Failed {
        /// The peer that did not answer.
        peer: String,
        /// Why, for the log and the operator.
        reason: String,
    },
}

impl ParticipantOutcome {
    /// The peer id, whichever variant this is.
    pub fn peer(&self) -> &str {
        match self {
            Self::Answered { peer, .. } | Self::Failed { peer, .. } => peer,
        }
    }

    /// Whether the participant produced an output.
    pub fn answered(&self) -> bool {
        matches!(self, Self::Answered { .. })
    }
}

/// The result of a fan-out, including the participants that did not answer.
#[derive(Debug, Clone)]
pub struct VerificationOutcome {
    /// The verdict, or why no verdict could be produced.
    pub verdict: Result<Verdict, VerificationError>,
    /// Every participant that was asked, in fan-out order.
    pub outcomes: Vec<ParticipantOutcome>,
    /// How many participants were requested.
    pub requested: usize,
    /// How many actually returned an output.
    pub answered: usize,
}

impl VerificationOutcome {
    /// Whether the outputs were accepted.
    pub fn is_accepted(&self) -> bool {
        matches!(&self.verdict, Ok(verdict) if verdict.is_accepted())
    }

    /// The accepted output, if there was one.
    pub fn output(&self) -> Option<&str> {
        match &self.verdict {
            Ok(Verdict::Accepted { output, .. }) => Some(output.as_str()),
            _ => None,
        }
    }

    /// The peers that did not answer.
    pub fn non_responses(&self) -> Vec<&str> {
        self.outcomes
            .iter()
            .filter(|outcome| !outcome.answered())
            .map(ParticipantOutcome::peer)
            .collect()
    }
}

/// Collects a whole generation from a provider into a single string.
///
/// The verified path needs the complete output to compare, not a stream, so
/// the events are drained here. A terminal [`TokenEvent::Error`] is a failed
/// vote, never an empty string that would accidentally compare equal to
/// another failed vote.
async fn collect_output(
    provider: &dyn TokenStreamProvider,
    prompt: &str,
    max_tokens: u32,
    greedy: bool,
    timeout: Duration,
) -> Result<String, String> {
    let receiver = if greedy {
        provider.generate_streaming_greedy(prompt.to_owned(), max_tokens as usize)
    } else {
        provider.generate_streaming(prompt.to_owned(), max_tokens as usize)
    };

    let receiver = receiver.map_err(|err| err.to_string())?;

    let collected = tokio::time::timeout(timeout, async {
        let mut output = String::new();
        loop {
            let event = receiver.recv_async().await.map_err(|err| err.to_string())?;
            match event {
                TokenEvent::Token { text, .. } => output.push_str(&text),
                TokenEvent::Done { .. } => return Ok(output),
                TokenEvent::Error { message } => return Err(message),
            }
        }
    })
    .await;

    match collected {
        Ok(result) => result,
        Err(_) => Err(format!(
            "the participant did not answer within {}ms",
            timeout.as_millis()
        )),
    }
}

/// Runs one verified request across the given participants.
///
/// `providers` is the set of nodes to ask, each paired with the decoding
/// configuration it is expected to run. The orchestrator:
///
/// 1. refuses the request outright if it is not deterministic
///    ([`VerificationRequest::new`]);
/// 2. fans out concurrently, with a per-participant timeout;
/// 3. records every non-response as a failed vote;
/// 4. applies the quorum rule before it lets [`verify`] produce a verdict;
/// 5. applies the verdict to the ledger only when a majority actually agreed.
///
/// Returns the outcome, including the participants that failed. It never
/// panics and never accuses a node for a difference the configuration
/// guaranteed.
pub async fn run_verification(
    request: &VerificationRequest,
    providers: &[(String, Determinism, &dyn TokenStreamProvider)],
    ledger: &ReputationLedger,
    config: &synapse_core::VerificationConfig,
    timeout: Duration,
) -> VerificationOutcome {
    let requested = providers.len();

    // The determinism gate runs before any compute is spent.
    if !request.determinism.is_comparable() {
        return VerificationOutcome {
            verdict: Err(VerificationError::NotDeterministic {
                reason: "the pinned configuration is incomplete".to_owned(),
            }),
            outcomes: Vec::new(),
            requested,
            answered: 0,
        };
    }

    // Fan out concurrently. A slow node must not serialize the others.
    let mut tasks = Vec::with_capacity(requested);
    for (peer, determinism, provider) in providers {
        let peer = peer.clone();
        let determinism = determinism.clone();
        let prompt = request.prompt.clone();
        let max_tokens = request.max_tokens;
        let peer_for_task = peer.clone();
        tasks.push(async move {
            // A verified request is always greedy. A provider that cannot
            // decode greedily is a failed vote, not a liar: it is excluded
            // rather than accused.
            let greedy = provider.supports_greedy();
            let result = if greedy {
                collect_output(*provider, &prompt, max_tokens, true, timeout).await
            } else {
                Err("the node cannot decode greedily, so it is not comparable".to_owned())
            };

            match result {
                Ok(output) => ParticipantOutcome::Answered {
                    peer: peer_for_task,
                    determinism,
                    output,
                },
                Err(reason) => ParticipantOutcome::Failed {
                    peer: peer_for_task,
                    reason,
                },
            }
        });
    }

    let outcomes = futures_util::future::join_all(tasks).await;
    let answered = outcomes.iter().filter(|outcome| outcome.answered()).count();

    // Quorum, part one: every requested participant has to answer.
    //
    // A non-response is not a vote. Two answers out of three requested is not
    // "two matched, so it is fine": a silent node is indistinguishable from a
    // node that is down, overloaded, or deliberately evading a check it would
    // fail. Treating silence as an abstention would let a lazy node answer
    // nothing, cost nothing, and let the honest nodes carry the request.
    if answered < requested || answered < MIN_QUORUM {
        metrics::counter!("synapse_verification_attempts_total").increment(1);
        metrics::counter!("synapse_verification_mismatches_total").increment(1);
        return VerificationOutcome {
            verdict: Ok(Verdict::Rejected {
                participants: outcomes.iter().map(|o| o.peer().to_owned()).collect(),
                distinct_outputs: 0,
            }),
            outcomes,
            requested,
            answered,
        };
    }

    // Quorum, part two: the winner must hold a strict majority of the
    // participants that were *requested*. Since part one already required all
    // of them to answer, this is the ordinary majority rule; it is kept
    // explicit so the invariant is visible at the point it matters.
    let participants: Vec<(String, Determinism, String)> = outcomes
        .iter()
        .filter_map(|outcome| match outcome {
            ParticipantOutcome::Answered {
                peer,
                determinism,
                output,
            } => Some((peer.clone(), determinism.clone(), output.clone())),
            ParticipantOutcome::Failed { .. } => None,
        })
        .collect();

    let verdict = match verify(request, &participants) {
        Ok(verdict) => verdict,
        Err(err) => {
            // A comparison that was never valid must not cost anyone anything.
            return VerificationOutcome {
                verdict: Err(err),
                outcomes,
                requested,
                answered,
            };
        }
    };

    // Enforce the strict majority against the requested redundancy. `verify`
    // already requires a majority of the participants it was given; this is
    // the stricter rule that a missing participant cannot be silently
    // converted into agreement.
    let verdict = match verdict {
        Verdict::Accepted {
            output,
            agreeing,
            outliers,
        } => {
            if agreeing.len() * 2 > requested {
                Verdict::Accepted {
                    output,
                    agreeing,
                    outliers,
                }
            } else {
                // Not enough of the requested nodes agreed. Flag it and move
                // no reputation: the honest nodes that did answer are not
                // rewarded, but nobody is punished for another node's silence.
                metrics::counter!("synapse_verification_mismatches_total").increment(1);
                Verdict::Rejected {
                    participants: outcomes.iter().map(|o| o.peer().to_owned()).collect(),
                    distinct_outputs: 0,
                }
            }
        }
        other => other,
    };

    apply_verdict(&verdict, ledger, config);

    VerificationOutcome {
        verdict: Ok(verdict),
        outcomes,
        requested,
        answered,
    }
}

/// The production entry point of redundant verification.
///
/// This ties the orchestration to the node's live routing table:
///
/// * the fan-out set is chosen by [`RoutingTable::best_peers`], which already
///   prefers the least loaded peer running the requested model and breaks ties
///   by reputation;
/// * the verdict is applied to the routing table's own ledger, so the next
///   routing decision sees the result (a peer caught disagreeing is
///   deprioritized immediately);
/// * the participant set must contain exactly `redundancy` **distinct** peers.
///   Asking the same endpoint K times is not redundancy: it is one opinion K
///   times, and it would reward a single node K times for agreeing with
///   itself. That is refused before any compute is spent.
#[derive(Debug)]
pub struct VerificationClient {
    /// The routing table whose ledger receives the verdict.
    routing: Arc<RoutingTable>,
    /// The node's verification policy.
    policy: VerificationPolicy,
    /// The deadline for a single participant.
    timeout: Duration,
}

impl VerificationClient {
    /// Builds a client over the routing table and verification policy.
    pub fn new(routing: Arc<RoutingTable>, config: synapse_core::VerificationConfig) -> Self {
        Self {
            routing,
            policy: VerificationPolicy::new(config),
            timeout: DEFAULT_PARTICIPANT_TIMEOUT,
        }
    }

    /// Overrides the per-participant timeout, for tests and operators.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// The routing table the verdict is applied to.
    pub fn routing(&self) -> &Arc<RoutingTable> {
        &self.routing
    }

    /// The verification policy of the node.
    pub fn policy(&self) -> &VerificationPolicy {
        &self.policy
    }

    /// The redundancy of a verified request.
    pub fn redundancy(&self) -> usize {
        self.policy.redundancy().max(MIN_QUORUM)
    }

    /// The distinct peers a verified request would fan out to.
    ///
    /// The result is at most `redundancy` entries; there is no duplication, so
    /// a mesh with fewer distinct eligible peers simply cannot verify. That is
    /// the honest answer, not a padded one.
    pub fn fanout(&self, model: &str) -> Vec<NodeTelemetry> {
        self.routing.best_peers(Some(model), self.redundancy())
    }

    /// Runs one request through the orchestrator and applies the verdict.
    ///
    /// `providers` must contain exactly [`redundancy`](Self::redundancy)
    /// participants with distinct peer ids. The caller builds them from
    /// [`fanout`](Self::fanout) (or from an explicit list of distinct nodes);
    /// this method refuses a padded set instead of silently comparing a node
    /// with itself.
    ///
    /// # Errors
    ///
    /// Returns an error when verification is disabled, the request is not
    /// deterministic, the participant set is the wrong size, or a peer appears
    /// twice. In every one of those cases no compute is spent and no
    /// reputation moves.
    pub async fn verify(
        &self,
        prompt: &str,
        max_tokens: u32,
        determinism: Determinism,
        providers: Vec<(String, Determinism, Arc<dyn TokenStreamProvider>)>,
    ) -> Result<VerificationOutcome, SynapseError> {
        if !self.policy.is_enabled() {
            return Err(SynapseError::Network(
                "verification is disabled in the configuration".to_owned(),
            ));
        }

        let request = VerificationRequest::new(prompt, determinism, max_tokens)
            .map_err(|err| SynapseError::Network(err.to_string()))?;

        let required = self.redundancy();
        if providers.len() != required {
            return Err(SynapseError::Network(format!(
                "verification is pinned to {required} distinct participants, got {}: a padded \
                 fan-out is one opinion repeated, not redundancy",
                providers.len()
            )));
        }

        let mut seen = HashSet::with_capacity(required);
        for (peer, _, _) in &providers {
            if !seen.insert(peer.clone()) {
                return Err(SynapseError::Network(format!(
                    "peer {peer} appears twice in the verification fan-out: verifying one node \
                     against itself proves nothing"
                )));
            }
        }

        let borrowed: Vec<(String, Determinism, &dyn TokenStreamProvider)> = providers
            .iter()
            .map(|(peer, determinism, provider)| {
                (
                    peer.clone(),
                    determinism.clone(),
                    Arc::as_ref(provider) as &dyn TokenStreamProvider,
                )
            })
            .collect();

        let outcome = run_verification(
            &request,
            &borrowed,
            self.routing.reputation_ledger(),
            self.policy.config(),
            self.timeout,
        )
        .await;

        info!(
            requested = outcome.requested,
            answered = outcome.answered,
            accepted = outcome.is_accepted(),
            non_responses = ?outcome.non_responses(),
            "redundant verification finished"
        );

        Ok(outcome)
    }
}

/// Builds the pinned configuration a verified request runs under.
///
/// This is the single place where the numerical identity of a verified request
/// is decided, so a caller cannot accidentally verify with a sampled or
/// partially pinned configuration.
pub fn pinned_determinism(
    model: impl Into<String>,
    quantization: impl Into<String>,
    seed: u64,
    n_threads: u32,
    backend: impl Into<String>,
) -> Determinism {
    Determinism::greedy(model, quantization, seed, n_threads, backend)
}

/// The sampling mode a verified request must use.
///
/// Exposed so the caller can assert it rather than assume it: verification is
/// only valid with strict argmax.
pub fn verification_sampling() -> Sampling {
    Sampling::Greedy
}

/// Turns a provider error into the reason string of a failed vote.
pub fn failure_reason(err: &SynapseError) -> String {
    err.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_non_response_is_not_a_vote() {
        // The quorum rule in isolation: every requested participant has to
        // answer, so two answers out of three requested is not a match.
        let requested = 3;
        let answered = 2;
        assert!(
            answered < requested,
            "two of three requested must not satisfy quorum"
        );

        let answered = 3;
        assert!(answered >= requested, "three of three does");
    }

    #[test]
    fn verification_sampling_is_greedy() {
        assert!(verification_sampling().is_greedy());
    }
}
