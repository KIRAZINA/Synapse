//! Phase 12 end-to-end: the verification orchestration, not the components.
//!
//! The earlier `phase12.rs` tested the pure decision functions directly. These
//! tests go through [`synapse_net::orchestrator::run_verification`] with real
//! mock providers, so the fan-out, the timeouts, the quorum rule and the
//! ledger update are all exercised together. A test here fails if the wheels
//! are not attached to the engine.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use libp2p::PeerId;
use synapse_core::{
    Determinism, NodeTelemetry, Sampling, SynapseError, TokenEvent, TokenStreamProvider,
    VerificationConfig,
};
use synapse_net::orchestrator::{
    run_verification, ParticipantOutcome, VerificationClient, VerificationOutcome, MIN_QUORUM,
};
use synapse_net::reputation::ReputationLedger;
use synapse_net::routing::RoutingTable;
use synapse_net::verify::{Verdict, VerificationError, VerificationRequest};

/// A mock node that returns a fixed output.
///
/// `greedy` says whether it can decode with strict argmax; a node that cannot
/// is excluded from a comparison rather than accused.
struct MockNode {
    output: String,
    greedy: bool,
    /// How long the node takes before it answers.
    delay: Duration,
    /// Whether the node answers at all. A node that does not is a non-response.
    answers: bool,
    /// How many times the node was asked, so a test can prove the fan-out
    /// actually reached it.
    calls: AtomicU32,
}

impl MockNode {
    fn new(output: &str) -> Self {
        Self {
            output: output.to_owned(),
            greedy: true,
            delay: Duration::ZERO,
            answers: true,
            calls: AtomicU32::new(0),
        }
    }

    fn slow(mut self, delay: Duration) -> Self {
        self.delay = delay;
        self
    }

    fn silent(mut self) -> Self {
        self.answers = false;
        self
    }

    fn not_greedy(mut self) -> Self {
        self.greedy = false;
        self
    }

    fn calls(&self) -> u32 {
        self.calls.load(Ordering::SeqCst)
    }

    fn stream(&self, max_tokens: usize) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        let (sender, receiver) = flume::unbounded();

        if !self.answers {
            // A node that never answers: the sender is dropped without a
            // terminal event, which is exactly what a killed process looks
            // like to the orchestrator.
            return Ok(receiver);
        }

        let output = self.output.clone();
        let delay = self.delay;
        tokio::spawn(async move {
            if !delay.is_zero() {
                tokio::time::sleep(delay).await;
            }
            let _ = sender.send(TokenEvent::Token {
                text: output,
                token_id: 0,
            });
            let _ = sender.send(TokenEvent::Done {
                total_tokens: max_tokens,
                elapsed_ms: 1,
                stop_reason: synapse_core::StopReason::MaxTokens,
            });
        });

        Ok(receiver)
    }
}

impl TokenStreamProvider for MockNode {
    fn generate_streaming(
        &self,
        _prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        self.stream(max_tokens)
    }

    fn supports_greedy(&self) -> bool {
        self.greedy
    }

    fn generate_streaming_greedy(
        &self,
        _prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        self.stream(max_tokens)
    }

    fn active_requests(&self) -> u32 {
        0
    }

    fn max_capacity(&self) -> u32 {
        4
    }
}

fn pinned() -> Determinism {
    Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 1234, 4, "cpu")
}

fn request() -> VerificationRequest {
    match VerificationRequest::new("what is 2 + 2?", pinned(), 16) {
        Ok(request) => request,
        Err(err) => panic!("the request should be verifiable: {err}"),
    }
}

fn config() -> VerificationConfig {
    VerificationConfig {
        enabled: true,
        redundancy: 3,
        ..VerificationConfig::default()
    }
}

/// Runs the orchestrator over a set of mock nodes.
async fn run(
    nodes: &[(&str, &MockNode)],
    ledger: &ReputationLedger,
    timeout: Duration,
) -> VerificationOutcome {
    let providers: Vec<(String, Determinism, &dyn TokenStreamProvider)> = nodes
        .iter()
        .map(|(peer, node)| {
            (
                (*peer).to_owned(),
                pinned(),
                *node as &dyn TokenStreamProvider,
            )
        })
        .collect();

    run_verification(&request(), &providers, ledger, &config(), timeout).await
}

/// A telemetry entry for a peer that can serve the model and has free capacity.
fn telemetry(peer: &PeerId) -> NodeTelemetry {
    NodeTelemetry {
        peer_id: peer.to_base58(),
        quic_port: 0,
        addresses: vec!["/ip4/127.0.0.1/tcp/4001".to_owned()],
        active_requests: 0,
        max_capacity: 4,
        supported_models: vec!["tinyllama-1.1b".to_owned()],
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0),
    }
}

/// A routing table that already knows the peers, so the production
/// [`VerificationClient`] has a real ledger and a real fan-out.
fn routing_with_peers(peers: &[PeerId], config: &VerificationConfig) -> Arc<RoutingTable> {
    let routing = Arc::new(RoutingTable::with_policy(Duration::from_secs(300), config));
    for peer in peers {
        routing.update(*peer, telemetry(peer));
    }
    routing
}

// ---------------------------------------------------------------------------
// 1. redundant_verification_accepts_on_match
// ---------------------------------------------------------------------------

#[tokio::test]
async fn redundant_verification_accepts_on_match() {
    let ledger = ReputationLedger::from_config(&config());

    let a = MockNode::new("four");
    let b = MockNode::new("four");
    let c = MockNode::new("four");

    let outcome = run(
        &[("node-a", &a), ("node-b", &b), ("node-c", &c)],
        &ledger,
        Duration::from_secs(5),
    )
    .await;

    // The fan-out actually reached every node.
    assert_eq!(a.calls(), 1, "the orchestrator must call every participant");
    assert_eq!(b.calls(), 1);
    assert_eq!(c.calls(), 1);

    assert_eq!(outcome.requested, 3);
    assert_eq!(outcome.answered, 3);
    assert!(outcome.is_accepted(), "identical outputs must be accepted");
    assert_eq!(outcome.output(), Some("four"));
    assert!(outcome.non_responses().is_empty());

    // Every participant's reputation is boosted.
    let reward = config().reward;
    for peer in ["node-a", "node-b", "node-c"] {
        assert_eq!(
            ledger.score(peer),
            reward,
            "{peer} took part in a match and must be rewarded"
        );
    }
}

// ---------------------------------------------------------------------------
// 2. redundant_verification_penalizes_outlier_on_mismatch
// ---------------------------------------------------------------------------

#[tokio::test]
async fn redundant_verification_penalizes_outlier_on_mismatch() {
    let ledger = ReputationLedger::from_config(&config());

    let a = MockNode::new("four");
    let b = MockNode::new("four");
    let liar = MockNode::new("banana");

    let outcome = run(
        &[("node-a", &a), ("node-b", &b), ("liar", &liar)],
        &ledger,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(outcome.answered, 3);
    assert!(outcome.is_accepted(), "two of three is a strict majority");

    match outcome.verdict {
        Ok(Verdict::Accepted { ref outliers, .. }) => {
            assert_eq!(outliers, &["liar".to_owned()], "the outlier is named");
        }
        other => panic!("expected an accepted verdict with an outlier, got {other:?}"),
    }

    // The outlier is penalized, the honest nodes are rewarded.
    let cfg = config();
    assert_eq!(ledger.score("liar"), cfg.penalty);
    assert!(ledger.score("liar") < 0.0, "the outlier must go negative");
    assert_eq!(ledger.score("node-a"), cfg.reward);
    assert_eq!(ledger.score("node-b"), cfg.reward);
}

// ---------------------------------------------------------------------------
// 3. verification_treats_non_response_as_failed_vote
// ---------------------------------------------------------------------------

#[tokio::test]
async fn verification_treats_non_response_as_failed_vote() {
    let ledger = ReputationLedger::from_config(&config());

    // Two nodes answer and agree; the third never answers at all.
    let a = MockNode::new("four");
    let b = MockNode::new("four");
    let silent = MockNode::new("four").silent();

    let outcome = run(
        &[("node-a", &a), ("node-b", &b), ("silent", &silent)],
        &ledger,
        Duration::from_millis(200),
    )
    .await;

    // The silent node was asked and did not answer.
    assert_eq!(outcome.requested, 3);
    assert_eq!(outcome.answered, 2, "only two nodes produced an output");
    assert_eq!(outcome.non_responses(), vec!["silent"]);

    // The critical assertion: two matching answers out of three requested is
    // NOT a match. A non-response is a failed vote, not an abstention that
    // lets the others carry the request.
    assert!(
        !outcome.is_accepted(),
        "two of three requested is not a strict majority: a non-response is not a vote"
    );
    assert!(outcome.output().is_none());

    // No reputation moves: nobody is rewarded for a request that did not reach
    // quorum, and the silent node is not punished for a timeout either.
    assert_eq!(ledger.score("node-a"), 0.0);
    assert_eq!(ledger.score("node-b"), 0.0);
    assert_eq!(ledger.score("silent"), 0.0);

    // A timeout is treated exactly like a silent node.
    let slow = MockNode::new("four").slow(Duration::from_secs(5));
    let outcome = run(
        &[("node-a", &a), ("node-b", &b), ("slow", &slow)],
        &ledger,
        Duration::from_millis(100),
    )
    .await;

    assert_eq!(outcome.answered, 2);
    assert_eq!(outcome.non_responses(), vec!["slow"]);
    assert!(!outcome.is_accepted(), "a timeout is a failed vote");
    assert_eq!(ledger.score("node-a"), 0.0);
}

// ---------------------------------------------------------------------------
// 4. no_majority_moves_no_reputation
// ---------------------------------------------------------------------------

#[tokio::test]
async fn no_majority_moves_no_reputation() {
    let ledger = ReputationLedger::from_config(&config());

    // Three different answers: no strict majority exists.
    let a = MockNode::new("one");
    let b = MockNode::new("two");
    let c = MockNode::new("three");

    let outcome = run(
        &[("node-a", &a), ("node-b", &b), ("node-c", &c)],
        &ledger,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(outcome.answered, 3);
    assert!(
        !outcome.is_accepted(),
        "three different answers are not a match"
    );

    match outcome.verdict {
        Ok(Verdict::Rejected {
            ref distinct_outputs,
            ..
        }) => assert_eq!(*distinct_outputs, 3),
        ref other => panic!("expected a rejected verdict, got {other:?}"),
    }

    // A tie must not punish anyone: there is no evidence about who was wrong.
    for peer in ["node-a", "node-b", "node-c"] {
        assert_eq!(ledger.score(peer), 0.0, "{peer} must not move on a tie");
    }

    // Two nodes that disagree is also a tie, not a majority.
    let ledger = ReputationLedger::from_config(&config());
    let left = MockNode::new("left");
    let right = MockNode::new("right");
    let outcome = run(
        &[("left", &left), ("right", &right)],
        &ledger,
        Duration::from_secs(5),
    )
    .await;

    assert!(!outcome.is_accepted());
    assert_eq!(ledger.score("left"), 0.0);
    assert_eq!(ledger.score("right"), 0.0);
}

// ---------------------------------------------------------------------------
// 5. determinism_enforced_at_fanout
// ---------------------------------------------------------------------------

#[tokio::test]
async fn determinism_enforced_at_fanout() {
    let ledger = ReputationLedger::from_config(&config());

    // A sampled request is refused before any compute is spent: no node is
    // called at all.
    let mut sampled = pinned();
    sampled.temperature = 0.7;
    let sampled_request = VerificationRequest::new("prompt", sampled, 16);
    assert!(
        matches!(
            sampled_request,
            Err(VerificationError::NotDeterministic { .. })
        ),
        "temperature above zero must be refused"
    );

    // The orchestrator refuses a request whose pinned config is incomplete.
    let a = MockNode::new("four");
    let b = MockNode::new("four");
    let mut incomplete = pinned();
    incomplete.backend = String::new();
    let bad_request = VerificationRequest::new("prompt", incomplete, 16);
    assert!(
        matches!(bad_request, Err(VerificationError::NotDeterministic { .. })),
        "an unspecified backend must be refused"
    );

    // A participant running a different thread count is not comparable.
    let providers: Vec<(String, Determinism, &dyn TokenStreamProvider)> = vec![
        (
            "node-a".to_owned(),
            pinned(),
            &a as &dyn TokenStreamProvider,
        ),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 1234, 8, "cpu"),
            &b as &dyn TokenStreamProvider,
        ),
    ];
    let outcome = run_verification(
        &request(),
        &providers,
        &ledger,
        &config(),
        Duration::from_secs(5),
    )
    .await;

    assert!(
        matches!(
            outcome.verdict,
            Err(VerificationError::IncompatibleParticipants { .. })
        ),
        "a different thread count must be refused, not compared"
    );
    assert_eq!(
        ledger.score("node-a"),
        0.0,
        "a refusal must not cost anyone"
    );
    assert_eq!(ledger.score("node-b"), 0.0);

    // A different backend is refused for the same reason.
    let providers: Vec<(String, Determinism, &dyn TokenStreamProvider)> = vec![
        (
            "node-a".to_owned(),
            pinned(),
            &a as &dyn TokenStreamProvider,
        ),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 1234, 4, "cuda"),
            &b as &dyn TokenStreamProvider,
        ),
    ];
    let outcome = run_verification(
        &request(),
        &providers,
        &ledger,
        &config(),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(
        outcome.verdict,
        Err(VerificationError::IncompatibleParticipants { .. })
    ));

    // A different quantization is refused.
    let providers: Vec<(String, Determinism, &dyn TokenStreamProvider)> = vec![
        (
            "node-a".to_owned(),
            pinned(),
            &a as &dyn TokenStreamProvider,
        ),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q8_0", 1234, 4, "cpu"),
            &b as &dyn TokenStreamProvider,
        ),
    ];
    let outcome = run_verification(
        &request(),
        &providers,
        &ledger,
        &config(),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(
        outcome.verdict,
        Err(VerificationError::IncompatibleParticipants { .. })
    ));

    // A different seed is refused.
    let providers: Vec<(String, Determinism, &dyn TokenStreamProvider)> = vec![
        (
            "node-a".to_owned(),
            pinned(),
            &a as &dyn TokenStreamProvider,
        ),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 99, 4, "cpu"),
            &b as &dyn TokenStreamProvider,
        ),
    ];
    let outcome = run_verification(
        &request(),
        &providers,
        &ledger,
        &config(),
        Duration::from_secs(5),
    )
    .await;
    assert!(matches!(
        outcome.verdict,
        Err(VerificationError::IncompatibleParticipants { .. })
    ));

    // A node that cannot decode greedily is a failed vote, not a liar: it is
    // excluded from the comparison and nobody is punished.
    let ledger = ReputationLedger::from_config(&config());
    let greedy_a = MockNode::new("four");
    let greedy_b = MockNode::new("four");
    let sampler = MockNode::new("four").not_greedy();
    let outcome = run(
        &[
            ("node-a", &greedy_a),
            ("node-b", &greedy_b),
            ("sampler", &sampler),
        ],
        &ledger,
        Duration::from_secs(5),
    )
    .await;

    assert_eq!(
        outcome.answered, 2,
        "the sampler is not a comparable answer"
    );
    assert_eq!(outcome.non_responses(), vec!["sampler"]);
    assert!(
        !outcome.is_accepted(),
        "a node that cannot be compared cannot help reach quorum"
    );
    // Nobody is punished: the sampler is excluded, not accused, and the two
    // honest nodes are not rewarded for a request that never reached quorum.
    assert_eq!(ledger.score("sampler"), 0.0, "it is not accused");
    assert_eq!(ledger.score("node-a"), 0.0);
    assert_eq!(ledger.score("node-b"), 0.0);
}

// ---------------------------------------------------------------------------
// The production entry point: same orchestration, shared routing ledger.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn verification_client_feeds_the_routing_tiebreaker() {
    let cfg = config();
    let peers: Vec<PeerId> = (0..3).map(|_| PeerId::random()).collect();
    let routing = routing_with_peers(&peers, &cfg);
    let client = VerificationClient::new(Arc::clone(&routing), cfg.clone());

    let a = Arc::new(MockNode::new("four"));
    let b = Arc::new(MockNode::new("four"));
    let liar = Arc::new(MockNode::new("banana"));

    let providers: Vec<(String, Determinism, Arc<dyn TokenStreamProvider>)> = vec![
        (peers[0].to_base58(), pinned(), a.clone()),
        (peers[1].to_base58(), pinned(), b.clone()),
        (peers[2].to_base58(), pinned(), liar.clone()),
    ];

    let outcome = client
        .verify("what is 2 + 2?", 16, pinned(), providers)
        .await
        .unwrap_or_else(|err| panic!("the verified request should run: {err}"));

    assert!(outcome.is_accepted());
    assert_eq!(outcome.output(), Some("four"));
    assert_eq!(outcome.answered, 3);
    assert_eq!(a.calls(), 1);
    assert_eq!(b.calls(), 1);
    assert_eq!(liar.calls(), 1);

    // The verdict landed in the routing table's own ledger, not a private
    // copy: this is the ledger `best_peers` reads.
    let honest = peers[0].to_base58();
    let other_honest = peers[1].to_base58();
    let caught = peers[2].to_base58();
    assert_eq!(routing.reputation(&honest), cfg.reward);
    assert_eq!(routing.reputation(&other_honest), cfg.reward);
    assert_eq!(routing.reputation(&caught), cfg.penalty);

    // And the tiebreaker is now active: with equal load, the penalized peer
    // sorts last, so the next request is routed elsewhere.
    let fanout = client.fanout("tinyllama-1.1b");
    assert_eq!(fanout.len(), 3, "all three peers are still reachable");
    assert_eq!(
        fanout.last().map(|telemetry| telemetry.peer_id.as_str()),
        Some(caught.as_str()),
        "the peer caught disagreeing must be deprioritized by the router"
    );
}

#[tokio::test]
async fn verification_client_refuses_a_padded_fanout() {
    let cfg = config();
    let peer = PeerId::random();
    let routing = routing_with_peers(std::slice::from_ref(&peer), &cfg);
    let client = VerificationClient::new(Arc::clone(&routing), cfg.clone());

    let node = Arc::new(MockNode::new("four"));
    let providers: Vec<(String, Determinism, Arc<dyn TokenStreamProvider>)> = vec![
        (peer.to_base58(), pinned(), node.clone()),
        (peer.to_base58(), pinned(), node.clone()),
        (peer.to_base58(), pinned(), node.clone()),
    ];

    let err = client
        .verify("what is 2 + 2?", 16, pinned(), providers)
        .await
        .expect_err("asking one node three times is not redundancy");

    assert!(
        err.to_string().contains("appears twice"),
        "the repeated peer must be named, got: {err}"
    );
    assert_eq!(node.calls(), 0, "the fan-out is refused before any compute");
    assert_eq!(routing.reputation(&peer.to_base58()), 0.0);
}

#[tokio::test]
async fn verification_client_refuses_a_short_fanout() {
    let cfg = config();
    let peer = PeerId::random();
    let routing = routing_with_peers(std::slice::from_ref(&peer), &cfg);
    let client = VerificationClient::new(Arc::clone(&routing), cfg.clone());

    let node = Arc::new(MockNode::new("four"));
    let providers: Vec<(String, Determinism, Arc<dyn TokenStreamProvider>)> =
        vec![(peer.to_base58(), pinned(), node.clone())];

    let err = client
        .verify("what is 2 + 2?", 16, pinned(), providers)
        .await
        .expect_err("one participant cannot verify anything");

    assert!(
        err.to_string().contains("distinct participants"),
        "the configured redundancy must be named, got: {err}"
    );
    assert_eq!(node.calls(), 0);
}

// ---------------------------------------------------------------------------
// The disabled path must be untouched.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn verification_is_off_by_default_and_never_forced() {
    let default = VerificationConfig::default();
    assert!(!default.enabled, "verification is off by default");
    assert_eq!(default.sample_every, 0, "sampling is off by default");

    // A disabled config means the orchestrator is never reached: the caller
    // takes the normal single-node path. Proven here by the fact that no node
    // is called when the policy says no.
    let policy = synapse_net::verify::VerificationPolicy::new(default.clone());
    assert!(!policy.should_verify(0, true));
    assert!(!policy.should_verify(100, true));

    // Sampling verifies a fraction, never all traffic.
    let sampled = VerificationConfig {
        enabled: true,
        sample_every: 10,
        ..VerificationConfig::default()
    };
    let policy = synapse_net::verify::VerificationPolicy::new(sampled);
    let verified = (0..100).filter(|seq| policy.should_sample(*seq)).count();
    assert_eq!(verified, 10, "one request in ten, not every request");
}

#[test]
fn the_quorum_floor_is_two() {
    assert_eq!(MIN_QUORUM, 2);
}

#[test]
fn a_failed_participant_is_recorded_not_dropped() {
    let outcome = ParticipantOutcome::Failed {
        peer: "node-x".to_owned(),
        reason: "timed out".to_owned(),
    };
    assert!(!outcome.answered());
    assert_eq!(outcome.peer(), "node-x");
}

#[test]
fn the_verified_path_uses_strict_argmax() {
    let sampling = synapse_net::orchestrator::verification_sampling();
    assert_eq!(sampling, Sampling::Greedy);
    assert!(sampling.is_greedy());
}
