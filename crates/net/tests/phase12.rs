//! Phase 12: compute integrity.
//!
//! Phase 11 proves *who* answered. These tests cover the layer above it: the
//! reputation ledger that remembers whether a peer's answers were right, the
//! routing preference that acts on it, and the redundant verification that
//! produces the verdicts in the first place.
//!
//! The determinism constraint is not a comment here: `verification_requires_determinism`
//! asserts that a sampled or under-specified request is refused, and that a
//! participant running a different quantization or seed is not compared.

use std::time::Duration;

use libp2p::PeerId;
use synapse_core::{Determinism, NodeTelemetry, VerificationConfig};
use synapse_net::reputation::ReputationLedger;
use synapse_net::routing::RoutingTable;
use synapse_net::verify::{
    apply_verdict, verify, VerificationError, VerificationPolicy, VerificationRequest, Verdict,
};

fn telemetry(peer: &PeerId, active: u32, cap: u32) -> NodeTelemetry {
    NodeTelemetry {
        peer_id: peer.to_base58(),
        quic_port: 0,
        active_requests: active,
        max_capacity: cap,
        supported_models: vec!["tinyllama-1.1b".to_owned()],
        addresses: Vec::new(),
        timestamp_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_millis() as u64)
            .unwrap_or(0),
    }
}

fn deterministic() -> Determinism {
    Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 1234)
}

fn request() -> VerificationRequest {
    match VerificationRequest::new("what is 2 + 2?", deterministic(), 16) {
        Ok(request) => request,
        Err(err) => panic!("the request should be verifiable: {err}"),
    }
}

// ---------------------------------------------------------------------------
// 1. reputation_updates_on_match
// ---------------------------------------------------------------------------

#[test]
fn reputation_updates_on_match() {
    let table = RoutingTable::new();
    let peer = PeerId::random().to_base58();

    assert_eq!(table.reputation(&peer), 0.0, "an unknown peer is neutral");

    let config = VerificationConfig::default();
    let participants = vec![
        (peer.clone(), deterministic(), "four".to_owned()),
        ("honest-b".to_owned(), deterministic(), "four".to_owned()),
        ("honest-c".to_owned(), deterministic(), "four".to_owned()),
    ];

    let verdict = match verify(&request(), &participants) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };
    assert!(verdict.is_accepted());

    let changed = apply_verdict(&verdict, table.reputation_ledger(), &config);
    assert_eq!(changed, 3, "every agreeing peer is updated");

    assert_eq!(
        table.reputation(&peer),
        config.reward,
        "a matching output raises the reputation by the reward"
    );

    // A second agreement accumulates rather than replacing the score.
    apply_verdict(&verdict, table.reputation_ledger(), &config);
    assert_eq!(table.reputation(&peer), config.reward * 2.0);

    let entry = table
        .reputation_ledger()
        .entry(&peer)
        .unwrap_or_else(|| panic!("the peer should have a ledger entry"));
    assert_eq!(entry.observations, 2);
    assert_eq!(entry.agreements, 2);
    assert_eq!(entry.agreement_rate(), Some(1.0));
}

// ---------------------------------------------------------------------------
// 2. reputation_penalizes_on_mismatch
// ---------------------------------------------------------------------------

#[test]
fn reputation_penalizes_on_mismatch() {
    let table = RoutingTable::new();
    let config = VerificationConfig::default();

    let liar = "liar".to_owned();
    let honest_a = "honest-a".to_owned();
    let honest_b = "honest-b".to_owned();

    // Two honest nodes agree, one node returns something else.
    let participants = vec![
        (honest_a.clone(), deterministic(), "four".to_owned()),
        (honest_b.clone(), deterministic(), "four".to_owned()),
        (liar.clone(), deterministic(), "banana".to_owned()),
    ];

    let verdict = match verify(&request(), &participants) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };

    // The majority output is accepted, and the outlier is named.
    assert!(verdict.is_accepted());
    assert_eq!(verdict.outliers(), [liar.as_str()]);
    assert_eq!(verdict.agreeing().len(), 2);

    apply_verdict(&verdict, table.reputation_ledger(), &config);

    // The outlier is penalized, the honest nodes are rewarded.
    assert_eq!(
        table.reputation(&liar),
        config.penalty,
        "the mismatched outlier is penalized"
    );
    assert!(table.reputation(&liar) < 0.0);
    assert_eq!(table.reputation(&honest_a), config.reward);
    assert_eq!(table.reputation(&honest_b), config.reward);

    let liar_entry = table
        .reputation_ledger()
        .entry(&liar)
        .unwrap_or_else(|| panic!("the liar should have a ledger entry"));
    assert_eq!(liar_entry.observations, 1);
    assert_eq!(liar_entry.agreements, 0);
    assert_eq!(liar_entry.agreement_rate(), Some(0.0));

    // A tie has no majority, so it must not punish anyone.
    let tie = vec![
        ("left".to_owned(), deterministic(), "a".to_owned()),
        ("right".to_owned(), deterministic(), "b".to_owned()),
    ];
    let verdict = match verify(&request(), &tie) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };
    assert!(matches!(verdict, Verdict::Rejected { .. }));
    assert_eq!(
        apply_verdict(&verdict, table.reputation_ledger(), &config),
        0,
        "a tie moves no reputation"
    );
    assert_eq!(table.reputation("left"), 0.0);
    assert_eq!(table.reputation("right"), 0.0);
}

// ---------------------------------------------------------------------------
// 3. routing_prefers_high_reputation
// ---------------------------------------------------------------------------

#[test]
fn routing_prefers_high_reputation() {
    let table = RoutingTable::new();
    let good = PeerId::random();
    let bad = PeerId::random();

    // Equal load and equal capacity: only reputation can separate them.
    table.update(good, telemetry(&good, 0, 4));
    table.update(bad, telemetry(&bad, 0, 4));

    // Without any reputation the choice falls back to the peer id, so it is
    // stable but arbitrary: either peer is a legitimate answer here.
    let neutral = table
        .get_best_peer(Some("tinyllama-1.1b"))
        .unwrap_or_else(|| panic!("a peer should be available"));
    assert!(
        neutral.peer_id == good.to_base58() || neutral.peer_id == bad.to_base58(),
        "a neutral table picks one of the known peers"
    );

    // Give the good peer a positive history and the bad one a negative one.
    table.record_verification(&good.to_base58(), 5.0, true);
    table.record_verification(&bad.to_base58(), -5.0, false);

    let best = table
        .get_best_peer(Some("tinyllama-1.1b"))
        .unwrap_or_else(|| panic!("a peer should be available"));

    assert_eq!(
        best.peer_id,
        good.to_base58(),
        "at equal load the higher-reputation node is selected"
    );
    assert_ne!(best.peer_id, bad.to_base58());

    // Load still dominates reputation: a saturated honest node loses to an
    // idle one, because reputation is a tie-breaker and not a bypass.
    table.update(good, telemetry(&good, 4, 4));
    let best = table
        .get_best_peer(Some("tinyllama-1.1b"))
        .unwrap_or_else(|| panic!("a peer should be available"));
    assert_eq!(
        best.peer_id,
        bad.to_base58(),
        "a node at capacity is skipped even when its reputation is higher"
    );

    // The fan-out used by verification returns distinct peers, best first.
    table.update(good, telemetry(&good, 0, 4));
    let fanout = table.best_peers(Some("tinyllama-1.1b"), 2);
    assert_eq!(fanout.len(), 2);
    assert_eq!(fanout[0].peer_id, good.to_base58());
    assert_eq!(fanout[1].peer_id, bad.to_base58());
}

// ---------------------------------------------------------------------------
// 4. verification_accepts_matching_outputs
// ---------------------------------------------------------------------------

#[test]
fn verification_accepts_matching_outputs() {
    let participants = vec![
        ("node-a".to_owned(), deterministic(), "the answer".to_owned()),
        ("node-b".to_owned(), deterministic(), "the answer".to_owned()),
        ("node-c".to_owned(), deterministic(), "the answer".to_owned()),
    ];

    let verdict = match verify(&request(), &participants) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };

    match verdict {
        Verdict::Accepted {
            output,
            agreeing,
            outliers,
        } => {
            assert_eq!(output, "the answer");
            assert_eq!(agreeing.len(), 3);
            assert!(outliers.is_empty());
        }
        Verdict::Rejected { .. } => panic!("identical outputs must be accepted"),
    }

    // Two participants that agree are a strict majority of two.
    let pair = vec![
        ("node-a".to_owned(), deterministic(), "same".to_owned()),
        ("node-b".to_owned(), deterministic(), "same".to_owned()),
    ];
    let verdict = match verify(&request(), &pair) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };
    assert!(verdict.is_accepted());
}

// ---------------------------------------------------------------------------
// 5. verification_rejects_mismatch
// ---------------------------------------------------------------------------

#[test]
fn verification_rejects_mismatch() {
    // Three different answers: no strict majority exists, so the request is
    // flagged instead of being accepted on a coin flip.
    let participants = vec![
        ("node-a".to_owned(), deterministic(), "one".to_owned()),
        ("node-b".to_owned(), deterministic(), "two".to_owned()),
        ("node-c".to_owned(), deterministic(), "three".to_owned()),
    ];

    let verdict = match verify(&request(), &participants) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };

    match verdict {
        Verdict::Rejected {
            participants,
            distinct_outputs,
        } => {
            assert_eq!(participants.len(), 3);
            assert_eq!(distinct_outputs, 3);
        }
        Verdict::Accepted { .. } => panic!("three different outputs must not be accepted"),
    }

    // A single liar against a majority is accepted but flagged as an outlier,
    // which is what a caller acts on.
    let with_liar = vec![
        ("node-a".to_owned(), deterministic(), "one".to_owned()),
        ("node-b".to_owned(), deterministic(), "one".to_owned()),
        ("node-c".to_owned(), deterministic(), "garbage".to_owned()),
    ];
    let verdict = match verify(&request(), &with_liar) {
        Ok(verdict) => verdict,
        Err(err) => panic!("verification failed: {err}"),
    };
    assert!(verdict.is_accepted());
    assert_eq!(verdict.outliers(), ["node-c".to_owned()]);
}

// ---------------------------------------------------------------------------
// 6. verification_requires_determinism
// ---------------------------------------------------------------------------

#[test]
fn verification_requires_determinism() {
    // A sampled request cannot be compared: two honest nodes would disagree.
    let mut sampled = deterministic();
    sampled.temperature = 0.8;
    assert!(
        matches!(
            VerificationRequest::new("prompt", sampled, 16),
            Err(VerificationError::NotDeterministic { .. })
        ),
        "a request above temperature 0 must be refused"
    );

    // An unspecified model or quantization is not comparable either.
    let no_model = Determinism::greedy("", "Q4_K_M", 1234);
    assert!(matches!(
        VerificationRequest::new("prompt", no_model, 16),
        Err(VerificationError::NotDeterministic { .. })
    ));

    let no_quant = Determinism::greedy("tinyllama-1.1b", "", 1234);
    assert!(matches!(
        VerificationRequest::new("prompt", no_quant, 16),
        Err(VerificationError::NotDeterministic { .. })
    ));

    // A participant running a different quantization is not part of the
    // comparison, even though its output happens to match.
    let different_quant = vec![
        ("node-a".to_owned(), deterministic(), "same".to_owned()),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q8_0", 1234),
            "same".to_owned(),
        ),
    ];
    assert!(
        matches!(
            verify(&request(), &different_quant),
            Err(VerificationError::IncompatibleParticipants { .. })
        ),
        "a different quantization must not be compared"
    );

    // A different seed is a different token sequence for the same prompt.
    let different_seed = vec![
        ("node-a".to_owned(), deterministic(), "same".to_owned()),
        (
            "node-b".to_owned(),
            Determinism::greedy("tinyllama-1.1b", "Q4_K_M", 7),
            "same".to_owned(),
        ),
    ];
    assert!(matches!(
        verify(&request(), &different_seed),
        Err(VerificationError::IncompatibleParticipants { .. })
    ));

    // A different model is not comparable either.
    let different_model = vec![
        ("node-a".to_owned(), deterministic(), "same".to_owned()),
        (
            "node-b".to_owned(),
            Determinism::greedy("llama-3-8b", "Q4_K_M", 1234),
            "same".to_owned(),
        ),
    ];
    assert!(matches!(
        verify(&request(), &different_model),
        Err(VerificationError::IncompatibleParticipants { .. })
    ));

    // The happy path still works: greedy, same model, same quantization, same
    // seed is comparable.
    assert!(deterministic().is_comparable());
    assert!(deterministic().is_compatible_with(&deterministic()));

    // Verification is opt-in: off by default, and never forced on all traffic.
    let default_policy = VerificationPolicy::new(VerificationConfig::default());
    assert!(!default_policy.is_enabled(), "verification is off by default");
    assert!(!default_policy.should_verify(0, true));

    let policy = VerificationPolicy::new(VerificationConfig {
        enabled: true,
        sample_every: 10,
        ..VerificationConfig::default()
    });
    assert!(policy.should_sample(0));
    assert!(!policy.should_sample(1), "sampling must not verify every request");
    assert!(policy.should_sample(10));
    assert!(
        policy.should_verify(3, true),
        "an explicit request is verified even when the sample misses"
    );
}

// ---------------------------------------------------------------------------
// The ledger is bounded and prune-able (the 11b idle-pruning pattern).
// ---------------------------------------------------------------------------

#[test]
fn the_reputation_ledger_is_bounded_and_prunable() {
    let ledger = ReputationLedger::new(-10.0, 10.0, Duration::from_millis(0), 2);

    ledger.record("a", 1.0, true);
    ledger.record("b", 1.0, true);
    ledger.record("c", 1.0, true);

    assert_eq!(ledger.len(), 2, "the ledger never exceeds max_entries");

    // A zero time to live makes every entry immediately prunable.
    assert_eq!(ledger.prune(), 2);
    assert!(ledger.is_empty());

    // Scores are clamped, so a peer cannot farm unlimited credit.
    let clamped = ReputationLedger::new(-10.0, 10.0, Duration::from_secs(60), 8);
    for _ in 0..100 {
        clamped.record("peer", 1.0, true);
    }
    assert_eq!(clamped.score("peer"), 10.0);
}
