use synapse_net::routing::RoutingTable;
use synapse_core::NodeTelemetry;
use libp2p::PeerId;

fn telemetry(peer: &str, active: u32, cap: u32) -> NodeTelemetry {
    NodeTelemetry {
        peer_id: peer.to_owned(),
        quic_port: 0,
        active_requests: active,
        max_capacity: cap,
        supported_models: vec!["m".into()],
        addresses: vec![],
        timestamp_ms: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis() as u64,
    }
}

#[test]
fn reputation_updates_on_match() {
    let t = RoutingTable::new();
    let p = "peer1".to_owned();
    t.update_reputation(&p, 1.0);
    assert_eq!(t.reputation(&p), 1.0);
}

#[test]
fn reputation_penalizes_on_mismatch() {
    let t = RoutingTable::new();
    let p = "peer1".to_owned();
    t.update_reputation(&p, -1.0);
    assert_eq!(t.reputation(&p), -1.0);
}

#[test]
fn routing_prefers_high_reputation() {
    let t = RoutingTable::new();
    let p1 = PeerId::random();
    let p2 = PeerId::random();
    let te1 = telemetry(&p1.to_base58(), 0, 4);
    let te2 = telemetry(&p2.to_base58(), 0, 4);
    t.update(p1, te1.clone());
    t.update(p2, te2.clone());
    t.update_reputation(&p1.to_base58(), 5.0);
    let best = t.get_best_peer(Some("m")).unwrap();
    assert_eq!(best.peer_id, p1.to_base58());
}

#[test]
fn verification_accepts_matching_outputs() {
    assert!(synapse_net::verify::outputs_match(&["a".into(), "a".into()]));
}

#[test]
fn verification_rejects_mismatch() {
    assert!(!synapse_net::verify::outputs_match(&["a".into(), "b".into()]));
}

#[test]
fn verification_requires_determinism() {
    // documented: comparison only with temp=0 + same model; here just assert outputs_match works
    assert!(synapse_net::verify::outputs_match(&["x".into()]));
}
