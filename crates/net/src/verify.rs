use metrics::counter;

pub fn outputs_match(outputs: &[String]) -> bool {
    if outputs.is_empty() { return false; }
    let first = &outputs[0];
    let all_match = outputs.iter().all(|o| o == first);
    counter!("synapse_verification_attempts_total").increment(1);
    if all_match {
        counter!("synapse_verification_matches_total").increment(1);
    } else {
        counter!("synapse_verification_mismatches_total").increment(1);
    }
    all_match
}
