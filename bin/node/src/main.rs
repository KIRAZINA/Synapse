//! Synapse node daemon entry point.

use anyhow::Result;
use synapse_engine::actor::InferenceActor;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(true).init();

    let actor = InferenceActor::spawn();
    info!("Synapse Node started successfully");

    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received, stopping inference actor");

    actor.shutdown()?;

    info!("Synapse Node stopped gracefully");
    Ok(())
}
