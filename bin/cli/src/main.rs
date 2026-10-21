//! Synapse command line client entry point.

use anyhow::Result;
use clap::{Parser, Subcommand};

#[derive(Debug, Parser)]
#[command(name = "synapse-cli", version, about = "Synapse command line client")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Send a prompt to a Synapse node.
    Chat {
        /// Text of the prompt to send.
        #[arg(long)]
        prompt: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt().with_target(true).init();

    let cli = Cli::parse();

    match cli.command {
        Command::Chat { prompt } => {
            println!("prompt: {prompt}");
        }
    }

    Ok(())
}
