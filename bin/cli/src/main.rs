//! Synapse command line client entry point.

use anyhow::Result;
use clap::{CommandFactory, Parser, Subcommand};

/// Command line arguments of the client.
#[derive(Debug, Parser)]
#[command(name = "synapse-cli", version, about = "Synapse command line client")]
struct Cli {
    /// Multiaddr of a bootstrap node that is dialed on start up.
    #[arg(long, value_name = "MULTIADDR")]
    bootstrap: Option<String>,

    #[command(subcommand)]
    command: Option<Command>,
}

/// The subcommands of the client.
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
        Some(Command::Chat { prompt }) => {
            println!("prompt: {prompt}");
        }
        None => match cli.bootstrap.as_deref() {
            Some(address) => {
                println!("bootstrap: {address}");
            }
            None => {
                println!("{}", Cli::command().render_help());
            }
        },
    }

    Ok(())
}
