//! The data plane has to stop a generation when the client goes away.
//!
//! A client that is killed, that crashes or that loses its network can no
//! longer send a stream reset. The server then has only two signals left: a
//! failing write, and the closure of the connection by its idle timeout. Both
//! are exercised here, together with the graceful abort that has to keep
//! working.

use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Duration;

use quinn::{Endpoint, VarInt};
use synapse_core::{SynapseError, TokenEvent, TokenStreamProvider};
use synapse_net::client::RemoteInferenceClient;
use synapse_net::server::InferenceServer;
use synapse_net::transport::{
    build_client_config, install_crypto_provider, QuicTransport, SERVER_NAME,
};
use synapse_net::wire;
use tokio::net::UdpSocket;

/// Token delay of the fake engine.
///
/// It is long enough that no generation of these tests can complete on its
/// own while a test is waiting for an abort.
const TOKEN_DELAY: Duration = Duration::from_millis(250);

/// Tokens requested by the tests. At [`TOKEN_DELAY`] the generation would run
/// for fifty seconds, so an abort is never confused with a completion.
const MAX_TOKENS: u32 = 200;

/// How long the tests wait for a generation to be aborted after the client
/// went silent.
///
/// The transport idle timeout is ten seconds; the margin covers the time the
/// QUIC timers and the test runtime need on top of it.
const ABORT_TIMEOUT: Duration = Duration::from_secs(20);

/// How long the tests wait for the abort of a graceful reset.
const RESET_TIMEOUT: Duration = Duration::from_secs(5);

/// Size of the receive buffer of the test relay.
const UDP_BUFFER_BYTES: usize = 64 * 1024;

/// A provider that emits one token every [`TOKEN_DELAY`].
///
/// It mirrors the engine in the only aspect that matters here: it stops as
/// soon as the receiver of the event stream is dropped, and it counts the
/// tokens it produced so that a test can tell an abort from a completion.
struct SlowProvider {
    /// The number of generations that are currently running.
    active: Arc<AtomicU32>,
    /// The number of tokens that were produced.
    produced: Arc<AtomicU32>,
}

impl SlowProvider {
    fn new() -> Self {
        Self {
            active: Arc::new(AtomicU32::new(0)),
            produced: Arc::new(AtomicU32::new(0)),
        }
    }

    /// The number of generations that are currently running.
    fn active(&self) -> u32 {
        self.active.load(Ordering::Relaxed)
    }

    /// The number of tokens that were produced.
    fn produced(&self) -> u32 {
        self.produced.load(Ordering::Relaxed)
    }
}

impl TokenStreamProvider for SlowProvider {
    fn generate_streaming(
        &self,
        _prompt: String,
        max_tokens: usize,
    ) -> Result<flume::Receiver<TokenEvent>, SynapseError> {
        let (sender, receiver) = flume::unbounded();
        let active = Arc::clone(&self.active);
        let produced = Arc::clone(&self.produced);

        active.fetch_add(1, Ordering::Relaxed);
        tokio::spawn(async move {
            for _ in 0..max_tokens {
                tokio::time::sleep(TOKEN_DELAY).await;

                let event = TokenEvent::Token {
                    text: "x".to_owned(),
                    token_id: 0,
                };

                // A failing send means that the server stopped listening.
                if sender.send(event).is_err() {
                    break;
                }

                produced.fetch_add(1, Ordering::Relaxed);
            }

            active.fetch_sub(1, Ordering::Relaxed);
        });

        Ok(receiver)
    }

    fn active_requests(&self) -> u32 {
        self.active()
    }

    fn max_capacity(&self) -> u32 {
        1
    }
}

/// Starts an inference server with the given provider.
///
/// The transport is returned as well: it owns the endpoint of the server.
async fn start_server(
    provider: Arc<SlowProvider>,
) -> Result<(QuicTransport, SocketAddr), Box<dyn std::error::Error>> {
    let transport = QuicTransport::new().await?;
    let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, transport.local_port()));

    let security = synapse_core::SecurityConfig {
        allow_loopback: true,
        ..Default::default()
    };
    let server = InferenceServer::new(
        transport.endpoint().clone(),
        provider as Arc<dyn TokenStreamProvider>,
        security,
        std::sync::Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
    );
    tokio::spawn(server.run());

    Ok((transport, addr))
}

/// Waits until the provider reports the given number of running generations.
async fn wait_for_active(provider: &SlowProvider, count: u32, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;

    while tokio::time::Instant::now() < deadline {
        if provider.active() == count {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    provider.active() == count
}

/// Waits until the provider has produced the given number of tokens.
async fn wait_for_produced(provider: &SlowProvider, count: u32, timeout: Duration) -> bool {
    let deadline = tokio::time::Instant::now() + timeout;

    while tokio::time::Instant::now() < deadline {
        if provider.produced() >= count {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }

    provider.produced() >= count
}

/// Forwards datagrams between a single client and the server.
///
/// The relay stands in for the network between the two. As long as it runs,
/// the client and the server talk to each other; aborting it is what a client
/// that is killed or disconnected does to its traffic.
async fn relay_packets(socket: Arc<UdpSocket>, server: SocketAddr) {
    let mut client: Option<SocketAddr> = None;
    let mut buffer = vec![0_u8; UDP_BUFFER_BYTES];

    loop {
        let Ok((length, from)) = socket.recv_from(&mut buffer).await else {
            return;
        };

        let target = if from == server {
            match client {
                Some(client) => client,
                // The server only answers once the client has spoken.
                None => continue,
            }
        } else {
            client = Some(from);
            server
        };

        let _ = socket.send_to(&buffer[..length], target).await;
    }
}

/// Keeps the port of a dead client bound and swallows every datagram.
///
/// Closing the socket would make the operating system answer with an ICMP
/// error instead of silence, and the test is about silence: a peer that is
/// gone without saying goodbye leaves the packets of the server unanswered.
async fn swallow_packets(socket: Arc<UdpSocket>) {
    let mut buffer = vec![0_u8; UDP_BUFFER_BYTES];

    loop {
        if socket.recv_from(&mut buffer).await.is_err() {
            return;
        }
    }
}

/// A client that goes silent without resetting its stream stops the server.
#[tokio::test]
async fn a_dead_client_aborts_the_generation() -> Result<(), Box<dyn std::error::Error>> {
    install_crypto_provider();

    let provider = Arc::new(SlowProvider::new());
    let (transport, server_addr) = start_server(Arc::clone(&provider) as Arc<_>).await?;

    // The client talks to the relay, not to the server: the relay is the only
    // path between them, so aborting it silences the connection.
    let relay_socket = Arc::new(UdpSocket::bind((Ipv4Addr::LOCALHOST, 0)).await?);
    let relay_addr = relay_socket.local_addr()?;
    let relay = tokio::spawn(relay_packets(Arc::clone(&relay_socket), server_addr));

    let endpoint = Endpoint::client((Ipv4Addr::LOCALHOST, 0).into())?;
    let config = build_client_config(transport.cert_der().to_vec())?;
    let connection = endpoint.connect_with(config, relay_addr, SERVER_NAME)?.await?;

    let (send, _recv) = connection.open_bi().await?;
    let mut send = send;
    wire::write_request(&mut send, "hello", MAX_TOKENS).await?;

    assert!(
        wait_for_produced(&provider, 2, Duration::from_secs(5)).await,
        "the server did not start the generation"
    );
    assert_eq!(
        provider.active(),
        1,
        "the server did not report a running generation"
    );

    // The client process dies: nothing of it reaches the server any more, and
    // no stream reset is sent either. The socket of the relay stays bound, so
    // the server gets silence instead of an error from the operating system.
    relay.abort();
    let _ = relay.await;
    tokio::spawn(swallow_packets(Arc::clone(&relay_socket)));

    // The server has to fall back on the idle timeout of the connection.
    assert!(
        wait_for_active(&provider, 0, ABORT_TIMEOUT).await,
        "the server kept the generation of a dead client running"
    );

    let produced = provider.produced();
    assert!(
        produced < MAX_TOKENS,
        "the generation ran to the token limit instead of being aborted after {produced} tokens"
    );

    Ok(())
}

/// The abort handle of the real client stops the generation immediately.
///
/// This is the graceful path of the command line client: an interrupt aborts
/// the generation through `AbortHandle`, which resets the stream. It has to
/// stay immediate, and it must not wait for the idle timeout.
#[tokio::test]
async fn the_client_abort_handle_aborts_the_generation() -> Result<(), Box<dyn std::error::Error>> {
    install_crypto_provider();

    let provider = Arc::new(SlowProvider::new());
    let (transport, server_addr) = start_server(Arc::clone(&provider) as Arc<_>).await?;

    let keypair = synapse_net::generate_keypair();
    let client = RemoteInferenceClient::new(server_addr, transport.cert_der().to_vec(), keypair, [0u8; 32])?;
    let (events, abort) = client.generate("hello".to_owned(), MAX_TOKENS).await?;

    // Wait for the first tokens, so that the generation is really running.
    assert!(
        wait_for_produced(&provider, 2, Duration::from_secs(5)).await,
        "the server did not produce any token"
    );
    let _ = events.recv_timeout(Duration::from_secs(5))?;

    // What the interrupt handler of the command line client does.
    let signalled = tokio::time::Instant::now();
    abort.abort();

    assert!(
        wait_for_active(&provider, 0, Duration::from_secs(2)).await,
        "the server did not abort the generation after the client aborted it"
    );
    assert!(
        signalled.elapsed() < Duration::from_secs(2),
        "the abort took {:?}, which is far more than a stream reset needs",
        signalled.elapsed()
    );
    assert!(
        provider.produced() < MAX_TOKENS,
        "the generation ran to the token limit instead of being aborted"
    );

    Ok(())
}

/// A client that resets its stream aborts the generation right away.
///
/// This is the graceful abort at the transport level: the connection stays
/// alive and only the stream of the generation is reset.
#[tokio::test]
async fn a_stream_reset_aborts_the_generation() -> Result<(), Box<dyn std::error::Error>> {
    install_crypto_provider();

    let provider = Arc::new(SlowProvider::new());
    let (transport, server_addr) = start_server(Arc::clone(&provider) as Arc<_>).await?;

    let endpoint = Endpoint::client((Ipv4Addr::LOCALHOST, 0).into())?;
    let config = build_client_config(transport.cert_der().to_vec())?;
    let connection = endpoint.connect_with(config, server_addr, SERVER_NAME)?.await?;

    let (mut send, mut recv) = connection.open_bi().await?;
    wire::write_request(&mut send, "hello", MAX_TOKENS).await?;

    // Wait for the first tokens, so that the generation is really running.
    assert!(
        wait_for_produced(&provider, 2, Duration::from_secs(5)).await,
        "the server did not produce any token"
    );
    let _ = wire::read_event(&mut recv).await?;

    // The abort of a live client: the stream is reset, the connection stays.
    send.reset(VarInt::from_u32(1))?;

    assert!(
        wait_for_active(&provider, 0, RESET_TIMEOUT).await,
        "the server did not abort the generation after the stream was reset"
    );
    assert!(
        provider.produced() < MAX_TOKENS,
        "the generation ran to the token limit instead of being aborted"
    );

    Ok(())
}
