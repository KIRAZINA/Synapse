//! The QUIC data plane of a Synapse node.
//!
//! The data plane is deliberately independent from the libp2p control plane:
//! the only thing the two share is the [`HandshakeInfo`] structure that the
//! control plane hands over, so the QUIC stack can be replaced without
//! touching the swarm.
//!
//! Every peer presents a self signed certificate that is generated at start
//! up. The certificate is published over the control plane and the client
//! pins it, which means that a peer can only be talked to if it owns the
//! private key of the certificate it announced.

use std::net::SocketAddr;
use std::sync::Arc;

use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use quinn::{ClientConfig, Endpoint, ServerConfig, VarInt};
use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
use rustls::crypto::{verify_tls12_signature, verify_tls13_signature};
use rustls::pki_types::{CertificateDer, PrivatePkcs8KeyDer, ServerName, UnixTime};
use rustls::{DigitallySignedStruct, Error as TlsError, SignatureScheme};
use synapse_core::SynapseError;
use tracing::{info, warn};

/// The payload of the data plane smoke test.
const PING: &[u8] = b"PING";

/// The answer to a [`PING`].
const PONG: &[u8] = b"PONG";

/// ALPN identifier of the Synapse data plane.
const ALPN: &[u8] = b"synapse/1";

/// The server name that is used for the TLS handshake.
///
/// The certificate of a peer is self signed for `localhost`; the identity is
/// established by pinning, not by the name.
pub const SERVER_NAME: &str = "localhost";

/// The QUIC data plane of a node.
#[derive(Debug, Clone)]
pub struct QuicTransport {
    endpoint: Endpoint,
    cert_der: Vec<u8>,
    local_port: u16,
}

impl QuicTransport {
    /// Generates a certificate, binds the endpoint and prepares the TLS
    /// configuration.
    pub async fn new() -> Result<Self, SynapseError> {
        let certified =
            rcgen::generate_simple_self_signed(vec![SERVER_NAME.to_owned()]).map_err(|err| {
                SynapseError::Network(format!("failed to generate the TLS certificate: {err}"))
            })?;

        let cert_der = certified.cert.der().to_vec();
        let key_der = certified.signing_key.serialize_der();

        let mut server_tls = rustls::ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(
                vec![CertificateDer::from(cert_der.clone())],
                PrivatePkcs8KeyDer::from(key_der).into(),
            )
            .map_err(|err| {
                SynapseError::Network(format!("failed to build the TLS server config: {err}"))
            })?;
        server_tls.alpn_protocols = vec![ALPN.to_vec()];

        let server_crypto = QuicServerConfig::try_from(server_tls).map_err(|err| {
            SynapseError::Network(format!("failed to build the QUIC server config: {err}"))
        })?;
        let server_config = ServerConfig::with_crypto(Arc::new(server_crypto));

        let bind_addr: SocketAddr = ([0, 0, 0, 0], 0).into();
        let endpoint = Endpoint::server(server_config, bind_addr).map_err(|err| {
            SynapseError::Network(format!("failed to bind the QUIC endpoint: {err}"))
        })?;

        let local_port = endpoint
            .local_addr()
            .map_err(|err| SynapseError::Network(format!("failed to read the local port: {err}")))?
            .port();

        info!(
            port = local_port,
            certificate_bytes = cert_der.len(),
            "quic data plane ready"
        );

        Ok(Self {
            endpoint,
            cert_der,
            local_port,
        })
    }

    /// The UDP port the endpoint listens on.
    pub fn local_port(&self) -> u16 {
        self.local_port
    }

    /// The DER encoded certificate that peers have to pin.
    pub fn cert_der(&self) -> &[u8] {
        &self.cert_der
    }

    /// The underlying QUIC endpoint.
    pub fn endpoint(&self) -> &Endpoint {
        &self.endpoint
    }

    /// Connects to a peer, sends a `PING` and waits for the `PONG`.
    pub async fn connect(&self, addr: SocketAddr, cert_der: Vec<u8>) -> Result<(), SynapseError> {
        connect(self.endpoint.clone(), addr, cert_der).await
    }
}

/// Connects to a peer, sends a `PING` and waits for the `PONG`.
///
/// This is the smoke test of the data plane; the generation protocol lives in
/// [`crate::client`].
pub async fn connect(
    endpoint: Endpoint,
    addr: SocketAddr,
    cert_der: Vec<u8>,
) -> Result<(), SynapseError> {
    let client_config = build_client_config(cert_der)?;

    let connecting = endpoint
        .connect_with(client_config, addr, SERVER_NAME)
        .map_err(|err| {
            SynapseError::Network(format!("failed to start the quic connection: {err}"))
        })?;

    let connection = connecting.await.map_err(|err| {
        SynapseError::Network(format!("the quic connection to {addr} failed: {err}"))
    })?;
    info!(%addr, "quic connection established");

    let (mut send, mut recv) = connection
        .open_bi()
        .await
        .map_err(|err| SynapseError::Network(format!("failed to open a quic stream: {err}")))?;

    send.write_all(&[crate::wire::CMD_PING])
        .await
        .map_err(|err| SynapseError::Network(format!("failed to send the ping: {err}")))?;
    send.write_all(PING)
        .await
        .map_err(|err| SynapseError::Network(format!("failed to send the ping: {err}")))?;

    // The answer is read before the stream is closed: the peer keeps the
    // connection alive until this side is finished.
    let mut answer = [0_u8; 4];
    recv.read_exact(&mut answer)
        .await
        .map_err(|err| SynapseError::Network(format!("failed to read the pong: {err}")))?;

    send.finish()
        .map_err(|err| SynapseError::Network(format!("failed to finish the ping: {err}")))?;

    if answer == PONG {
        info!(%addr, "quic ping/pong succeeded");
    } else {
        warn!(%addr, "quic ping/pong failed: unexpected answer");
        return Err(SynapseError::Network(format!(
            "unexpected answer from {addr}: {answer:?}"
        )));
    }

    connection.close(VarInt::from_u32(0), b"done");
    Ok(())
}

/// Builds a client configuration that pins the certificate of the peer.
pub fn build_client_config(cert_der: Vec<u8>) -> Result<ClientConfig, SynapseError> {
    let mut tls = rustls::ClientConfig::builder()
        .dangerous()
        .with_custom_certificate_verifier(Arc::new(PinnedCertificateVerifier {
            expected: cert_der,
        }))
        .with_no_client_auth();
    tls.alpn_protocols = vec![ALPN.to_vec()];

    let crypto = QuicClientConfig::try_from(tls).map_err(|err| {
        SynapseError::Network(format!("failed to build the QUIC client config: {err}"))
    })?;

    Ok(ClientConfig::new(Arc::new(crypto)))
}

/// The signature schemes that are accepted for the pinned certificate.
fn supported_algorithms() -> rustls::crypto::WebPkiSupportedAlgorithms {
    rustls::crypto::ring::default_provider().signature_verification_algorithms
}

/// Verifies that the peer presents exactly the pinned certificate.
#[derive(Debug)]
struct PinnedCertificateVerifier {
    /// The DER encoded certificate of the peer.
    expected: Vec<u8>,
}

impl ServerCertVerifier for PinnedCertificateVerifier {
    fn verify_server_cert(
        &self,
        end_entity: &CertificateDer<'_>,
        _intermediates: &[CertificateDer<'_>],
        _server_name: &ServerName<'_>,
        _ocsp_response: &[u8],
        _now: UnixTime,
    ) -> Result<ServerCertVerified, TlsError> {
        if end_entity.as_ref() == self.expected.as_slice() {
            Ok(ServerCertVerified::assertion())
        } else {
            Err(TlsError::General(
                "the peer certificate does not match the pinned certificate".to_owned(),
            ))
        }
    }

    fn verify_tls12_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls12_signature(message, cert, dss, &supported_algorithms())
    }

    fn verify_tls13_signature(
        &self,
        message: &[u8],
        cert: &CertificateDer<'_>,
        dss: &DigitallySignedStruct,
    ) -> Result<HandshakeSignatureValid, TlsError> {
        verify_tls13_signature(message, cert, dss, &supported_algorithms())
    }

    fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
        vec![
            SignatureScheme::ED25519,
            SignatureScheme::ECDSA_NISTP256_SHA256,
            SignatureScheme::ECDSA_NISTP384_SHA384,
            SignatureScheme::RSA_PSS_SHA256,
            SignatureScheme::RSA_PKCS1_SHA256,
        ]
    }
}
