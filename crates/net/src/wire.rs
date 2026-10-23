//! The binary wire protocol of the Synapse data plane.
//!
//! The protocol is deliberately minimal: a command byte, a few big endian
//! integers and the UTF-8 payload. There is no serialisation framework
//! involved, so a token event costs exactly one small allocation.
//!
//! # Request (client to server)
//!
//! ```text
//! CMD_GENERATE | max_tokens: u32 | prompt_len: u32 | prompt: [u8]
//! ```
//!
//! # Events (server to client)
//!
//! ```text
//! EVT_TOKEN  | token_id: i32 | text_len: u16 | text: [u8]
//! EVT_DONE   | total_tokens: u32 | elapsed_ms: u64 | stop_reason: u8
//! EVT_ERROR  | msg_len: u16 | msg: [u8]
//! ```

use std::io;

use synapse_core::{StopReason, TokenEvent};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

/// Command: generate tokens for a prompt.
pub const CMD_GENERATE: u8 = 0x01;

/// Command: the data plane smoke test.
pub const CMD_PING: u8 = 0x02;

/// Command: authenticate the connection.
pub const CMD_AUTH: u8 = 0x03;

/// Event: a generated token.
pub const EVT_TOKEN: u8 = 0x01;

/// Event: the generation completed.
pub const EVT_DONE: u8 = 0x02;

/// Event: the generation failed.
pub const EVT_ERROR: u8 = 0x03;

/// Event: authentication rejected.
pub const EVT_AUTH_REJECTED: u8 = 0x04;

/// Largest prompt that is accepted, in bytes.
pub const MAX_PROMPT_BYTES: usize = 1024 * 1024;

/// Largest token piece or error message that is accepted, in bytes.
pub const MAX_TEXT_BYTES: usize = 64 * 1024;

/// Writes a generation request.
pub async fn write_request<W>(writer: &mut W, prompt: &str, max_tokens: u32) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let prompt_bytes = prompt.as_bytes();
    if prompt_bytes.len() > MAX_PROMPT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "the prompt is too long",
        ));
    }

    let mut frame = Vec::with_capacity(1 + 4 + 4 + prompt_bytes.len());
    frame.push(CMD_GENERATE);
    frame.extend_from_slice(&max_tokens.to_be_bytes());
    frame.extend_from_slice(&(prompt_bytes.len() as u32).to_be_bytes());
    frame.extend_from_slice(prompt_bytes);

    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads a generation request, including its command byte.
pub async fn read_request<R>(reader: &mut R) -> io::Result<(String, u32)>
where
    R: AsyncRead + Unpin,
{
    let mut command = [0_u8; 1];
    reader.read_exact(&mut command).await?;

    if command[0] != CMD_GENERATE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown command {:#04x}", command[0]),
        ));
    }

    read_request_body(reader).await
}

/// Reads the body of a generation request.
///
/// Used by a server that already consumed the command byte in order to
/// dispatch between the supported commands.
pub async fn read_request_body<R>(reader: &mut R) -> io::Result<(String, u32)>
where
    R: AsyncRead + Unpin,
{
    let mut max_tokens = [0_u8; 4];
    reader.read_exact(&mut max_tokens).await?;

    let mut prompt_len = [0_u8; 4];
    reader.read_exact(&mut prompt_len).await?;
    let prompt_len = u32::from_be_bytes(prompt_len) as usize;

    if prompt_len > MAX_PROMPT_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "the prompt is too long",
        ));
    }

    let mut prompt = vec![0_u8; prompt_len];
    reader.read_exact(&mut prompt).await?;

    let prompt =
        String::from_utf8(prompt).map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

    Ok((prompt, u32::from_be_bytes(max_tokens)))
}

/// Writes a single event of a generation.
pub async fn write_event<W>(writer: &mut W, event: &TokenEvent) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let frame = match event {
        TokenEvent::Token { text, token_id } => {
            let bytes = text.as_bytes();
            if bytes.len() > MAX_TEXT_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "the token piece is too long",
                ));
            }

            let mut frame = Vec::with_capacity(1 + 4 + 2 + bytes.len());
            frame.push(EVT_TOKEN);
            frame.extend_from_slice(&token_id.to_be_bytes());
            frame.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
            frame.extend_from_slice(bytes);
            frame
        }
        TokenEvent::Done {
            total_tokens,
            elapsed_ms,
            stop_reason,
        } => {
            let mut frame = Vec::with_capacity(1 + 4 + 8 + 1);
            frame.push(EVT_DONE);
            frame.extend_from_slice(&(*total_tokens as u32).to_be_bytes());
            frame.extend_from_slice(&elapsed_ms.to_be_bytes());
            frame.push(stop_reason_to_u8(*stop_reason));
            frame
        }
        TokenEvent::Error { message } => {
            let bytes = message.as_bytes();
            if bytes.len() > MAX_TEXT_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "the error message is too long",
                ));
            }

            let mut frame = Vec::with_capacity(1 + 2 + bytes.len());
            frame.push(EVT_ERROR);
            frame.extend_from_slice(&(bytes.len() as u16).to_be_bytes());
            frame.extend_from_slice(bytes);
            frame
        }
    };

    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads a single event of a generation.
pub async fn read_event<R>(reader: &mut R) -> io::Result<TokenEvent>
where
    R: AsyncRead + Unpin,
{
    let mut kind = [0_u8; 1];
    reader.read_exact(&mut kind).await?;

    match kind[0] {
        EVT_TOKEN => {
            let mut token_id = [0_u8; 4];
            reader.read_exact(&mut token_id).await?;

            let mut text_len = [0_u8; 2];
            reader.read_exact(&mut text_len).await?;
            let text_len = u16::from_be_bytes(text_len) as usize;

            let mut text = vec![0_u8; text_len];
            reader.read_exact(&mut text).await?;
            let text = String::from_utf8(text)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

            Ok(TokenEvent::Token {
                text,
                token_id: i32::from_be_bytes(token_id),
            })
        }
        EVT_DONE => {
            let mut total_tokens = [0_u8; 4];
            reader.read_exact(&mut total_tokens).await?;

            let mut elapsed_ms = [0_u8; 8];
            reader.read_exact(&mut elapsed_ms).await?;

            let mut stop_reason = [0_u8; 1];
            reader.read_exact(&mut stop_reason).await?;

            Ok(TokenEvent::Done {
                total_tokens: u32::from_be_bytes(total_tokens) as usize,
                elapsed_ms: u64::from_be_bytes(elapsed_ms),
                stop_reason: stop_reason_from_u8(stop_reason[0])?,
            })
        }
        EVT_ERROR => {
            let mut message_len = [0_u8; 2];
            reader.read_exact(&mut message_len).await?;
            let message_len = u16::from_be_bytes(message_len) as usize;

            let mut message = vec![0_u8; message_len];
            reader.read_exact(&mut message).await?;
            let message = String::from_utf8(message)
                .map_err(|err| io::Error::new(io::ErrorKind::InvalidData, err))?;

            Ok(TokenEvent::Error { message })
        }
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown event {other:#04x}"),
        )),
    }
}

/// Maps a stop reason onto its wire representation.
pub fn stop_reason_to_u8(reason: StopReason) -> u8 {
    match reason {
        StopReason::Eos => 0,
        StopReason::MaxTokens => 1,
        StopReason::ClientAbort => 2,
    }
}

/// Maps a wire representation back onto a stop reason.
pub fn stop_reason_from_u8(value: u8) -> io::Result<StopReason> {
    match value {
        0 => Ok(StopReason::Eos),
        1 => Ok(StopReason::MaxTokens),
        2 => Ok(StopReason::ClientAbort),
        other => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown stop reason {other:#04x}"),
        )),
    }
}

/// Writes an AUTH frame.
pub async fn write_auth<W>(writer: &mut W, client_peer_id: &[u8], public_key: &[u8], nonce: &[u8; 32], signature: &[u8]) -> io::Result<()>
where
    W: AsyncWrite + Unpin,
{
    let mut frame = Vec::with_capacity(1 + 2 + client_peer_id.len() + 2 + public_key.len() + 32 + 2 + signature.len());
    frame.push(CMD_AUTH);
    frame.extend_from_slice(&(client_peer_id.len() as u16).to_be_bytes());
    frame.extend_from_slice(client_peer_id);
    frame.extend_from_slice(&(public_key.len() as u16).to_be_bytes());
    frame.extend_from_slice(public_key);
    frame.extend_from_slice(nonce);
    frame.extend_from_slice(&(signature.len() as u16).to_be_bytes());
    frame.extend_from_slice(signature);
    writer.write_all(&frame).await?;
    writer.flush().await?;
    Ok(())
}

/// Reads an AUTH frame body.
pub async fn read_auth<R>(reader: &mut R) -> io::Result<(Vec<u8>, Vec<u8>, [u8; 32], Vec<u8>)>
where
    R: AsyncRead + Unpin,
{
    let mut pid_len = [0u8; 2];
    reader.read_exact(&mut pid_len).await?;
    let pid_len = u16::from_be_bytes(pid_len) as usize;
    if pid_len > 256 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "peer id too long"));
    }
    let mut pid = vec![0u8; pid_len];
    reader.read_exact(&mut pid).await?;
    let mut pk_len = [0u8; 2];
    reader.read_exact(&mut pk_len).await?;
    let pk_len = u16::from_be_bytes(pk_len) as usize;
    if pk_len > 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "public key too long"));
    }
    let mut pk = vec![0u8; pk_len];
    reader.read_exact(&mut pk).await?;
    let mut nonce = [0u8; 32];
    reader.read_exact(&mut nonce).await?;
    let mut sig_len = [0u8; 2];
    reader.read_exact(&mut sig_len).await?;
    let sig_len = u16::from_be_bytes(sig_len) as usize;
    if sig_len > 1024 {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "signature too long"));
    }
    let mut sig = vec![0u8; sig_len];
    reader.read_exact(&mut sig).await?;
    Ok((pid, pk, nonce, sig))
}
