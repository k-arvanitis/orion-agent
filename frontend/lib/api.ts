// Thin fetch wrappers around the FastAPI surface.

import type { StreamEvent } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8088";

/**
 * POST /api/chat — streamed NDJSON.
 *
 * Yields each parsed event as it arrives. The final event is always a
 * `{ type: "trace", ... }` carrying the tool / sql / chunks / latency / guard
 * data the sidebar needs.
 */
export async function* streamChat(
  message: string,
  sessionId: string,
  signal?: AbortSignal,
): AsyncGenerator<StreamEvent> {
  const r = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ message, session_id: sessionId }),
    signal,
  });
  if (!r.ok || !r.body) {
    throw new Error(`Chat request failed: HTTP ${r.status}`);
  }

  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let nlIdx;
    while ((nlIdx = buffer.indexOf("\n")) !== -1) {
      const line = buffer.slice(0, nlIdx).trim();
      buffer = buffer.slice(nlIdx + 1);
      if (!line) continue;
      try {
        yield JSON.parse(line) as StreamEvent;
      } catch {
        // skip malformed line — protocol expects newline-delimited JSON
      }
    }
  }
  // Flush any tail
  const tail = buffer.trim();
  if (tail) {
    try {
      yield JSON.parse(tail) as StreamEvent;
    } catch {
      /* ignore */
    }
  }
}

/** POST /api/transcribe — multipart audio upload, returns transcript text. */
export async function transcribeAudio(audio: Blob): Promise<string> {
  const fd = new FormData();
  // Send a generic webm filename — Groq Whisper sniffs the bytes anyway.
  fd.append("file", audio, "recording.webm");
  const r = await fetch(`${API_BASE}/api/transcribe`, {
    method: "POST",
    body: fd,
  });
  if (!r.ok) {
    throw new Error(`Transcription failed: HTTP ${r.status}`);
  }
  const data = (await r.json()) as { text: string };
  return data.text;
}

/** POST /api/tts — returns an MP3 Blob you can play with `new Audio(URL.createObjectURL(blob))`. */
export async function synthesizeSpeech(text: string): Promise<Blob> {
  const r = await fetch(`${API_BASE}/api/tts`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!r.ok) {
    throw new Error(`TTS failed: HTTP ${r.status}`);
  }
  return await r.blob();
}
