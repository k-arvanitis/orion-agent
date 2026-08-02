// Thin fetch wrappers around the FastAPI surface.

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8088";

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
