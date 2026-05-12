"use client";

import { useEffect, useRef, useState } from "react";

import { streamChat, synthesizeSpeech, transcribeAudio } from "@/lib/api";
import type { Message as MessageT, Trace } from "@/lib/types";

import Message from "./Message";
import SampleQuestions from "./SampleQuestions";
import VoiceRecorder from "./VoiceRecorder";

type Props = {
  sessionId: string;
  onTrace: (t: Trace) => void;
};

export default function ChatPanel({ sessionId, onTrace }: Props) {
  const [messages, setMessages] = useState<MessageT[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [errorBanner, setErrorBanner] = useState<string | null>(null);
  // Track which assistant message id needs autoplay (only when input was voice).
  const [autoplayId, setAutoplayId] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Reset chat when the parent rotates session_id (New Conversation).
  useEffect(() => {
    setMessages([]);
    setAutoplayId(null);
    setErrorBanner(null);
  }, [sessionId]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function send(prompt: string, isVoice: boolean) {
    if (!prompt.trim() || busy) return;
    setErrorBanner(null);
    setBusy(true);

    const userMsg: MessageT = {
      id: crypto.randomUUID(),
      role: "user",
      content: prompt,
      isVoice,
    };
    const assistantId = crypto.randomUUID();
    const assistantMsg: MessageT = {
      id: assistantId,
      role: "assistant",
      content: "",
    };
    setMessages((prev) => [...prev, userMsg, assistantMsg]);

    let accumulated = "";
    try {
      for await (const event of streamChat(prompt, sessionId)) {
        if (event.type === "token") {
          accumulated += event.content;
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantId ? { ...m, content: accumulated } : m,
            ),
          );
        } else if (event.type === "trace") {
          onTrace({
            tools: event.tools,
            sql: event.sql,
            chunks: event.chunks,
            latency: event.latency,
            guard_fired: event.guard_fired,
          });
        } else if (event.type === "error") {
          setErrorBanner(event.message);
        }
      }
    } catch (e) {
      setErrorBanner(e instanceof Error ? e.message : "Chat request failed.");
    }

    // TTS is generated only when the originating user message was voice.
    if (isVoice && accumulated) {
      try {
        const blob = await synthesizeSpeech(accumulated);
        const url = URL.createObjectURL(blob);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, audioUrl: url } : m,
          ),
        );
        setAutoplayId(assistantId);
      } catch (e) {
        setErrorBanner(
          `Voice playback failed (text response is shown above): ${
            e instanceof Error ? e.message : String(e)
          }`,
        );
      }
    }

    setBusy(false);
  }

  async function handleRecorded(blob: Blob) {
    setBusy(true);
    setErrorBanner(null);
    let transcript = "";
    try {
      transcript = await transcribeAudio(blob);
    } catch (e) {
      setErrorBanner(
        `Transcription failed: ${e instanceof Error ? e.message : String(e)}`,
      );
      setBusy(false);
      return;
    }
    if (!transcript.trim()) {
      setErrorBanner("Could not transcribe — try speaking again.");
      setBusy(false);
      return;
    }
    setBusy(false);
    await send(transcript, true);
  }

  return (
    <section className="flex h-full min-h-0 flex-1 flex-col">
      <div
        ref={scrollRef}
        className="scrollbar-thin flex-1 overflow-y-auto px-6 py-4"
      >
        <div className="mx-auto flex max-w-3xl flex-col gap-3">
          {messages.length === 0 && (
            <SampleQuestions onPick={(q) => send(q, false)} disabled={busy} />
          )}
          {messages.map((m) => (
            <Message key={m.id} msg={m} autoplayAudio={m.id === autoplayId} />
          ))}
          {errorBanner && (
            <div className="sticky bottom-0 rounded border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-800">
              <strong>Error.</strong> {errorBanner}
            </div>
          )}
        </div>
      </div>

      <div className="border-t border-ink-200 bg-surface px-6 py-3">
        <div className="mx-auto flex max-w-3xl flex-col gap-2">
          <VoiceRecorder onRecorded={handleRecorded} disabled={busy} />
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const v = input.trim();
              if (!v) return;
              setInput("");
              void send(v, false);
            }}
            className="flex items-center gap-2"
          >
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={busy}
              placeholder="Type your message — or use the mic above to speak…"
              className="flex-1 rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-sm text-ink-800 focus:border-brand focus:outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={busy || !input.trim()}
              className="rounded-md bg-brand px-4 py-2 text-sm font-medium text-white hover:bg-brand-dark disabled:cursor-not-allowed disabled:opacity-50"
            >
              {busy ? "…" : "Send"}
            </button>
          </form>
        </div>
      </div>
    </section>
  );
}
