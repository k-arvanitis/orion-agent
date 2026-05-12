"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  onRecorded: (audio: Blob) => void;
  disabled?: boolean;
};

/**
 * Press-to-record mic button. Click to start, click again to stop.
 * On stop, the recorded webm blob is handed to `onRecorded` for upload.
 *
 * Uses the browser MediaRecorder API. Requires the user to grant microphone
 * permission once per origin.
 */
export default function VoiceRecorder({ onRecorded, disabled }: Props) {
  const [recording, setRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<BlobPart[]>([]);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    return () => {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    };
  }, []);

  async function start() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("Microphone API not available in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const recorder = new MediaRecorder(stream);
      chunksRef.current = [];
      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        stream.getTracks().forEach((t) => t.stop());
        streamRef.current = null;
        if (blob.size > 0) onRecorded(blob);
      };
      recorder.start();
      recorderRef.current = recorder;
      setRecording(true);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not access microphone.");
    }
  }

  function stop() {
    recorderRef.current?.stop();
    recorderRef.current = null;
    setRecording(false);
  }

  return (
    <div className="flex items-center gap-2">
      <button
        type="button"
        onClick={recording ? stop : start}
        disabled={disabled}
        aria-pressed={recording}
        title={recording ? "Stop recording" : "Tap to record"}
        className={`flex h-9 w-9 items-center justify-center rounded-full text-white transition disabled:cursor-not-allowed disabled:opacity-50 ${
          recording
            ? "animate-pulse bg-red-600 hover:bg-red-500"
            : "bg-ink-700 hover:bg-ink-600"
        }`}
      >
        {recording ? "■" : "🎤"}
      </button>
      <span className="text-[11px] text-ink-500">
        {recording ? (
          <span className="font-medium text-red-600">listening…</span>
        ) : (
          "Tap mic to speak"
        )}
      </span>
      {error && (
        <span className="text-[11px] text-red-600" role="alert">
          {error}
        </span>
      )}
    </div>
  );
}
