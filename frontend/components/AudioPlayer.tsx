"use client";

import { useEffect, useRef, useState } from "react";

type Props = {
  src: string;
  autoplay?: boolean;
};

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export default function AudioPlayer({ src, autoplay }: Props) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [playing, setPlaying] = useState(false);
  const [current, setCurrent] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    if (autoplay && audioRef.current) {
      audioRef.current.play().catch(() => {
        /* autoplay may be blocked — user can press play */
      });
    }
  }, [autoplay, src]);

  function toggle() {
    const a = audioRef.current;
    if (!a) return;
    if (a.paused) a.play();
    else a.pause();
  }

  function seek(e: React.MouseEvent<HTMLDivElement>) {
    const a = audioRef.current;
    if (!a || !duration) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    a.currentTime = Math.max(0, Math.min(duration, ratio * duration));
  }

  const progress = duration ? (current / duration) * 100 : 0;

  return (
    <div className="mt-2 flex h-10 w-full items-center gap-3 rounded-xl bg-slate-900 px-3 text-slate-100">
      <button
        type="button"
        onClick={toggle}
        aria-label={playing ? "Pause" : "Play"}
        className="flex h-6 w-6 flex-shrink-0 items-center justify-center text-slate-100 transition hover:text-brand-light"
      >
        {playing ? (
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
            <rect x="6" y="5" width="4" height="14" rx="1" />
            <rect x="14" y="5" width="4" height="14" rx="1" />
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="currentColor" className="h-5 w-5">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
      </button>

      <div
        className="relative h-1 flex-1 cursor-pointer rounded-full bg-slate-700"
        onClick={seek}
      >
        <div
          className="absolute left-0 top-0 h-full rounded-full bg-brand transition-[width] duration-100"
          style={{ width: `${progress}%` }}
        />
      </div>

      <span className="flex-shrink-0 font-mono text-[11px] tabular-nums text-slate-400">
        {formatTime(current)} / {formatTime(duration)}
      </span>

      <audio
        ref={audioRef}
        src={src}
        className="hidden"
        onPlay={() => setPlaying(true)}
        onPause={() => setPlaying(false)}
        onEnded={() => setPlaying(false)}
        onTimeUpdate={(e) => setCurrent(e.currentTarget.currentTime)}
        onLoadedMetadata={(e) => setDuration(e.currentTarget.duration)}
      />
    </div>
  );
}
