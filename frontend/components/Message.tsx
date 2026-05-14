"use client";

import type { Message as MessageT } from "@/lib/types";

import AudioPlayer from "./AudioPlayer";

type Props = {
  msg: MessageT;
  autoplayAudio?: boolean;
};

export default function Message({ msg, autoplayAudio }: Props) {
  const isUser = msg.role === "user";
  const isLoading = !isUser && msg.content === "" && !msg.audioUrl;

  return (
    <div
      className={`flex flex-col ${isUser ? "items-end" : "items-start"}`}
    >
      <div
        className={`max-w-[85%] whitespace-pre-wrap rounded-2xl px-4 py-2 text-sm ${
          isUser
            ? "bg-brand text-white"
            : "border border-ink-200 bg-surface text-ink-800"
        }`}
      >
        {isUser && msg.isVoice && (
          <span
            className="mr-2 text-xs"
            title="This message came from the microphone"
          >
            🎤
          </span>
        )}
        {isLoading ? (
          <span className="block h-2 w-12 animate-pulse rounded-full bg-ink-200" />
        ) : (
          <span>{msg.content}</span>
        )}
        {!isUser && msg.audioUrl && (
          <AudioPlayer src={msg.audioUrl} autoplay={autoplayAudio} />
        )}
      </div>
      <div className="mt-0.5 px-1 text-[11px] text-ink-400">
        {isUser ? (msg.isVoice ? "you · voice" : "you") : "Orion"}
      </div>
    </div>
  );
}
