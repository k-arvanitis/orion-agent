"use client";

import { useEffect, useState } from "react";

import ChatPanel from "@/components/ChatPanel";
import ThemeToggle from "@/components/ThemeToggle";
import TraceSidebar from "@/components/TraceSidebar";
import type { Trace } from "@/lib/types";

const SESSION_KEY = "orion-session-id";

export default function Home() {
  const [sessionId, setSessionId] = useState<string>("");
  const [trace, setTrace] = useState<Trace | null>(null);

  // Generate / restore session_id on mount. localStorage so the agent's
  // per-thread history survives a page reload during a demo.
  useEffect(() => {
    const existing =
      typeof window !== "undefined"
        ? window.localStorage.getItem(SESSION_KEY)
        : null;
    if (existing) {
      setSessionId(existing);
    } else {
      const fresh = crypto.randomUUID();
      window.localStorage.setItem(SESSION_KEY, fresh);
      setSessionId(fresh);
    }
  }, []);

  function newSession() {
    const fresh = crypto.randomUUID();
    window.localStorage.setItem(SESSION_KEY, fresh);
    setSessionId(fresh);
    setTrace(null);
  }

  if (!sessionId) {
    return null; // wait for hydration
  }

  return (
    <div className="flex h-screen flex-col">
      <header className="flex items-center justify-between border-b border-ink-200 bg-surface px-6 py-3">
        <div>
          <h1 className="text-xl font-bold tracking-tight text-ink-900">
            ShopNova Customer Support
          </h1>
          <p className="text-[11px] text-ink-500">
            Orion · LangGraph · Qdrant · Supabase · voice mode
          </p>
        </div>
        <ThemeToggle />
      </header>
      <div className="flex flex-1 overflow-hidden">
        <ChatPanel sessionId={sessionId} onTrace={setTrace} />
        <TraceSidebar
          trace={trace}
          sessionId={sessionId}
          onNewSession={newSession}
        />
      </div>
    </div>
  );
}
