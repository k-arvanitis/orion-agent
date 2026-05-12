"use client";

import { useState } from "react";

import type { Trace } from "@/lib/types";

const TOOL_LABELS: Record<string, { label: string; cls: string }> = {
  query_database: { label: "🗄️ SQL", cls: "bg-blue-50 text-blue-700" },
  search_policies: {
    label: "📄 RAG",
    cls: "bg-emerald-50 text-emerald-700",
  },
  escalate: { label: "🚨 Escalation", cls: "bg-red-50 text-red-700" },
};

type Props = {
  trace: Trace | null;
  sessionId: string;
  onNewSession: () => void;
};

export default function TraceSidebar({
  trace,
  sessionId,
  onNewSession,
}: Props) {
  return (
    <aside className="scrollbar-thin hidden w-[300px] flex-shrink-0 flex-col gap-3 overflow-y-auto border-l border-ink-200 bg-ink-50 p-3 lg:flex">
      {!trace && (
        <Panel title="Tool trace">
          <p className="text-xs text-ink-500">
            Ask a question — the agent&apos;s tools, SQL, retrieved chunks and
            latency for the last turn show up here.
          </p>
        </Panel>
      )}

      {trace && (
        <Panel
          title="Tool trace"
          right={
            <span className="text-[11px] text-ink-500">
              {trace.latency.toFixed(2)}s
            </span>
          }
        >
          <div className="space-y-3">
            {trace.guard_fired && (
              <div className="rounded border border-amber-200 bg-amber-50 px-2.5 py-2 text-xs text-amber-800">
                <strong>Hallucination guard triggered.</strong> The first
                answer contained numbers not present in the tool output; the
                agent was re-prompted and the corrected reply is shown.
              </div>
            )}

            <div className="space-y-1.5">
              <div className="text-[11px] font-medium text-ink-600">
                Tools used
              </div>
              {trace.tools.length === 0 ? (
                <div className="rounded bg-ink-100 px-2 py-1.5 text-[11px] text-ink-500">
                  No tools called — answered from context.
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {trace.tools.map((t) => {
                    const meta = TOOL_LABELS[t] ?? {
                      label: t,
                      cls: "bg-ink-100 text-ink-600",
                    };
                    return (
                      <span
                        key={t}
                        className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${meta.cls}`}
                      >
                        {meta.label}
                      </span>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        </Panel>
      )}

      {trace?.sql && (
        <Panel title="Generated SQL" collapsible defaultOpen>
          <pre className="overflow-x-auto rounded bg-slate-900 p-2 font-mono text-[10px] text-slate-50">
            {trace.sql}
          </pre>
        </Panel>
      )}

      {trace?.chunks && trace.chunks.length > 0 && (
        <Panel
          title={`Retrieved chunks (${trace.chunks.length})`}
          collapsible
          defaultOpen
        >
          <ul className="space-y-3">
            {trace.chunks.map((c, i) => (
              <li
                key={i}
                className="border-l-2 border-ink-200 pl-2.5 text-xs"
              >
                <div className="font-semibold text-ink-800">
                  {i + 1}. {c.heading}
                </div>
                <div className="text-[11px] text-ink-400">📄 {c.source}</div>
                <p className="mt-1 text-ink-600">
                  {c.content.length > 300
                    ? c.content.slice(0, 300) + "…"
                    : c.content}
                </p>
              </li>
            ))}
          </ul>
        </Panel>
      )}

      <div className="mt-auto rounded-lg border border-ink-200 bg-surface p-3">
        <div className="text-[11px] font-medium text-ink-600">Session</div>
        <code className="font-mono text-[10px] text-ink-400">
          {sessionId.slice(0, 8)}…
        </code>
        <button
          onClick={onNewSession}
          className="mt-2 w-full rounded-md border border-ink-200 bg-ink-50 px-3 py-1 text-[11px] text-ink-700 hover:bg-ink-100"
        >
          New conversation
        </button>
      </div>
    </aside>
  );
}

function Panel({
  title,
  right,
  collapsible,
  defaultOpen,
  children,
}: {
  title: string;
  right?: React.ReactNode;
  collapsible?: boolean;
  defaultOpen?: boolean;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(!!defaultOpen);
  const body = (!collapsible || open) && (
    <div className="p-3">{children}</div>
  );

  return (
    <div className="rounded-lg border border-ink-200 bg-surface">
      {collapsible ? (
        <button
          type="button"
          onClick={() => setOpen((o) => !o)}
          className="flex w-full items-center justify-between border-b border-ink-100 px-3 py-2 text-xs font-semibold uppercase tracking-wide text-ink-500 hover:bg-ink-50"
        >
          <span>{title}</span>
          <span className="text-ink-400">{open ? "−" : "+"}</span>
        </button>
      ) : (
        <div className="flex items-center justify-between border-b border-ink-100 px-3 py-2">
          <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
            {title}
          </span>
          {right}
        </div>
      )}
      {body}
    </div>
  );
}
