"use client";

const SAMPLES: { label: string; question: string }[] = [
  {
    label: "🗄️ Order lookup (SQL)",
    question:
      "What is the status of order 416e49799e9260d93c8f636ce6661a55?",
  },
  {
    label: "📄 Policy question (RAG)",
    question: "How long do I have to return a product?",
  },
  {
    label: "🔀 Mixed (SQL + RAG)",
    question: "My order arrived late — am I eligible for a refund?",
  },
  {
    label: "🚨 Escalation",
    question:
      "This is the third time my order is wrong. I want to speak to a human. My email is customer@example.com.",
  },
];

type Props = {
  onPick: (q: string) => void;
  disabled?: boolean;
};

export default function SampleQuestions({ onPick, disabled }: Props) {
  return (
    <div className="rounded-lg border border-ink-200 bg-surface">
      <div className="flex items-center justify-between border-b border-ink-100 px-3 py-2">
        <span className="text-xs font-semibold uppercase tracking-wide text-ink-500">
          Try a sample question
        </span>
      </div>
      <div className="space-y-3 p-3">
        <p className="text-xs text-ink-500">
          Don&apos;t know the dataset? Click any of these to see the agent route
          to the right tool.
        </p>
        <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-4">
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              onClick={() => onPick(s.question)}
              disabled={disabled}
              title={s.question}
              className="rounded-md border border-ink-200 bg-ink-50 px-3 py-2 text-left text-xs text-ink-700 hover:bg-ink-100 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {s.label}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
