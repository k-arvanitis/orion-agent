// TypeScript counterparts to the Pydantic schemas in api/schemas.py.

export type Chunk = {
  source: string;
  heading: string;
  content: string;
};

export type Trace = {
  tools: string[];
  sql: string | null;
  chunks: Chunk[] | null;
  latency: number;
  guard_fired: boolean;
};

export type Message = {
  id: string;
  role: "user" | "assistant";
  content: string;
  // Set on the assistant message when the corresponding user input was voice.
  audioUrl?: string;
  isVoice?: boolean; // true if this user message came from the mic
};

export type StreamEvent =
  | { type: "token"; content: string }
  | {
      type: "trace";
      tools: string[];
      sql: string | null;
      chunks: Chunk[] | null;
      latency: number;
      guard_fired: boolean;
    }
  | { type: "error"; message: string };
