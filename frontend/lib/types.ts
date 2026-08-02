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
