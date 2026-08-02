"use client";

import {
  ChevronDownIcon,
  CodeXmlIcon,
  DatabaseIcon,
  FileSearchIcon,
  WrenchIcon,
} from "lucide-react";

import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Separator } from "@/components/ui/separator";
import type { TicketTechnicalDetails } from "@/lib/support-data";
import type { Trace } from "@/lib/types";

type Props = {
  details: TicketTechnicalDetails;
  trace?: Trace | null;
};

const TOOL_LABELS: Record<string, string> = {
  query_database: "Checked order data",
  search_policies: "Reviewed policy",
  escalate: "Escalated to team",
};

export default function TechnicalDetails({
  details,
  trace,
}: Props) {
  return (
    <Collapsible className="rounded-xl border bg-card">
      <CollapsibleTrigger className="group flex w-full items-center justify-between gap-3 rounded-xl px-3 py-3 text-left outline-none transition-colors hover:bg-muted focus-visible:ring-2 focus-visible:ring-ring">
        <span className="flex min-w-0 items-center gap-2">
          <WrenchIcon className="size-4 shrink-0 text-muted-foreground" />
          <span>
            <span className="block text-sm font-medium">Technical details</span>
            <span className="block text-xs text-muted-foreground">
              {trace
                ? `${trace.latency.toFixed(2)}s · latest Orion run`
                : details.documents.length > 0
                  ? `${details.tools.length} tool · vector sources`
                  : `${details.tools.length} tools · SQL records`}
            </span>
          </span>
        </span>
        <ChevronDownIcon className="size-4 shrink-0 text-muted-foreground transition-transform group-data-panel-open:rotate-180" />
      </CollapsibleTrigger>

      <CollapsibleContent>
        <Separator />
        <div className="flex flex-col gap-4 p-3">
          {trace ? (
            <>
              <section className="flex flex-col gap-2">
                <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <DatabaseIcon className="size-3.5" />
                  Actions taken
                </h4>
                <div className="flex flex-wrap gap-1.5">
                  {trace.tools.length > 0 ? (
                    trace.tools.map((tool) => (
                      <Badge key={tool} variant="secondary">
                        {TOOL_LABELS[tool] ?? tool}
                      </Badge>
                    ))
                  ) : (
                    <span className="text-xs text-muted-foreground">
                      Answered from conversation context.
                    </span>
                  )}
                </div>
              </section>

              {trace.sql && (
                <section className="flex flex-col gap-2">
                  <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <CodeXmlIcon className="size-3.5" />
                    Generated SQL
                  </h4>
                  <pre className="scrollbar-warm overflow-x-auto rounded-lg bg-foreground p-3 font-mono text-[10px] leading-relaxed text-background">
                    {trace.sql}
                  </pre>
                </section>
              )}

              {trace.chunks && trace.chunks.length > 0 && (
                <section className="flex flex-col gap-2">
                  <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <FileSearchIcon className="size-3.5" />
                    Sources ({trace.chunks.length})
                  </h4>
                  <div className="flex flex-col gap-2">
                    {trace.chunks.map((chunk, index) => (
                      <div key={`${chunk.source}-${index}`} className="rounded-lg bg-muted p-2.5">
                        <p className="text-xs font-medium">{chunk.heading}</p>
                        <p className="mt-0.5 text-[11px] text-muted-foreground">
                          {chunk.source}
                        </p>
                        <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                          {chunk.content}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <p className="text-[11px] text-muted-foreground">
                Safety check: {trace.guard_fired ? "response corrected" : "clear"}
              </p>
            </>
          ) : (
            <>
              <section className="flex flex-col gap-2">
                <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                  <WrenchIcon className="size-3.5" />
                  Tools called
                </h4>
                <div className="flex flex-col gap-2">
                  {details.tools.map((tool) => (
                    <div key={tool.name} className="rounded-lg bg-muted px-2.5 py-2">
                      <div className="flex items-center justify-between gap-2">
                        <Badge variant="secondary" className="font-mono text-[10px]">
                          {tool.name}
                        </Badge>
                        <span className="truncate font-mono text-[10px] text-muted-foreground">
                          {tool.result}
                        </span>
                      </div>
                      <p className="mt-1 text-[11px] text-muted-foreground">{tool.label}</p>
                    </div>
                  ))}
                </div>
              </section>

              {details.records.length > 0 && (
                <section className="flex flex-col gap-2">
                  <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <DatabaseIcon className="size-3.5" />
                    Retrieved from SQL
                  </h4>
                  {details.records.map((record) => (
                    <p key={`${record.source}-${record.record}`} className="font-mono text-[10px] text-muted-foreground">
                      {record.source} · {record.record}
                    </p>
                  ))}
                </section>
              )}

              {details.documents.length > 0 && (
                <section className="flex flex-col gap-2">
                  <h4 className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                    <FileSearchIcon className="size-3.5" />
                    Retrieved from vector store
                  </h4>
                  {details.documents.map((document) => (
                    <div key={`${document.source}-${document.heading}`} className="rounded-lg bg-muted p-2.5">
                      <p className="text-xs font-medium">{document.heading}</p>
                      <p className="mt-0.5 font-mono text-[10px] text-muted-foreground">
                        {document.source} · score {document.score.toFixed(3)}
                      </p>
                    </div>
                  ))}
                </section>
              )}
            </>
          )}
        </div>
      </CollapsibleContent>
    </Collapsible>
  );
}
