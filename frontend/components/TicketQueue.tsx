"use client";

import { useMemo, useState } from "react";
import { InboxIcon, SearchIcon } from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupInput,
} from "@/components/ui/input-group";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { cn } from "@/lib/utils";
import type { Ticket, TicketStatus } from "@/lib/support-data";

type Props = {
  tickets: Ticket[];
  selectedId: string;
  onSelect: (ticket: Ticket) => void;
  compact?: boolean;
};

function statusVariant(status: TicketStatus) {
  return status === "Waiting for support" ? ("warning" as const) : ("success" as const);
}

export default function TicketQueue({
  tickets,
  selectedId,
  onSelect,
  compact,
}: Props) {
  const [query, setQuery] = useState("");
  const [filter, setFilter] = useState("waiting");

  const waitingCount = tickets.filter(
    (ticket) => ticket.status === "Waiting for support",
  ).length;

  const visibleTickets = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    return tickets.filter((ticket) => {
      const matchesFilter =
        filter === "all" ||
        (filter === "waiting" && ticket.status === "Waiting for support") ||
        (filter === "resolved" && ticket.status !== "Waiting for support");
      const matchesQuery =
        !normalizedQuery ||
        ticket.customer.name.toLowerCase().includes(normalizedQuery) ||
        ticket.customer.email.toLowerCase().includes(normalizedQuery) ||
        ticket.subject.toLowerCase().includes(normalizedQuery) ||
        ticket.id.toLowerCase().includes(normalizedQuery) ||
        ticket.customer.orders.some(
          (order) =>
            order.id.toLowerCase().includes(normalizedQuery) ||
            order.parcelId.toLowerCase().includes(normalizedQuery),
        );
      return matchesFilter && matchesQuery;
    });
  }, [filter, query, tickets]);

  return (
    <section
      aria-label="Ticket queue"
      className={cn(
        "flex h-full min-h-0 flex-col bg-sidebar text-sidebar-foreground",
        !compact && "border-r border-sidebar-border",
      )}
    >
      <div className="flex flex-col gap-3 px-4 pb-3 pt-4">
        <div className="flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2.5">
            <div className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground">
              <InboxIcon className="size-4" />
            </div>
            <div className="min-w-0">
              <h2 className="font-heading text-sm font-semibold">Support queue</h2>
              <p className="text-xs text-muted-foreground">
                {waitingCount} need a person
              </p>
            </div>
          </div>
        </div>

        <InputGroup>
          <InputGroupAddon>
            <SearchIcon />
          </InputGroupAddon>
          <InputGroupInput
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Name, email, order or parcel"
            aria-label="Search conversations"
          />
        </InputGroup>

        <div className="flex items-center gap-2">
          <ToggleGroup
            value={[filter]}
            onValueChange={(value) => setFilter(value[0] ?? "waiting")}
            size="sm"
            spacing={1}
            aria-label="Filter tickets"
          >
            <ToggleGroupItem value="waiting">Needs support</ToggleGroupItem>
            <ToggleGroupItem value="resolved">Resolved</ToggleGroupItem>
            <ToggleGroupItem value="all">All</ToggleGroupItem>
          </ToggleGroup>
        </div>
      </div>

      <Separator />

      <ScrollArea className="min-h-0 flex-1">
        <div className="flex flex-col gap-0.5 p-2">
          {visibleTickets.map((ticket) => {
            const isSelected = ticket.id === selectedId;
            return (
              <button
                key={ticket.id}
                type="button"
                data-selected={isSelected}
                onClick={() => onSelect(ticket)}
                className={cn(
                  "group flex w-full gap-3 rounded-xl px-2.5 py-2 text-left transition-colors hover:bg-sidebar-accent focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-sidebar-ring data-[selected=true]:bg-sidebar-accent",
                  isSelected && "shadow-[inset_3px_0_0_var(--primary)]",
                )}
              >
                <div className="mt-0.5 shrink-0">
                  <Avatar>
                    <AvatarFallback>{ticket.customer.initials}</AvatarFallback>
                  </Avatar>
                </div>

                <div className="min-w-0 flex-1">
                  <div className="flex items-center justify-between gap-2">
                    <span className="truncate text-sm font-medium">
                      {ticket.customer.name}
                    </span>
                    <span className="shrink-0 text-[11px] text-muted-foreground">
                      {ticket.time}
                    </span>
                  </div>
                  <p className="mt-0.5 truncate text-xs font-medium">
                    {ticket.subject}
                  </p>
                  <p className="mt-0.5 line-clamp-2 text-xs leading-4 text-muted-foreground">
                    {ticket.preview}
                  </p>
                  <div className="mt-2 flex items-center gap-1.5">
                    <Badge variant={statusVariant(ticket.status)}>
                      {ticket.status}
                    </Badge>
                    {ticket.source === "demo" && (
                      <Badge variant="secondary">Live</Badge>
                    )}
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </ScrollArea>

      <Separator />
      <div className="flex items-center justify-between gap-3 px-4 py-3 text-xs text-muted-foreground">
        <span>{waitingCount} waiting</span>
        <span>{tickets.length - waitingCount} resolved</span>
      </div>
    </section>
  );
}
