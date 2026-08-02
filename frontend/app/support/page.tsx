"use client";

import { useEffect, useRef, useState } from "react";
import {
  PanelLeftIcon,
  PanelRightIcon,
} from "lucide-react";

import ConversationPanel from "@/components/ConversationPanel";
import CustomerSidebar from "@/components/CustomerSidebar";
import OrionLogo from "@/components/OrionLogo";
import ThemeToggle from "@/components/ThemeToggle";
import TicketQueue from "@/components/TicketQueue";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Avatar, AvatarBadge, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from "@/components/ui/sheet";
import { Skeleton } from "@/components/ui/skeleton";
import {
  deleteSupportConversation,
  fetchSupportTickets,
  finishSupportConversation,
  markSupportConversationRead,
} from "@/lib/support-api";
import type { Ticket } from "@/lib/support-data";

export default function Home() {
  const [tickets, setTickets] = useState<Ticket[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [loading, setLoading] = useState(true);
  const [dataError, setDataError] = useState<string | null>(null);
  const latestDemoTurn = useRef<string | null>(null);

  const selectedTicket =
    tickets.find((ticket) => ticket.id === selectedId) ?? tickets[0];
  const waitingCount = tickets.filter(
    (ticket) => ticket.status === "Waiting for support",
  ).length;

  useEffect(() => {
    if (!selectedTicket?.unread) return;

    let active = true;
    void markSupportConversationRead(selectedTicket.id)
      .then((updated) => {
        if (!active) return;
        setTickets((current) =>
          current.map((ticket) => (ticket.id === updated.id ? updated : ticket)),
        );
      })
      .catch((caught) => {
        if (!active) return;
        setDataError(
          caught instanceof Error ? caught.message : "Conversation could not be marked read.",
        );
      });

    return () => {
      active = false;
    };
  }, [selectedTicket?.id, selectedTicket?.unread]);

  useEffect(() => {
    let active = true;

    async function syncTickets() {
      try {
        const nextTickets = await fetchSupportTickets();
        if (!active) return;
        const newestDemo = nextTickets.find((ticket) => ticket.source === "demo");
        const newestDemoTurn = newestDemo
          ? `${newestDemo.id}:${newestDemo.messages.length}`
          : null;
        setTickets(nextTickets);
        setDataError(null);
        if (newestDemo && newestDemoTurn !== latestDemoTurn.current) {
          latestDemoTurn.current = newestDemoTurn;
          setSelectedId(newestDemo.id);
        } else {
          if (!newestDemo) latestDemoTurn.current = null;
          setSelectedId((current) =>
            nextTickets.some((ticket) => ticket.id === current)
              ? current
              : (nextTickets[0]?.id ?? ""),
          );
        }
      } catch (caught) {
        if (!active) return;
        setDataError(
          caught instanceof Error ? caught.message : "Support data could not be loaded.",
        );
      } finally {
        if (active) setLoading(false);
      }
    }

    void syncTickets();
    const interval = window.setInterval(syncTickets, 1500);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  function selectTicket(ticket: Ticket) {
    setSelectedId(ticket.id);
  }

  async function deleteTicket(ticketId: string) {
    await deleteSupportConversation(ticketId);
    setTickets((current) => current.filter((ticket) => ticket.id !== ticketId));
    setSelectedId((current) => (current === ticketId ? "" : current));
  }

  async function finishTicket(ticketId: string) {
    const updated = await finishSupportConversation(ticketId);
    setTickets((current) =>
      current.map((ticket) => (ticket.id === updated.id ? updated : ticket)),
    );
  }

  if (loading) {
    return (
      <div className="grid h-screen grid-cols-[300px_minmax(0,1fr)_320px] bg-background">
        <Skeleton className="h-full rounded-none" />
        <div className="flex flex-col gap-4 p-8">
          <Skeleton className="h-12 w-2/3" />
          <Skeleton className="h-28 w-3/4" />
          <Skeleton className="ml-auto h-24 w-2/3" />
        </div>
        <Skeleton className="h-full rounded-none" />
      </div>
    );
  }

  if (!selectedTicket) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-background p-6">
        <Alert className="max-w-lg">
          <AlertTitle>Support database unavailable</AlertTitle>
          <AlertDescription>
            {dataError ?? "No conversations were found. Start the API with make api."}
          </AlertDescription>
        </Alert>
      </div>
    );
  }

  const mobileQueue = (
    <Sheet>
      <SheetTrigger
        render={
          <Button
            size="icon-sm"
            variant="ghost"
            className="md:hidden"
            aria-label="Open ticket queue"
          />
        }
      >
        <PanelLeftIcon />
      </SheetTrigger>
      <SheetContent side="left" className="w-[min(360px,92vw)] gap-0 p-0 sm:max-w-[360px]">
        <SheetHeader className="sr-only">
          <SheetTitle>Ticket queue</SheetTitle>
          <SheetDescription>Browse and select support conversations.</SheetDescription>
        </SheetHeader>
        <TicketQueue
          tickets={tickets}
          selectedId={selectedTicket.id}
          onSelect={selectTicket}
          compact
        />
      </SheetContent>
    </Sheet>
  );

  const mobileDetails = (
    <Sheet>
      <SheetTrigger
        render={
          <Button
            size="icon-sm"
            variant="ghost"
            className="xl:hidden"
            aria-label="Open customer details"
          />
        }
      >
        <PanelRightIcon />
      </SheetTrigger>
      <SheetContent side="right" className="w-[min(380px,94vw)] gap-0 p-0 sm:max-w-[380px]">
        <SheetHeader className="sr-only">
          <SheetTitle>Customer details</SheetTitle>
          <SheetDescription>Customer profile, orders, and technical details.</SheetDescription>
        </SheetHeader>
        <CustomerSidebar ticket={selectedTicket} compact />
      </SheetContent>
    </Sheet>
  );

  return (
    <div className="flex h-screen min-h-0 flex-col overflow-hidden bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b bg-card px-3 sm:px-4">
        <div className="flex min-w-0 items-center gap-6">
          <div className="flex items-center gap-2.5" aria-label="Orion support workspace">
            <OrionLogo />
            <div className="min-w-0">
              <p className="font-heading text-sm font-bold leading-none">Orion</p>
              <p className="mt-1 text-[10px] leading-none text-muted-foreground">Operator workspace</p>
            </div>
          </div>

          <nav className="hidden items-center gap-1 md:flex" aria-label="Workspace navigation">
            <Badge variant="warning">
              Needs support
              <span>{waitingCount}</span>
            </Badge>
          </nav>
        </div>

        <div className="flex shrink-0 items-center gap-1">
          <ThemeToggle />
          <Avatar className="ml-1">
            <AvatarFallback>AK</AvatarFallback>
            <AvatarBadge />
          </Avatar>
          <span className="hidden text-sm font-medium sm:inline">Alex Kim</span>
        </div>
      </header>

      <div className="grid min-h-0 flex-1 grid-cols-1 md:grid-cols-[300px_minmax(0,1fr)] xl:grid-cols-[300px_minmax(0,1fr)_320px]">
        <div className="hidden min-h-0 md:block">
          <TicketQueue
            tickets={tickets}
            selectedId={selectedTicket.id}
            onSelect={selectTicket}
          />
        </div>

        <ConversationPanel
          key={selectedTicket.id}
          ticket={selectedTicket}
          queueTrigger={mobileQueue}
          detailsTrigger={mobileDetails}
          onFinish={() => finishTicket(selectedTicket.id)}
          onDelete={() => deleteTicket(selectedTicket.id)}
        />

        <div className="hidden min-h-0 xl:block">
          <CustomerSidebar ticket={selectedTicket} />
        </div>
      </div>
    </div>
  );
}
