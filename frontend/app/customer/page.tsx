"use client";

import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import {
  ArrowUpRightIcon,
  CheckCircle2Icon,
  DatabaseIcon,
  HeadphonesIcon,
  MailIcon,
  MapPinIcon,
  MessageSquarePlusIcon,
  PackageIcon,
  SendIcon,
  ShoppingBagIcon,
  SparklesIcon,
  WrenchIcon,
} from "lucide-react";

import TechnicalDetails from "@/components/TechnicalDetails";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Field, FieldGroup, FieldLabel } from "@/components/ui/field";
import {
  InputGroup,
  InputGroupAddon,
  InputGroupTextarea,
} from "@/components/ui/input-group";
import { Marker, MarkerContent, MarkerIcon } from "@/components/ui/marker";
import {
  Message,
  MessageAvatar,
  MessageContent,
  MessageFooter,
  MessageHeader,
} from "@/components/ui/message";
import { Spinner } from "@/components/ui/spinner";
import {
  fetchSupportTickets,
  sendCustomerMessage,
  SupportApiError,
} from "@/lib/support-api";
import type { Ticket } from "@/lib/support-data";

const CURRENT_TICKET_KEY = "orion-customer-database-ticket-v1";

export default function CustomerPage() {
  const [ticket, setTicket] = useState<Ticket | null>(null);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [clarification, setClarification] = useState<string | null>(null);
  const threadEndRef = useRef<HTMLDivElement | null>(null);

  const customer = ticket?.customer;
  const order = customer?.orders[0];
  const isVisitor = customer?.id === "CUS-VISITOR";

  useEffect(() => {
    let active = true;

    async function load() {
      try {
        const conversations = await fetchSupportTickets();
        if (!active) return;
        const currentId = window.localStorage.getItem(CURRENT_TICKET_KEY);
        setTicket(
          conversations.find(
            (conversation) =>
              conversation.id === currentId && conversation.source === "demo",
          ) ?? null,
        );
        setError(null);
      } catch (caught) {
        if (active) {
          setError(
            caught instanceof Error
              ? caught.message
              : "The customer database could not be loaded.",
          );
        }
      } finally {
        if (active) setReady(true);
      }
    }

    async function syncConversation() {
      const currentId = window.localStorage.getItem(CURRENT_TICKET_KEY);
      if (!currentId) return;
      try {
        const conversations = await fetchSupportTickets();
        if (!active) return;
        setTicket(
          conversations.find((conversation) => conversation.id === currentId) ?? null,
        );
      } catch {
        // Keep the last successful conversation visible during a transient poll failure.
      }
    }

    void load();
    const interval = window.setInterval(syncConversation, 1500);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      threadEndRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [clarification, error, ticket?.messages.length]);

  async function sendMessage(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;

    setBusy(true);
    setError(null);
    setClarification(null);
    try {
      const nextTicket = await sendCustomerMessage(content, ticket?.id);
      window.localStorage.setItem(CURRENT_TICKET_KEY, nextTicket.id);
      setTicket(nextTicket);
      setDraft("");
    } catch (caught) {
      if (caught instanceof SupportApiError && caught.status === 422) {
        setClarification(caught.message);
      } else {
        setError(
          caught instanceof Error ? caught.message : "Orion could not send the message.",
        );
      }
    } finally {
      setBusy(false);
    }
  }

  function newConversation() {
    if (busy) return;
    setError(null);
    setClarification(null);
    window.localStorage.removeItem(CURRENT_TICKET_KEY);
    setTicket(null);
    setDraft("");
  }

  const visibleMessages =
    ticket?.messages.filter((message) => message.role !== "handoff") ?? [];

  return (
    <div className="flex h-screen min-h-0 flex-col overflow-hidden bg-background">
      <header className="flex h-14 shrink-0 items-center justify-between gap-3 border-b bg-card px-3 sm:px-5">
        <div className="flex items-center gap-2.5">
          <div className="flex size-9 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm">
            <ShoppingBagIcon className="size-4" />
          </div>
          <div>
            <p className="font-heading text-sm font-bold leading-none">ShopNova</p>
            <p className="mt-1 text-[11px] leading-none text-muted-foreground">Customer help</p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={newConversation} disabled={!ready || busy}>
            <MessageSquarePlusIcon data-icon="inline-start" />
            <span className="hidden sm:inline">New conversation</span>
          </Button>
          <Button
            variant="outline"
            size="sm"
            render={<Link href="/support" target="_blank" />}
            nativeButton={false}
          >
            <span className="hidden sm:inline">Open support view</span>
            <span className="sm:hidden">Support</span>
            <ArrowUpRightIcon data-icon="inline-end" />
          </Button>
        </div>
      </header>

      <main className="mx-auto grid min-h-0 w-full max-w-6xl flex-1 gap-4 p-3 lg:grid-cols-[280px_minmax(0,1fr)] lg:p-4">
        <aside className="scrollbar-warm hidden min-h-0 flex-col gap-3 overflow-y-auto lg:flex">
          {customer ? (
            <>
              <Card size="sm">
                <CardHeader>
                  <div className="flex items-center gap-3">
                    <Avatar>
                      <AvatarFallback>{customer.initials}</AvatarFallback>
                    </Avatar>
                    <div className="min-w-0">
                      <CardTitle>{isVisitor ? "Policy visitor" : customer.name}</CardTitle>
                      <CardDescription>
                        {isVisitor ? "No account details required" : `Customer since ${customer.since}`}
                      </CardDescription>
                    </div>
                  </div>
                </CardHeader>
                <CardContent className="flex flex-col gap-2.5">
                  {isVisitor && (
                    <Badge variant="success" className="w-fit">
                      Policy question
                    </Badge>
                  )}
                  {!isVisitor && (
                    <>
                      <div className="flex items-start gap-2 text-xs">
                        <MailIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                        <span className="truncate">{customer.email}</span>
                      </div>
                      <div className="flex items-start gap-2 text-xs">
                        <MapPinIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                        <span>{customer.location}</span>
                      </div>
                    </>
                  )}
                </CardContent>
              </Card>

              {order && (
                <Card size="sm">
                  <CardHeader className="min-w-0 gap-y-0.5">
                    <CardTitle>Matched order</CardTitle>
                    <CardDescription className="truncate font-mono text-[10px]">
                      {order.id}
                    </CardDescription>
                    <CardAction>
                      <Badge variant={order.status === "Delivered" ? "success" : "warning"}>
                        {order.status}
                      </Badge>
                    </CardAction>
                  </CardHeader>
                  <CardContent className="flex min-w-0 flex-col gap-1.5">
                    <p className="truncate text-xs font-medium">{order.item}</p>
                    <p className="truncate text-[11px] text-muted-foreground">{order.detail}</p>
                    <p className="truncate font-mono text-[10px] text-muted-foreground">
                      Parcel {order.parcelId}
                    </p>
                  </CardContent>
                </Card>
              )}

              <TechnicalDetails details={ticket.technicalDetails} />
            </>
          ) : (
            <Card size="sm">
              <CardHeader>
                <div className="flex items-center gap-2">
                  <div className="flex size-8 items-center justify-center rounded-lg bg-accent text-accent-foreground">
                    <WrenchIcon className="size-3.5" />
                  </div>
                  <div>
                    <CardTitle>Technical details</CardTitle>
                    <CardDescription>No tools called yet</CardDescription>
                  </div>
                </div>
              </CardHeader>
              <CardContent className="flex items-center gap-2 text-xs text-muted-foreground">
                <DatabaseIcon className="size-3.5" />
                SQL or vector activity appears after the first message.
              </CardContent>
            </Card>
          )}
        </aside>

        <Card className="h-full min-h-0 overflow-hidden">
          <CardHeader className="shrink-0 py-3.5">
            <div className="flex items-center gap-3">
              <div className="flex size-9 items-center justify-center rounded-xl bg-accent text-accent-foreground">
                <SparklesIcon className="size-4" />
              </div>
              <div>
                <CardTitle>Chat with ShopNova</CardTitle>
                <CardDescription>Orion is here to help with your order.</CardDescription>
              </div>
            </div>
          </CardHeader>

          <CardContent className="scrollbar-warm min-h-0 flex-1 overflow-y-auto px-4">
            <div className="mx-auto flex max-w-2xl flex-col gap-3 py-2">
              <Marker variant="separator">
                <MarkerContent>Customer conversation</MarkerContent>
              </Marker>

              {customer && !isVisitor && (
                <Marker>
                  <MarkerIcon>
                    <CheckCircle2Icon />
                  </MarkerIcon>
                  <MarkerContent>Account and order matched securely</MarkerContent>
                </Marker>
              )}

              {visibleMessages.map((message) => {
                const isCustomer = message.role === "customer";
                const isAgent = message.role === "agent";
                return (
                  <Message key={message.id} align={isCustomer ? "end" : "start"}>
                    <MessageAvatar>
                      <Avatar size="sm">
                        <AvatarFallback>
                          {isCustomer
                            ? (customer?.initials ?? "You")
                            : isAgent
                              ? <HeadphonesIcon className="size-3.5" />
                              : <SparklesIcon className="size-3.5" />}
                        </AvatarFallback>
                      </Avatar>
                    </MessageAvatar>
                    <MessageContent>
                      <MessageHeader>
                        {isCustomer ? "You" : isAgent ? "Alex Kim" : "Orion"}
                      </MessageHeader>
                      <Bubble
                        align={isCustomer ? "end" : "start"}
                        variant={
                          isCustomer
                            ? "outline"
                            : isAgent
                              ? "secondary"
                              : "default"
                        }
                      >
                        <BubbleContent>
                          {message.content}
                          <MessageFooter className="justify-end px-0 text-[10px] text-current opacity-60">
                            {message.time}
                          </MessageFooter>
                        </BubbleContent>
                      </Bubble>
                    </MessageContent>
                  </Message>
                );
              })}

              {error && (
                <Alert variant="destructive">
                  <AlertTitle>I couldn’t complete that request</AlertTitle>
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {clarification && (
                <Message align="start">
                  <MessageAvatar>
                      <Avatar size="sm">
                      <AvatarFallback>
                        <SparklesIcon className="size-3.5" />
                      </AvatarFallback>
                    </Avatar>
                  </MessageAvatar>
                  <MessageContent>
                    <MessageHeader>Orion</MessageHeader>
                    <Bubble align="start" variant="default">
                      <BubbleContent>{clarification}</BubbleContent>
                    </Bubble>
                  </MessageContent>
                </Message>
              )}

              {!error && !clarification && ticket?.status === "Waiting for support" && (
                <Alert>
                  <AlertTitle>Your message was sent to a support teammate</AlertTitle>
                  <AlertDescription>
                    Please wait here for Alex Kim’s reply. Your conversation and order
                    details have already been shared with them.
                  </AlertDescription>
                </Alert>
              )}

              {!error &&
                !clarification &&
                ticket?.status === "Resolved by support" && (
                  <Alert className="flex flex-col items-center gap-1.5 px-4 py-4 text-center">
                    <CheckCircle2Icon />
                    <AlertTitle>Conversation closed</AlertTitle>
                    <AlertDescription>
                      Our support team has closed this conversation. Start a new
                      conversation if you need more help.
                    </AlertDescription>
                  </Alert>
                )}

              {ticket && (
                <div className="lg:hidden">
                  <TechnicalDetails details={ticket.technicalDetails} />
                </div>
              )}

              <div ref={threadEndRef} aria-hidden="true" />
            </div>
          </CardContent>

          <CardFooter className="shrink-0 bg-card px-4 py-3">
            <form onSubmit={sendMessage} className="w-full">
              <FieldGroup>
                <Field>
                  <FieldLabel htmlFor="customer-message" className="sr-only">
                    Message ShopNova
                  </FieldLabel>
                  <InputGroup>
                    <InputGroupTextarea
                      id="customer-message"
                      rows={2}
                      value={draft}
                      onChange={(event) => setDraft(event.target.value)}
                      onKeyDown={(event) => {
                        if (
                          event.key === "Enter" &&
                          !event.shiftKey &&
                          !event.nativeEvent.isComposing
                        ) {
                          event.preventDefault();
                          event.currentTarget.form?.requestSubmit();
                        }
                      }}
                      enterKeyHint="send"
                      placeholder="Message ShopNova…"
                      disabled={busy}
                    />
                    <InputGroupAddon align="block-end" className="justify-between">
                      <span className="flex items-center gap-1 text-xs text-muted-foreground">
                        Enter to send · Shift+Enter for a new line
                      </span>
                      <Button type="submit" size="sm" disabled={busy || !draft.trim()}>
                        {busy ? (
                          <Spinner data-icon="inline-start" />
                        ) : (
                          <SendIcon data-icon="inline-start" />
                        )}
                        {busy ? "Sending…" : "Send"}
                      </Button>
                    </InputGroupAddon>
                  </InputGroup>
                </Field>
              </FieldGroup>
            </form>
          </CardFooter>
        </Card>
      </main>
    </div>
  );
}
