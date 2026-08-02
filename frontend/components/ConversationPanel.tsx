"use client";

import { useEffect, useRef, useState, type ReactNode } from "react";
import {
  BotIcon,
  CheckCircle2Icon,
  SendIcon,
  Trash2Icon,
} from "lucide-react";
import ReactMarkdown from "react-markdown";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Bubble, BubbleContent } from "@/components/ui/bubble";
import { Button } from "@/components/ui/button";
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
import { transcribeAudio } from "@/lib/api";
import { sendSupportReply } from "@/lib/support-api";
import type { SupportMessage, Ticket } from "@/lib/support-data";

import VoiceRecorder from "./VoiceRecorder";

type Props = {
  ticket: Ticket;
  queueTrigger?: ReactNode;
  detailsTrigger?: ReactNode;
  onFinish: () => Promise<void>;
  onDelete: () => Promise<void>;
};

function MessageBody({ children }: { children: string }) {
  return (
    <ReactMarkdown
      components={{
        a: ({ children: linkChildren, ...props }) => (
          <a {...props} target="_blank" rel="noreferrer">
            {linkChildren}
          </a>
        ),
      }}
    >
      {children}
    </ReactMarkdown>
  );
}

export default function ConversationPanel({
  ticket,
  queueTrigger,
  detailsTrigger,
  onFinish,
  onDelete,
}: Props) {
  const [messages, setMessages] = useState<SupportMessage[]>(ticket.messages);
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [finishing, setFinishing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMessages(ticket.messages);
  }, [ticket.messages]);

  useEffect(() => {
    setDraft("");
    setError(null);
  }, [ticket.id]);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function addMessage(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const content = draft.trim();
    if (!content || busy) return;

    setBusy(true);
    setError(null);
    try {
      const updated = await sendSupportReply(ticket.id, content);
      setMessages(updated.messages);
      setDraft("");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The reply could not be sent.");
    } finally {
      setBusy(false);
    }
  }

  async function handleRecorded(blob: Blob) {
    setBusy(true);
    setError(null);
    try {
      const transcript = await transcribeAudio(blob);
      setDraft((current) => (current ? `${current} ${transcript}` : transcript));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "The recording could not be transcribed.");
    } finally {
      setBusy(false);
    }
  }

  async function deleteConversation() {
    setDeleting(true);
    setError(null);
    try {
      await onDelete();
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "The conversation could not be deleted.",
      );
      setDeleting(false);
    }
  }

  async function finishConversation() {
    setFinishing(true);
    setError(null);
    try {
      await onFinish();
      setFinishing(false);
    } catch (caught) {
      setError(
        caught instanceof Error ? caught.message : "The conversation could not be finished.",
      );
      setFinishing(false);
    }
  }

  return (
    <main className="flex h-full min-h-0 min-w-0 flex-1 flex-col bg-background">
      <header className="flex min-h-16 items-center justify-between gap-3 border-b bg-card px-3 py-2.5 sm:px-4">
        <div className="flex min-w-0 items-center gap-2">
          {queueTrigger}
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h1 className="truncate font-heading text-sm font-semibold sm:text-base">
                {ticket.subject}
              </h1>
            </div>
            <p className="mt-0.5 truncate text-xs text-muted-foreground">
              {ticket.customer.name} · {ticket.id} · {ticket.channel}
            </p>
          </div>
        </div>

        <div className="flex shrink-0 items-center gap-1.5">
          <Badge
            variant={ticket.status === "Waiting for support" ? "warning" : "success"}
            className="hidden sm:inline-flex"
          >
            {ticket.status}
          </Badge>

          {ticket.status !== "Resolved by Orion" ? (
            <div className="flex h-8 items-center gap-2 rounded-lg border bg-card px-2.5 text-xs font-medium">
              <Avatar size="sm">
                <AvatarFallback>AK</AvatarFallback>
              </Avatar>
              <span className="hidden sm:inline">Alex Kim</span>
            </div>
          ) : (
            <Badge variant="secondary">
              <BotIcon data-icon="inline-start" />
              Orion
            </Badge>
          )}

          {ticket.status === "Waiting for support" && (
            <Button
              type="button"
              size="sm"
              onClick={finishConversation}
              disabled={busy || finishing}
            >
              {finishing ? (
                <Spinner data-icon="inline-start" />
              ) : (
                <CheckCircle2Icon data-icon="inline-start" />
              )}
              {finishing ? "Closing…" : "Close"}
            </Button>
          )}

          {detailsTrigger}

          <AlertDialog>
            <AlertDialogTrigger
              render={
                <Button
                  size="icon-sm"
                  variant="ghost"
                  aria-label="Delete conversation"
                />
              }
            >
              <Trash2Icon />
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete this conversation?</AlertDialogTitle>
                <AlertDialogDescription>
                  This removes the conversation and its messages from the support queue.
                  The customer and order records will stay in the database.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel disabled={deleting}>Keep conversation</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  disabled={deleting}
                  onClick={deleteConversation}
                >
                  {deleting && <Spinner data-icon="inline-start" />}
                  {deleting ? "Deleting…" : "Delete conversation"}
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      </header>

      <div ref={scrollRef} className="scrollbar-warm min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex w-full max-w-3xl flex-col gap-4 p-4 sm:p-5">
          <Marker variant="separator">
            <MarkerContent>Today</MarkerContent>
          </Marker>

          <Marker>
            <MarkerIcon>
              <BotIcon />
            </MarkerIcon>
            <MarkerContent>
              {ticket.customer.id === "CUS-VISITOR"
                ? "Orion searched the policy library and resolved this question"
                : ticket.status === "Resolved by support"
                ? `Alex Kim closed the conversation with ${ticket.customer.name}`
                : ticket.status === "Resolved by Orion"
                ? `Orion matched ${ticket.customer.name} and resolved this conversation`
                : `Orion matched ${ticket.customer.name} and sent this case to support`}
            </MarkerContent>
          </Marker>

          {messages.map((message) => {
            if (message.role === "handoff") {
              return (
                <div
                  key={message.id}
                  className="rounded-xl border border-note-border bg-note p-3 text-note-foreground"
                >
                  <Marker variant="border">
                    <MarkerIcon>
                      <BotIcon />
                    </MarkerIcon>
                    <MarkerContent>
                      Orion handoff · {message.time}
                    </MarkerContent>
                  </Marker>
                  {ticket.actionNeeded && (
                    <p className="mt-2 text-sm font-semibold">
                      Action needed: {ticket.actionNeeded}
                    </p>
                  )}
                  <p className="mt-1 text-sm leading-relaxed">{message.content}</p>
                </div>
              );
            }

            const isCustomer = message.role === "customer";
            const isOrion = message.role === "orion";

            return (
              <Message key={message.id} align={isCustomer ? "start" : "end"}>
                <MessageAvatar>
                  <Avatar size="sm">
                    <AvatarFallback>
                      {isCustomer ? ticket.customer.initials : isOrion ? <BotIcon className="size-3.5" /> : "AK"}
                    </AvatarFallback>
                  </Avatar>
                </MessageAvatar>
                <MessageContent>
                  <MessageHeader>
                    {message.sender}
                    {isOrion && <Badge variant="success" className="ml-2">AI teammate</Badge>}
                  </MessageHeader>
                  <Bubble
                    align={isCustomer ? "start" : "end"}
                    variant={isCustomer ? "outline" : isOrion ? "tinted" : "secondary"}
                  >
                    <BubbleContent>
                      <div className="flex flex-col gap-2 leading-relaxed [&_a]:font-medium [&_a]:underline [&_ol]:list-decimal [&_ol]:pl-4 [&_p]:whitespace-pre-wrap [&_strong]:font-semibold [&_ul]:list-disc [&_ul]:pl-4">
                        <MessageBody>{message.content}</MessageBody>
                      </div>
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
              <AlertTitle>Orion needs another try</AlertTitle>
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}
        </div>
      </div>

      <section className="border-t bg-card p-3 sm:p-4">
        {ticket.status !== "Waiting for support" ? (
          <Alert className="mx-auto flex max-w-3xl flex-col items-center gap-1.5 px-4 py-4 text-center">
            <CheckCircle2Icon />
            <AlertTitle>
              {ticket.status === "Resolved by support"
                ? "Conversation closed"
                : "Resolved by Orion"}
            </AlertTitle>
            <AlertDescription>
              {ticket.status === "Resolved by support"
                ? "Closed by Alex Kim. No further support action is required."
                : "Orion answered the customer and no support action is needed."}
            </AlertDescription>
          </Alert>
        ) : (
          <form onSubmit={addMessage} className="mx-auto max-w-3xl">
            <FieldGroup>
              <Field>
                <FieldLabel htmlFor="support-reply" className="sr-only">
                  Reply to customer
                </FieldLabel>
                <InputGroup>
                  <InputGroupTextarea
                    id="support-reply"
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
                    disabled={busy}
                    rows={2}
                    placeholder="Reply to the customer…"
                  />

                  <InputGroupAddon align="block-end" className="justify-between">
                    <VoiceRecorder onRecorded={handleRecorded} disabled={busy} />

                    <Button type="submit" size="sm" disabled={busy || !draft.trim()}>
                      <SendIcon data-icon="inline-start" />
                      Send reply
                    </Button>
                  </InputGroupAddon>
                </InputGroup>
              </Field>
            </FieldGroup>
          </form>
        )}
      </section>
    </main>
  );
}
