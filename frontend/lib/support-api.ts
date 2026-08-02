import type { Customer, Ticket } from "@/lib/support-data";

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8088";

export class SupportApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "content-type": "application/json",
      ...init?.headers,
    },
    cache: "no-store",
  });

  if (!response.ok) {
    let message = `Support request failed: HTTP ${response.status}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) message = body.detail;
    } catch {
      // Keep the status-based fallback when the response is not JSON.
    }
    throw new SupportApiError(message, response.status);
  }

  return (await response.json()) as T;
}

export function fetchSupportCustomers(signal?: AbortSignal) {
  return request<Customer[]>("/api/support/customers", { signal });
}

export function fetchSupportTickets(signal?: AbortSignal) {
  return request<Ticket[]>("/api/support/conversations", { signal });
}

export function sendCustomerMessage(
  message: string,
  conversationId?: string | null,
) {
  return request<Ticket>("/api/support/conversations/messages", {
    method: "POST",
    body: JSON.stringify({
      message,
      conversation_id: conversationId || null,
    }),
  });
}

export function sendSupportReply(conversationId: string, message: string) {
  return request<Ticket>(
    `/api/support/conversations/${encodeURIComponent(conversationId)}/reply`,
    {
      method: "POST",
      body: JSON.stringify({ message }),
    },
  );
}

export function markSupportConversationRead(conversationId: string) {
  return request<Ticket>(
    `/api/support/conversations/${encodeURIComponent(conversationId)}/read`,
    { method: "POST" },
  );
}

export function finishSupportConversation(conversationId: string) {
  return request<Ticket>(
    `/api/support/conversations/${encodeURIComponent(conversationId)}/finish`,
    { method: "POST" },
  );
}

export function deleteSupportConversation(conversationId: string) {
  return request<{ deleted: string }>(
    `/api/support/conversations/${encodeURIComponent(conversationId)}`,
    { method: "DELETE" },
  );
}

export function resetSupportDemo() {
  return request<Ticket[]>("/api/support/demo/reset", { method: "POST" });
}
