export type TicketStatus =
  | "Waiting for support"
  | "Resolved by Orion"
  | "Resolved by support";
export type MessageRole = "customer" | "orion" | "agent" | "handoff";

export type SupportMessage = {
  id: string;
  role: MessageRole;
  sender: string;
  time: string;
  content: string;
};

export type CustomerOrder = {
  id: string;
  parcelId: string;
  date: string;
  total: string;
  status: string;
  item: string;
  detail: string;
};

export type Customer = {
  id: string;
  name: string;
  initials: string;
  email: string;
  phone: string;
  location: string;
  since: string;
  orderCount: number;
  totalSpent: string;
  tags: string[];
  matchedBy: string;
  orders: CustomerOrder[];
};

export type TicketTechnicalDetails = {
  tools: Array<{
    name: string;
    label: string;
    result: string;
  }>;
  records: Array<{
    source: string;
    record: string;
  }>;
  documents: Array<{
    source: string;
    heading: string;
    score: number;
  }>;
};

export type Ticket = {
  id: string;
  subject: string;
  preview: string;
  time: string;
  unread: number;
  status: TicketStatus;
  channel: "Email" | "Chat" | "Voice";
  actionNeeded?: string;
  source?: "seed" | "demo";
  customer: Customer;
  technicalDetails: TicketTechnicalDetails;
  messages: SupportMessage[];
};
