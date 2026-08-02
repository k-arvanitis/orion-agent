"use client";

import {
  BadgeCheckIcon,
  FingerprintIcon,
  MailIcon,
  MapPinIcon,
  PackageIcon,
  PhoneIcon,
  ShoppingBagIcon,
} from "lucide-react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import type { Ticket } from "@/lib/support-data";
import type { Trace } from "@/lib/types";

import TechnicalDetails from "./TechnicalDetails";

type Props = {
  ticket: Ticket;
  trace?: Trace | null;
  compact?: boolean;
};

function orderVariant(status: string) {
  if (status === "Delivered") return "success" as const;
  if (status === "Delayed") return "coral" as const;
  return "warning" as const;
}

export default function CustomerSidebar({
  ticket,
  trace,
  compact,
}: Props) {
  const customer = ticket.customer;
  const isVisitor = customer.id === "CUS-VISITOR";

  return (
    <aside
      aria-label="Customer details"
      className={compact ? "h-full bg-background" : "h-full border-l bg-background"}
    >
      <ScrollArea className="h-full">
        <div className="flex flex-col gap-4 p-4">
          <section className="flex flex-col items-center gap-3 py-2 text-center">
            <Avatar size="lg">
              <AvatarFallback>{customer.initials}</AvatarFallback>
            </Avatar>
            <div>
              <h2 className="font-heading text-base font-semibold">
                {isVisitor ? "Policy visitor" : customer.name}
              </h2>
              <p className="mt-0.5 text-xs text-muted-foreground">
                {isVisitor ? "No account details required" : `Customer since ${customer.since}`}
              </p>
            </div>
            <div className="flex flex-col items-center gap-1">
              <Badge variant="success">
                <BadgeCheckIcon data-icon="inline-start" />
                {isVisitor ? "Policy question" : "Verified customer"}
              </Badge>
              <p className="text-[11px] text-muted-foreground">
                {isVisitor
                  ? "Orion can answer general policies without personal data"
                  : `Found using ${customer.matchedBy.toLowerCase()}`}
              </p>
              {!isVisitor && (
                <p className="font-mono text-[10px] text-muted-foreground">
                  {customer.id}
                </p>
              )}
            </div>
            <div className="flex flex-wrap justify-center gap-1.5">
              {customer.tags.map((tag) => (
                <Badge key={tag} variant="secondary">
                  {tag}
                </Badge>
              ))}
            </div>
          </section>

          {!isVisitor && <div className="grid grid-cols-2 gap-2">
            <div className="rounded-xl border bg-card p-3 text-center">
              <p className="font-heading text-base font-semibold">{customer.orderCount}</p>
              <p className="text-[11px] text-muted-foreground">Total orders</p>
            </div>
            <div className="rounded-xl border bg-card p-3 text-center">
              <p className="font-heading text-base font-semibold">{customer.totalSpent}</p>
              <p className="text-[11px] text-muted-foreground">Lifetime spend</p>
            </div>
          </div>}

          {!isVisitor && <Card size="sm">
            <CardHeader>
              <CardTitle>Contact</CardTitle>
              <CardDescription>Matched customer record</CardDescription>
            </CardHeader>
            <CardContent>
              <dl className="flex flex-col gap-3 text-xs">
                <div className="flex items-start gap-2.5">
                  <FingerprintIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0">
                    <dt className="text-muted-foreground">Customer ID</dt>
                    <dd className="truncate font-mono text-[11px] font-medium">
                      {customer.id}
                    </dd>
                  </div>
                </div>
                <div className="flex items-start gap-2.5">
                  <MailIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                  <div className="min-w-0">
                    <dt className="text-muted-foreground">Email</dt>
                    <dd className="truncate font-medium">{customer.email}</dd>
                  </div>
                </div>
                <div className="flex items-start gap-2.5">
                  <PhoneIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                  <div>
                    <dt className="text-muted-foreground">Phone</dt>
                    <dd className="font-medium">{customer.phone}</dd>
                  </div>
                </div>
                <div className="flex items-start gap-2.5">
                  <MapPinIcon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" />
                  <div>
                    <dt className="text-muted-foreground">Location</dt>
                    <dd className="font-medium">{customer.location}</dd>
                  </div>
                </div>
              </dl>
            </CardContent>
          </Card>}

          <TechnicalDetails
            details={ticket.technicalDetails}
            trace={trace}
          />

          {customer.orders.length > 0 && <Separator />}

          {customer.orders.length > 0 && <section className="flex flex-col gap-2.5">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="font-heading text-sm font-semibold">Recent orders</h3>
                <p className="text-xs text-muted-foreground">Purchase and delivery context</p>
              </div>
              <ShoppingBagIcon className="size-4 text-muted-foreground" />
            </div>

            {customer.orders.map((order) => (
              <Card key={order.id} size="sm">
                <CardHeader>
                  <CardTitle>{order.item}</CardTitle>
                  <CardDescription>{order.date} · {order.total}</CardDescription>
                  <CardAction>
                    <Badge variant={orderVariant(order.status)}>{order.status}</Badge>
                  </CardAction>
                </CardHeader>
                <CardContent>
                  <div className="flex items-start gap-2 text-xs text-muted-foreground">
                    <PackageIcon className="mt-0.5 size-3.5 shrink-0" />
                    <div className="min-w-0">
                      <p>{order.detail}</p>
                      <p className="mt-1 truncate font-mono text-[10px]">
                        Parcel {order.parcelId}
                      </p>
                      <p className="truncate font-mono text-[10px]">
                        Order {order.id}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </section>}

        </div>
      </ScrollArea>
    </aside>
  );
}
