import { BotIcon } from "lucide-react";

export default function OrionLogo() {
  return (
    <span
      className="flex size-8 shrink-0 items-center justify-center rounded-xl bg-primary text-primary-foreground shadow-sm"
      role="img"
      aria-label="Orion"
    >
      <BotIcon className="size-[18px]" />
    </span>
  );
}
