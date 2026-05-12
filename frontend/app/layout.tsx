import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ShopNova Support — Orion AI Agent",
  description:
    "Customer support agent with text and voice modes, hybrid RAG, Text2SQL, and live tool tracing.",
};

const themeInit = `try{var t=localStorage.getItem('theme');if(t==='dark'||(!t&&window.matchMedia('(prefers-color-scheme:dark)').matches))document.documentElement.classList.add('dark')}catch(e){}`;

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInit }} />
      </head>
      <body>{children}</body>
    </html>
  );
}
