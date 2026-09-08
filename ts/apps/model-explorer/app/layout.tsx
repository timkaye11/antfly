import type { Metadata } from "next";
import { SiteNav } from "@/components/site-nav";
import { Providers } from "./providers";
import "./globals.css";

export const metadata: Metadata = {
  title: "Antfly Model Explorer",
  description:
    "How data flows through Antfly's Zig inference runtime — GLiNER2, Gemma4 E2B/E4B, Qwen3 Embedding, and Qwen3-VL, from tokenization to Metal kernels.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body style={{ fontFamily: "var(--font-sans)" }}>
        <Providers>
          <SiteNav />
          <main className="min-h-screen">{children}</main>
        </Providers>
      </body>
    </html>
  );
}
