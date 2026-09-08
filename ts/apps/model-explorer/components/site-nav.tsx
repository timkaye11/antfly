"use client";

import {
  Button,
  cn,
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@antfly/design-system";
import { ChevronDown, Moon, Sun } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useTheme } from "next-themes";
import { useEffect, useState } from "react";

const MODELS = [
  { slug: "gemma4-e4b", name: "Gemma4 E2B/E4B", hook: "PLE + iSWA + MoE + MTP" },
  { slug: "gliner2", name: "GLiNER2", hook: "disentangled attention + span head" },
  { slug: "qwen3-embedding", name: "Qwen3 Embedding", hook: "last-token pooling at 8k" },
  { slug: "qwen3-vl", name: "Qwen3-VL", hook: "pixels → m-RoPE" },
];

const SYSTEMS = [
  { href: "/systems/perf", name: "Perf & roofline" },
  { href: "/systems/kernels", name: "Kernel routing" },
  { href: "/systems/timeline", name: "Frame timeline" },
  { href: "/systems/kv", name: "KV cache" },
  { href: "/systems/flags", name: "Env flags" },
];

function ThemeToggle() {
  const { resolvedTheme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);
  if (!mounted) return <div className="size-8" />;
  return (
    <Button
      variant="ghost"
      size="icon"
      aria-label="Toggle theme"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
    >
      {resolvedTheme === "dark" ? <Sun className="size-4" /> : <Moon className="size-4" />}
    </Button>
  );
}

export function SiteNav() {
  const pathname = usePathname();
  const active = (prefix: string) => pathname?.startsWith(prefix);

  return (
    <header className="sticky top-0 z-50 border-b bg-background/90 backdrop-blur">
      <div className="mx-auto flex h-14 max-w-7xl items-center gap-1 px-4">
        <Link href="/" className="mr-4 font-semibold tracking-tight">
          Antfly <span className="text-muted-foreground">Model Explorer</span>
        </Link>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant={active("/models") ? "outline" : "ghost"} size="sm">
              Models <ChevronDown className="size-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="w-72">
            {MODELS.map((m) => (
              <DropdownMenuItem key={m.slug} asChild>
                <Link href={`/models/${m.slug}`} className="flex flex-col items-start gap-0.5">
                  <span className="font-medium">{m.name}</span>
                  <span className="text-xs text-muted-foreground">{m.hook}</span>
                </Link>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant={active("/runtime") ? "outline" : "ghost"} size="sm" asChild>
          <Link href="/runtime">Runtime</Link>
        </Button>
        <Button variant={active("/explore") ? "outline" : "ghost"} size="sm" asChild>
          <Link href="/explore/gemma4-e4b">Explorer</Link>
        </Button>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <Button variant={active("/systems") ? "outline" : "ghost"} size="sm">
              Systems <ChevronDown className="size-3" />
            </Button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start">
            {SYSTEMS.map((s) => (
              <DropdownMenuItem key={s.href} asChild>
                <Link href={s.href}>{s.name}</Link>
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
        <Button variant={active("/legend") ? "outline" : "ghost"} size="sm" asChild>
          <Link href="/legend">Legend</Link>
        </Button>
        <div className={cn("ml-auto")}>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}
