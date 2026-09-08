"use client";

import { cn } from "@antfly/design-system";
import Link from "next/link";
import {
  Children,
  createContext,
  isValidElement,
  type ReactElement,
  type ReactNode,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

/* ------------------------------------------------------------------ */
/* Scene                                                               */
/* ------------------------------------------------------------------ */

export interface SceneProps {
  id: string;
  /** The pinned graphic shown while this scene's prose is centered. */
  graphic: ReactNode;
  children: ReactNode;
}

/** Marker component — rendered by ScrollyChapter, never directly. */
export function Scene(_props: SceneProps): null {
  return null;
}

/* ------------------------------------------------------------------ */
/* Chapter                                                             */
/* ------------------------------------------------------------------ */

interface ChapterContextValue {
  activeScene: string | null;
}
const ChapterContext = createContext<ChapterContextValue>({ activeScene: null });
export const useChapter = () => useContext(ChapterContext);

export interface ScrollyChapterProps {
  id: string;
  number?: number;
  title: string;
  /** Optional lead paragraph rendered above the scenes. */
  intro?: ReactNode;
  children: ReactNode;
  className?: string;
}

/**
 * Distill-style scrollytelling: prose scrolls on the left, the active
 * scene's graphic is pinned (CSS sticky) on the right and crossfades as
 * scenes activate. Degrades to a stacked layout on small screens and
 * under prefers-reduced-motion (see .scrolly-graphic in globals.css).
 */
export function ScrollyChapter({ id, number, title, intro, children, className }: ScrollyChapterProps) {
  const scenes = useMemo(
    () =>
      Children.toArray(children).filter(
        (c): c is ReactElement<SceneProps> => isValidElement(c) && c.type === Scene,
      ),
    [children],
  );

  const [activeScene, setActiveScene] = useState<string | null>(scenes[0]?.props.id ?? null);
  const proseRefs = useRef(new Map<string, HTMLElement>());

  const registerProse = useCallback((sceneId: string, el: HTMLElement | null) => {
    if (el) proseRefs.current.set(sceneId, el);
    else proseRefs.current.delete(sceneId);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        // Pick the entry closest to the viewport's upper-middle band.
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length === 0) return;
        let best: IntersectionObserverEntry | null = null;
        for (const entry of visible) {
          if (!best || entry.boundingClientRect.top < best.boundingClientRect.top) best = entry;
        }
        const sceneId = best?.target.getAttribute("data-scene");
        if (sceneId) setActiveScene(sceneId);
      },
      { rootMargin: "-25% 0px -55% 0px" },
    );
    for (const el of proseRefs.current.values()) observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const activeIndex = Math.max(
    0,
    scenes.findIndex((s) => s.props.id === activeScene),
  );

  return (
    <section id={id} className={cn("scroll-mt-20 border-b py-12", className)}>
      <ChapterContext.Provider value={{ activeScene }}>
        <div className="mx-auto max-w-7xl px-4">
          <header className="mb-8 max-w-2xl">
            <h2 className="text-2xl font-bold tracking-tight">
              {number !== undefined && (
                <span className="mr-3 font-mono text-lg text-primary">{String(number).padStart(2, "0")}</span>
              )}
              {title}
            </h2>
            {intro && <div className="mt-3 text-muted-foreground">{intro}</div>}
          </header>

          <div className="grid gap-8 lg:grid-cols-[2fr_3fr]">
            {/* Prose column */}
            <div>
              {scenes.map((scene) => (
                <div
                  key={scene.props.id}
                  id={`${id}-${scene.props.id}`}
                  data-scene={scene.props.id}
                  ref={(el) => registerProse(scene.props.id, el)}
                  className={cn(
                    "scroll-mt-32 border-l-2 py-10 pl-5 pr-2 transition-colors lg:min-h-[55vh]",
                    activeScene === scene.props.id ? "border-primary" : "border-border/60",
                  )}
                >
                  <div className="prose-sm space-y-3 text-[15px] leading-relaxed [&_code]:font-mono [&_code]:text-[13px]">
                    {scene.props.children}
                  </div>
                  {/* Mobile / reduced-motion: graphic inline under its prose */}
                  <div className="mt-6 lg:hidden">{scene.props.graphic}</div>
                </div>
              ))}
            </div>

            {/* Pinned graphic pane */}
            <div className="scrolly-pane">
              <div className="scrolly-graphic flex flex-col">
                <div className="relative min-h-0 flex-1 overflow-hidden rounded-lg border bg-card">
                  {scenes.map((scene, i) => (
                    <div
                      key={scene.props.id}
                      aria-hidden={i !== activeIndex}
                      className={cn(
                        "absolute inset-0 overflow-auto p-4 transition-opacity duration-300",
                        i === activeIndex ? "opacity-100" : "pointer-events-none opacity-0",
                      )}
                    >
                      {scene.props.graphic}
                    </div>
                  ))}
                </div>
                {/* Step dots */}
                {scenes.length > 1 && (
                  <div className="mt-3 flex items-center justify-center gap-2">
                    {scenes.map((scene, i) => (
                      <Link
                        key={scene.props.id}
                        href={`#${id}-${scene.props.id}`}
                        aria-label={`Scene ${i + 1}`}
                        className={cn(
                          "size-2.5 rounded-full transition-colors",
                          i === activeIndex ? "bg-primary" : "bg-muted-foreground/30 hover:bg-muted-foreground/60",
                        )}
                      />
                    ))}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </ChapterContext.Provider>
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* "Where Antfly diverges" callout                                     */
/* ------------------------------------------------------------------ */

export function Divergence({
  others,
  antfly,
  link,
  className,
}: {
  /** llama.cpp / vLLM / PyTorch side (muted). */
  others: ReactNode;
  /** Antfly side (full color). */
  antfly: ReactNode;
  /** One link per callout: a measurable or linkable claim. */
  link?: ReactNode;
  className?: string;
}) {
  return (
    <aside
      className={cn("my-4 border-l-4 border-double bg-muted/30 py-3 pl-4 pr-3 text-sm", className)}
      style={{ borderLeftColor: "var(--diverge-accent)" }}
    >
      <div className="mb-2 font-mono text-[11px] font-semibold uppercase tracking-wider text-primary">
        ⟂ Where Antfly diverges
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        <div className="text-muted-foreground">
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wide opacity-70">
            llama.cpp / vLLM / PyTorch
          </div>
          {others}
        </div>
        <div>
          <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-primary/70">Antfly</div>
          {antfly}
        </div>
      </div>
      {link && <div className="mt-2">{link}</div>}
    </aside>
  );
}
