"use client";

import { useEffect, useState } from "react";
import { motion } from "motion/react";
import { Paperclip, Loader2, Play, FlaskConical } from "lucide-react";
import { AnalyzePage } from "@/components/analyze-page";
import { ThemeToggle } from "@/components/theme-toggle";
import type { WorkspaceFile } from "@/lib/api";

interface DemoManifest {
  datasetName: string;
  prompt: string;
  reportTheme: string;
  reportUrl: string;
  transcriptUrl?: string;
  artifacts: WorkspaceFile[];
}

const ease = [0.22, 1, 0.36, 1] as const;

export default function DemoPage() {
  const [started, setStarted] = useState(false);
  const [manifest, setManifest] = useState<DemoManifest | null>(null);
  const [transcript, setTranscript] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/demo/manifest.json");
        if (!res.ok) throw new Error("Demo manifest not found");
        const data: DemoManifest = await res.json();
        setManifest(data);
        const tRes = await fetch(data.transcriptUrl ?? "/demo/transcript.json");
        if (!tRes.ok) throw new Error("Demo transcript not found");
        setTranscript(await tRes.text());
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to load demo");
      }
    })();
  }, []);

  if (error) {
    return (
      <main className="min-h-[100dvh] flex flex-col items-center justify-center bg-background text-foreground px-6 text-center">
        <FlaskConical className="size-6 text-muted-foreground/40 mb-4" />
        <p className="font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground">
          {error}
        </p>
      </main>
    );
  }

  if (!manifest || !transcript) {
    return (
      <main className="min-h-[100dvh] flex items-center justify-center bg-background text-foreground">
        <Loader2 className="size-5 animate-spin text-primary" />
      </main>
    );
  }

  if (started) {
    return (
      <AnalyzePage
        prompt={manifest.prompt}
        files={[]}
        reportTheme={manifest.reportTheme || "literature"}
        presetId={null}
        planningEnabled={false}
        routerEnabled={false}
        engine="gemini"
        sessionId="demo-proof-session"
        demo={{
          content: transcript,
          artifacts: manifest.artifacts,
          reportUrl: manifest.reportUrl,
          datasetName: manifest.datasetName,
        }}
      />
    );
  }

  // ── Intro screen ──
  return (
    <main className="relative min-h-[100dvh] w-full flex flex-col items-center justify-center bg-background text-foreground overflow-hidden selection:bg-primary/20 px-4">
      <div className="fixed inset-0 z-0 pointer-events-none">
        <div className="absolute top-[10%] left-[10%] w-[50vw] h-[50vw] md:w-[35vw] md:h-[35vw] bg-primary/10 rounded-full blur-[100px] md:blur-[140px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[60vw] h-[45vw] bg-[#E5A84B]/10 dark:bg-[#F5C76A]/10 rounded-full blur-[120px]" />
      </div>

      <div className="absolute top-4 sm:top-6 left-4 sm:left-8 z-40">
        <button
          onClick={() => history.back()}
          className="font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.25em] text-muted-foreground/60 hover:text-foreground transition-colors"
        >
          ← About swaylytics
        </button>
      </div>
      <div className="absolute top-4 sm:top-6 right-4 sm:right-8 z-40">
        <ThemeToggle />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.9, ease }}
        className="relative z-10 w-full max-w-3xl border border-border/50 bg-background/80 backdrop-blur-md shadow-2xl shadow-primary/5"
      >
        {/* corner ticks */}
        <div className="absolute top-0 left-0 w-3 h-3 border-t border-l border-primary/50 pointer-events-none" />
        <div className="absolute top-0 right-0 w-3 h-3 border-t border-r border-primary/50 pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-3 h-3 border-b border-l border-primary/50 pointer-events-none" />
        <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-primary/50 pointer-events-none" />

        <div className="px-6 sm:px-10 pt-8 pb-7">
          <div className="flex items-center gap-2">
            <span className="size-1.5 bg-amber-500 rotate-45" />
            <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-amber-600 dark:text-amber-400 font-bold">
              Recorded session · pre-completed
            </span>
          </div>

          <h1 className="mt-4 font-display text-3xl sm:text-5xl font-medium tracking-tight lowercase leading-tight">
            a real run,
            <br />
            frozen in <span className="text-primary italic font-bold">time</span>.
            <span className="text-primary">.</span>
          </h1>

          <p className="mt-4 text-sm sm:text-base text-muted-foreground leading-relaxed max-w-xl">
            Below is the exact prompt and dataset from a session this agent already
            completed — every reasoning step, line of code, execution result and chart.
            Hit execute to open it. When you&apos;re convinced,{" "}
            <a href="/try" className="text-primary font-medium hover:underline underline-offset-4">
              start your own live run
            </a>
            .
          </p>

          {/* Attached dataset */}
          <div className="mt-7">
            <p className="font-mono text-[9px] uppercase tracking-[0.25em] text-muted-foreground/60 mb-2">
              Attached_Dataset
            </p>
            <div className="inline-flex items-center gap-2 border border-primary/25 bg-primary/5 px-3 py-2 font-mono text-[11px] text-primary lowercase">
              <Paperclip className="size-3" />
              {manifest.datasetName}
            </div>
          </div>

          {/* Prompt preview */}
          <div className="mt-6 border-l-2 border-primary/30 pl-4 py-1">
            <p className="font-mono text-[9px] uppercase tracking-[0.25em] text-muted-foreground/60 mb-2">
              Prompt
            </p>
            <p className="text-sm sm:text-[15px] leading-relaxed text-foreground/85 whitespace-pre-wrap">
              {manifest.prompt}
            </p>
          </div>

          <button
            onClick={() => setStarted(true)}
            className="group mt-8 w-full relative overflow-hidden bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all px-7 py-4 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold shadow-lg"
          >
            <span className="flex items-center justify-center gap-2.5">
              <Play className="size-3.5" />
              Open the completed session
              <span className="opacity-50 group-hover:opacity-90 transition-opacity">— instant, free</span>
            </span>
          </button>

          <a
            href={manifest.reportUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-3 flex items-center justify-center gap-2 w-full border border-border hover:border-primary/40 hover:bg-primary/[0.04] transition-all px-7 py-3.5 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold text-foreground"
          >
            View generated report
            <span className="opacity-50">— HTML</span>
          </a>
          <p className="mt-2 text-center font-mono text-[9px] uppercase tracking-[0.15em] text-muted-foreground/50">
            {manifest.artifacts.length} artifacts · {manifest.datasetName} · report ready
          </p>
        </div>
      </motion.div>

      <motion.p
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="relative z-10 mt-6 font-mono text-[8px] sm:text-[9px] uppercase tracking-[0.25em] text-muted-foreground/40"
      >
        want your own results instead? head to{" "}
        <a href="/try" className="text-primary hover:underline underline-offset-4">/try</a>
      </motion.p>
    </main>
  );
}
