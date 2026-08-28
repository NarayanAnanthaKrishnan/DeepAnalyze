"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "motion/react";
import {
  DitheringBackground,
} from "@/components/ui/dithering-background";
import { StaticBackground } from "@/components/ui/static-background";
import { ThemeToggle } from "@/components/theme-toggle";
import {
  Upload,
  Terminal,
  FileBarChart2,
  BrainCircuit,
  Wrench,
  ShieldCheck,
  Palette,
  FolderOutput,
  Activity,
  Play,
  ArrowRight,
  ZapOff,
  Zap,
  Sparkles,
  BarChart3,
  MessageCircle,
  Workflow,
  Database,
  Shield,
  GitBranch,
} from "lucide-react";

const ease = [0.22, 1, 0.36, 1] as const;

const STEPS = [
  {
    icon: Upload,
    step: "01",
    title: "Attach any dataset",
    body: "CSV, Excel, SQLite, JSON — drop it in and describe what you want to know in plain language.",
  },
  {
    icon: Terminal,
    step: "02",
    title: "The agent takes over",
    body: "It plans an analysis, writes Python, executes it in a sandbox, reads the results and iterates — fixing its own errors along the way.",
  },
  {
    icon: FileBarChart2,
    step: "03",
    title: "Get answers & artifacts",
    body: "Charts, cleaned tables, statistical tests — streamed live — then compiled into a publication-ready HTML report.",
  },
];

const FEATURES = [
  {
    icon: BrainCircuit,
    title: "Genuinely agentic",
    body: "Not a chatbot bolted onto pandas. A multi-round loop where the model reasons, writes code, observes real execution output and decides what to do next — until it reaches a conclusion.",
    accent: true,
  },
  {
    icon: Wrench,
    title: "Self-healing execution",
    body: "Failed run? A senior-analyst router reviews the traceback, prescribes corrected code and forces the agent to fix it before moving on.",
  },
  {
    icon: ShieldCheck,
    title: "Sandboxed by design",
    body: "Every snippet runs in an isolated subprocess with hardened defaults, path-traversal guards and workspace scoping.",
  },
  {
    icon: FolderOutput,
    title: "Real artifacts",
    body: "Everything the agent produces — plots, transformed tables, reports — lands in a downloadable gallery you can inspect and reuse.",
  },
  {
    icon: Palette,
    title: "Themed reports",
    body: "The final analysis is rewritten by AI into a self-contained HTML report — pick 1920s newspaper, brutalist academic, redacted dossier or engineering blueprint.",
  },
  {
    icon: Activity,
    title: "Watch it think",
    body: "Every reasoning trace, line of code and execution result streams live over SSE. No black box — you see exactly what happens and when.",
  },
];

const FUTURE = [
  {
    icon: BarChart3,
    title: "Power BI, one click",
    body: "Turn any generated CSV into a DAX measure pack + relationship schema. Export a .json bundle and a playbook — no manual DAX writing, direct to Power BI import.",
  },
  {
    icon: MessageCircle,
    title: "Ask your data",
    body: "Every chart, table and transcript chunk gets embedded. A floating “Ask the data” panel answers with citations to the actual files, not hallucinations.",
  },
  {
    icon: Workflow,
    title: "True multi-agent",
    body: "Planner → coder → critic as separate agents with explicit tool-calling, rather than one model juggling tags. Cleaner traces, better recovery.",
  },
  {
    icon: Database,
    title: "Deeper EDA lab",
    body: "Auto feature engineering, smart statistical test selection, a SQL workbench over your SQLite uploads, and anomaly scoring — beyond the current pass.",
  },
  {
    icon: Shield,
    title: "Built for prod",
    body: "Cost caps per run, workspace TTL cleanup, real Docker sandbox, and server-side session auth — so the demo scales without surprises.",
  },
  {
    icon: GitBranch,
    title: "Measured quality",
    body: "DS1000 harness from benchmarks/ds1000 wired into CI, per-engine prompt tuning, and report-quality evals — so every change is proven.",
  },
];

const STACK = [
  "Next.js 16",
  "React 19",
  "FastAPI",
  "Google Gemini",
  "SSE Streaming",
  "Python Sandbox",
  "Tailwind v4",
  "shiki Syntax Highlighting",
];

export default function ShowcasePage() {
  const router = useRouter();
  const [dynamicBgEnabled, setDynamicBgEnabled] = useState(false);

  useEffect(() => {
    const storedBg = localStorage.getItem("dynamicBgEnabled");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- post-hydration localStorage read
    if (storedBg === "true") setDynamicBgEnabled(true);
    router.prefetch("/try");
    router.prefetch("/demo");
  }, [router]);

  useEffect(() => {
    localStorage.setItem("dynamicBgEnabled", String(dynamicBgEnabled));
  }, [dynamicBgEnabled]);

  return (
    <main className="relative min-h-[100dvh] w-full flex flex-col bg-background text-foreground overflow-x-hidden selection:bg-primary/20">

      {/* ── Background ambience ── */}
      <div className="fixed inset-0 z-0 pointer-events-none">
        <AnimatePresence mode="wait">
          {dynamicBgEnabled ? (
            <motion.div
              key="dynamic-bg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1 }}
              className="absolute inset-0 opacity-40 dark:opacity-60 saturate-50 mix-blend-luminosity dark:mix-blend-screen"
            >
              <DitheringBackground />
            </motion.div>
          ) : (
            <motion.div
              key="static-bg"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 1 }}
              className="absolute inset-0"
            >
              <StaticBackground />
            </motion.div>
          )}
        </AnimatePresence>
        <div className="absolute top-[5%] left-[8%] w-[50vw] h-[50vw] md:w-[35vw] md:h-[35vw] bg-primary/10 rounded-full blur-[100px] md:blur-[140px]" />
        <div className="absolute top-[45%] right-[-15%] w-[60vw] h-[40vw] bg-[#E5A84B]/10 dark:bg-[#F5C76A]/10 rounded-full blur-[120px] md:blur-[160px]" />
        <div className="absolute inset-0 z-0 opacity-[0.03] bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] mix-blend-overlay" />
      </div>

      {/* ── Header ── */}
      <header className="absolute top-4 sm:top-6 left-4 sm:left-8 right-4 sm:right-8 z-40 flex items-center justify-between">
        <span className="font-display text-sm sm:text-base font-medium tracking-tight lowercase">
          <span className="text-primary italic font-bold">sway</span>lytics
          <span className="text-primary ml-0.5">.</span>
        </span>
        <div className="flex items-center gap-3">
          <button
            onClick={() => setDynamicBgEnabled(!dynamicBgEnabled)}
            className="flex items-center justify-center size-8 text-muted-foreground/70 hover:text-foreground transition-all duration-200 border border-border/20 hover:border-primary/40 hover:bg-primary/5"
            title={dynamicBgEnabled ? "Disable Dynamic Background" : "Enable Dynamic Background"}
          >
            {dynamicBgEnabled ? <Zap className="size-4" /> : <ZapOff className="size-4" />}
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* ── Hero ── */}
      <section className="relative z-10 flex-1 flex flex-col items-center justify-center w-full max-w-6xl mx-auto px-4 sm:px-6 pt-32 pb-20 min-h-[92vh]">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, ease }}
          className="px-4 py-1 rounded-none border border-primary/30 bg-primary/5 backdrop-blur-sm shadow-sm"
        >
          <span className="font-mono text-[8px] sm:text-[10px] uppercase tracking-[0.3em] text-primary font-semibold">
            Autonomous EDA · Multi-Agent Pipeline · One free run
          </span>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, scale: 0.98, y: 18 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.08, ease }}
          className="mt-8 font-display font-medium text-[13vw] sm:text-7xl md:text-8xl lg:text-[7.5rem] tracking-tighter leading-[0.9] text-center lowercase relative"
        >
          <span className="relative inline-block group">
            <span className="absolute -inset-6 bg-primary/20 blur-3xl rounded-full opacity-0 sm:opacity-100 transition-opacity duration-1000 group-hover:opacity-60 mix-blend-screen" />
            <span className="relative text-primary italic font-bold pr-1">sway</span>
          </span>
          <span className="text-foreground/90 ml-[-0.05em]">lytics</span>
          <span className="text-primary ml-1 translate-y-1">.</span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 14 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.16, ease }}
          className="mt-6 max-w-2xl text-center text-base sm:text-lg text-muted-foreground leading-relaxed"
        >
          Upload a dataset. Watch an AI agent plan the analysis, write and execute real
          Python, recover from its own errors, and deliver charts, tables and a themed
          report — all on its own.
        </motion.p>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.9, delay: 0.24 }}
          className="mt-3 max-w-xl text-center font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.25em] text-muted-foreground/60"
        >
          not a wrapper · a full perceive&nbsp;→&nbsp;code&nbsp;→&nbsp;execute&nbsp;→&nbsp;repair loop
        </motion.p>

        {/* CTAs */}
        <motion.div
          initial={{ opacity: 0, y: 18 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.3, ease }}
          className="mt-10 flex flex-col sm:flex-row items-stretch sm:items-center gap-3 w-full max-w-lg"
        >
          {/* Primary: try it */}
          <button
            onClick={() => router.push("/try")}
            className="group flex-1 relative overflow-hidden bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all px-7 py-4 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold shadow-xl shadow-primary/15"
          >
            <span className="absolute top-0 right-0 w-6 h-6 bg-primary/20 -translate-x-4 -translate-y-4 rotate-45 group-hover:translate-x-0 group-hover:translate-y-0 transition-transform" />
            <span className="flex items-center justify-center gap-2.5">
              Try it live — free
              <ArrowRight className="size-3.5 group-hover:translate-x-1 transition-transform" />
            </span>
          </button>
          {/* Secondary: demo */}
          <button
            onClick={() => router.push("/demo")}
            className="group flex-1 border border-border hover:border-primary/40 hover:bg-primary/[0.04] transition-all px-7 py-4 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold text-foreground"
          >
            <span className="flex items-center justify-center gap-2.5">
              <Play className="size-3 text-primary" />
              See a real session
            </span>
          </button>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 1, delay: 0.45 }}
          className="mt-4 font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground/50"
        >
          no signup to look around · one email-verified run per person
        </motion.p>

        {/* Stats strip */}
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1, delay: 0.55, ease }}
          className="mt-14 grid grid-cols-3 gap-px bg-border/30 border border-border/30 w-full max-w-2xl"
        >
          {[
            ["multi-round", "agent loop"],
            ["self-repair", "on errors"],
            ["5 themes", "html reports"],
          ].map(([top, bottom]) => (
            <div key={top} className="bg-background/70 backdrop-blur-md px-4 py-5 text-center">
              <p className="font-display text-lg sm:text-2xl font-medium tracking-tight lowercase">
                {top}
                <span className="text-primary">.</span>
              </p>
              <p className="mt-1 font-mono text-[8px] sm:text-[9px] uppercase tracking-[0.25em] text-muted-foreground/70">
                {bottom}
              </p>
            </div>
          ))}
        </motion.div>
      </section>

      {/* ── How it works ── */}
      <section className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t border-border/30">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.8, ease }}
        >
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-primary font-bold">
            How it works
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-5xl font-medium tracking-tight lowercase">
            three steps. zero code.
            <span className="text-primary">.</span>
          </h2>
        </motion.div>

        <div className="mt-14 grid md:grid-cols-3 gap-px bg-border/30 border border-border/30">
          {STEPS.map((step, i) => (
            <motion.div
              key={step.step}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ duration: 0.7, delay: i * 0.12, ease }}
              className="group relative bg-background/80 backdrop-blur-md p-7 sm:p-8 hover:bg-background transition-colors"
            >
              <div className="absolute top-0 right-0 w-10 h-10 border-t border-r border-primary/20 group-hover:border-primary/60 transition-colors" />
              <div className="flex items-start justify-between">
                <div className="size-10 flex items-center justify-center border border-primary/30 bg-primary/5">
                  <step.icon className="size-4 text-primary" />
                </div>
                <span className="font-mono text-[10px] text-muted-foreground/40 tracking-widest">{step.step}</span>
              </div>
              <h3 className="mt-6 font-display text-xl font-medium tracking-tight lowercase">{step.title}</h3>
              <p className="mt-3 text-sm leading-relaxed text-muted-foreground">{step.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Features ── */}
      <section className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t border-border/30">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.8, ease }}
        >
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-primary font-bold">
            Why it&apos;s different
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-5xl font-medium tracking-tight lowercase">
            built like an agent.
            <span className="text-primary">.</span>
            not a demo.
          </h2>
        </motion.div>

        <div className="mt-14 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FEATURES.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.65, delay: (i % 3) * 0.1, ease }}
              className={`relative p-6 sm:p-7 border transition-all hover:-translate-y-0.5 ${
                feature.accent
                  ? "border-primary/40 bg-primary/[0.04]"
                  : "border-border/40 bg-background/60 hover:border-primary/25"
              }`}
            >
              {feature.accent && (
                <Sparkles className="absolute top-5 right-5 size-3.5 text-primary/70" />
              )}
              <feature.icon className={`size-5 ${feature.accent ? "text-primary" : "text-muted-foreground/60"}`} />
              <h3 className="mt-4 font-display text-lg font-medium tracking-tight lowercase">
                {feature.title}
                <span className={feature.accent ? "text-primary" : ""}>.</span>
              </h3>
              <p className="mt-2.5 text-sm leading-relaxed text-muted-foreground">{feature.body}</p>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Future Scope ── */}
      <section className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t border-border/30">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.8, ease }}
        >
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-primary font-bold">
            What&apos;s next
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-5xl font-medium tracking-tight lowercase">
            built to grow.
            <span className="text-primary">.</span>
          </h2>
          <p className="mt-4 max-w-2xl text-sm sm:text-base text-muted-foreground leading-relaxed">
            The seams are already there — workspace + generated index + Gemini as the stable core. Here&apos;s where it goes next, in plain terms.
          </p>
        </motion.div>

        <div className="mt-14 grid sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {FUTURE.map((item, i) => (
            <motion.div
              key={item.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.65, delay: (i % 3) * 0.1, ease }}
              className="relative p-6 sm:p-7 border border-border/40 bg-background/60 hover:border-primary/25 hover:bg-background transition-all hover:-translate-y-0.5"
            >
              <item.icon className="size-5 text-primary/70" />
              <h3 className="mt-4 font-display text-lg font-medium tracking-tight lowercase">
                {item.title}
                <span className="text-primary">.</span>
              </h3>
              <p className="mt-2.5 text-sm leading-relaxed text-muted-foreground">{item.body}</p>
            </motion.div>
          ))}
        </div>

        <motion.p
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.2 }}
          className="mt-10 text-center font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground/50"
        >
          Designed to be extended — not a one-off demo.
        </motion.p>
      </section>

      {/* ── Visual slots (screenshots / GIF placeholders) ── */}
      <section className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-24 border-t border-border/30">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-80px" }}
          transition={{ duration: 0.8, ease }}
        >
          <p className="font-mono text-[9px] uppercase tracking-[0.3em] text-primary font-bold">
            The interface
          </p>
          <h2 className="mt-3 font-display text-3xl sm:text-5xl font-medium tracking-tight lowercase">
            watch every move.
            <span className="text-primary">.</span>
          </h2>
        </motion.div>

        <div className="mt-14 grid lg:grid-cols-2 gap-4">
          {[
            { label: "Live streaming session — reasoning, code & output", href: "/demo", cta: "Open live example" },
            { label: "Final themed HTML report", href: "/try", cta: "Generate your own" },
          ].map((slot, i) => (
            <motion.div
              key={slot.label}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-40px" }}
              transition={{ duration: 0.7, delay: i * 0.12, ease }}
              className="group relative aspect-video border border-dashed border-border/50 bg-secondary/10 hover:border-primary/30 transition-colors overflow-hidden"
            >
              <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-6 text-center">
                <Terminal className="size-6 text-muted-foreground/30 group-hover:text-primary/50 transition-colors" />
                <p className="font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.2em] text-muted-foreground/50 leading-relaxed max-w-xs">
                  {slot.label}
                </p>
                <Link
                  href={slot.href}
                  className="mt-1 inline-flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.25em] font-bold text-primary hover:underline underline-offset-4"
                >
                  {slot.cta} <ArrowRight className="size-3" />
                </Link>
              </div>
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── Final CTA ── */}
      <section className="relative z-10 w-full max-w-6xl mx-auto px-4 sm:px-6 py-28 border-t border-border/30 flex flex-col items-center text-center">
        <motion.h2
          initial={{ opacity: 0, y: 18 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, ease }}
          className="font-display text-4xl sm:text-6xl font-medium tracking-tight lowercase"
        >
          bring a dataset.
          <br />
          <span className="text-primary italic font-bold">leave</span> with answers.
          <span className="text-primary">.</span>
        </motion.h2>
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.8, delay: 0.15, ease }}
          className="mt-10 flex flex-col sm:flex-row gap-3 w-full max-w-lg"
        >
          <button
            onClick={() => router.push("/try")}
            className="group flex-1 bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all px-7 py-4 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold shadow-xl shadow-primary/15"
          >
            <span className="flex items-center justify-center gap-2.5">
              Start your free run
              <ArrowRight className="size-3.5 group-hover:translate-x-1 transition-transform" />
            </span>
          </button>
          <button
            onClick={() => router.push("/demo")}
            className="flex-1 border border-border hover:border-primary/40 hover:bg-primary/[0.04] transition-all px-7 py-4 font-mono text-[10px] sm:text-[11px] uppercase tracking-[0.25em] font-bold text-foreground"
          >
            <span className="flex items-center justify-center gap-2.5">
              <Play className="size-3 text-primary" />
              See a real session
            </span>
          </button>
        </motion.div>
      </section>

      {/* ── Footer ── */}
      <footer className="relative z-10 w-full border-t border-border/30">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-10 flex flex-col sm:flex-row items-center justify-between gap-6">
          <div>
            <span className="font-display text-sm font-medium tracking-tight lowercase">
              <span className="text-primary italic font-bold">sway</span>lytics
              <span className="text-primary ml-0.5">.</span>
            </span>
            <p className="mt-1 font-mono text-[8px] uppercase tracking-[0.25em] text-muted-foreground/50">
              autonomous exploratory data analysis
            </p>
          </div>
          <div className="flex flex-wrap items-center justify-center gap-x-6 gap-y-2">
            {STACK.map((tech) => (
              <span key={tech} className="font-mono text-[8px] uppercase tracking-[0.2em] text-muted-foreground/40">
                {tech}
              </span>
            ))}
          </div>
        </div>
      </footer>
    </main>
  );
}
