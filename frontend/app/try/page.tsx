"use client";

import { useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { motion, AnimatePresence } from "motion/react";
import { DitheringBackground } from "@/components/ui/dithering-background";
import { StaticBackground } from "@/components/ui/static-background";
import { TextScramble } from "@/components/ui/text-scrammble";
import { ThemeToggle } from "@/components/theme-toggle";
import { AuthGateModal } from "@/components/auth-gate-modal";
import { PromptInputEnhanced } from "@/components/prompt-input-enhanced";
import { PresetSelector } from "@/components/preset-selector";
import { storeTransfer, type EngineType } from "@/lib/transfer-store";
import { startTrial, GateError } from "@/lib/trial";
import { getVerifiedEmail, isAuthConfigured, signOutLocally } from "@/lib/supabase";
import { Zap, ZapOff, Lock, ArrowLeft } from "lucide-react";

const ease = [0.22, 1, 0.36, 1] as const;

function makeSessionId(): string {
  return `t-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function TryPage() {
  const router = useRouter();
  const [input, setInput] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [reportTheme, setReportTheme] = useState("literature");
  const [selectedPresetId, setSelectedPresetId] = useState<string | null>(null);
  const [planRouterEnabled, setPlanRouterEnabled] = useState(false);
  const [engine, setEngine] = useState<EngineType>("deepanalyze");
  const [dynamicBgEnabled, setDynamicBgEnabled] = useState(false);

  // Gate state
  const [gateOpen, setGateOpen] = useState(false);
  const [lockedOpen, setLockedOpen] = useState(false);
  const [claimingTrial, setClaimingTrial] = useState(false);
  const pendingTidRef = useRef<string | null>(null);

  // Hydrate from localStorage after mount (avoids SSR mismatch)
  useEffect(() => {
    const storedPlan = localStorage.getItem("planRouterEnabled");
    if (storedPlan === "true") setPlanRouterEnabled(true);

    const storedEngine = localStorage.getItem("engine");
    if (storedEngine === "gemini") setEngine("gemini");

    const storedBg = localStorage.getItem("dynamicBgEnabled");
    if (storedBg === "true") setDynamicBgEnabled(true);

    router.prefetch("/analyze");
  }, [router]);

  useEffect(() => {
    localStorage.setItem("planRouterEnabled", String(planRouterEnabled));
  }, [planRouterEnabled]);

  useEffect(() => {
    localStorage.setItem("engine", engine);
  }, [engine]);

  useEffect(() => {
    localStorage.setItem("dynamicBgEnabled", String(dynamicBgEnabled));
  }, [dynamicBgEnabled]);

  const handlePresetSelect = (presetId: string, promptText: string) => {
    setSelectedPresetId(presetId);
    setInput(promptText);
  };

  const launch = (
    tid: string,
    promptText: string,
    filesSnapshot: File[],
    theme: string,
    preset: string | null,
    routerEnabled: boolean,
    engineChoice: EngineType
  ) => {
    storeTransfer(
      {
        prompt: promptText,
        files: filesSnapshot,
        reportTheme: theme,
        presetId: preset,
        planRouterEnabled: engineChoice === "gemini" ? false : routerEnabled,
        engine: engineChoice,
      },
      tid
    );
    router.push(`/analyze?tid=${tid}`);
  };

  /** Claim the trial for a pre-generated tid/session pair and navigate. */
  const claimAndLaunch = async (tid: string) => {
    const sid = makeSessionId();
    try {
      sessionStorage.setItem(`session:${tid}`, sid);
    } catch { /* noop */ }
    try {
      await startTrial(sid);
      launch(tid, input.trim(), files, reportTheme, selectedPresetId, planRouterEnabled, engine);
      return true;
    } catch (err) {
      if (err instanceof GateError && err.code === "trial_used") {
        setLockedOpen(true);
      } else if (err instanceof GateError && (err.code === "invalid_or_expired" || err.code === "unauthenticated")) {
        signOutLocally();
        setGateOpen(true);
      } else {
        try { sessionStorage.removeItem(`session:${tid}`); } catch { /* noop */ }
        console.error(err);
        const msg = err instanceof Error ? err.message : String(err);
        alert(`Could not start analysis: ${msg}\n\nCheck browser console (F12) for details. If this persists, try signing out and back in.`);
      }
      return false;
    }
  };

  const handleAnalyze = () => {
    if (!input.trim() || files.length === 0 || claimingTrial) return;
    const tid = crypto.randomUUID();

    if (!isAuthConfigured()) {
      // Gate not configured (local dev) — run directly.
      launch(tid, input.trim(), files, reportTheme, selectedPresetId, planRouterEnabled, engine);
      return;
    }

    (async () => {
      setClaimingTrial(true);
      try {
        const email = await getVerifiedEmail().catch(() => null);
        if (!email) {
          pendingTidRef.current = tid;
          setGateOpen(true);
          return;
        }
        await claimAndLaunch(tid);
      } finally {
        setClaimingTrial(false);
      }
    })();
  };

  const handleVerified = async () => {
    setGateOpen(false);
    const tid = pendingTidRef.current;
    pendingTidRef.current = null;
    if (!tid) return;
    setClaimingTrial(true);
    try {
      await claimAndLaunch(tid);
    } finally {
      setClaimingTrial(false);
    }
  };

  return (
    <main className="relative min-h-[100dvh] w-full flex flex-col bg-background text-foreground overflow-x-hidden selection:bg-primary/20">

      {/* Background Ambience */}
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

        <div className="absolute top-[10%] left-[10%] w-[60vw] h-[60vw] md:w-[40vw] md:h-[40vw] bg-primary/10 rounded-full blur-[80px] md:blur-[120px] mix-blend-normal" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[70vw] h-[50vw] bg-[#E5A84B]/10 dark:bg-[#F5C76A]/10 rounded-full blur-[100px] md:blur-[140px]" />
        <div className="absolute inset-0 z-0 opacity-[0.03] bg-[url('https://www.transparenttextures.com/patterns/stardust.png')] mix-blend-overlay" />
      </div>

      {/* Extreme Minimal Universal Header */}
      <div className="absolute top-4 sm:top-6 left-4 sm:left-8 z-40">
        <button
          onClick={() => router.push("/")}
          className="flex items-center gap-2 font-mono text-[9px] sm:text-[10px] uppercase tracking-[0.25em] text-muted-foreground/60 hover:text-foreground transition-colors"
        >
          <ArrowLeft className="size-3" />
          About swaylytics
        </button>
      </div>
      <div className="absolute top-4 sm:top-6 right-4 sm:right-8 z-40 flex items-center gap-0.5 sm:gap-1 pointer-events-auto">
        <button
          onClick={() => setDynamicBgEnabled(!dynamicBgEnabled)}
          className="flex items-center justify-center size-8 text-muted-foreground/70 hover:text-foreground transition-all duration-200 border border-border/20 hover:border-primary/40 hover:bg-primary/5"
          title={dynamicBgEnabled ? "Disable Dynamic Background" : "Enable Dynamic Background"}
        >
          {dynamicBgEnabled ? <Zap className="size-4" /> : <ZapOff className="size-4" />}
        </button>
        <ThemeToggle />
      </div>

      {/* Main Centered Hero */}
      <div className="flex-1 flex flex-col items-center justify-center relative z-10 w-full max-w-5xl mx-auto px-4 sm:px-6 pt-16 pb-8 sm:pb-16 min-h-[500px]">

        {/* Typography */}
        <motion.div
          initial={{ opacity: 0, scale: 0.98, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 1, ease: [0.16, 1, 0.3, 1] }}
          className="w-full flex flex-col items-center text-center mb-4 sm:mb-8"
        >
          <div className="px-4 py-1 rounded-none border border-primary/30 bg-primary/5 backdrop-blur-sm shadow-sm">
            <TextScramble
              phrases={[
                "Autonomous Data Science",
                "Upload. Analyze. Discover.",
                "Generate Instant Insights"
              ]}
              pauseMs={3500}
              loop
              autoStart
              textClass="font-mono text-[8px] sm:text-[10px] uppercase tracking-[0.25em] text-primary font-semibold"
              dudClass="text-primary/30"
            />
          </div>

          <div className="flex flex-col items-center mt-6 sm:mt-8">
            <h1 className="font-display font-medium text-7xl sm:text-[7rem] md:text-[9rem] tracking-tighter leading-[0.85] text-foreground lowercase relative z-10 flex items-center justify-center">
              <span className="relative inline-block group">
                <span className="absolute -inset-6 bg-primary/20 blur-3xl rounded-full opacity-0 sm:opacity-100 transition-opacity duration-1000 group-hover:opacity-60 mix-blend-screen" />
                <span className="relative text-primary italic font-bold pr-1">sway</span>
              </span>
              <span className="text-foreground/90 ml-[-0.05em]">lytics</span>
              <span className="text-primary ml-1 translate-y-1">.</span>
            </h1>

            <div className="mt-6 sm:mt-8 flex items-center justify-center relative z-20">
              <div className="group flex items-center gap-3 sm:gap-5 text-[10px] sm:text-[15px] font-mono uppercase tracking-[0.3em] text-muted-foreground/30 hover:text-muted-foreground/70 transition-colors duration-700 cursor-default">
                <span className="flex items-center gap-2">
                  <span className="italic text-primary/40 group-hover:text-primary/70 lowercase tracking-widest transition-colors duration-700">free trial</span>
                  <span className="opacity-30">/</span>
                  <span className="opacity-80">one run</span>
                </span>
                <span className="w-4 sm:w-12 h-[1px] bg-gradient-to-r from-transparent via-muted-foreground/20 to-transparent" />
                <span className="flex items-center gap-2">
                  <span className="italic text-foreground/40 group-hover:text-foreground/70 lowercase tracking-widest transition-colors duration-700">no card</span>
                  <span className="opacity-30">/</span>
                  <span className="opacity-80">full pipeline</span>
                </span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Elevated Input Card w/ Ambient Pedestal */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
          className="relative w-full max-w-3xl"
        >
          {/* Glowing Pedestal Line */}
          <div className="absolute -inset-x-4 -bottom-4 sm:-bottom-6 flex justify-center pointer-events-none">
            <div className="w-1/2 h-[1px] bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
            <div className="absolute top-0 w-2/3 h-10 bg-primary/15 blur-2xl" />
          </div>

          <div className="relative z-10 drop-shadow-xl">
            <PromptInputEnhanced
              input={input}
              onInputChange={setInput}
              files={files}
              onFilesChange={setFiles}
              reportTheme={reportTheme}
              onReportThemeChange={setReportTheme}
              planRouterEnabled={planRouterEnabled}
              onPlanRouterEnabledChange={setPlanRouterEnabled}
              engine={engine}
              onEngineChange={setEngine}
              isLoading={claimingTrial}
              onSubmit={handleAnalyze}
            />
          </div>
        </motion.div>

        {/* Preset Selector */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.8, delay: 0.4 }}
          className="mt-8 sm:mt-12 w-full max-w-3xl flex justify-center"
        >
          <PresetSelector
            selectedId={selectedPresetId}
            onSelect={handlePresetSelect}
          />
        </motion.div>

      </div>

      {/* Auth gate */}
      <AuthGateModal
        open={gateOpen}
        onClose={() => { setGateOpen(false); pendingTidRef.current = null; }}
        onVerified={handleVerified}
      />

      {/* Trial already used */}
      <AnimatePresence>
        {lockedOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[100] flex items-center justify-center bg-background/80 backdrop-blur-md px-4"
            onClick={() => setLockedOpen(false)}
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.96, y: 12 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.96, y: 12 }}
              transition={{ duration: 0.35, ease }}
              className="relative w-full max-w-md border border-destructive/30 bg-background shadow-2xl px-7 pt-8 pb-7"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center gap-2 mb-1">
                <Lock className="size-3.5 text-destructive" />
                <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-destructive font-bold">
                  Trial Used
                </span>
              </div>
              <h2 className="font-display text-2xl font-medium tracking-tight text-foreground">
                This email has had its run.
              </h2>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                Each email gets one full agent pipeline. Want another look? Watch the
                recorded session on the demo page — it shows everything the agent does.
              </p>
              <div className="mt-6 flex items-center gap-2">
                <button
                  onClick={() => { setLockedOpen(false); router.push("/demo"); }}
                  className="flex-1 py-3 border border-border hover:border-primary/40 hover:bg-primary/5 transition-all font-mono text-[10px] uppercase tracking-[0.25em] font-bold text-foreground"
                >
                  View Demo
                </button>
                <button
                  onClick={() => setLockedOpen(false)}
                  className="px-6 py-3 font-mono text-[10px] uppercase tracking-[0.25em] text-muted-foreground hover:text-foreground transition-colors"
                >
                  Close
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </main>
  );
}
