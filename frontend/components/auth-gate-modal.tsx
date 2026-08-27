"use client";

import { useEffect, useRef, useState } from "react";
import { motion, AnimatePresence } from "motion/react";
import { Mail, Loader2, ArrowLeft, ArrowRight, X, ShieldCheck } from "lucide-react";
import { getSupabase } from "@/lib/supabase";
import { GateError } from "@/lib/trial";

interface AuthGateModalProps {
  open: boolean;
  onClose: () => void;
  onVerified: (email: string) => void;
}

const ease = [0.22, 1, 0.36, 1] as const;

export function AuthGateModal({ open, onClose, onVerified }: AuthGateModalProps) {
  const [step, setStep] = useState<"email" | "code">("email");
  const [email, setEmail] = useState("");
  const [code, setCode] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resendIn, setResendIn] = useState(0);
  const codeInputRef = useRef<HTMLInputElement>(null);

  // Reset when reopened
  useEffect(() => {
    if (open) {
      setStep("email");
      setCode("");
      setError(null);
      setBusy(false);
      setResendIn(0);
    }
  }, [open]);

  // Resend countdown
  useEffect(() => {
    if (resendIn <= 0) return;
    const t = setTimeout(() => setResendIn((v) => v - 1), 1000);
    return () => clearTimeout(t);
  }, [resendIn]);

  // Auto-focus code input
  useEffect(() => {
    if (step === "code") codeInputRef.current?.focus();
  }, [step]);

  const validateEmail = (value: string) =>
    /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/.test(value.trim());

  // Magic-link fallback: if user clicks the link in the email, Supabase (PKCE) will
  // create a session and trigger SIGNED_IN — auto-verify without typing the code.
  useEffect(() => {
    if (!open) return;
    const supabase = getSupabase();
    if (!supabase) return;
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === "SIGNED_IN" && session?.user?.email) {
        onVerified(session.user.email);
      }
    });
    return () => sub.subscription.unsubscribe();
  }, [open, onVerified]);

  const handleSendCode = async () => {
    if (!validateEmail(email)) {
      setError("Enter a valid email address");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const supabase = getSupabase();
      if (!supabase) throw new Error("Auth not configured");
      const { error: otpError } = await supabase.auth.signInWithOtp({
        email: email.trim(),
        options: { shouldCreateUser: true },
      });
      if (otpError) throw otpError;
      setStep("code");
      setResendIn(45);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to send code");
    } finally {
      setBusy(false);
    }
  };

  const handleVerify = async () => {
    if (code.trim().length < 8) {
      setError("Enter the 8-digit code");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const supabase = getSupabase();
      if (!supabase) throw new Error("Auth not configured");
      const { data, error: verifyError } = await supabase.auth.verifyOtp({
        email: email.trim(),
        token: code.trim(),
        type: "email",
      });
      if (verifyError) throw verifyError;
      const verifiedEmail = data.user?.email || email.trim();
      onVerified(verifiedEmail);
    } catch (err) {
      if (err instanceof GateError) setError(err.message);
      else setError(err instanceof Error ? err.message : "Verification failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-[100] flex items-center justify-center bg-background/80 backdrop-blur-md px-4"
          onClick={busy ? undefined : onClose}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.96, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.96, y: 12 }}
            transition={{ duration: 0.35, ease }}
            className="relative w-full max-w-md border border-primary/25 bg-background shadow-2xl shadow-primary/10"
            onClick={(e) => e.stopPropagation()}
          >
            {/* corner ticks */}
            <div className="absolute top-0 left-0 w-3 h-3 border-t border-l border-primary/60" />
            <div className="absolute top-0 right-0 w-3 h-3 border-t border-r border-primary/60" />
            <div className="absolute bottom-0 left-0 w-3 h-3 border-b border-l border-primary/60" />
            <div className="absolute bottom-0 right-0 w-3 h-3 border-b border-r border-primary/60" />

            <button
              onClick={onClose}
              disabled={busy}
              className="absolute top-3 right-3 text-muted-foreground/50 hover:text-foreground transition-colors disabled:opacity-30"
            >
              <X className="size-4" />
            </button>

            <div className="px-7 pt-8 pb-7">
              <div className="flex items-center gap-2 mb-1">
                <ShieldCheck className="size-3.5 text-primary" />
                <span className="font-mono text-[9px] uppercase tracking-[0.3em] text-primary font-bold">
                  Free Trial Verification
                </span>
              </div>
              <h2 className="font-display text-2xl font-medium tracking-tight text-foreground">
                {step === "email" ? "One analysis, on us." : "Check your inbox."}
              </h2>
              <p className="mt-2 text-sm text-muted-foreground leading-relaxed">
                {step === "email" ? (
                  <>
                    Verify your email to unlock <span className="text-foreground font-medium">one full agent run</span> — code,
                    execution, artifacts and report. No card required.
                  </>
                ) : (
                  <>
                    We sent an 8-digit code to{" "}
                    <span className="text-foreground font-medium">{email}</span>.
                  </>
                )}
              </p>

              {step === "email" ? (
                <div className="mt-6 space-y-3">
                  <div className="flex items-center gap-2 border border-border bg-secondary/20 focus-within:border-primary/50 transition-colors px-3">
                    <Mail className="size-4 text-muted-foreground/50 shrink-0" />
                    <input
                      type="email"
                      autoFocus
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      onKeyDown={(e) => e.key === "Enter" && !busy && handleSendCode()}
                      placeholder="you@work.com"
                      className="w-full bg-transparent py-3 text-sm text-foreground placeholder:text-muted-foreground/40 outline-none"
                    />
                  </div>
                  {error && (
                    <p className="text-xs text-destructive font-mono tracking-wide">{error}</p>
                  )}
                  <button
                    onClick={handleSendCode}
                    disabled={busy}
                    className="w-full flex items-center justify-center gap-2 bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all py-3 font-mono text-[10px] uppercase tracking-[0.25em] font-bold disabled:opacity-50"
                  >
                    {busy ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <>
                        Send Code <ArrowRight className="size-3.5" />
                      </>
                    )}
                  </button>
                </div>
              ) : (
                <div className="mt-6 space-y-3">
                  <input
                    ref={codeInputRef}
                    inputMode="numeric"
                    autoComplete="one-time-code"
                    value={code}
                    onChange={(e) => {
                      setCode(e.target.value.replace(/\D/g, "").slice(0, 8));
                      setError(null);
                    }}
                    onKeyDown={(e) => e.key === "Enter" && !busy && handleVerify()}
                    placeholder="••••••••"
                    className="w-full bg-transparent border border-border focus:border-primary/50 transition-colors py-3 text-center font-mono text-xl tracking-[0.5em] text-foreground outline-none placeholder:tracking-[0.5em]"
                  />
                  {error && (
                    <p className="text-xs text-destructive font-mono tracking-wide">{error}</p>
                  )}
                  <button
                    onClick={handleVerify}
                    disabled={busy}
                    className="w-full flex items-center justify-center gap-2 bg-foreground text-background hover:bg-primary hover:text-primary-foreground transition-all py-3 font-mono text-[10px] uppercase tracking-[0.25em] font-bold disabled:opacity-50"
                  >
                    {busy ? (
                      <Loader2 className="size-3.5 animate-spin" />
                    ) : (
                      <>
                        Verify & Start <ArrowRight className="size-3.5" />
                      </>
                    )}
                  </button>
                  <div className="flex items-center justify-between pt-1">
                    <button
                      onClick={() => { setStep("email"); setError(null); }}
                      disabled={busy}
                      className="flex items-center gap-1.5 text-[11px] font-mono uppercase tracking-widest text-muted-foreground/70 hover:text-foreground transition-colors disabled:opacity-40"
                    >
                      <ArrowLeft className="size-3" /> Email
                    </button>
                    <button
                      onClick={handleSendCode}
                      disabled={busy || resendIn > 0}
                      className="text-[11px] font-mono uppercase tracking-widest text-muted-foreground/70 hover:text-primary transition-colors disabled:opacity-40"
                    >
                      {resendIn > 0 ? `Resend in ${resendIn}s` : "Resend code"}
                    </button>
                  </div>
                </div>
              )}

              <p className="mt-6 pt-4 border-t border-border/40 text-[10px] font-mono uppercase tracking-[0.15em] text-muted-foreground/50 leading-relaxed">
                1 full pipeline per email · your data stays in an isolated workspace
              </p>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
