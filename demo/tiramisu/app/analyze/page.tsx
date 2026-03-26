"use client";

import { Suspense } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { consumeTransfer, popActiveSession, type TransferData } from "@/lib/transfer-store";
import { stopGeneration } from "@/lib/api";
import { AnalyzePage } from "@/components/analyze-page";

function AnalyzeContent() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const [transfer, setTransfer] = useState<TransferData | null>(null);
  const consumedRef = useRef(false);
  const [sessionId] = useState(
    () => `t-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
  );

  useEffect(() => {
    // Guard against React Strict Mode double-invocation
    if (consumedRef.current) return;

    const tid = searchParams.get("tid");

    // Helper: stop any lingering backend stream, then redirect to landing
    const stopAndRedirect = async () => {
      const staleSession = popActiveSession();
      if (staleSession) {
        // Await so the fetch completes before navigation cancels it
        await stopGeneration(staleSession).catch(() => {});
      }
      router.replace("/");
    };

    if (!tid) {
      stopAndRedirect();
      return;
    }
    const data = consumeTransfer(tid);
    if (!data) {
      // Refresh detected — transfer already consumed
      stopAndRedirect();
      return;
    }
    consumedRef.current = true;
    setTransfer(data);
  }, [searchParams, router]);

  if (!transfer) {
    return (
      <div className="flex h-screen items-center justify-center bg-background">
        <div className="animate-pulse font-display text-lg text-muted-foreground">
          Preparing analysis...
        </div>
      </div>
    );
  }

  return (
    <AnalyzePage
      prompt={transfer.prompt}
      files={transfer.files}
      reportTheme={transfer.reportTheme}
      presetId={transfer.presetId}
      sessionId={sessionId}
    />
  );
}

export default function AnalyzeRoute() {
  return (
    <Suspense
      fallback={
        <div className="flex h-screen items-center justify-center bg-background">
          <div className="animate-pulse font-display text-lg text-muted-foreground">
            Loading...
          </div>
        </div>
      }
    >
      <AnalyzeContent />
    </Suspense>
  );
}
