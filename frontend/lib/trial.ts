"use client";

import { BACKEND_URL } from "./config";
import { getAccessToken, isAuthConfigured } from "./supabase";

export class GateError extends Error {
  code: "trial_used" | "no_trial" | "unauthenticated" | "invalid_or_expired" | "unknown";
  constructor(
    code: GateError["code"],
    message: string
  ) {
    super(message);
    this.code = code;
    this.name = "GateError";
  }
}

/**
 * Bind the verified email to this session and claim its single trial run.
 * Throws GateError("trial_used") when the email already consumed its run.
 */
export async function startTrial(sessionId: string): Promise<void> {
  if (!isAuthConfigured()) return; // gate disabled (local dev)

  const token = await getAccessToken();
  if (!token) throw new GateError("unauthenticated", "Not signed in");

  let res: Response;
  try {
    res = await fetch(`${backendUrl()}/gate/start`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ session_id: sessionId }),
    });
  } catch (e) {
    const msg = e instanceof TypeError && e.message.includes("Failed to fetch")
      ? `Backend not reachable at ${backendUrl()} — is it running? (python -m uvicorn backend_app.app:app --port 8200)`
      : e instanceof Error ? e.message : String(e);
    throw new GateError("unknown", msg);
  }

  if (res.ok) return;

  let detail = "";
  let errorCode: GateError["code"] = "unknown";
  try {
    const body = await res.json();
    detail = body.detail?.error || body.detail || "";
  } catch {
    /* ignore */
  }
  if (detail === "trial_used") errorCode = "trial_used";
  else if (res.status === 401)
    errorCode = detail === "unauthenticated" ? "unauthenticated" : "invalid_or_expired";
  throw new GateError(errorCode, detail || `Gate request failed (${res.status})`);
}

function backendUrl(): string {
  return BACKEND_URL;
}

/** Lightweight status probe (used to show locked state early). */
export async function fetchTrialStatus(): Promise<{
  configured: boolean;
  authenticated: boolean;
  used: boolean | null;
} | null> {
  if (!isAuthConfigured()) return null;
  try {
    const token = await getAccessToken();
    const res = await fetch(`${BACKEND_URL}/gate/status`, {
      headers: token ? { Authorization: `Bearer ${token}` } : undefined,
    });
    if (!res.ok) return null;
    return res.json();
  } catch {
    return null;
  }
}
