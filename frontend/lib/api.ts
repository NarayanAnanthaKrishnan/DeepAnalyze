import { BACKEND_URL } from "./config";
import type { EngineType } from "./transfer-store";

/** Auth header for gated backend calls (Supabase access token). */
export async function authHeaders(): Promise<Record<string, string>> {
  try {
    const { getAccessToken } = await import("./supabase");
    const token = await getAccessToken();
    return token ? { Authorization: `Bearer ${token}` } : {};
  } catch {
    return {};
  }
}

/** Thrown when the backend rejects a call due to trial gating (401/403). */
export class TrialGateError extends Error {
  code: string;
  status: number;
  constructor(status: number, code: string) {
    super(code === "trial_used" ? "Your free analysis has already been used." : `Access denied (${status})`);
    this.name = "TrialGateError";
    this.code = code;
    this.status = status;
  }
}

async function guardResponse(res: Response): Promise<void> {
  if (res.status !== 401 && res.status !== 403) return;
  let code = "denied";
  try {
    const body = await res.json();
    code = body?.detail?.error || body?.detail || code;
    if (typeof code !== "string") code = "denied";
  } catch {
    /* ignore */
  }
  throw new TrialGateError(res.status, code);
}

export async function uploadFiles(
  sessionId: string,
  files: File[]
): Promise<void> {
  const formData = new FormData();
  for (const file of files) formData.append("files", file);
  const res = await fetch(
    `${BACKEND_URL}/workspace/upload?session_id=${sessionId}`,
    { method: "POST", body: formData, headers: await authHeaders() }
  );
  await guardResponse(res);
  if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
}

export interface PlanResult {
  plan: string | null;
  data_profile?: Record<string, unknown>;
  error?: string;
  reason?: string;
}

export async function planAnalysis(
  sessionId: string,
  prompt: string,
  workspace: string[]
): Promise<PlanResult> {
  const res = await fetch(`${BACKEND_URL}/chat/plan`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeaders()) },
    body: JSON.stringify({ session_id: sessionId, prompt, workspace }),
  });
  await guardResponse(res);
  if (!res.ok) return { plan: null, error: `Plan request failed: ${res.status}` };
  return res.json();
}

export async function startChatStream(
  sessionId: string,
  messages: { role: string; content: string }[],
  workspace: string[],
  signal: AbortSignal,
  plan?: string | null,
  routerEnabled?: boolean,
  engine?: EngineType
): Promise<Response> {
  const isGemini = engine === "gemini";
  const res = await fetch(`${BACKEND_URL}/chat/completions`, {
    method: "POST",
    signal,
    headers: { "Content-Type": "application/json", ...(await authHeaders()) },
    body: JSON.stringify({
      model: isGemini ? "gemini-3-flash-preview" : "DeepAnalyze-8B",
      provider: isGemini ? "gemini" : "local",
      temperature: isGemini ? 1.0 : 0.4,
      messages,
      workspace,
      stream: true,
      session_id: sessionId,
      ...(plan ? { plan } : {}),
      ...(routerEnabled ? { router_enabled: true } : {}),
    }),
  });
  await guardResponse(res);
  return res;
}

export async function stopGeneration(sessionId: string): Promise<void> {
  await fetch(`${BACKEND_URL}/chat/stop`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export interface WorkspaceFile {
  name: string;
  path: string;
  size: number;
  is_generated?: boolean;
}

export async function fetchWorkspaceFiles(
  sessionId: string
): Promise<WorkspaceFile[]> {
  const res = await fetch(
    `${BACKEND_URL}/workspace/files?session_id=${sessionId}`
  );
  const data = await res.json();
  return data.files || [];
}

export function getDownloadUrl(sessionId: string, path: string): string {
  return `${BACKEND_URL}/workspace/download?session_id=${sessionId}&path=${encodeURIComponent(path)}`;
}

export function getPreviewUrl(sessionId: string, path: string): string {
  return `${BACKEND_URL}/workspace/preview?session_id=${sessionId}&path=${encodeURIComponent(path)}`;
}

export function getDownloadBundleUrl(sessionId: string, category: string = "all"): string {
  return `${BACKEND_URL}/workspace/download-bundle?session_id=${sessionId}&category=${category}`;
}

export async function clearWorkspace(sessionId: string): Promise<void> {
  await fetch(`${BACKEND_URL}/workspace/clear?session_id=${sessionId}`, {
    method: "DELETE",
  });
}

// ─── HTML Report Generation ──────────────────────────────────────────

export interface HtmlReportResult {
  message: string;
  html_file: string;
  view_url: string;
  rel_path: string;
  model_used?: string;
  fallback?: boolean;
}

export async function generateHtmlReport(
  sessionId: string,
  messages: { role: string; content: string }[],
  title: string,
  reportTheme: string,
  artifacts: { name: string; path: string }[],
  signal?: AbortSignal
): Promise<HtmlReportResult> {
  const res = await fetch(`${BACKEND_URL}/export/report/html`, {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(await authHeaders()) },
    signal,
    body: JSON.stringify({
      session_id: sessionId,
      messages,
      title,
      report_theme: reportTheme,
      artifacts,
    }),
  });
  await guardResponse(res);
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = body.detail || JSON.stringify(body);
    } catch {
      detail = await res.text().catch(() => "");
    }
    throw new Error(`Report generation failed (${res.status}): ${detail}`);
  }
  return res.json();
}
