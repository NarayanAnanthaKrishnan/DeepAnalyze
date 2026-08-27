# Swaylytics — Agent Context

> **Swaylytics** (codename: *Tiramisu*) is an autonomous multi-agent Exploratory Data Analysis (EDA) tool. Upload a dataset, pick a preset prompt, and watch a Gemini-driven agent write and execute Python code iteratively until it reaches a conclusion.

---

## 1. Project Overview

- **Frontend**: Next.js 16 (React 19, Tailwind v4) — port **3000**
- **Backend**: FastAPI + Uvicorn — port **8200**
- **Primary model**: **Google Gemini** (default `gemini-3-flash-preview`)
- **Secondary model**: DeepAnalyze-8B via vLLM (legacy / optional, requires SSH tunnel)
- **Communication**: Server-Sent Events (SSE) for streaming agent output
- **Code execution**: Sandboxed `subprocess.run` of `python <tmpfile>.py` per code block (stateless)
- **Workspace**: Per-session directory under `./workspace/<session_id>/`
- **Vision**: Build out into a multi-agent EDA platform with Power BI integration and a RAG layer for asking questions over analyzed data

---

## 2. High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  Frontend (Next.js, :3000)                                         │
│  ┌──────────┐  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  │
│  │ Home     │→ │ AnalyzePage │→ │ TransferStore│→ │ sessionStg │  │
│  │ (page.tsx)│  │ (1531 lines)│  │  (Map+SS)    │  │  (recovery)│  │
│  └──────────┘  └──────┬──────┘  └──────────────┘  └────────────┘  │
│                       │ SSE stream                                 │
└───────────────────────┼────────────────────────────────────────────┘
                        │ /chat/completions, /workspace/*, /export/*
                        ▼
┌────────────────────────────────────────────────────────────────────┐
│  Backend (FastAPI, :8200)                                          │
│                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐   │
│  │ /chat      │  │ /workspace │  │ /export    │  │ /execute   │   │
│  │ router     │  │ router     │  │ router     │  │  (direct)  │   │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘   │
│        ▼               ▼               ▼               ▼          │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    ┌────────────┐      │
│  │  chat.py │   │workspace │   │exporter  │    │ execution  │      │
│  │ (agent   │   │  .py     │   │  .py     │    │   .py      │      │
│  │  loop)   │   │(files)   │   │(HTML rpt)│    │(sandbox)   │      │
│  └────┬─────┘   └──────────┘   └────┬─────┘    └────────────┘      │
│       │                              │                              │
│       ▼                              │                              │
│  ┌──────────┐                        │                              │
│  │ planner  │ ←─ error recovery     │                              │
│  │  .py     │ ←─ checkpoint review  │                              │
│  └────┬─────┘                        │                              │
└───────┼──────────────────────────────┼──────────────────────────────┘
        ▼                              ▼
   Gemini API                  workspace/<sid>/
   (REST/SSE)                  + generated/
                               + generated/reports/
```

---

## 3. Directory Structure

```
DeepAnalyze/
├── backend.py                    # Uvicorn entry point
├── requirements.txt              # Python deps
├── start.bat / start.sh          # Launchers (Windows / Unix)
├── stop.bat / stop.sh            # Stop scripts
├── .env                          # GEMINI_API_KEY
│
├── backend_app/                  # ── Backend (FastAPI) ──────────
│   ├── __init__.py
│   ├── app.py                    # FastAPI app factory, CORS, router wiring
│   ├── settings.py               # Env-driven Settings dataclass
│   ├── routers/
│   │   ├── chat.py               # /execute, /chat/plan, /chat/completions, /chat/stop
│   │   ├── workspace.py          # /workspace/* (files, upload, download, preview, tree)
│   │   └── export.py             # /export/report, /export/report/html
│   └── services/
│       ├── chat.py               # ★ Core multi-turn agent orchestrator (bot_stream)
│       ├── planner.py            # Data profiler + Gemini plan + error recovery + checkpoints + code validator
│       ├── execution.py          # Subprocess code sandbox, artifact collection, file block builder
│       ├── workspace.py          # File ops, previews, uploads, bundles, tree, proxy
│       └── exporter.py           # Markdown export + Gemini-driven HTML report (5 themes)
│
├── frontend/                     # ── Frontend (Next.js) ─────────
│   ├── package.json              # Next 16, React 19, Tailwind v4, motion, shiki
│   ├── app/
│   │   ├── layout.tsx            # Fonts (Geist, Syne), ThemeProvider
│   │   ├── page.tsx              # Landing page
│   │   ├── globals.css
│   │   └── analyze/
│   │       ├── page.tsx          # Route handler: tid → transfer, session id, snapshot recovery
│   │       └── loading.tsx
│   ├── components/
│   │   ├── analyze-page.tsx      # ★ Main streaming UI (~1531 lines)
│   │   ├── prompt-input-enhanced.tsx  # Input bar with engine + theme + router toggles
│   │   ├── preset-selector.tsx   # Preset prompt chips
│   │   ├── theme-selector.tsx    # Report aesthetic picker
│   │   ├── theme-toggle.tsx
│   │   └── ui/                   # Atomic UI primitives (button, popover, code-block, markdown, prompt-input, backgrounds…)
│   └── lib/
│       ├── api.ts                # Typed REST client (upload, chat stream, plan, report, etc.)
│       ├── config.ts             # BACKEND_URL (env-overridable)
│       ├── prompt-presets.ts     # 7 preset prompts (EDA, cleaning, viz, stats, SQL, feature, report)
│       ├── stream-parser.ts      # Tag-based section parser for streamed content
│       ├── transfer-store.ts     # Cross-page transfer (in-memory Map + sessionStorage fallback)
│       └── utils.ts              # cn() className helper
│
├── workspace/                    # Per-session working directories (gitignored runtime)
├── logs/                         # backend.log, tiramisu.log
├── benchmarks/ds1000/            # DS1000 eval harness (legacy DeepAnalyze)
└── AGENTS.md                     # ← you are here
```

---

## 4. Backend Deep-Dive

### 4.1 `app.py` — Application Factory
- Creates FastAPI with CORS (`*`), registers three routers, defines a no-op `lifespan`.
- Three router groups: `workspace`, `chat`, `export`.

### 4.2 `settings.py` — Configuration
All env-driven; loaded eagerly from `.env` via `_load_demo_env()`.

| Setting | Env | Default | Notes |
|---|---|---|---|
| `api_base` | `DEEPANALYZE_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible vLLM endpoint |
| `model_path` | `DEEPANALYZE_MODEL_PATH` | `DeepAnalyze-8B` | Local model id |
| `workspace_base_dir` | `DEEPANALYZE_WORKSPACE_BASE` | `workspace` | Per-session subfolders |
| `http_server_host/port` | `DEEPANALYZE_FILE_SERVER_*` | `localhost:8100` | File URL base |
| `backend_host/port` | `DEEPANALYZE_BACKEND_*` | `0.0.0.0:8200` | FastAPI bind |
| `execution_timeout_sec` | `DEEPANALYZE_EXECUTION_TIMEOUT_SEC` | `120` | Subprocess kill |
| `gemini_api_key` | `GEMINI_API_KEY` | `""` | **Required for Gemini** |
| `gemini_model` | `GEMINI_MODEL` | `gemini-3-flash-preview` | Default model |
| `router_error_recovery` | `ROUTER_ERROR_RECOVERY` | `true` | Hybrid router: error fix-ups |
| `router_checkpoints` | `ROUTER_CHECKPOINTS` | `true` | Hybrid router: periodic reviews |
| `router_checkpoint_interval` | `ROUTER_CHECKPOINT_INTERVAL` | `3` | Rounds between checkpoints |

Derived properties:
- `planning_enabled` / `router_active` — `True` iff `gemini_api_key` is non-empty.
- `file_server_base` — full URL prefix.

### 4.3 `services/chat.py` — The Core Orchestrator ★

This is the heart of the system. Key building blocks:

#### Provider abstraction
- `ChatRuntimeConfig` (`provider` ∈ {`local`, `heywhale`, `gemini`}, `temperature`, `model`, `api_key`, `api_base`).
- Three stream iterators:
  - `_iter_local_stream` — OpenAI SDK against vLLM endpoint with stop token ids and 32k max tokens.
  - `_iter_heywhale_stream` — legacy third-party endpoint (kept but unused going forward).
  - `_iter_gemini_stream` — Gemini `streamGenerateContent?alt=sse` with retries.

#### Gemini chat system prompt (`GEMINI_SYSTEM_PROMPT`)
Forces Gemini into an XML-tagged protocol:
```
<Analyze>     reasoning
<Understand>  data comprehension
<Code>        ```python … ```  ← stops here, system executes
<Execute>     injected by system only (do NOT generate)
<File>        linked artifacts
<Answer>      final conclusion
```
Critical rule: "After writing `</Code>`, you MUST stop. Do NOT write `<Execute>` or guess what the output might be."

#### `bot_stream()` — the multi-turn loop
Located at `services/chat.py:423`. Yields OpenAI-style JSON chunks. Flow per round:

1. **Build user prompt** — append `# Instruction` + (optional) `# Analysis Plan` + `# Data` (file listings + sizes).
2. **Stream from provider** (Gemini by default):
   - Handles Gemini's `thought` parts by wrapping in `<Thinking>…</Thinking>`.
   - Auto-prefixes `<Analyze>\n` if the first non-blank content doesn't start with a structured tag.
   - Streams deltas straight to the client.
   - Breaks early on `</Code>` (so the backend can execute) or `</Answer>` (loop ends).
3. **Extract code** from `<Code>…```python …```</Code>` (`_extract_code_to_execute`).
4. **Pre-execution validation** (`validate_code_before_execution`):
   - AST-parse; skip on syntax error.
   - Auto-prepend standard imports (`pd`, `np`, `plt`, `sns`, `stats`) if referenced but not imported.
   - Auto-prepend `df = pd.read_csv(...)` (or xlsx) when `df` is used but no read call exists.
   - Warn about unknown column references by scanning `data_context` (the plan) for `"name": "…"` entries.
   - Patch `open(..., "w")` calls without `encoding=` to include `encoding='utf-8'` (Windows cp1252 fix).
5. **Execute code** via `execute_code_safe` in subprocess.
6. **Snapshot workspace** before/after, diff for new + modified files, collect artifact paths.
7. **Yield `<Execute>…</Execute>` + `<File>…</File>`** back to client (file links use `/workspace/download?session_id=…&path=…`).
8. **Append to conversation** as `role: "execute"` (so next turn's context includes results).
9. **Hybrid Router: Error recovery** — if execution errored and `router_active`:
   - `call_gemini_error_recovery` is invoked with the failed code, error output, recent conversation, and data context.
   - Gemini returns a "Senior Analyst Guidance" block including a `### Corrected Code` section.
   - A retry directive is appended so the model must regenerate the code immediately.
10. **Hybrid Router: Checkpoint** — every `ROUTER_CHECKPOINT_INTERVAL` successful rounds:
    - `call_gemini_checkpoint` reviews progress and emits steering guidance.
    - Injected as `role: "execute"` so the model adapts in the next round.
11. **Update workspace** — newly created files are added to the file list for future context.

Stop conditions:
- `</Answer>` emitted by model.
- `finish_reason == "stop"` with a missing closing tag → auto-close the tag.
- `stop_event` set via `/chat/stop`.
- Process exits (no explicit max-rounds guard in code, but the model is instructed to converge).

### 4.4 `services/planner.py` — Pre-Analysis + Hybrid Router

- **`build_profiling_script(file_names)`** — emits a self-contained Python script that opens each file, infers type (CSV, TSV, XLSX, XLS, JSON, SQLite, text), and prints a structured JSON profile between `__PROFILE_JSON_START__` / `__PROFILE_JSON_END__` markers. CSV/XLSX: shape, per-column dtype/null/unique, numeric describe, sample rows (first 15). JSON: length, sample entries, top-level keys. SQLite: table list + columns + row counts.
- **`generate_plan(session_id, user_prompt, workspace_files)`** — runs the profiler (30s timeout) → calls `call_gemini_planner` (async, Gemini `medium` thinking) with `GEMINI_PROMPT_TEMPLATE` to produce a 5-section markdown plan: Data Understanding, Hypotheses (2-8), Analysis Steps (5-10), Potential Pitfalls, Key Visualizations.
- **`call_gemini_error_recovery`** — synchronously invokes Gemini (`temperature=0.3`, `medium` thinking) with the recent conversation, failed code, and error → returns corrected code in markdown.
- **`call_gemini_checkpoint`** — sync Gemini call (`temperature=0.5`, `medium` thinking) for steering review.
- **`validate_code_before_execution`** — see chat.py section above.
- **`_call_gemini_sync`** — internal sync helper with retry/backoff. `GEMINI_API_URL` = `…/v1beta/models/{model}:generateContent`.
- **`_ERROR_INDICATORS`** — tuple of Python error substrings (`Traceback`, `ModuleNotFoundError`, `KeyError`, etc.) used by `is_execution_error`.

### 4.5 `services/execution.py` — Code Sandbox
- **`execute_code_safe(code, workspace_dir, session_id, timeout_sec)`** — writes code to a tempfile in the session workspace, runs `python <tmpfile>` with `subprocess.run` (capture stdout+stderr, text, timeout). Env: forces `MPLBACKEND=Agg`, `QT_QPA_PLATFORM=offscreen`, drops `DISPLAY`. Returns combined stdout/stderr; on timeout/exception returns `[Timeout]: …` or `[Error]: …`.
- **`snapshot_workspace_files(dir)`** — `{resolved_path: (size, mtime_ns)}` map.
- **`collect_artifact_paths(before, after, generated_dir, session_id)`** — diffs snapshots, copies new files into `generated/`, overwrites modified ones in place (no `_modified` suffix), and registers them in the session's generated index.
- **`build_file_block(paths, workspace_dir, session_id)`** — builds `<File>` markdown block with download URLs (`![name](url)` for images, `[name](url)` for others).

### 4.6 `services/workspace.py` — File Management
- **`get_session_workspace(session_id)`** — creates/returns `./workspace/<session_id>/`.
- **Generated index** — `generated/.deepanalyze_generated.json` tracks all artifact paths so the UI can show the "Generated Files" panel.
- **Previews** — supports CSV/TSV (paginated), XLSX/XLS (sheet-aware), SQLite (table list + paginated rows), JSON, text/markdown, logs. 50kB text cap, 500-char cell cap.
- **`collect_file_info`** — emits the "File 1: `{name, size}`" blocks prepended to user prompts.
- **`uniquify_path`** — appends ` (1)`, ` (2)`, … when names collide.
- **`resolve_workspace_path`** — path-traversal guard.
- **`list_workspace_files` / `build_tree`** — UI listings, with `is_generated` flag derived from the index.
- **`upload_files_to_workspace` / `…_to_dir`** — multi-file uploads; `.py` blocked.
- **`download_generated_bundle`** — zips `generated/` (optionally filtered by category) into `tables/`, `images/`, `others/` for the "Download All" button.
- **`proxy_external_file`** — generic CORS proxy used by the UI.
- **Classifiers** — `TABLE_EXTENSIONS`, `IMAGE_EXTENSIONS`, `TEXT_PREVIEW_EXTENSIONS`, `BLOCKED_UPLOAD_EXTENSIONS={".py"}`.
- **Icons** — `get_file_icon(ext)` maps extensions to emoji (🖼️ 📃 📌 📝 📑 📳 🗽 etc.) for the file tree.

### 4.7 `services/exporter.py` — Report Generation
- **`export_report_from_body`** — extracts `<Analyze>/<Understand>/<Code>/<Execute>/<Answer>` segments from the message log (regex-based), appends an "Appendix: Detailed Process" section, writes `<title>_<timestamp>.md` to `generated/reports/`, registers it.
- **`extract_full_analysis_content`** — richer extraction (used by HTML report) including `RouterGuidance` and `Thinking`, with consecutive failed-execute deduplication.
- **`collect_artifact_images_base64`** — base64-encodes up to 20 images from `generated/` for the multimodal HTML report.
- **HTML report pipeline** (`export_html_report_from_body`):
  1. Extract analysis sections.
  2. Collect images.
  3. Build mega-prompt via `_build_html_report_prompt` (injects a long, opinionated theme directive).
  4. Call Gemini — **primary**: `gemini-3.1-pro-preview` (`medium` thinking), **fallback**: `gemini-3-flash-preview` (`high` thinking).
  5. Post-process HTML: `_inject_base64_images` replaces `<img src="filename.png">` placeholders with `data:image/png;base64,…` URIs.
  6. Save as `<title>_<timestamp>.html` and return a `view_url` (preview, not attachment).
- **Themes** — `_THEME_INSTRUCTIONS` defines 5 detailed aesthetic guides:
  - `literature` — 1920s newspaper, off-white + sepia, Playfair Display, justified text, drop caps.
  - `academic` — brutalist IBM research paper, forest-green, STIX/Computer Modern, rigid grid.
  - `dossier` — 1970s redacted file, manila beige, Special Elite, "Classified" redactions.
  - `blueprint` — drafting terminal, blueprint blue + cyan, JetBrains Mono, dimension lines.
  - `surprise` — Gemini has free creative reign with a curated inspiration list.
- **`_FRONTEND_DESIGN_SKILL`** — a system-prompt-style "skill" block reinforcing distinctive, intentional design (forbidding Inter/Roboto/generic purple gradients).
- **Retry** — 2 attempts with 2s/4s backoff for transient Gemini failures on report calls.

### 4.8 Routers

#### `routers/chat.py`
- `POST /execute` — direct code execution (no streaming). Used for ad-hoc execution.
- `POST /chat/plan` — body: `{session_id, prompt, workspace[]}` → returns `{plan, data_profile?, error?}`.
- `POST /chat/completions` — body: `{messages, workspace, session_id, plan?, router_enabled?, provider, model, temperature}` → SSE stream of OpenAI-format JSON chunks.
- `POST /chat/stop` — sets the per-session stop event (threading.Event).

#### `routers/workspace.py`
- `GET /workspace/files?session_id` — list with icons/categories/preview/download URLs.
- `GET /workspace/tree?session_id` — nested tree.
- `GET /workspace/download?path&session_id&download` — `FileResponse` (inline or attachment).
- `GET /workspace/download-bundle?category&session_id` — zipped `generated/`.
- `GET /workspace/preview?path&session_id&page&page_size&table_name&sheet_name` — typed JSON preview.
- `POST /workspace/upload` / `POST /workspace/upload-to?dir` — multipart uploads.
- `DELETE /workspace/file?path&session_id`, `POST /workspace/move?src&dst_dir`, `DELETE /workspace/dir?path&recursive`, `DELETE /workspace/clear`.
- `GET /proxy?url` — external CORS proxy.

#### `routers/export.py`
- `POST /export/report` — markdown extraction.
- `POST /export/report/html` — full Gemini HTML report. 502 on Gemini HTTP errors, 500 with traceback tail on internal failure.

---

## 5. Frontend Deep-Dive

### 5.1 Routes
- `/` — `app/page.tsx` — landing page with hero ("swaylytics."), dynamic background toggle, prompt input card, preset selector. On submit → `storeTransfer(...)` + `router.push('/analyze?tid=...')`.
- `/analyze` — `app/analyze/page.tsx` (Suspense) → `AnalyzeContent` reads `?tid=`, derives/loads `sessionId` from `sessionStorage[session:<tid>]`, looks for `snapshot:<sessionId>` for HMR/reload recovery, then consumes the transfer and renders `<AnalyzePage>`.

### 5.2 `lib/transfer-store.ts` — Cross-Page State
- **Primary**: in-memory `Map<tid, TransferData>` (preserves `File` objects across SPA nav).
- **Fallback**: `sessionStorage[transfer:<tid>]` (text fields only; `File[]` becomes empty → backend already has them from prior upload).
- `consumeTransfer(tid)` checks memory first, then sessionStorage.
- `setActiveSession(id)` + `popActiveSession()` — bookkeeping for `/chat/stop` on refresh.
- `clearTransfer(tid)` — used on "Clear workspace" button.
- Stale `transfer:*` keys are wiped on every new `storeTransfer` so a fresh analysis in the same tab doesn't inherit prior runs.

### 5.3 `lib/api.ts` — Typed REST Client
- `uploadFiles(sessionId, files)` — multipart POST to `/workspace/upload`.
- `planAnalysis(sessionId, prompt, workspace)` — POST to `/chat/plan`.
- `startChatStream(sessionId, messages, workspace, signal, plan?, routerEnabled?, engine)` — POST to `/chat/completions`, returns raw `Response` (caller reads the SSE body).
- `stopGeneration(sessionId)` — POST to `/chat/stop`.
- `fetchWorkspaceFiles(sessionId)`, `getDownloadUrl`, `getPreviewUrl`, `getDownloadBundleUrl`, `clearWorkspace`.
- `generateHtmlReport(sessionId, messages, title, reportTheme, artifacts, signal?)` — POST to `/export/report/html`.

### 5.4 `lib/stream-parser.ts` — Tag-Based Section Parser
- Tags recognized: `Analyze`, `Understand`, `Code`, `Execute`, `Answer`, `File`, `RouterGuidance`, `Thinking`.
- `parseSections(content)` walks opening tags, tracks `round` (increments on every `Analyze` after an `Execute`/`Answer`), marks `isComplete` when closing tag found, breaks on the still-streaming section.
- `getPreTagContent(content)` returns text before the first tag (rare, e.g. unprefixed stream lead).

### 5.5 `lib/prompt-presets.ts` — 7 Presets
| ID | Label | Purpose |
|---|---|---|
| `eda` | Exploratory Analysis | schema, quality, distributions, anomalies, correlations |
| `cleaning` | Data Cleaning | missing/duplicates/types/outliers + cleaning plan |
| `viz` | Visualization Report | presentation-ready charts with interpretation |
| `stats` | Statistical Testing | hypotheses, method selection, significance |
| `sql` | SQL Analysis | SQLite queries + result interpretation |
| `feature` | Feature Review | modeling prep, target candidates, feature quality |
| `report` | Executive Summary | stakeholder-ready summary |

### 5.6 `components/analyze-page.tsx` — Main UI ★
~1531 lines, organized into:

#### Phases
`uploading → planning → streaming → complete | error` (with `reportStatus` for the HTML report lifecycle: `idle | generating | ready | error | cancelled`).

#### Streaming loop
- Maintains `pendingContentRef` and `displayedContentRef`; a `requestAnimationFrame` tick gradually reveals content (step ≈ `diff / 5` chars per frame) for a smooth "type-on" feel.
- Reads SSE body line-by-line, parses OpenAI-style chunks, appends to `pendingContentRef`.
- On `finish_reason === "stop"`, breaks and finalizes.
- On AbortError (user stop), finalizes whatever was accumulated.

#### Recovery
- Reads `sessionStorage[snapshot:<sessionId>]` on mount; if found, restores as `complete` (snapshot was either finished or interrupted mid-stream).
- `writeSnapshot` on completion + every 3s during streaming (best-effort, swallows quota errors).
- Saved snapshot fields: prompt, reportTheme, presetId, phase, accumulatedContent, completedTurns, messages, workspaceFileNames, plan, engine, reportStatus, reportUrl, reportFallback.

#### Section rendering
- Each parsed section is rendered with a timeline indicator (`section-row` with dot + label + spine).
- Special renderers: `Thinking` (collapsible, blue), `Analyze/Understand` (plain prose), `Code` (terminal-card with shiki highlight + copy button), `Execute` (stdout-styled block), `Answer` (clean prose), `File` (image/file cards), `RouterGuidance` (amber "Senior Analyst" callout with code fences stripped).
- `Code` followed by a complete `Execute` is fused: code block + attached stdout.

#### Right panel (lg+)
- Auto-opens when analysis completes and artifacts exist.
- Top: file list with image previews, "Download All" → `getDownloadBundleUrl`.
- Bottom: report lifecycle (generating with progress bar + cancel, ready with "View Report" link, error with retry, cancelled with "Generate Report" button).

#### Auto HTML report
- `useEffect` watching `phase === "complete" && reportStatus === "idle"` triggers `generateHtmlReport(...)`.
- Stale-response guard via `reportGenIdRef` so a slow first call doesn't overwrite a newer one.
- AbortController for cancellation.

#### Follow-up turns
- `handleSendFollowUp` snapshots the current assistant turn into `completedTurns`, sends `[...messages, assistantTurn, newUserMsg]` back to `/chat/completions`, resets the streaming buffer, aborts any in-flight report.

#### Other controls
- `handleStop` — calls `stopGeneration(sessionId)` + aborts local stream; finalizes as `complete`.
- `handleClearWorkspace` — abort, stop, clear, remove snapshots, redirect home.
- Scroll-to-bottom button (`showScrollBtn` flips based on scroll position).
- Theme switcher inline (overrides initial choice).

### 5.7 `components/prompt-input-enhanced.tsx` — Input Bar
- File attach (multi), engine selector (Popover with **DeepAnalyze-8B** vs **Gemini 3 Flash** — blue accent for Gemini), report theme selector, plan+router toggle (only shown with DeepAnalyze engine, amber accent), mic (placeholder), Execute/Halt button.
- Stops in-flight when `isLoading`.
- Sends `engine` so the backend can pick the right provider.

### 5.8 `components/preset-selector.tsx`
Renders the 7 presets as monospace chips; clicking fills the input with the preset's prompt text.

### 5.9 `components/theme-selector.tsx`
Popover with the 5 report themes (Literature, Academic, Surprise me, Old School, Engineering). "Surprise me" gets a Sparkles icon.

### 5.10 `components/ui/`
Atomic primitives: `button`, `popover`, `prompt-input`, `code-block` (shiki), `markdown`, `text-shimmer`, `text-scrammble`, `static-background`, `dithering-background`, `cross-scroll-background`, `loading-screen`, `bg-pattern`, `badge`, `avatar`, `select`, `tooltip`, `scroll-button`, `message`, `response-stream`, `reasoning`, `special-text`, `text-marquee`, `textarea`.

---

## 6. The Multi-Agent Loop in Practice

> **Concretely:** the system prompt forces Gemini to behave as a single agent emitting structured tags. The "multi-agent" effect comes from the **Hybrid Router** (when DeepAnalyze is selected) where Gemini also plays the role of Senior Analyst reviewer.

### 6.1 Pure-Gemini flow (`engine === "gemini"`)
1. User uploads files + writes prompt (or picks a preset) on `/`.
2. Files POSTed to `/workspace/upload`.
3. **No planning phase** (engine is Gemini → planner skipped; Gemini does its own planning inline).
4. `/chat/completions` opens SSE; `bot_stream` invokes `_iter_gemini_stream` with:
   - `systemInstruction: GEMINI_SYSTEM_PROMPT`
   - `generationConfig: { temperature: 1.0, maxOutputTokens: 65536, stopSequences: ["</Code>"], thinkingConfig: { thinkingLevel: "high", includeThoughts: true } }`
5. Stream deltas flow to UI; `</Code>` triggers extraction → `execute_code_safe` → result wrapped in `<Execute>` and fed back as `role: "execute"`.
6. Loop until `</Answer>` or stop.
7. On completion, `generateHtmlReport` is auto-called → renders into the right panel.

### 6.2 DeepAnalyze + Hybrid Router flow
Same as above but:
- `/chat/plan` is called first → analysis plan is injected into the system/user context.
- Model body is `DeepAnalyze-8B` via vLLM (OpenAI-compatible).
- On execution errors, `call_gemini_error_recovery` provides corrected code → injected as Senior Analyst guidance.
- Every 3 successful rounds, `call_gemini_checkpoint` reviews and steers.
- `router_enabled` is forced `false` when `engine === "gemini"` (frontend guard).

### 6.3 Why "stateless execution"
The system prompt explicitly warns: *"We are running each Code block in a separate session, so you can't rely on the previous session's state. So, you need to load the data and do all the preprocessing steps in each session."* This isolates failures but means each `Code` block should re-do its imports + data loading. The pre-execution validator automates this with import-patching and `df = pd.read_*` injection.

---

## 7. API Reference (cheat sheet)

| Method | Path | Body / Query | Returns |
|---|---|---|---|
| POST | `/execute` | `{code, session_id}` | `{success, result, message}` |
| POST | `/chat/plan` | `{session_id, prompt, workspace[]}` | `{plan, data_profile?, error?}` |
| POST | `/chat/completions` | `{messages, workspace, session_id, plan?, router_enabled?, provider, model, temperature}` | SSE stream |
| POST | `/chat/stop` | `{session_id}` | `{message, session_id}` |
| GET | `/workspace/files?session_id` | — | `{files: WorkspaceFile[]}` |
| GET | `/workspace/tree?session_id` | — | nested tree |
| GET | `/workspace/download?session_id&path&download` | — | file stream |
| GET | `/workspace/download-bundle?session_id&category` | — | zip |
| GET | `/workspace/preview?session_id&path&page&page_size&table_name&sheet_name` | — | typed preview |
| POST | `/workspace/upload` (multipart, `files`) | `?session_id` | `{message, files, rejected}` |
| POST | `/workspace/upload-to?session_id&dir` (multipart) | — | `{message, files, rejected}` |
| DELETE | `/workspace/file?session_id&path` | — | `{message: "deleted"}` |
| POST | `/workspace/move?session_id&src&dst_dir` | — | `{message, new_path}` |
| DELETE | `/workspace/dir?session_id&path&recursive` | — | `{message: "deleted"}` |
| DELETE | `/workspace/clear?session_id` | — | `{message}` |
| GET | `/proxy?url` | — | proxied bytes |
| POST | `/export/report` | `{messages, title, session_id}` | `{message, md, files, download_urls}` |
| POST | `/export/report/html` | `{messages, title, session_id, report_theme, artifacts?}` | `{message, html_file, view_url, rel_path, model_used, fallback}` |

---

## 8. Configuration Reference

### `.env`
```
GEMINI_API_KEY = <required-for-gemini>
GEMINI_MODEL = gemini-3-flash-preview          # default
ROUTER_ERROR_RECOVERY = true
ROUTER_CHECKPOINTS = true
ROUTER_CHECKPOINT_INTERVAL = 3
DEEPANALYZE_API_BASE = http://localhost:8000/v1
DEEPANALYZE_MODEL_PATH = DeepAnalyze-8B
DEEPANALYZE_WORKSPACE_BASE = workspace
DEEPANALYZE_FILE_SERVER_HOST = localhost
DEEPANALYZE_FILE_SERVER_PORT = 8100
DEEPANALYZE_BACKEND_HOST = 0.0.0.0
DEEPANALYZE_BACKEND_PORT = 8200
DEEPANALYZE_EXECUTION_TIMEOUT_SEC = 120
```

### `frontend/lib/config.ts`
```ts
export const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || "http://localhost:8200";
```

---

## 9. Key Design Decisions & Trade-offs

1. **Stateless code execution** → isolates failures, but every code block must re-load data. Mitigated by validator auto-injecting imports and `read_*` calls.
2. **Subprocess sandbox (not Docker)** → simple, but no isolation from the host FS. Workspace scoping + `uniquify_path` + path-traversal guard (`resolve_workspace_path`) provide containment.
3. **`.py` uploads blocked** — prevents the user from confusing the model by feeding it prior code. Workspace `generated/.deepanalyze_generated.json` is also hidden.
4. **Hardcoded 30-round intuition** — the README mentions 30 rounds; the code doesn't enforce it explicitly, relying on the model to converge via `</Answer>`. Consider adding an explicit guard if costs/latency become a concern.
5. **MPL backend forced to `Agg`** — prevents matplotlib from trying to open a display.
6. **`utf-8` open() patching** — Windows cp1252 default would crash on non-ASCII text/CSV; validator patches every `open(..., "w")` to include `encoding='utf-8'`.
7. **HTML report uses base64 inlining** — self-contained file, no asset hosting needed. Capped at 20 images.
8. **Gemini 3.1 Pro → 3 Flash fallback for reports** — graceful degradation when the larger model fails.
9. **Transfer store fallback** — sessionStorage preserves text state across HMR/reloads; the in-memory Map preserves `File` objects across SPA nav.
10. **Snapshot recovery restores `phase="complete"`** — even mid-stream snapshots show as completed, since the SSE connection is gone; users see the partial content and can continue.
11. **Hybrid Router is "Gemini-supervises-local-model"** — only active when `engine !== "gemini"`. For pure-Gemini flows, the model self-corrects via the system prompt.

---

## 10. Future Roadmap (Hooks Already in Place)

### 10.1 Power BI Integration
- **Where to plug in**: `services/export.py` is the natural home. Add a `services/powerbi.py` and a new route `POST /export/report/powerbi` that:
  1. Receives the same `{messages, title, session_id, artifacts}` payload.
  2. Uses Gemini (or a dedicated prompt) to author a **Power BI DAX measure set + a PBIX-ready schema** (tables, relationships, visual suggestions) as a downloadable `.json` bundle + a markdown playbook.
  3. Optionally render a static Power BI-style dashboard preview (HTML/CSS) inside the right panel.
- **Frontend hook**: add a "Power BI" option to the report theme selector (`lib/theme-selector.tsx`, `components/analyze-page.tsx` REPORT_THEMES) and a new button in the right panel "Export to Power BI".
- **Data flow**: workspace artifacts (CSVs in `generated/`) → Power BI's CSV import → DAX measures generated from analysis. The generated index already gives us a clean list of "publishable" files.

### 10.2 RAG over Analyzed Data
- **Indexing surface**: every `generated/` artifact (CSV/JSON/TXT/MD/PNG caption) is a candidate document. Conversation history (`messages` in the snapshot) is a secondary source.
- **Storage**: a new `services/rag.py` that:
  1. On each new artifact (`register_generated_paths` hook in `services/workspace.py`), chunks + embeds using a small embedding model (e.g., `text-embedding-3-small` or Gemini's `text-embedding-004`) and stores in a lightweight vector index (Chroma, FAISS, or sqlite-vss).
  2. Exposes `POST /rag/query` → retrieves top-k chunks → sends to Gemini with the conversation as context → returns a grounded answer with citations to workspace files.
- **Frontend hook**: a new floating "Ask the data" panel on `analyze-page.tsx` (or a new `/qa/<session_id>` route) backed by a small chat UI that calls `/rag/query`.
- **Why the pieces are already there**: the workspace is per-session and persists, the generated index is authoritative, and Gemini is already the LLM of record — the RAG layer is just an embedding store + a retrieval-augmented prompt.

### 10.3 Other natural next steps
- **Per-engine prompts** — keep `GEMINI_SYSTEM_PROMPT` for chat but add per-engine tweaks (e.g., an OpenAI/Anthropic provider).
- **Explicit max-rounds guard** in `bot_stream` for cost control.
- **Workspace cleanup cron** — currently `clear_workspace` is manual only.
- **Tool/function-calling** — replace the raw `</Code>` protocol with native tool calls for cleaner reasoning traces.
- **Multi-user** — `session_id` is client-supplied; add auth + a server-side session table.
- **DS1000 harness** — `benchmarks/ds1000/` is the legacy eval; integrate into CI.

---

## 11. Quick Debugging Pointers

- **Backend not reachable**: `http://localhost:8200` — check `logs/backend.log`.
- **Frontend not loading**: `http://localhost:3000` — check `logs/tiramisu.log`.
- **Gemini 429/5xx**: auto-retried 3x with backoff. Persistent failures show as `[Error] Gemini API request failed after 3 retries` inline in the stream.
- **Code execution hangs**: killed at `execution_timeout_sec` (default 120s); output is `[Timeout]: execution exceeded N seconds`.
- **"Unknown column" warning**: emitted as a `RouterGuidance` block; check the data profile for actual column names.
- **Report generation stuck**: cancel button in the right panel; `handleRetryReport` will re-trigger.
- **Session not recovering after refresh**: the snapshot is in `sessionStorage`, so it only survives same-tab refreshes. New tabs get a fresh session.
- **HMR loses the stream**: snapshot every 3s means worst-case 3s of lost tokens on a refresh.
