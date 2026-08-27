from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


os.environ.setdefault("MPLBACKEND", "Agg")


def _load_demo_env() -> None:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _get_bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


_load_demo_env()


PREVIEWABLE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".pdf",
    ".txt",
    ".doc",
    ".docx",
    ".csv",
    ".xlsx",
}


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


@dataclass(frozen=True)
class Settings:
    api_base: str = os.getenv("DEEPANALYZE_API_BASE", "http://localhost:8000/v1")
    model_path: str = os.getenv("DEEPANALYZE_MODEL_PATH", "DeepAnalyze-8B")
    workspace_base_dir: str = os.getenv("DEEPANALYZE_WORKSPACE_BASE", "workspace")
    http_server_host: str = os.getenv("DEEPANALYZE_FILE_SERVER_HOST", "localhost")
    http_server_port: int = int(os.getenv("DEEPANALYZE_FILE_SERVER_PORT", "8100"))
    backend_host: str = os.getenv("DEEPANALYZE_BACKEND_HOST", "0.0.0.0")
    backend_port: int = int(os.getenv("PORT") or os.getenv("DEEPANALYZE_BACKEND_PORT", "8200"))
    execution_timeout_sec: int = int(os.getenv("DEEPANALYZE_EXECUTION_TIMEOUT_SEC", "120"))
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
    router_error_recovery: bool = _get_bool_env("ROUTER_ERROR_RECOVERY", True)
    router_checkpoints: bool = _get_bool_env("ROUTER_CHECKPOINTS", True)
    router_checkpoint_interval: int = int(os.getenv("ROUTER_CHECKPOINT_INTERVAL", "3"))

    # ── Public-demo gating ──
    demo_max_rounds: int = int(os.getenv("DEMO_MAX_ROUNDS", "8"))
    gate_enforced: bool = _get_bool_env("GATE_ENFORCED", True)
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_jwt_secret: str = os.getenv("SUPABASE_JWT_SECRET", "")
    supabase_service_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")
    supabase_anon_key: str = os.getenv("SUPABASE_ANON_KEY", "") or os.getenv("NEXT_PUBLIC_SUPABASE_ANON_KEY", "")
    trial_db_path: str = os.getenv("TRIAL_DB_PATH", "data/trials.db")
    admin_key: str = os.getenv("ADMIN_KEY", "")
    admin_emails_raw: str = os.getenv("ADMIN_EMAILS", "")
    allowed_origins: str = os.getenv("ALLOWED_ORIGINS", "*")
    enable_proxy: bool = _get_bool_env("ENABLE_PROXY", False)

    @property
    def file_server_base(self) -> str:
        return f"http://{self.http_server_host}:{self.http_server_port}"

    @property
    def planning_enabled(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def router_active(self) -> bool:
        return bool(self.gemini_api_key.strip())

    @property
    def cors_origin_list(self) -> list[str]:
        raw = self.allowed_origins.strip()
        if not raw or raw == "*":
            return ["*"]
        return [origin.strip() for origin in raw.split(",") if origin.strip()]

    @property
    def admin_emails(self) -> frozenset[str]:
        return frozenset(
            email.strip().lower()
            for email in self.admin_emails_raw.split(",")
            if email.strip()
        )

    @property
    def gate_active(self) -> bool:
        """Gate requires Supabase credentials; GATE_ENFORCED=false disables it (local dev)."""
        if not self.gate_enforced:
            return False
        return bool(
            self.supabase_url.strip() and self.supabase_service_key.strip()
        )


settings = Settings()
