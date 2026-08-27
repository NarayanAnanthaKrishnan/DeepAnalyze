from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from fastapi import HTTPException, Request

from .auth import extract_bearer_token, verify_supabase_token
from ..settings import settings

log = logging.getLogger(__name__)

_TABLE = "trials"

_ANONYMOUS = "__anonymous__"
_ADMIN = "__admin__"


@dataclass(frozen=True)
class TrialInfo:
    email: str
    session_id: str | None
    created_at: str | None
    consumed_at: str | None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_admin_request(request: Request) -> bool:
    key = settings.admin_key.strip()
    if key and request.headers.get("x-admin-key", "") == key:
        return True
    token = extract_bearer_token(request)
    if token:
        claims = verify_supabase_token(token)
        email = str(claims.get("email") or "").lower() if claims else ""
        if email and email in settings.admin_emails:
            return True
    return False


def _email_from_request(request: Request) -> str:
    """Return the verified email for this request, raising 401 if invalid."""
    token = extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=401, detail={"error": "unauthenticated"})
    claims = verify_supabase_token(token)
    if not claims:
        raise HTTPException(status_code=401, detail={"error": "invalid_or_expired"})
    email = str(claims.get("email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=401, detail={"error": "no_email_claim"})
    return email


# ─── Supabase Postgres backend ────────────────────────────────────────


def _sb_headers() -> dict[str, str]:
    return {
        "apikey": settings.supabase_service_key,
        "Authorization": f"Bearer {settings.supabase_service_key}",
        "Content-Type": "application/json",
    }


async def _sb_get_trial(email: str) -> TrialInfo | None:
    url = (
        f"{settings.supabase_url.rstrip('/')}/rest/v1/{_TABLE}"
        f"?select=email,session_id,created_at,consumed_at&email=eq.{email}"
    )
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers=_sb_headers())
        resp.raise_for_status()
        rows = resp.json()
    except Exception as exc:  # noqa: BLE001 - fall back to open gate on infra errors?
        # Fail CLOSED: if Supabase is configured but unreachable, don't hand out free runs.
        log.error("Supabase trial lookup failed: %s", exc)
        raise HTTPException(
            status_code=503, detail={"error": "quota_backend_unavailable"}
        ) from exc
    if not rows:
        return None
    row = rows[0]
    return TrialInfo(
        email=row.get("email") or email,
        session_id=row.get("session_id"),
        created_at=row.get("created_at"),
        consumed_at=row.get("consumed_at"),
    )


async def _sb_insert(email: str, session_id: str) -> bool:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{_TABLE}"
    payload = {"email": email, "session_id": session_id, "created_at": _utc_now_iso()}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(url, headers=_sb_headers(), json=payload)
    except Exception as exc:  # noqa: BLE001
        log.error("Supabase trial insert failed: %s", exc)
        raise HTTPException(
            status_code=503, detail={"error": "quota_backend_unavailable"}
        ) from exc
    if resp.status_code in (200, 201):
        return True
    if resp.status_code == 409:  # unique violation (concurrent claim race)
        return False
    log.error("Supabase trial insert unexpected status %s: %s", resp.status_code, resp.text[:300])
    raise HTTPException(status_code=503, detail={"error": "quota_backend_unavailable"})


async def _sb_patch(
    email: str,
    values: dict,
    only_unconsumed: bool = True,
) -> None:
    url = f"{settings.supabase_url.rstrip('/')}/rest/v1/{_TABLE}?email=eq.{email}"
    if only_unconsumed:
        url += "&consumed_at=is.null"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.patch(url, headers=_sb_headers(), json=values)
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        log.error("Supabase trial patch failed: %s", exc)


# ─── SQLite fallback (local dev without Supabase) ─────────────────────

_sqlite_lock = threading.Lock()


def _sqlite_conn() -> sqlite3.Connection:
    from pathlib import Path

    db_path = Path(settings.trial_db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {_TABLE} (
            email TEXT PRIMARY KEY,
            session_id TEXT,
            created_at TEXT,
            consumed_at TEXT
        )"""
    )
    return conn


async def _sqlite_get_trial(email: str) -> TrialInfo | None:
    with _sqlite_lock:
        conn = _sqlite_conn()
        try:
            row = conn.execute(
                f"SELECT email, session_id, created_at, consumed_at FROM {_TABLE} WHERE email = ?",
                (email,),
            ).fetchone()
        finally:
            conn.close()
    if not row:
        return None
    return TrialInfo(email=row[0], session_id=row[1], created_at=row[2], consumed_at=row[3])


async def _sqlite_insert(email: str, session_id: str) -> bool:
    with _sqlite_lock:
        conn = _sqlite_conn()
        try:
            conn.execute(
                f"INSERT INTO {_TABLE} (email, session_id, created_at) VALUES (?, ?, ?)",
                (email, session_id, _utc_now_iso()),
            )
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()


async def _sqlite_patch(email: str, values: dict, only_unconsumed: bool = True) -> None:
    sets = ", ".join(f"{k} = ?" for k in values)
    sql = f"UPDATE {_TABLE} SET {sets} WHERE email = ?"
    params = [*values.values(), email]
    if only_unconsumed:
        sql += " AND consumed_at IS NULL"
    with _sqlite_lock:
        conn = _sqlite_conn()
        try:
            conn.execute(sql, params)
            conn.commit()
        finally:
            conn.close()


# ─── Public API ───────────────────────────────────────────────────────


def _backend():
    return (_sb_get_trial, _sb_insert, _sb_patch) if settings.gate_active else (
        _sqlite_get_trial,
        _sqlite_insert,
        _sqlite_patch,
    )


async def get_trial(email: str) -> TrialInfo | None:
    getter, _, _ = _backend()
    return await getter(email)


async def claim_trial(email: str, session_id: str) -> tuple[bool, TrialInfo]:
    """Claim a trial for email+session.

    Returns (newly_created, info). Existing unconsumed trials may rebind to a
    new session_id (user abandoned first attempt before running anything).
    Consumed trials are immutable.
    """
    existing = await get_trial(email)
    if existing:
        if existing.consumed_at:
            return False, existing
        if (existing.session_id or "") != session_id:
            _, _, patcher = _backend()
            await patcher(email, {"session_id": session_id}, only_unconsumed=True)
            existing = TrialInfo(
                email=existing.email,
                session_id=session_id,
                created_at=existing.created_at,
                consumed_at=None,
            )
        return False, existing

    _, inserter, _ = _backend()
    if await inserter(email, session_id):
        info = await get_trial(email)
        return True, info or TrialInfo(email, session_id, None, None)

    # Lost a concurrent-claim race — treat as existing.
    existing = await get_trial(email)
    if existing and not existing.consumed_at and (existing.session_id or "") != session_id:
        _, _, patcher = _backend()
        await patcher(email, {"session_id": session_id})
        existing = TrialInfo(email, session_id, existing.created_at, None)
    return False, existing or TrialInfo(email, session_id, None, None)


async def consume_trial(email: str, session_id: str) -> None:
    """Mark a trial consumed (idempotent). No-op for anonymous/admin callers."""
    if email in (_ANONYMOUS, _ADMIN):
        return
    _, _, patcher = _backend()
    await patcher(email, {"consumed_at": _utc_now_iso(), "session_id": session_id})


async def enforce_access(request: Request, session_id: str | None = None) -> str:
    """FastAPI dependency-style guard. Returns the acting identity (email).

    Raises 401/403/503 HTTPExceptions on denial.
    """
    if not settings.gate_active:
        return _ANONYMOUS
    if is_admin_request(request):
        return _ADMIN

    email = _email_from_request(request)
    if email in settings.admin_emails:
        return email

    info = await get_trial(email)
    if not info:
        raise HTTPException(status_code=403, detail={"error": "no_trial"})

    if session_id and (info.session_id or "") != session_id:
        if info.consumed_at:
            raise HTTPException(status_code=403, detail={"error": "trial_used"})
        # Not yet consumed: allow rebinding (fresh attempt / second tab).
        _, _, patcher = _backend()
        await patcher(email, {"session_id": session_id})

    return email


async def trial_status_for_request(request: Request) -> dict:
    authenticated = False
    used: bool | None = None
    email: str | None = None
    if settings.gate_active:
        token = extract_bearer_token(request)
        if token:
            claims = verify_supabase_token(token)
            if claims and claims.get("email"):
                email = str(claims["email"]).lower()
                authenticated = True
                if email in settings.admin_emails or is_admin_request(request):
                    used = False
                else:
                    info = await get_trial(email)
                    used = bool(info and info.consumed_at)
    return {
        "configured": settings.gate_active,
        "authenticated": authenticated,
        "used": used,
        "email": email,
    }
