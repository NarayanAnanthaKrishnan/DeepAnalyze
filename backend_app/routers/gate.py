from __future__ import annotations

import logging

import shutil
import time
from pathlib import Path

from fastapi import APIRouter, Body, HTTPException, Query, Request

from ..services.auth import extract_bearer_token, verify_supabase_token
from ..services.quota import (
    _ADMIN,
    _ANONYMOUS,
    claim_trial,
    enforce_access,
    is_admin_request,
    trial_status_for_request,
)
from ..settings import settings

log = logging.getLogger(__name__)

router = APIRouter()


def _email_from_request_or_none(request: Request) -> str | None:
    token = extract_bearer_token(request)
    if not token:
        return None
    claims = verify_supabase_token(token)
    if not claims:
        return None
    email = str(claims.get("email") or "").strip().lower()
    return email or None


@router.post("/gate/start")
async def gate_start(request: Request, body: dict = Body(default={})):
    """Bind a verified (OTP-verified) email to a session and claim its single trial.

    Returns 403 {"error": "trial_used"} when this email already consumed its run.
    """
    session_id = str(body.get("session_id") or "").strip()
    if not session_id:
        raise HTTPException(status_code=400, detail={"error": "missing_session_id"})

    if not settings.gate_active:
        return {"ok": True, "gate": "disabled", "email": None}

    if is_admin_request(request):
        return {"ok": True, "admin": True, "email": None}

    email = _email_from_request_or_none(request)
    if not email:
        raise HTTPException(status_code=401, detail={"error": "invalid_or_expired"})

    if email in settings.admin_emails:
        return {"ok": True, "admin": True, "email": email}

    _, info = await claim_trial(email, session_id)
    if info.consumed_at:
        raise HTTPException(status_code=403, detail={"error": "trial_used"})

    log.info("Trial claimed: %s -> session %s", email, session_id)
    return {"ok": True, "email": email, "session_id": session_id}


@router.get("/gate/status")
async def gate_status(request: Request):
    return await trial_status_for_request(request)


@router.get("/admin/stats")
async def admin_stats(request: Request):
    identity = await enforce_access(request) if settings.gate_active else _ANONYMOUS
    if identity != _ADMIN and not is_admin_request(request):
        raise HTTPException(status_code=403, detail={"error": "forbidden"})

    if settings.gate_active:
        import httpx

        url = (
            f"{settings.supabase_url.rstrip('/')}/rest/v1/trials"
            "?select=email,session_id,created_at,consumed_at"
            "&order=created_at.desc&limit=100"
        )
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url, headers={
                "apikey": settings.supabase_service_key,
                "Authorization": f"Bearer {settings.supabase_service_key}",
            })
            resp.raise_for_status()
            rows = resp.json()
    else:
        from ..services.quota import _sqlite_conn

        conn = _sqlite_conn()
        rows = [
            {
                "email": r[0],
                "session_id": r[1],
                "created_at": r[2],
                "consumed_at": r[3],
            }
            for r in conn.execute(
                "SELECT email, session_id, created_at, consumed_at FROM trials "
                "ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        ]
        conn.close()

    return {
        "total_trials": len(rows),
        "consumed": sum(1 for r in rows if r.get("consumed_at")),
        "recent": rows[:50],
        "max_rounds": settings.demo_max_rounds,
    }


@router.post("/admin/cleanup")
async def admin_cleanup(
    request: Request,
    days: int = Query(7, ge=1, le=30, description="delete sessions older than N days"),
    dry_run: bool = Query(False, description="if true, only report without deleting"),
):
    if not is_admin_request(request):
        raise HTTPException(status_code=403, detail={"error": "forbidden — X-Admin-Key required"})
    workspace_base = Path(settings.workspace_base_dir).resolve()
    if not workspace_base.exists():
        return {"deleted": 0, "kept": 0, "dry_run": dry_run, "days": days}
    cutoff = time.time() - days * 86400
    deleted = kept = 0
    candidates: list[str] = []
    for child in workspace_base.iterdir():
        if not child.is_dir():
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            kept += 1
            continue
        if mtime < cutoff:
            candidates.append(child.name)
            if not dry_run:
                try:
                    shutil.rmtree(child)
                    deleted += 1
                except OSError as exc:
                    log.warning("cleanup failed for %s: %s", child, exc)
                    kept += 1
            else:
                deleted += 1
        else:
            kept += 1
    log.info("cleanup dry_run=%s days=%s deleted=%s kept=%s", dry_run, days, deleted, kept)
    return {"deleted": deleted, "kept": kept, "dry_run": dry_run, "days": days, "candidates": candidates[:20]}

