from __future__ import annotations

import logging

from fastapi import APIRouter, Body, HTTPException, Request

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

