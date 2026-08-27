from __future__ import annotations

import logging
from typing import Any

import jwt

from ..settings import settings

log = logging.getLogger(__name__)

_JWKS_ALGOS = ["ES256", "RS256"]


def _jwks_url() -> str:
    return f"{settings.supabase_url.rstrip('/')}/auth/v1/.well-known/jwks.json"


_jwks_client: jwt.PyJWKClient | None = None


def _get_jwks_client() -> jwt.PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        # PyJWKClient fetches and caches keys internally.
        _jwks_client = jwt.PyJWKClient(_jwks_url(), cache_keys=True)
    return _jwks_client


def verify_supabase_token(token: str) -> dict[str, Any] | None:
    """Verify a Supabase access token and return its claims, or None.

    Tries asymmetric verification via JWKS (ES256/RS256, default on current
    Supabase projects), then falls back to HS256 with SUPABASE_JWT_SECRET
    for legacy projects, then finally validates via Supabase Auth API
    (works without any secret, covers HS256 projects where secret isn't configured).
    """
    if not token or not settings.supabase_url.strip():
        return None

    try:
        signing_key = _get_jwks_client().get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=_JWKS_ALGOS,
            options={"verify_aud": False, "verify_iss": False},
        )
    except Exception as exc:  # noqa: BLE001 - any JWKS/signature failure falls through
        log.debug("Supabase JWKS verification failed: %s", exc)

    secret = settings.supabase_jwt_secret.strip()
    if secret:
        try:
            return jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                options={"verify_aud": False, "verify_iss": False},
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("Supabase HS256 verification failed: %s", exc)

    try:
        import httpx

        anon_key = settings.supabase_anon_key.strip() or settings.supabase_service_key.strip()
        with httpx.Client(timeout=8) as client:
            resp = client.get(
                f"{settings.supabase_url.rstrip('/')}/auth/v1/user",
                headers={
                    "apikey": anon_key,
                    "Authorization": f"Bearer {token}",
                },
            )
        if resp.status_code == 200:
            user = resp.json()
            email = user.get("email") or (user.get("user") or {}).get("email") if isinstance(user, dict) else None
            if isinstance(user, dict) and user.get("email"):
                return {"email": user["email"], "sub": user.get("id"), "role": user.get("role")}
            if isinstance(user, dict) and isinstance(user.get("user"), dict) and user["user"].get("email"):
                return {"email": user["user"]["email"], "sub": user["user"].get("id")}
            log.debug("Supabase Auth API returned no email: %s", str(user)[:300])
        else:
            log.debug("Supabase Auth API verify failed %s: %s", resp.status_code, resp.text[:300])
    except Exception as exc:  # noqa: BLE001
        log.debug("Supabase Auth API fallback failed: %s", exc)

    return None


def extract_bearer_token(request) -> str:
    header = request.headers.get("authorization") or ""
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return ""
