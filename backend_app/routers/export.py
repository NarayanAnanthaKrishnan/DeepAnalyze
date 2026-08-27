from __future__ import annotations

import logging
import traceback

from fastapi import APIRouter, Body, HTTPException, Request
from fastapi.responses import JSONResponse

import httpx

from ..services.exporter import export_html_report_from_body, export_report_from_body
from ..services.quota import enforce_access

log = logging.getLogger(__name__)

router = APIRouter()


@router.post("/export/report")
async def export_report(request: Request, body: dict = Body(...)):
    await enforce_access(request, body.get("session_id"))
    try:
        return JSONResponse(export_report_from_body(body))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/export/report/html")
async def export_report_html(request: Request, body: dict = Body(...)):
    await enforce_access(request, body.get("session_id"))
    try:
        result = await export_html_report_from_body(body)
        return JSONResponse(result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {exc.response.status_code} — {exc.response.text[:500]}",
        ) from exc
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("HTML report generation failed")
        raise HTTPException(
            status_code=500,
            detail=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-500:]}",
        ) from exc
