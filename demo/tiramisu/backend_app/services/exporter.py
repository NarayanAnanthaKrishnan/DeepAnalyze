from __future__ import annotations

import base64
import logging
import mimetypes
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ..settings import settings, IMAGE_EXTENSIONS
from .workspace import (
    build_download_url,
    build_preview_url,
    get_session_workspace,
    register_generated_paths,
    uniquify_path,
)

log = logging.getLogger(__name__)

GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_REPORT_MODEL = "gemini-3.1-pro-preview"
_MAX_REPORT_IMAGES = 20


def extract_sections_from_messages(messages: list[dict[str, Any]]) -> str:
    if not isinstance(messages, list):
        return ""

    parts: list[str] = []
    appendix: list[str] = []
    tag_pattern = r"<(Analyze|Understand|Code|Execute|File|Answer)>([\s\S]*?)</\1>"

    for message in messages:
        if (message or {}).get("role") != "assistant":
            continue

        content = str((message or {}).get("content") or "")
        step = 1
        for match in re.finditer(tag_pattern, content, re.DOTALL):
            tag, segment = match.groups()
            segment = segment.strip()
            if tag == "Answer":
                parts.append(f"{segment}\n")
            appendix.append(f"\n### Step {step}: {tag}\n\n{segment}\n")
            step += 1

    final_text = "".join(parts).strip()
    if appendix:
        final_text += (
            "\n\n---\n\n# Appendix: Detailed Process\n"
            + "".join(appendix).strip()
        )
    return final_text


def save_md(md_text: str, base_name: str, workspace_dir: str) -> Path:
    target_dir = Path(workspace_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    md_path = uniquify_path(target_dir / f"{base_name}.md")
    md_path.write_text(md_text, encoding="utf-8")
    return md_path


def _sanitize_filename_component(
    raw: str,
    *,
    fallback: str,
    max_length: int = 80,
) -> str:
    text = str(raw or "").strip()
    if not text:
        return fallback

    # Forbidden Windows filename characters + control characters
    text = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip(" ._")

    if not text:
        text = fallback

    if len(text) > max_length:
        text = text[:max_length].rstrip(" ._") or fallback

    return text


def _build_export_base_name(title: str, *, prefix: str, timestamp: str) -> str:
    safe_title = _sanitize_filename_component(title, fallback=prefix, max_length=80)
    return f"{safe_title}_{timestamp}"


def _to_file_meta(
    session_id: str,
    workspace_root: Path,
    file_path: Path | None,
) -> dict[str, Any] | None:
    if file_path is None:
        return None
    rel_path = file_path.relative_to(workspace_root).as_posix()
    return {
        "name": file_path.name,
        "path": rel_path,
        "download_url": build_download_url(f"{session_id}/{rel_path}"),
    }


def export_report_from_body(body: dict[str, Any]) -> dict[str, Any]:
    messages = body.get("messages", [])
    if not isinstance(messages, list):
        raise ValueError("messages must be a list")

    title = (body.get("title") or "").strip()
    session_id = body.get("session_id", "default")
    workspace_dir = get_session_workspace(session_id)
    workspace_root = Path(workspace_dir)

    md_text = extract_sections_from_messages(messages)
    if not md_text:
        md_text = "(No <Analyze>/<Understand>/<Code>/<Execute>/<Answer> sections found.)"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = _build_export_base_name(title, prefix="Report", timestamp=timestamp)

    export_dir = workspace_root / "generated" / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)

    md_path = save_md(md_text, base_name, str(export_dir))
    register_generated_paths(
        session_id,
        [md_path.relative_to(workspace_root).as_posix()],
    )

    md_meta = _to_file_meta(session_id, workspace_root, md_path)

    return {
        "message": "exported",
        "md": md_path.name,
        "files": {
            "md": md_meta,
        },
        "download_urls": {
            "md": md_meta["download_url"] if md_meta else None,
        },
    }


# ─── HTML Report Generation via Gemini 3.1 Pro ─────────────────────────


def extract_full_analysis_content(
    messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Extract all tagged sections from the conversation, preserving flow."""
    tag_pattern = r"<(Analyze|Understand|Code|Execute|File|Answer|RouterGuidance|Thinking)>([\s\S]*?)</\1>"
    sections: list[dict[str, str]] = []
    prev_was_failed_exec = False

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "")
        content = str(msg.get("content") or "")

        if role == "user":
            sections.append({"role": "user", "tag": "User", "content": content.strip()})
            prev_was_failed_exec = False
            continue

        if role not in ("assistant",):
            continue

        for match in re.finditer(tag_pattern, content, re.DOTALL):
            tag, segment = match.groups()
            segment = segment.strip()
            if not segment:
                continue

            # Filter consecutive failed Execute blocks (noise from retries)
            is_exec_error = (
                tag == "Execute"
                and any(
                    kw in segment
                    for kw in ("Traceback", "Error", "[Timeout]", "Exception")
                )
            )
            if is_exec_error and prev_was_failed_exec:
                continue  # skip duplicate retry noise
            prev_was_failed_exec = is_exec_error

            sections.append({"role": "assistant", "tag": tag, "content": segment})

    return sections


def collect_artifact_images_base64(
    session_id: str,
    artifacts: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Collect workspace images as base64 for embedding in the report."""
    workspace_dir = Path(get_session_workspace(session_id))
    seen: set[str] = set()
    results: list[dict[str, str]] = []

    def _add_image(path: Path) -> None:
        if len(results) >= _MAX_REPORT_IMAGES:
            return
        name = path.name
        if name in seen or not path.is_file():
            return
        suffix = path.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS:
            return
        seen.add(name)
        mime = mimetypes.guess_type(str(path))[0] or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        results.append({"filename": name, "base64_data": data, "mime_type": mime})

    # Scan generated/ directory
    gen_dir = workspace_dir / "generated"
    if gen_dir.is_dir():
        for p in sorted(gen_dir.rglob("*")):
            _add_image(p)

    # Also check explicitly listed artifacts
    if artifacts:
        for art in artifacts:
            art_path = workspace_dir / art.get("path", "")
            _add_image(art_path)

    # Scan workspace root for images not in generated/
    for p in sorted(workspace_dir.iterdir()):
        _add_image(p)

    return results


def _inject_base64_images(html: str, images: list[dict[str, str]]) -> str:
    """Post-process generated HTML to replace image placeholders with base64 data URIs.

    Gemini can see the images (multimodal) but cannot emit the raw base64 string.
    It uses placeholder src attributes that we replace here.

    Strategy: for every <img> whose src does NOT start with "data:", find the best
    matching image from our collection and inject the base64 data URI.
    """
    if not images:
        return html

    # Build lookup by filename (case-insensitive, without extension too)
    by_name: dict[str, dict[str, str]] = {}
    by_stem: dict[str, dict[str, str]] = {}
    for img in images:
        by_name[img["filename"].lower()] = img
        stem = Path(img["filename"]).stem.lower()
        by_stem[stem] = img

    def _find_image(src_value: str) -> dict[str, str] | None:
        """Match a src/alt/data-artifact attribute to a collected image."""
        val = src_value.strip().lower()
        # Direct filename match
        for name, img in by_name.items():
            if name in val:
                return img
        # Stem match (without extension)
        for stem, img in by_stem.items():
            if stem in val:
                return img
        return None

    def _replace_img(match: re.Match) -> str:
        tag = match.group(0)
        # Skip images that already have base64 data URIs
        if "data:" in tag and ";base64," in tag:
            return tag

        # Try to find the referenced image from src, alt, or data-artifact
        img = None
        for attr in ("src", "alt", "data-artifact", "data-src"):
            attr_match = re.search(rf'{attr}=["\']([^"\']*)["\']', tag)
            if attr_match:
                img = _find_image(attr_match.group(1))
                if img:
                    break

        if not img:
            # Fallback: try matching any filename mentioned anywhere in the tag
            for name, candidate in by_name.items():
                if name.split(".")[0] in tag.lower():
                    img = candidate
                    break

        if img:
            data_uri = f"data:{img['mime_type']};base64,{img['base64_data']}"
            # Replace or inject src attribute
            if re.search(r'src=["\']', tag):
                tag = re.sub(
                    r'src=["\'][^"\']*["\']',
                    f'src="{data_uri}"',
                    tag,
                    count=1,
                )
            else:
                tag = tag.replace("<img", f'<img src="{data_uri}"', 1)

        return tag

    return re.sub(r"<img\b[^>]*>", _replace_img, html, flags=re.IGNORECASE)


_THEME_INSTRUCTIONS: dict[str, str] = {
    "modern": (
        "MODERN theme: Clean, contemporary design. Use a sophisticated gradient "
        "color scheme (deep indigos to teals or warm sunset tones). Card-based layout "
        "with subtle shadows and rounded corners. Sans-serif typography (use Google Fonts "
        "like 'Plus Jakarta Sans' or 'Outfit'). Smooth section transitions. Glass-morphism "
        "effects for key metrics cards. Accent color for highlights and CTAs."
    ),
    "literature": (
        "FRASURBANE DIGITAL MANUSCRIPT: Warm vintage parchment (#FAF8F2 with ultra-subtle paper-fiber texture overlay) meets muted elegant nostalgic 70s energy. Elegant serif stack: Playfair Display for majestic headings + Lora for body (1.9 line-height, generous margins). Illuminated drop caps with soft ink-spread fade-in animation. Decorative horizontal rules reimagined as glowing golden vine illustrations that subtly bloom on scroll. Earth-tone palette with deep burgundy, antique gold foil, and faint holographic ink accents on data highlights. Marginalia-style footnotes that pop as elegant glass popovers. Feels like a beautifully typeset private journal from a secret library, but with modern micro-animations. Think: dark-academia meets Frasurbane — intimate, scholarly, and enchantingly timeless. You may also use perfect curves or rounded corners if you wish, and make sure that the large starting character of the report is visible."
    ),
    "academic": (
        "BRUTALIST GEOMETRIC RESEARCH LAB: Crisp white canvas with single hyper-bold forest-green accent (#0A3D2A) and Bauhaus geometric precision. Typography: STIX Two Text or Computer Modern for razor-sharp academic authority. Strict modular grid with numbered sections (1.1, 1.2…) in heavy geometric weights. Floating sidebar table of contents. Abstract 3D paper-fold effects and subtle concrete-texture overlays on figure cards. Summary box rendered as a stark, high-contrast brutalist panel. Footnote references appear as clean inline glass tooltips. Data tables and charts use raw alignment with zero decoration — let the geometry speak. Think: 1960s IBM research paper crossed with brutalist web design — rigorous, intellectual, and powerfully unconventional. Make sure that the display text is always visible while reading and when selecting text, regardless of the background you are using."
    ),
    "minimal": (
        "EXTREME BRUTALIST SWISS VOID: Absolute monochrome (#000 on #fff) with one shocking electric-cyan accent that pulses only on interaction. Zero shadows, zero rounded corners — razor-sharp geometric edges and flawless International Typographic grid. Headings in massive bold Inter or Helvetica Neue; body text (15–16px) for pure data clarity. 6rem+ whitespace everywhere. No decorative elements whatsoever — typography and perfect alignment do all the heavy lifting. Data cards are strict floating rectangles with surgical spacing. Think: Dieter Rams product manual meets high-end brutalist portfolio — austere, commanding, and breathtakingly pure. Make sure that if you add any hover effect, the content or images should still be viewable properly."
    ),
    "business": (
        "BUSINESS theme: Executive/corporate report style. Professional navy (#1a365d) "
        "and white color scheme with gold (#d4a853) or teal accents. Executive summary "
        "prominently at the top. KPI metric cards with large numbers. Charts and data "
        "visualizations given maximum space. Clean table styling with alternating row "
        "colors. Sans-serif fonts ('DM Sans' or 'Nunito Sans'). Company-report feel "
        "with header/footer. Page-break hints for printing."
    ),
    "surprise": (
        "SURPRISE ME theme: You have COMPLETE creative freedom. Pick a bold, unexpected, "
        "and design direction that nobody would expect for a data analysis report. "
        "Some ideas to inspire (but invent your own): retro terminal/hacker aesthetic, "
        "newspaper front-page layout, vintage science magazine, cyberpunk neon, hand-drawn "
        "sketchbook feel, brutalist web design, art deco, vaporwave, comic book panels, "
        "dark academia, space exploration mission brief. Whatever you choose, commit FULLY "
        "to the aesthetic and make it genuinely stunning. This should make someone say 'wow'."
    ),
}

_FRONTEND_DESIGN_SKILL = """\
---
name: frontend-design
description: Create distinctive, production-grade frontend interfaces with high design quality. Use this skill when the user asks to build web components, pages, artifacts, posters, or applications (examples include websites, landing pages, dashboards, React components, HTML/CSS layouts, or when styling/beautifying any web UI). Generates creative, polished code and UI design that avoids generic AI aesthetics.
license: Complete terms in LICENSE.txt
---

This skill guides creation of distinctive, production-grade frontend interfaces that avoid generic "AI slop" aesthetics. Implement real working code with exceptional attention to aesthetic details and creative choices.

The user provides frontend requirements: a component, page, application, or interface to build. They may include context about the purpose, audience, or technical constraints.

## Design Thinking

Before coding, understand the context and commit to a BOLD aesthetic direction:
- **Purpose**: What problem does this interface solve? Who uses it?
- **Tone**: Pick an extreme: brutally minimal, maximalist chaos, retro-futuristic, organic/natural, luxury/refined, playful/toy-like, editorial/magazine, brutalist/raw, art deco/geometric, soft/pastel, industrial/utilitarian, etc. There are so many flavors to choose from. Use these for inspiration but design one that is true to the aesthetic direction.
- **Constraints**: Technical requirements (framework, performance, accessibility).
- **Differentiation**: What makes this UNFORGETTABLE? What's the one thing someone will remember?

**CRITICAL**: Choose a clear conceptual direction and execute it with precision. Bold maximalism and refined minimalism both work - the key is intentionality, not intensity.

Then implement working code (HTML/CSS/JS, React, Vue, etc.) that is:
- Production-grade and functional
- Visually striking and memorable
- Cohesive with a clear aesthetic point-of-view
- Meticulously refined in every detail

## Frontend Aesthetics Guidelines

Focus on:
- **Typography**: Choose fonts that are beautiful, unique, and interesting. Avoid generic fonts like Arial and Inter; opt instead for distinctive choices that elevate the frontend's aesthetics; unexpected, characterful font choices. Pair a distinctive display font with a refined body font.
- **Color & Theme**: Commit to a cohesive aesthetic. Use CSS variables for consistency. Dominant colors with sharp accents outperform timid, evenly-distributed palettes.
- **Motion**: Use animations for effects and micro-interactions. Prioritize CSS-only solutions for HTML. Use Motion library for React when available. Focus on high-impact moments: one well-orchestrated page load with staggered reveals (animation-delay) creates more delight than scattered micro-interactions. Use scroll-triggering and hover states that surprise.
- **Spatial Composition**: Unexpected layouts. Asymmetry. Overlap. Diagonal flow. Grid-breaking elements. Generous negative space OR controlled density.
- **Backgrounds & Visual Details**: Create atmosphere and depth rather than defaulting to solid colors. Add contextual effects and textures that match the overall aesthetic. Apply creative forms like gradient meshes, noise textures, geometric patterns, layered transparencies, dramatic shadows, decorative borders, custom cursors, and grain overlays.

NEVER use generic AI-generated aesthetics like overused font families (Inter, Roboto, Arial, system fonts), cliched color schemes (particularly purple gradients on white backgrounds), predictable layouts and component patterns, and cookie-cutter design that lacks context-specific character.

Interpret creatively and make unexpected choices that feel genuinely designed for the context. No design should be the same. Vary between light and dark themes, different fonts, different aesthetics. NEVER converge on common choices (Space Grotesk, for example) across generations.

**IMPORTANT**: Match implementation complexity to the aesthetic vision. Maximalist designs need elaborate code with extensive animations and effects. Minimalist or refined designs need restraint, precision, and careful attention to spacing, typography, and subtle details. Elegance comes from executing the vision well.

Remember: Gemini is capable of extraordinary creative work. Don't hold back, show what can truly be created when thinking outside the box and committing fully to a distinctive vision.
"""


def _build_html_report_prompt(
    analysis_content: list[dict[str, str]],
    images: list[dict[str, str]],
    report_theme: str,
    title: str,
) -> str:
    """Build the mega-prompt for Gemini to generate the HTML report."""

    # Format the analysis content — exclude Code blocks, keep insights and outputs
    content_parts: list[str] = []
    for section in analysis_content:
        tag = section["tag"]
        content = section["content"]
        if tag == "User":
            content_parts.append(f"## USER REQUEST\n{content}\n")
        elif tag in ("Analyze", "Understand", "Thinking"):
            content_parts.append(f"## AGENT REASONING\n{content}\n")
        elif tag == "Execute":
            content_parts.append(f"## EXECUTION OUTPUT\n```\n{content}\n```\n")
        elif tag == "Answer":
            content_parts.append(f"## FINAL ANSWER\n{content}\n")
        elif tag == "File":
            content_parts.append(f"## GENERATED FILES\n{content}\n")
        elif tag == "RouterGuidance":
            content_parts.append(f"## SENIOR ANALYST GUIDANCE\n{content}\n")
        # NOTE: Code blocks are intentionally excluded from the report content

    analysis_text = "\n---\n\n".join(content_parts)

    # Image reference list — tell Gemini to use placeholder src we'll replace
    image_refs = ""
    if images:
        img_list = "\n".join(
            f"  - {img['filename']}"
            for img in images
        )
        image_refs = f"""
## Available Visualizations/Charts (CRITICAL — MUST USE ALL)
The following images were generated during the analysis. They are provided as inline
images in this request so you can SEE them. You MUST include ALL of them prominently
in the report — they are the most important part.

For each image, use an <img> tag with the EXACT filename as the src attribute:
  <img src="{{filename}}" alt="descriptive caption" class="..." />

For example: <img src="scatter_plot.png" alt="Scatter plot of X vs Y" />

The system will automatically replace these filenames with the actual embedded image
data after generation. Just use the exact filename as src.

Images to embed:
{img_list}

Place each image at the most contextually appropriate location in the report.
Add descriptive figure captions and interpretations for each visualization.
Make the visualizations the CENTERPIECE of the report — they should be large,
prominent, and beautifully framed within the design.
"""

    theme_instruction = _THEME_INSTRUCTIONS.get(
        report_theme, _THEME_INSTRUCTIONS["modern"]
    )

    report_title = title.strip() or "Data Analysis Report"

    return textwrap.dedent(f"""\
        You are an elite frontend designer AND expert data analyst creating a stunning
        web-based report. Your task: transform the raw analysis data below into a
        beautiful, self-contained HTML page that a data analyst would be proud to share.

        # REPORT TITLE
        {report_title}

        # DESIGN THEME
        {theme_instruction}

        # FRONTEND DESIGN SKILL
        Use your frontend design skill for achieving below results:
        {_FRONTEND_DESIGN_SKILL}

        # TECHNICAL REQUIREMENTS
        - Output a single, complete, self-contained HTML file
        - Use Tailwind CSS via CDN: <script src="https://cdn.tailwindcss.com"></script>
        - Use Google Fonts via CDN for typography
        - For images: use the exact filename as the src attribute (they will be
          replaced with base64 data URIs automatically after generation)
        - Responsive design (works on mobile and desktop)
        - Print-friendly (use @media print for clean printing)
        - No external JavaScript dependencies beyond Tailwind
        - Start with <!DOCTYPE html> and include proper <head> with meta tags

        Always think extra and ultra unique and creative designs, layout and big fonts and never present any generic design.

        # CONTENT INSTRUCTIONS
        - Read through ALL the analysis content below carefully
        - Extract the key insights, findings, and conclusions and detailed report.
        - Filter out debugging noise, error messages, and redundant retry attempts
        - This is just a high level guidelines, always use your judgement and can structure the report in any way considering the dataset and analysis.
        - Add your own knowledge and insights to the report to make it detailed and comprehensive.
        - DO NOT include any code snippets or code blocks in the report — this is
          for a non-technical audience. Focus on insights, not implementation.
        - Make data visualizations (embedded images) the CENTERPIECE of the report —
          they should be large, prominent, and beautifully framed
        - Make sure the images are large enough so as the visualizations are clearly readable and understandable.
        - Add contextual captions and interpretations for each visualization
        - Use data tables where appropriate to present key metrics
        - Include ALL generated artifacts/visualizations — do not skip any
        - The target audience is a data analyst but make it very easy to understand and present in a creative way.
                
        {image_refs}

        # FULL ANALYSIS CONTENT
        Below is the complete agent analysis session. Parse it, extract what matters,
        and transform it into the report:

        {analysis_text}

        # OUTPUT
        Return ONLY the complete HTML code. No explanations, no markdown fences — just
        the raw HTML starting with <!DOCTYPE html>.
    """)


async def _call_gemini_report(
    prompt_text: str,
    images: list[dict[str, str]],
) -> str:
    """Call Gemini 3.1 Pro Preview to generate the HTML report."""
    url = GEMINI_API_URL.format(model=GEMINI_REPORT_MODEL)

    parts: list[dict[str, Any]] = [{"text": prompt_text}]
    for img in images:
        parts.append(
            {
                "inline_data": {
                    "mime_type": img["mime_type"],
                    "data": img["base64_data"],
                }
            }
        )

    payload: dict[str, Any] = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": 65536,
            "thinkingConfig": {"thinkingLevel": "high"},
        },
    }

    timeout = httpx.Timeout(connect=30, read=300, write=30, pool=30)
    async with httpx.AsyncClient(timeout=timeout) as client:
        log.info("Calling Gemini %s for HTML report (%d parts)...", GEMINI_REPORT_MODEL, len(parts))
        resp = await client.post(
            url,
            headers={"x-goog-api-key": settings.gemini_api_key},
            json=payload,
        )
        if resp.status_code != 200:
            log.error("Gemini API error %d: %s", resp.status_code, resp.text[:1000])
        resp.raise_for_status()
        data = resp.json()
        log.info("Gemini report response received, candidates=%d", len(data.get("candidates", [])))

    # Extract text from response — skip thinking parts
    text_parts: list[str] = []
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part and not part.get("thought"):
                text_parts.append(part["text"])

    raw = "".join(text_parts).strip()

    # Strip markdown fences if present
    fence_match = re.search(r"```html\s*([\s\S]*?)\s*```", raw)
    if fence_match:
        return fence_match.group(1).strip()

    # Try to extract from <!DOCTYPE or <html
    doctype_match = re.search(r"(<!DOCTYPE[\s\S]*)", raw, re.IGNORECASE)
    if doctype_match:
        return doctype_match.group(1).strip()

    html_match = re.search(r"(<html[\s\S]*)", raw, re.IGNORECASE)
    if html_match:
        return html_match.group(1).strip()

    return raw


async def export_html_report_from_body(body: dict[str, Any]) -> dict[str, Any]:
    """Generate a beautiful HTML report from analysis messages via Gemini."""
    if not settings.gemini_api_key.strip():
        raise ValueError("GEMINI_API_KEY is not configured")

    messages = body.get("messages", [])
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    title = (body.get("title") or "").strip()
    session_id = body.get("session_id", "default")
    report_theme = body.get("report_theme", "modern")
    artifacts = body.get("artifacts") or []

    workspace_dir = get_session_workspace(session_id)
    workspace_root = Path(workspace_dir)

    # 1. Extract analysis content
    analysis_content = extract_full_analysis_content(messages)
    if not analysis_content:
        raise ValueError("No analysis content found in messages")

    # 2. Collect images
    images = collect_artifact_images_base64(session_id, artifacts)
    log.info(
        "HTML report: %d sections, %d images for session %s",
        len(analysis_content),
        len(images),
        session_id,
    )

    # 3. Build prompt
    prompt = _build_html_report_prompt(analysis_content, images, report_theme, title)

    # 4. Call Gemini to generate HTML
    html_content = await _call_gemini_report(prompt, images)

    # 5. Post-process: inject actual base64 data URIs into image placeholders
    html_content = _inject_base64_images(html_content, images)
    log.info("Post-processed HTML: injected base64 images")

    # 6. Save HTML
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = _build_export_base_name(title, prefix="Report", timestamp=timestamp)
    export_dir = workspace_root / "generated" / "reports"
    export_dir.mkdir(parents=True, exist_ok=True)
    html_path = uniquify_path(export_dir / f"{base_name}.html")
    html_path.write_text(html_content, encoding="utf-8")

    rel_path = html_path.relative_to(workspace_root).as_posix()
    register_generated_paths(session_id, [rel_path])

    # Use preview URL (inline, no download header) so browser renders the HTML
    view_url = build_preview_url(f"{session_id}/{rel_path}")

    return {
        "message": "html_report_generated",
        "html_file": html_path.name,
        "view_url": view_url,
        "rel_path": rel_path,
    }
