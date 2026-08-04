#!/usr/bin/env python
"""Build the RADAR-PD hosted user manual.

The manual source is Markdown so it can be edited with normal repo tooling.
This script keeps the generated HTML and PDF in sync without adding a Markdown
package dependency to the app runtime.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import yaml
from jinja2 import Environment, FileSystemLoader

try:
    from PIL import Image as PILImage
except Exception:  # pragma: no cover - checked at runtime
    PILImage = None

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        Preformatted,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
except Exception:  # pragma: no cover - checked at runtime
    colors = None
    getSampleStyleSheet = None


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_ROOT = REPO_ROOT / "docs" / "user_manual"
DIST_DIR = DOC_ROOT / "dist"
SCREENSHOT_DIR = DOC_ROOT / "screenshots"

CHAPTERS = [
    "index.md",
    "01_quick_start.md",
    "02_workspaces_and_sessions.md",
    "03_input_files.md",
    "04_setup_options.md",
    "05_full_radar_pd.md",
    "06_rapid_hypothesis_mode.md",
    "07_custom_cif_libraries.md",
    "08_results_plots_and_interpretation.md",
    "09_api_usage.md",
    "10_troubleshooting.md",
    "11_tutorials.md",
    "glossary.md",
]

MANIFEST_FILE = DOC_ROOT / "screenshot_manifest.yaml"
HTML_OUT = DIST_DIR / "radar_pd_hosted_user_manual.html"
PDF_OUT = DIST_DIR / "radar_pd_hosted_user_manual.pdf"
COMBINED_MD_OUT = DIST_DIR / "radar_pd_hosted_user_manual.md"

IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")

DISALLOWED_LOCAL_SETUP_PHRASES = [
    "pip install -e",
    "streamlit run",
    "git clone",
    "docker run",
    "apt install",
    "conda install",
    "local installation",
]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _chapter_paths() -> list[Path]:
    return [DOC_ROOT / name for name in CHAPTERS]


def load_manifest() -> dict:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(f"Missing screenshot manifest: {MANIFEST_FILE}")
    data = yaml.safe_load(_read_text(MANIFEST_FILE)) or {}
    data.setdefault("screenshots", [])
    return data


def _is_external(path_text: str) -> bool:
    return path_text.startswith(("http://", "https://", "mailto:", "#"))


def _normalize_image_target(target: str) -> str:
    return target.split("#", 1)[0].split("?", 1)[0].strip()


def _validate_markdown_image_links(chapter: Path, problems: list[str]) -> None:
    text = _read_text(chapter)
    for _, target in IMAGE_RE.findall(text):
        clean = _normalize_image_target(target)
        if not clean or _is_external(clean):
            continue
        candidate = (chapter.parent / clean).resolve()
        if not candidate.exists():
            problems.append(f"{chapter.name}: missing image link {target}")


def _validate_no_local_install_docs(chapter: Path, problems: list[str]) -> None:
    text = _read_text(chapter).lower()
    for phrase in DISALLOWED_LOCAL_SETUP_PHRASES:
        if phrase in text:
            problems.append(
                f"{chapter.name}: user manual contains local setup phrase {phrase!r}"
            )


def check_manual() -> tuple[int, list[str]]:
    problems: list[str] = []

    for chapter in _chapter_paths():
        if not chapter.exists():
            problems.append(f"Missing chapter: {chapter.relative_to(REPO_ROOT)}")
            continue
        _validate_markdown_image_links(chapter, problems)
        _validate_no_local_install_docs(chapter, problems)

    manifest = load_manifest()
    seen_files: set[str] = set()
    for entry in manifest.get("screenshots", []):
        name = entry.get("file")
        if not name:
            if entry.get("required", False):
                problems.append("Required screenshot entry has no file value")
            continue
        if name in seen_files:
            problems.append(f"Duplicate screenshot manifest file: {name}")
        seen_files.add(name)
        if entry.get("required", False):
            candidate = SCREENSHOT_DIR / name
            if not candidate.exists():
                problems.append(f"Missing required screenshot: {candidate}")
            elif candidate.stat().st_size <= 0:
                problems.append(f"Empty screenshot file: {candidate}")

    return len(problems), problems


def concatenate_markdown() -> str:
    chunks = []
    for chapter in _chapter_paths():
        chunks.append(f"<!-- Source: {chapter.name} -->\n\n")
        chunks.append(_read_text(chapter).rstrip())
        chunks.append("\n\n")
    return "".join(chunks).rstrip() + "\n"


def _render_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)

    def _link(match: re.Match[str]) -> str:
        label = match.group(1)
        target = html.escape(match.group(2), quote=True)
        return f'<a href="{target}">{label}</a>'

    escaped = LINK_RE.sub(_link, escaped)
    return escaped


def _is_table_separator(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", c or "") for c in cells)


def _split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _html_image(
    alt: str, target: str, chapter_dir: Path, html_output_dir: Path
) -> str:
    clean = _normalize_image_target(target)
    if _is_external(clean):
        src = html.escape(clean, quote=True)
    else:
        abs_target = (chapter_dir / clean).resolve()
        src = Path(os.path.relpath(abs_target, html_output_dir)).as_posix()
        src = html.escape(src, quote=True)
    caption = html.escape(alt)
    return (
        '<figure class="manual-figure">'
        f'<img src="{src}" alt="{caption}" loading="lazy" />'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def markdown_to_html(md: str, chapter_dir: Path, html_output_dir: Path) -> str:
    lines = md.splitlines()
    out: list[str] = []
    i = 0
    in_ul = False
    in_ol = False

    def close_lists() -> None:
        nonlocal in_ul, in_ol
        if in_ul:
            out.append("</ul>")
            in_ul = False
        if in_ol:
            out.append("</ol>")
            in_ol = False

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            close_lists()
            i += 1
            continue

        if stripped.startswith("```"):
            close_lists()
            language = stripped[3:].strip()
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            lang_class = f' class="language-{html.escape(language)}"' if language else ""
            out.append(
                f"<pre><code{lang_class}>"
                + html.escape("\n".join(code_lines))
                + "</code></pre>"
            )
            continue

        table_starts = (
            "|" in stripped
            and i + 1 < len(lines)
            and "|" in lines[i + 1]
            and _is_table_separator(lines[i + 1])
        )
        if table_starts:
            close_lists()
            header = _split_table_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                rows.append(_split_table_row(lines[i]))
                i += 1
            out.append('<div class="table-wrap"><table>')
            out.append(
                "<thead><tr>"
                + "".join(f"<th>{_render_inline(c)}</th>" for c in header)
                + "</tr></thead>"
            )
            out.append("<tbody>")
            for row in rows:
                out.append(
                    "<tr>"
                    + "".join(f"<td>{_render_inline(c)}</td>" for c in row)
                    + "</tr>"
                )
            out.append("</tbody></table></div>")
            continue

        image_match = IMAGE_RE.fullmatch(stripped)
        if image_match:
            close_lists()
            out.append(
                _html_image(
                    image_match.group(1),
                    image_match.group(2),
                    chapter_dir,
                    html_output_dir,
                )
            )
            i += 1
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            close_lists()
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
            out.append(
                f'<h{level} id="{html.escape(slug, quote=True)}">'
                f"{_render_inline(text)}</h{level}>"
            )
            i += 1
            continue

        if stripped.startswith(">"):
            close_lists()
            quote = stripped.lstrip("> ")
            out.append(f"<blockquote>{_render_inline(quote)}</blockquote>")
            i += 1
            continue

        if re.match(r"^[-*]\s+", stripped):
            if not in_ul:
                close_lists()
                out.append("<ul>")
                in_ul = True
            item = re.sub(r"^[-*]\s+", "", stripped)
            out.append(f"<li>{_render_inline(item)}</li>")
            i += 1
            continue

        if re.match(r"^\d+\.\s+", stripped):
            if not in_ol:
                close_lists()
                out.append("<ol>")
                in_ol = True
            item = re.sub(r"^\d+\.\s+", "", stripped)
            out.append(f"<li>{_render_inline(item)}</li>")
            i += 1
            continue

        close_lists()
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if (
                next_stripped.startswith(("#", "```", ">", "-", "*"))
                or re.match(r"^\d+\.\s+", next_stripped)
                or IMAGE_RE.fullmatch(next_stripped)
                or (
                    "|" in next_stripped
                    and i + 1 < len(lines)
                    and _is_table_separator(lines[i + 1])
                )
            ):
                break
            paragraph.append(next_stripped)
            i += 1
        out.append(f"<p>{_render_inline(' '.join(paragraph))}</p>")

    close_lists()
    return "\n".join(out)


def _toc_from_chapters() -> list[dict[str, str]]:
    toc = []
    for chapter in _chapter_paths():
        for line in _read_text(chapter).splitlines():
            if line.startswith("# "):
                title = line[2:].strip()
                slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
                toc.append({"title": title, "href": f"#{slug}", "source": chapter.name})
                break
    return toc


def build_html() -> Path:
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    _write_text(COMBINED_MD_OUT, concatenate_markdown())
    env = Environment(loader=FileSystemLoader(str(DOC_ROOT)), autoescape=True)
    template = env.get_template("manual_template.html")
    css = _read_text(DOC_ROOT / "manual.css")
    body_parts = []
    for chapter in _chapter_paths():
        body_parts.append(
            markdown_to_html(_read_text(chapter), chapter.parent, DIST_DIR)
        )
    html_doc = template.render(
        title="RADAR-PD Hosted User Manual",
        generated_at=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        css=css,
        toc=_toc_from_chapters(),
        body_html="\n\n".join(body_parts),
        manifest=load_manifest(),
    )
    _write_text(HTML_OUT, html_doc)
    return HTML_OUT


def _pdf_inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", escaped)
    escaped = LINK_RE.sub(r"\1", escaped)
    return escaped


def _pdf_image_flowables(path: Path, caption: str, max_width: float) -> list:
    if PILImage is None:
        return []
    if not path.exists():
        return []
    with PILImage.open(path) as im:
        width_px, height_px = im.size
    width = min(max_width, 6.6 * inch)
    height = width * (height_px / max(1, width_px))
    max_height = 5.8 * inch
    if height > max_height:
        scale = max_height / height
        width *= scale
        height *= scale
    return [
        Spacer(1, 8),
        Image(str(path), width=width, height=height),
        Paragraph(html.escape(caption), _pdf_styles()["Caption"]),
        Spacer(1, 10),
    ]


def _pdf_styles() -> dict:
    base = getSampleStyleSheet()
    base.add(
        ParagraphStyle(
            name="ManualTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=30,
            textColor=colors.HexColor("#0f2f26"),
            spaceAfter=18,
        )
    )
    base.add(
        ParagraphStyle(
            name="H1Manual",
            parent=base["Heading1"],
            fontSize=18,
            leading=22,
            textColor=colors.HexColor("#12372e"),
            spaceBefore=14,
            spaceAfter=8,
        )
    )
    base.add(
        ParagraphStyle(
            name="H2Manual",
            parent=base["Heading2"],
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#174d3f"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="BodyManual",
            parent=base["BodyText"],
            fontSize=9.5,
            leading=13.5,
            spaceAfter=6,
        )
    )
    base.add(
        ParagraphStyle(
            name="Caption",
            parent=base["BodyText"],
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#596963"),
            spaceAfter=8,
        )
    )
    return base


def markdown_to_pdf_flowables(md: str, chapter_dir: Path, styles: dict) -> list:
    flow = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1
            flow.append(
                Preformatted(
                    "\n".join(code_lines),
                    ParagraphStyle(
                        "CodeBlock",
                        parent=styles["Code"],
                        fontName="Courier",
                        fontSize=7.5,
                        leading=9,
                        backColor=colors.HexColor("#f4f7f5"),
                        borderColor=colors.HexColor("#d9e6df"),
                        borderWidth=0.5,
                        borderPadding=6,
                    ),
                )
            )
            flow.append(Spacer(1, 8))
            continue
        if (
            "|" in stripped
            and i + 1 < len(lines)
            and "|" in lines[i + 1]
            and _is_table_separator(lines[i + 1])
        ):
            header = _split_table_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i].strip() and lines[i].strip():
                rows.append(_split_table_row(lines[i]))
                i += 1
            data = [[Paragraph(_pdf_inline(cell), styles["BodyManual"]) for cell in header]]
            data.extend(
                [[Paragraph(_pdf_inline(cell), styles["BodyManual"]) for cell in row] for row in rows]
            )
            table = Table(data, hAlign="LEFT", repeatRows=1)
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eaf4ef")),
                        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#12372e")),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8e3de")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 5),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            flow.append(table)
            flow.append(Spacer(1, 8))
            continue
        image_match = IMAGE_RE.fullmatch(stripped)
        if image_match:
            target = _normalize_image_target(image_match.group(2))
            image_path = (chapter_dir / target).resolve()
            flow.extend(_pdf_image_flowables(image_path, image_match.group(1), 6.6 * inch))
            i += 1
            continue
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            if level == 1:
                flow.append(Paragraph(_pdf_inline(text), styles["H1Manual"]))
            elif level == 2:
                flow.append(Paragraph(_pdf_inline(text), styles["H2Manual"]))
            else:
                flow.append(Paragraph(f"<b>{_pdf_inline(text)}</b>", styles["BodyManual"]))
            i += 1
            continue
        if re.match(r"^[-*]\s+", stripped):
            item = re.sub(r"^[-*]\s+", "", stripped)
            flow.append(Paragraph(f"- {_pdf_inline(item)}", styles["BodyManual"]))
            i += 1
            continue
        if re.match(r"^\d+\.\s+", stripped):
            flow.append(Paragraph(_pdf_inline(stripped), styles["BodyManual"]))
            i += 1
            continue
        paragraph = [stripped]
        i += 1
        while i < len(lines):
            next_stripped = lines[i].strip()
            if not next_stripped:
                break
            if (
                next_stripped.startswith(("#", "```", "-", "*", ">"))
                or re.match(r"^\d+\.\s+", next_stripped)
                or IMAGE_RE.fullmatch(next_stripped)
                or (
                    "|" in next_stripped
                    and i + 1 < len(lines)
                    and _is_table_separator(lines[i + 1])
                )
            ):
                break
            paragraph.append(next_stripped)
            i += 1
        flow.append(Paragraph(_pdf_inline(" ".join(paragraph)), styles["BodyManual"]))
    return flow


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#67736f"))
    canvas.drawString(inch * 0.7, 0.45 * inch, "RADAR-PD Hosted User Manual")
    canvas.drawRightString(7.8 * inch, 0.45 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_pdf() -> Path:
    if getSampleStyleSheet is None:
        raise RuntimeError("reportlab is required to build the PDF manual")
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    styles = _pdf_styles()
    doc = SimpleDocTemplate(
        str(PDF_OUT),
        pagesize=letter,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.7 * inch,
        bottomMargin=0.7 * inch,
        title="RADAR-PD Hosted User Manual",
    )
    story = [
        Paragraph("RADAR-PD Hosted User Manual", styles["ManualTitle"]),
        Paragraph(
            "Web app and API guide for hosted scientific diffraction analysis.",
            styles["BodyManual"],
        ),
        Spacer(1, 12),
    ]
    for idx, chapter in enumerate(_chapter_paths()):
        if idx:
            story.append(PageBreak())
        story.extend(markdown_to_pdf_flowables(_read_text(chapter), chapter.parent, styles))
    doc.build(story, onFirstPage=_footer, onLaterPages=_footer)
    return PDF_OUT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate manual inputs")
    group.add_argument("--html", action="store_true", help="build HTML manual")
    group.add_argument("--pdf", action="store_true", help="build PDF manual")
    group.add_argument("--all", action="store_true", help="validate and build all outputs")
    args = parser.parse_args(argv)

    if args.check or args.all:
        count, problems = check_manual()
        if problems:
            for problem in problems:
                print(f"ERROR: {problem}", file=sys.stderr)
            return 1
        print("Manual check passed.")

    if args.html or args.all:
        html_path = build_html()
        print(f"Wrote {html_path}")

    if args.pdf or args.all:
        pdf_path = build_pdf()
        print(f"Wrote {pdf_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
