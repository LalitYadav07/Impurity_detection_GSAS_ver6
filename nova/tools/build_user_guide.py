"""Build the bundled task guide without network access or scientific jobs.

Run from any directory: python nova/tools/build_user_guide.py
To add a reviewed UI capture to a task, use e.g.:
  --capture current-live-setup /path/to/capture.png
  --capture-context "Local 0.6.31 UI fixture; no scientific jobs."
Capture IDs current-<task-id> override that task's historical image or diagram.
Original image bytes, date, version and context are retained in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re

from user_guide_content import DOWNLOADS, TASKS, TROUBLESHOOTING

NOVA = Path(__file__).resolve().parents[1]
DATE = "2026-11-05"

CSS = r"""
:root{color-scheme:light;--ink:#203b32;--green:#195a40;--muted:#62776d;--line:#d4e2d9;--wash:#f2f6f3;--amber:#87541e;--soft:#edf5ee}
*{box-sizing:border-box}html{scroll-behavior:smooth;scroll-padding-top:24px}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}a{color:var(--green);text-underline-offset:3px}button,input{font:inherit}button,a,input,summary{touch-action:manipulation}button:focus-visible,a:focus-visible,summary:focus-visible,input:focus-visible{outline:3px solid #d38330;outline-offset:4px}.skip{position:absolute;left:16px;top:-100px;background:#fff;padding:12px;z-index:9}.skip:focus{top:10px}header.page-head{background:white;border-bottom:1px solid var(--line);padding:22px max(24px,calc((100vw - 1440px)/2));display:flex;justify-content:space-between;align-items:center;gap:20px}.brand{font-weight:760;font-size:20px;text-decoration:none}.version{font-size:13px;color:var(--muted)}.shell{max-width:1488px;margin:32px auto 70px;padding:0 24px;display:grid;grid-template-columns:226px minmax(0,1fr);gap:36px}.sidebar{position:sticky;top:20px;align-self:start;max-height:calc(100vh - 40px);overflow:auto;padding-right:12px}.sidebar label{font-size:13px;font-weight:700}.sidebar input{width:100%;border:1px solid #aebfb4;border-radius:6px;background:white;padding:10px 11px;margin:7px 0 16px}.nav-group{font-size:11px;text-transform:uppercase;letter-spacing:.1em;color:var(--muted);margin:20px 0 5px}.sidebar nav a{display:flex;gap:9px;align-items:baseline;padding:7px 0;text-decoration:none;font-size:13px;line-height:1.4}.nav-number{color:#788b80;font-size:11px;min-width:16px}.sidebar .utility{display:block;font-size:13px;margin-top:13px}.nav-note{font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:16px;margin-top:20px}main{min-width:0}.intro{padding:0 0 28px}.kicker{text-transform:uppercase;font-size:11px;letter-spacing:.11em;font-weight:760;color:var(--green);margin:0 0 10px}h1{font-size:clamp(31px,3.5vw,48px);line-height:1.12;letter-spacing:-.025em;margin:0 0 16px;max-width:820px}h2{font-size:clamp(24px,2.4vw,31px);line-height:1.2;letter-spacing:-.018em;margin:0 0 12px}h3{font-size:17px;line-height:1.35;margin:0 0 9px}p{margin:0 0 13px}.lead{font-size:18px;max-width:850px;color:#4c6759}.routes{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:22px 0}.route{padding:17px;border:1px solid var(--line);border-radius:8px;background:white;text-decoration:none;line-height:1.5}.route strong{display:block;font-size:16px;margin:0 0 6px}.route span{font-size:13px;color:var(--muted)}.scope-note{background:#e7f0e9;border-left:3px solid var(--green);padding:13px 16px;font-size:13px}.search-status{font-size:13px;color:var(--muted);min-height:1em}.empty{padding:20px;background:white;border:1px solid var(--line);border-radius:8px}.task{background:#fff;border:1px solid var(--line);border-radius:10px;padding:28px;margin:0 0 26px;scroll-margin-top:22px}.task-head{margin-bottom:20px}.task-head p:last-child{color:#546e60;max-width:850px;margin:0}.task-body{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:28px;align-items:start}.steps{counter-reset:step;list-style:none;padding:0;margin:0 0 16px}.steps li{counter-increment:step;position:relative;padding-left:32px;margin:0 0 16px;font-size:15px;line-height:1.65}.steps li:before{content:counter(step);position:absolute;left:0;top:2px;width:22px;height:22px;display:grid;place-items:center;background:#edf4ee;color:var(--green);font-size:12px;font-weight:750;border-radius:50%}.expected{border-left:3px solid #6b9d76;background:#f2f7f2;padding:13px 15px;font-size:14px}.expected strong{display:block;margin-bottom:4px}.expected p{margin:0}.screen{margin:0;min-width:0}.zoom{display:block;padding:0;width:100%;border:1px solid #d2e0d7;border-radius:7px;background:var(--wash);overflow:hidden;cursor:zoom-in}.zoom img{display:block;width:100%;height:auto}.screen figcaption{font-size:12px;line-height:1.55;color:var(--muted);padding-top:9px}.capture-badge{display:inline-block;background:#edf4ee;color:#3f6650;padding:2px 6px;border-radius:4px;font-size:11px;font-weight:650;margin-bottom:5px}.capture-badge.historical{background:#fff2dc;color:#7a561f}.screen-context{display:block;margin-top:4px}.task-details{margin-top:20px;border-top:1px solid var(--line)}details{border-bottom:1px solid #e0e8e2}summary{cursor:pointer;padding:13px 2px;color:var(--green);font-size:14px;font-weight:650;line-height:1.45}summary::marker{color:#789381}.detail-body{font-size:14px;padding:1px 0 15px 19px;max-width:950px}.detail-body p{margin:0}.extra-shots{display:grid;grid-template-columns:1fr 1fr;gap:20px;padding:8px 0 18px}.prompt{border:1px solid var(--line);border-radius:6px;background:#f5f8f5;padding:14px;margin-top:18px}.prompt-head{display:flex;gap:10px;align-items:center;justify-content:space-between;margin-bottom:8px}.prompt-head strong{font-size:13px}.prompt button,.dialog-bar button{color:var(--green);background:white;border:1px solid #b9cdbf;border-radius:5px;padding:6px 11px;cursor:pointer;font-size:12px}pre{white-space:pre-wrap;word-break:break-word;font:13px/1.65 ui-monospace,SFMono-Regular,Menlo,monospace;margin:0}.prompt p{font-size:12px;color:var(--muted);margin:9px 0 0}.task-footer{display:flex;justify-content:space-between;gap:16px;border-top:1px solid #e3eae5;margin-top:22px;padding-top:14px;font-size:12px}.task-footer a{text-decoration:none}.diagram{margin:0;padding:20px;background:linear-gradient(145deg,#edf5ef,#f8faf8);border:1px solid var(--line);border-radius:8px}.diagram-title{font-size:12px;letter-spacing:.04em;color:var(--green);font-weight:700;margin:0 0 16px}.flow{list-style:none;margin:0;padding:0}.flow li{position:relative;margin-bottom:24px;background:white;border:1px solid #bcd1c2;border-radius:7px;padding:13px 16px;line-height:1.45;font-size:14px}.flow li:not(:last-child):after{content:'↓';position:absolute;bottom:-23px;left:calc(50% - 5px);color:#679577;font-size:18px}.flow li:last-child{margin-bottom:0}.flow strong{display:block}.flow span{display:block;color:var(--muted);font-size:12px;margin-top:4px}.diagram figcaption{font-size:12px;color:var(--muted);margin-top:14px}.downloads,.help{background:white;border:1px solid var(--line);border-radius:10px;padding:28px;margin:0 0 26px}.download-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:18px 0}.download-item{border:1px solid var(--line);border-radius:7px;padding:14px;font-size:13px}.download-item a{font-weight:700}.download-item p{font-size:12px;color:var(--muted);margin:7px 0 0}.reference-callout{background:#fff7e9;border-left:3px solid #cfad6d;padding:14px 16px;font-size:14px}.foot{font-size:12px;color:var(--muted);padding:4px 0 20px}.toast{position:fixed;bottom:20px;left:50%;transform:translateX(-50%);padding:10px 16px;background:#183e2c;color:white;border-radius:6px;z-index:20;font-size:14px}dialog{border:0;border-radius:9px;padding:0;width:min(1500px,97vw);max-width:97vw;max-height:96vh;background:#fff}dialog::backdrop{background:#102a20e8}.dialog-bar{position:sticky;top:0;display:flex;gap:12px;align-items:center;border-bottom:1px solid var(--line);background:white;padding:12px 16px}.dialog-bar strong{font-size:14px;flex:1}.dialog-image{overflow:auto;max-height:calc(94vh - 150px)}.dialog-image img{display:block;width:100%;height:auto}.dialog-image.actual img{width:auto;max-width:none}.dialog-caption{font-size:12px;line-height:1.6;color:var(--muted);padding:12px 16px;margin:0}.screen figcaption a{white-space:nowrap}[hidden]{display:none!important}
@media(max-width:1120px){.shell{grid-template-columns:185px minmax(0,1fr);gap:24px}.task{padding:22px}.task-body{grid-template-columns:1fr}.task-visual{order:2}.directions{order:1}.routes{gap:9px}.route{padding:13px}}
@media(max-width:740px){header.page-head{padding:16px 20px;align-items:flex-start}.version{text-align:right;font-size:11px}.shell{display:block;margin-top:22px;padding:0 16px}.sidebar{position:static;max-height:none;padding:0 0 22px}.sidebar nav{display:grid;grid-template-columns:1fr 1fr;gap:0 15px}.nav-group{grid-column:1/-1;margin-top:11px}.sidebar input{margin-bottom:3px}.nav-note{margin:10px 0;padding-top:10px}.sidebar .utility{display:inline-block;margin:6px 14px 0 0}.routes{grid-template-columns:1fr}.route strong{margin-bottom:3px}.task{padding:20px;margin-bottom:20px}.task-body{gap:18px}.download-grid,.extra-shots{grid-template-columns:1fr}.downloads,.help{padding:20px}.task-footer{flex-direction:column;gap:8px}.dialog-bar{flex-wrap:wrap}.dialog-bar strong{flex-basis:100%}.dialog-image{max-height:70vh}h1{font-size:34px}}
@media(prefers-reduced-motion:reduce){html{scroll-behavior:auto}}
@media print{body{background:white}.page-head,.sidebar,.route,.task-footer,.prompt button,.zoom-hint{display:none}.shell{display:block;margin:0;padding:0}.task,.downloads,.help{border:0;padding:18px 0;break-inside:avoid}.task-body{grid-template-columns:1fr 1fr}.task-visual{order:0}details{display:block}.detail-body{display:block}details:not([open])>*:not(summary){display:block}.screen{break-inside:avoid}.foot{border-top:1px solid #aaa}a{color:#111}}
"""

JS = r"""
(() => {
  const query = document.getElementById('task-search');
  const tasks = [...document.querySelectorAll('article.task')];
  const navigation = [...document.querySelectorAll('[data-task-link]')];
  const status = document.getElementById('search-status');
  const empty = document.getElementById('search-empty');
  function filter() {
    const terms = query.value.toLowerCase().trim().split(/\s+/).filter(Boolean);
    let count = 0;
    tasks.forEach(task => {
      const match = terms.every(term => task.textContent.toLowerCase().includes(term));
      task.hidden = !match;
      navigation.filter(link => link.dataset.taskLink === task.id).forEach(link => link.hidden = !match);
      if (match) count++;
      task.querySelectorAll('details').forEach(detail => {
        if (terms.length && match && terms.every(term => detail.textContent.toLowerCase().includes(term))) {
          if (!detail.open) { detail.open = true; detail.dataset.searchOpened = 'true'; }
        } else if (detail.dataset.searchOpened) { detail.open = false; delete detail.dataset.searchOpened; }
      });
    });
    document.querySelectorAll('.nav-group').forEach(group => {
      group.hidden = !navigation.some(link => link.dataset.group === group.dataset.group && !link.hidden);
    });
    status.textContent = terms.length ? `${count} of ${tasks.length} tasks match. Clear search to show all tasks.` : '';
    empty.hidden = count > 0;
  }
  query.addEventListener('input', filter);
  function reveal(target) {
    if (!target) return;
    const task = target.closest('article.task');
    if (target.hidden || (task && task.hidden)) { query.value = ''; filter(); }
    if (target.tagName === 'DETAILS') target.open = true;
  }
  document.querySelectorAll('a[href^="#"]').forEach(link => link.addEventListener('click', () => {
    reveal(document.getElementById(link.getAttribute('href').slice(1)));
  }));
  window.addEventListener('hashchange', () => reveal(document.getElementById(location.hash.slice(1))));
  reveal(document.getElementById(location.hash.slice(1)));
  const toast = document.getElementById('copy-status'); let toastTimer;
  function notify(message) {
    toast.textContent = message; toast.hidden = false; window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => toast.hidden = true, 4000);
  }
  document.querySelectorAll('[data-copy]').forEach(button => button.addEventListener('click', async () => {
    const source = document.getElementById(button.dataset.copy);
    try { await navigator.clipboard.writeText(source.textContent); notify('Example copied. Review it for your experiment.'); }
    catch (_) {
      const range = document.createRange(); range.selectNodeContents(source);
      const selection = window.getSelection(); selection.removeAllRanges(); selection.addRange(range);
      notify('Example selected. Use your system Copy command.');
    }
  }));
  const dialog = document.getElementById('image-dialog');
  const image = document.getElementById('full-image');
  const holder = document.getElementById('dialog-image');
  let opener;
  document.querySelectorAll('[data-zoom]').forEach(button => button.addEventListener('click', () => {
    opener = button; holder.classList.remove('actual'); image.src = button.dataset.zoom;
    image.alt = button.querySelector('img').alt;
    document.getElementById('image-title').textContent = image.alt;
    document.getElementById('image-caption').textContent = button.closest('figure').querySelector('figcaption').textContent;
    document.getElementById('actual-size').textContent = 'Actual pixels'; dialog.showModal();
  }));
  document.getElementById('close-image').addEventListener('click', () => dialog.close());
  document.getElementById('actual-size').addEventListener('click', event => {
    const actual = holder.classList.toggle('actual'); event.target.textContent = actual ? 'Fit to window' : 'Actual pixels';
  });
  dialog.addEventListener('close', () => { if (opener) opener.focus(); });
  dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); });
})();
"""

DIAGRAMS = {
    "contact": ("A private report, separate from experiment data", [("Contact developer", "Write freely and choose screenshots."),("Review and send", "Verify the previews and sender identity."),("Wait for delivery reference", "Galaxy stores and privately shares the report."),("Retry or download if needed", "Keep the draft until delivery is confirmed.")]),
    "models": ("From one parent to a reviewed comparison", [
        ("Inspect exact parent", "Confirm the phase, sites and histogram."),
        ("Discover finite candidates", "Review k, child models, modes and omissions."),
        ("Build → approve → start", "Check acquisitions, starts and compute budget."),
        ("Compare fits and receipts", "Retain competing and unresolved outcomes.")]),
    "webmon": ("One pinned run, explicit reference assumptions", [
        ("View a POWGEN snapshot", "Confirm the run and producer mapping."),
        ("Review nuclear reference + candidates", "Inspect the source fit and fixed assumptions."),
        ("Prepare reviewed live models", "Review the exact revision before analysis."),
        ("Analyze updates for this run", "Recheck changed inputs and scientific qualifications.")]),
    "recovery": ("Choose the action that matches your intent", [
        ("Restore stopped watch", "Open the saved scope; monitoring remains stopped."),
        ("Refresh saved results", "Collect existing jobs and their published outputs."),
        ("Resume existing watch", "Continue discovery for the same saved scope.")]),
}


def app_version() -> str:
    match = re.search(r'^version\s*=\s*"([^"]+)"', (NOVA / "pyproject.toml").read_text(), re.M)
    if not match:
        raise ValueError("Cannot read application version from nova/pyproject.toml")
    return match.group(1)


def e(value: str) -> str:
    return escape(str(value), quote=True)


def import_capture(root: Path, captures: dict, capture_id: str, source: Path,
                   version: str, date: str, context: str) -> None:
    if not re.fullmatch(r"[a-z0-9-]+", capture_id):
        raise ValueError("Capture ID must contain lowercase letters, digits and hyphens")
    if capture_id.startswith("current-") and capture_id[8:] not in {task["id"] for task in TASKS}:
        raise ValueError("Current capture ID must name an existing task")
    raw = source.read_bytes()
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = "png"
    elif raw.startswith(b"\xff\xd8\xff"):
        extension = "jpg"
    else:
        raise ValueError("Captures must contain PNG or JPEG image bytes")
    path = "assets/" + hashlib.sha256(raw).hexdigest()[:20] + "." + extension
    (root / "assets").mkdir(exist_ok=True)
    (root / path).write_bytes(raw)
    captures[capture_id] = dict(path=path, application_version=version, captured=date,
                                context=context, role="interface")


def visual(task: dict, captures: dict, version: str) -> str:
    capture_id = "current-" + task["id"]
    is_current_override = capture_id in captures
    if capture_id not in captures:
        capture_id = task.get("shot")
    if capture_id:
        shot = captures[capture_id]
        historical = shot["role"] == "historical_example"
        label = ("Historical example" if historical else "Interface capture") + " · app " + shot["application_version"]
        guidance = "Follow the current control names in the steps; this recorded layout is older." if historical else ""
        alt = task["title"] if is_current_override else task.get("alt", task["title"])
        return f'''<figure class="screen task-visual">
<button class="zoom" type="button" data-zoom="{e(shot['path'])}" aria-label="Enlarge: {e(alt)}"><img src="{e(shot['path'])}" alt="{e(alt)}" data-capture="{e(capture_id)}" data-capture-version="{e(shot['application_version'])}" loading="lazy"></button>
<figcaption><span class="capture-badge{' historical' if historical else ''}">{e(label)}</span><br>{e(shot['captured'])} · {e(shot['context'])}<span class="screen-context">{e(guidance)} <span class="zoom-hint">Click to enlarge; Escape closes the image.</span></span></figcaption></figure>'''
    title, nodes = DIAGRAMS[task["diagram"]]
    steps = "".join(f"<li><strong>{e(title)}</strong><span>{e(detail)}</span></li>" for title, detail in nodes)
    return f'<figure class="diagram task-visual"><p class="diagram-title">{e(title)}</p><ol class="flow">{steps}</ol><figcaption>Workflow of the {e(version)} controls. Review the details at each step.</figcaption></figure>'


def build(root: Path) -> dict:
    version = app_version()
    captures = json.loads((root / "screenshot-sources.json").read_text())
    if not (root / "reference-0.4.34.html").is_file():
        raise ValueError("Keep the reviewed 0.4.34 reference page alongside this guide")
    nav = []
    cards = []
    previous_group = None
    for number, task in enumerate(TASKS, 1):
        if task["group"] != previous_group:
            nav.append(f'<p class="nav-group" data-group="{e(task["group"])}">{e(task["group"])}</p>')
            previous_group = task["group"]
        nav.append(f'<a data-task-link="{e(task["id"])}" data-group="{e(task["group"])}" href="#{e(task["id"])}"><span class="nav-number">{number:02}</span><span>{e(task["title"])}</span></a>')
        steps = "".join(f"<li>{text}</li>" for text in task["steps"])
        detail_parts = []
        for title, text in task.get("details", []):
            anchor = ' id="single-pattern"' if title == "Analyzing a single file instead" else ""
            detail_parts.append(f'<details{anchor}><summary>{e(title)}</summary><div class="detail-body"><p>{text}</p></div></details>')
        details = "".join(detail_parts)
        if task.get("extra_shots"):
            images = "".join(visual(dict(id=shot, title=alt, alt=alt, shot=shot), captures, version)
                             for shot, alt in task["extra_shots"])
            details += '<details><summary>See the current Live panel and launch form</summary><div class="extra-shots">' + images + '</div></details>'
        prompt = ""
        if task.get("prompt"):
            prompt = f'<div class="prompt"><div class="prompt-head"><strong>Example text to adapt</strong><button type="button" data-copy="prompt-{e(task["id"])}">Copy example</button></div><pre id="prompt-{e(task["id"])}">{e(task["prompt"])}</pre><p>{e(task["prompt_note"])}</p></div>'
        next_link = f'<a href="#{e(TASKS[number]["id"])}">Next: {e(TASKS[number]["title"])} →</a>' if number < len(TASKS) else '<a href="#downloads">Example files →</a>'
        cards.append(f'''<article class="task" id="{e(task['id'])}" data-task="{e(task['id'])}">
<div class="task-head"><p class="kicker">{e(task['group'])} · {number:02}</p><h2>{e(task['title'])}</h2><p>{e(task['purpose'])}</p></div>
<div class="task-body">{visual(task, captures, version)}<div class="directions"><ol class="steps">{steps}</ol><div class="expected"><strong>What you should see</strong><p>{task['expected']}</p></div>{prompt}</div></div>
<div class="task-details">{details}</div><div class="task-footer"><a href="#top">Back to start ↑</a>{next_link}</div></article>''')
    downloads = "".join(f'<div class="download-item"><a href="inputs/{e(name)}" download>{e(title)}</a><p>{e(note)}</p></div>' for name, title, note in DOWNLOADS)
    help_items = "".join(f'<details><summary>{e(title)}</summary><div class="detail-body"><p>{text}</p></div></details>' for title, text in TROUBLESHOOTING)
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="radar-pd-guide-version" content="{e(version)}"><title>RADAR-PD {e(version)} — user guide</title><link rel="stylesheet" href="guide.css"></head>
<body id="top"><a class="skip" href="#main">Skip to guide</a><header class="page-head"><a class="brand" href="#top">RADAR-PD user guide</a><span class="version">Guide for application {e(version)}<br>Updated 5 November 2026</span></header>
<div class="shell"><aside class="sidebar"><label for="task-search">Find a task or control</label><input id="task-search" type="search" placeholder="Try profile, restore, error bars…" aria-controls="main"><nav aria-label="Guide tasks">{''.join(nav)}</nav><a class="utility" href="#downloads">Example files</a><a class="utility" href="#help">Troubleshooting</a><a class="utility" href="reference-0.4.34.html">Archived AGES walkthrough</a><p class="nav-note">Keep your RADAR-PD session open while reading. This guide opens in a separate tab.</p></aside>
<main id="main"><section class="intro"><p class="kicker">From your data to an inspected result</p><h1>Start with one result.<br>Then follow your experiment.</h1><p class="lead">Choose your starting point, check the inputs, and inspect the fit. Refinement and model searches are optional next steps.</p>
<div class="routes"><a class="route" href="#live-setup"><strong>I have an IPTS</strong><span>Use Live and the guided setup. Existing scans work too.</span></a><a class="route" href="#single-pattern"><strong>I have a diffraction file</strong><span>Use Single pattern with a matching profile and candidate library.</span></a><a class="route" href="#recover"><strong>I have saved work</strong><span>Restore results or a stopped watch from the same Galaxy History.</span></a></div>
<p class="scope-note"><b>A first Live result:</b> follow tasks 1–6. The instructions use current {e(version)} controls. Each screenshot identifies its own version and date; the public AGES images document the older 0.4.34 example.</p></section>
<p id="search-status" class="search-status" role="status" aria-live="polite"></p><p id="search-empty" class="empty" hidden>No matching task. Try a visible control name or clear the search.</p>
{''.join(cards)}
<section class="downloads" id="downloads"><p class="kicker">Reference material</p><h2>Example files and the archived walkthrough</h2><p>These files belong to the recorded POWGEN IPTS-2767 Al₂O₃/MgO example. They demonstrate formats and choices; review them for your own experiment.</p><div class="download-grid">{downloads}</div><p>Prepare CIFs from a source you may use. Licensed ICSD exports are not included.</p><div class="reference-callout"><a href="reference-0.4.34.html"><b>Open the complete 0.4.34 AGES walkthrough →</b></a><br>17 screenshot-led lessons and 253 field-by-field choices, preserved as a dated reference. Its recorded results are historical evidence; current procedures are above.</div></section>
<section class="help" id="help"><p class="kicker">When something does not look right</p><h2>Find the next check</h2>{help_items}</section>
<footer class="foot"><p>RADAR-PD guide for {e(version)} · {DATE}. Guide generation submits no scientific jobs. Scientific results and source permissions remain tied to their original experiment. Screenshot provenance is recorded in <a href="screenshot-sources.json">screenshot sources</a>.</p></footer></main></div>
<dialog id="image-dialog" aria-labelledby="image-title"><div class="dialog-bar"><strong id="image-title">Recorded screen</strong><button id="actual-size" type="button">Actual pixels</button><button id="close-image" type="button">Close</button></div><div class="dialog-image" id="dialog-image"><img id="full-image" alt=""></div><p class="dialog-caption" id="image-caption"></p></dialog><div id="copy-status" class="toast" role="status" aria-live="polite" hidden></div><script src="guide.js" defer></script></body></html>
'''
    (root / "index.html").write_text(html, encoding="utf-8")
    (root / "guide.css").write_text(CSS.strip() + "\n", encoding="utf-8")
    (root / "guide.js").write_text(JS.strip() + "\n", encoding="utf-8")
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    manifest = dict(schema="radar-pd-user-guide/v2", application_version=version,
                    pages=["index.html", "reference-0.4.34.html"],
                    reference_version="0.4.34", reference_statistics=dict(lessons=17, options=253, rendered_options=254, screenshot_templates=53),
                    tasks=[task["id"] for task in TASKS],
                    screenshots=len([name for name in files if name.startswith("assets/")]),
                    captures=captures, files=files, inputs=[item[0] for item in DOWNLOADS],
                    licensed_cif_files_redistributed=False, scientific_jobs_submitted=0)
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=NOVA / "help")
    parser.add_argument("--capture", nargs=2, action="append", metavar=("ID", "FILE"), default=[])
    parser.add_argument("--capture-version", default=None)
    parser.add_argument("--capture-date", default=DATE)
    parser.add_argument("--capture-context", default="Reviewed application interface; no scientific jobs submitted for the guide.")
    args = parser.parse_args()
    if args.capture:
        capture_sources = json.loads((args.root / "screenshot-sources.json").read_text())
        for capture_id, path in args.capture:
            import_capture(args.root, capture_sources, capture_id, Path(path), args.capture_version or app_version(), args.capture_date, args.capture_context)
        (args.root / "screenshot-sources.json").write_text(json.dumps(capture_sources, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = build(args.root)
    print(json.dumps({"guide_version": manifest["application_version"], "tasks": len(manifest["tasks"]), "screenshots": manifest["screenshots"], "files": len(manifest["files"])}))
