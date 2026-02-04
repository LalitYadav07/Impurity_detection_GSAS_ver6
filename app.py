import streamlit as st
import os
import sys
import yaml
import json
import time
import datetime
from pathlib import Path
import re
import shutil
import queue
import html
import psutil
import gc
import random
import importlib.util
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# --- CONSTANTS ---
PERIODIC_TABLE = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn"
]
PROJECT_ROOT = str(Path(__file__).resolve().parent)
IS_HF_SPACES = "SPACE_ID" in os.environ

# --- SETUP PATHS ---
scripts_dir = str(Path(__file__).resolve().parent / "scripts")
if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

from config_builder import build_pipeline_config
# Lazy import runner later or import here if safe
from runner import PipelineRunner

# --- GSAS-II HEALTH CHECK ---
def check_gsas_installation():
    try:
        import GSASII
        import GSASII.GSASIIpath as G2path
        
        # Check for the core binary module
        try:
            import pyspg
            st.success("✅ GSAS-II and binaries confirmed loaded.")
            return True
        except ImportError as e:
            st.error(f"❌ GSAS-II binaries (pyspg) failed to load: {e}")
            st.info("The system may be rebuilding binaries. Please refresh in a minute.")
            return False
            
    except ImportError:
        st.error("❌ GSAS-II package not found. Please ensure it is in requirements.txt.")
        return False
    except Exception as e:
        st.error(f"⚠️ GSAS-II initialization error: {e}")
        return False

# Trigger Check (Runs once per session/reset)
if 'gsas_ready' not in st.session_state:
    st.session_state.gsas_ready = check_gsas_installation()
GSAS_READY = st.session_state.gsas_ready

# Fallback for Pixi detection (unused if running via pip)
def is_pixi_available():
    import shutil
    return shutil.which("pixi") is not None

if 'use_pixi' not in st.session_state:
    st.session_state.use_pixi = is_pixi_available()

# --- DATABASE STATUS CHECK ---
DB_DIR = Path(PROJECT_ROOT) / "data" / "database_aug"
# Essential files to check
REQUIRED_FILES = [
    "highsymm_metadata.json",
    "catalog_deduplicated.csv",
    "mp_experimental_stable.csv"
]
PROFILES_DIR = DB_DIR / "profiles64"

def check_db_integrity():
    if not DB_DIR.exists(): return False
    for f in REQUIRED_FILES:
        if not (DB_DIR / f).exists(): return False
    if not PROFILES_DIR.exists() or not any(PROFILES_DIR.iterdir()): return False
    return True

DB_EXISTS = check_db_integrity()

def get_gdrive_id(url):
    import re
    patterns = [
        r'/file/d/([^/]+)',
        r'id=([^&]+)',
        r'/open\?id=([^&]+)'
    ]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None

def download_and_extract_db(url):
    import requests
    import zipfile
    import os
    import gdown
    import shutil
    
    DB_DIR.mkdir(parents=True, exist_ok=True)
    try:
        with st.status("📥 Downloading Database Archive...", expanded=True) as status:
            zip_path = Path(PROJECT_ROOT) / "temp_db.zip"
            
            # --- 1. Download ---
            gdrive_id = get_gdrive_id(url)
            if gdrive_id:
                st.write(f"Detected Google Drive Link (ID: {gdrive_id})")
                gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
            else:
                st.write(f"Fetching from: {url}")
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=81920):
                        if chunk: f.write(chunk)
            
            # --- 2. Extract ---
            if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
                st.error("❌ Downloaded file is not a valid ZIP.")
                zip_path.unlink(missing_ok=True)
                return False

            st.write("📦 Extracting files...")
            temp_extract_dir = Path(PROJECT_ROOT) / "temp_extract"
            if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
            temp_extract_dir.mkdir()

            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(temp_extract_dir)
            
            # Smart move: find where the data actually is (handle 'database_aug/' prefix)
            extracted_db_path = temp_extract_dir
            if (temp_extract_dir / "database_aug").exists():
                extracted_db_path = temp_extract_dir / "database_aug"
            
            # Copy contents to final destination
            st.write("📂 Organizing folders...")
            for item in extracted_db_path.iterdir():
                dest = DB_DIR / item.name
                if dest.exists():
                    if dest.is_dir(): shutil.rmtree(dest)
                    else: dest.unlink()
                shutil.move(str(item), str(DB_DIR))
                
            # Cleanup
            zip_path.unlink()
            shutil.rmtree(temp_extract_dir)
            
            # 3. Validation
            if check_db_integrity():
                status.update(label="✅ Database Installed Successfully!", state="complete")
                return True
            else:
                status.update(label="❌ Missing files after extraction.", state="error")
                return False

    except Exception as e:
        st.error(f"Download/Extraction failed: {e}")
        return False

# --- GSAS-II CHECK & IMPORT ---
try:
    import GSASII.GSASIIscriptable as G2sc
    GSAS_AVAILABLE = True
except ImportError:
    GSAS_AVAILABLE = False

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="GSAS-II Impurity Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- PREMIUM CSS: ORNL-Inspired Calm Theme ---
st.markdown("""
<style>
    /* ============ Global Theme: Calm Light Mode ============ */
    :root {
        --ornl-green: #154734;
        --ornl-green-light: #1e6b4a;
        --accent-green: #4caf50;
        --bg-primary: #f8f9fa;
        --bg-secondary: #ffffff;
        --bg-tertiary: #e9ecef;
        --text-primary: #212529;
        --text-secondary: #495057;
        --border-color: #dee2e6;
    }
    
    /* ============ Stability & Layout ============ */
    /* Force vertical scrollbar to prevent horizontal jitter when content expands */
    html {
        overflow-y: scroll;
    }
    
    /* Stop the "fading" effect during reruns by minimizing transition noise */
    .stApp {
        background-color: var(--bg-primary);
        color: var(--text-primary);
        transition: none !important;
    }
</style>
""", unsafe_allow_html=True)
# --- PREMIUM CSS: Part 2 ---
st.markdown("""
<style>
    /* ============ Elegant Buttons ============ */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3em;
        background: linear-gradient(135deg, var(--ornl-green) 0%, var(--ornl-green-light) 100%);
        color: white;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(21, 71, 52, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(21, 71, 52, 0.3);
        background: linear-gradient(135deg, var(--ornl-green-light) 0%, var(--accent-green) 100%);
    }
    
    /* ============ Metrics Cards ============ */
    div[data-testid="stMetric"] {
        background-color: var(--bg-secondary);
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid var(--ornl-green);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.08);
    }
    div[data-testid="stMetric"] label {
        color: var(--text-secondary);
    }
    
    /* ============ Tabs ============ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: var(--bg-secondary);
        border-radius: 8px 8px 0 0;
        color: var(--text-secondary);
        padding: 0 20px;
        border: 1px solid var(--border-color);
        border-bottom: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--bg-tertiary);
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: var(--ornl-green);
        color: white;
        border-color: var(--ornl-green);
    }
    
    /* ============ Log Viewer ============ */
    .log-viewer {
        height: 500px; 
        overflow-y: scroll !important; 
        scroll-behavior: smooth;
        background-color: #1e293b; 
        color: #e2e8f0; 
        padding: 15px; 
        font-family: 'JetBrains Mono', 'Fira Code', monospace; 
        border-radius: 8px; 
        border: 1px solid #334155;
        font-size: 0.85em;
        line-height: 1.6;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    .log-header { color: #a78bfa; font-weight: bold; }
    .log-metric { color: #34d399; }

    /* ============ File Explorer ============ */
    .file-tree-item {
        padding: 6px 10px;
        border-radius: 6px;
        margin-bottom: 3px;
        transition: background-color 0.2s;
        border-bottom: 1px solid var(--border-color);
        background-color: var(--bg-secondary);
    }
    .file-tree-item:hover {
        background-color: var(--bg-tertiary);
    }
    .file-tree-folder { color: var(--ornl-green); font-weight: bold; }
    .file-tree-file { color: var(--text-secondary); }

    /* ============ Sidebar ============ */
    [data-testid="stSidebar"] {
        background-color: var(--bg-secondary);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--ornl-green);
    }

    /* ============ Expanders ============ */
    .streamlit-expanderHeader {
        background-color: var(--bg-secondary);
        border-radius: 8px;
        border: 1px solid var(--border-color);
    }
    .streamlit-expanderContent {
        background-color: var(--bg-secondary);
        border: 1px solid var(--border-color);
        border-top: none;
        border-radius: 0 0 8px 8px;
    }

    /* ============ Progress Bar ============ */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--ornl-green) 0%, var(--accent-green) 100%);
    }

    /* ============ Decision Engine (Knee Analysis) ============ */
    .decision-engine {
        height: 300px; 
        overflow-y: auto; 
        background-color: #0f172a; 
        color: #cbd5e1; 
        padding: 15px; 
        font-family: 'Inter', sans-serif; 
        border-radius: 12px; 
        border: 1px solid #1e293b;
        border-left: 6px solid #f59e0b;
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-top: 15px;
    }
    .decision-item {
        margin-bottom: 12px;
        padding: 10px;
        background-color: #1e293b;
        border-radius: 6px;
        border-left: 2px solid #f59e0b;
        animation: fadeIn 0.5s ease-out;
    }
    .decision-tag {
        color: #f59e0b;
        font-weight: 800;
        font-size: 0.7rem;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
        display: block;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* ============ Success/Warning/Error Boxes ============ */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
    }

    /* ============ Premium Timeline UI ============ */
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(21, 71, 52, 0.4); }
        70% { box-shadow: 0 0 0 10px rgba(21, 71, 52, 0); }
        100% { box-shadow: 0 0 0 0 rgba(21, 71, 52, 0); }
    }
    
    .timeline-container {
        padding: 10px 5px;
        font-family: 'Inter', sans-serif;
    }
    .timeline-item {
        position: relative;
        padding-left: 30px;
        padding-bottom: 20px;
        border-left: 2px solid var(--border-color);
        margin-left: 10px;
    }
    .timeline-item.last {
        border-left: none;
    }
    .timeline-item.active {
        border-left-color: var(--ornl-green);
    }
    .timeline-item.complete {
        border-left-color: var(--accent-green);
    }
    
    .timeline-dot {
        position: absolute;
        left: -8px;
        top: 0;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: var(--bg-tertiary);
        border: 2px solid var(--border-color);
        z-index: 1;
    }
    .timeline-item.active .timeline-dot {
        background: var(--ornl-green);
        border-color: var(--ornl-green);
        animation: pulse 2s infinite;
    }
    .timeline-item.complete .timeline-dot {
        background: var(--accent-green);
        border-color: var(--accent-green);
    }
    
    .timeline-content {
        top: -4px;
        position: relative;
    }
    .timeline-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--text-primary);
        margin-bottom: 2px;
    }
    .timeline-subtitle {
        font-size: 0.8rem;
        color: var(--text-secondary);
    }
    .timeline-item.active .timeline-title {
        color: var(--ornl-green);
    }
    .timeline-item.complete .timeline-title {
        color: var(--text-secondary);
        text-decoration: line-through;
        opacity: 0.8;
    }

    /* Sub-steps (Pass stages) */
    .sub-steps {
        margin-top: 10px;
        padding-left: 5px;
        border-left: 1px dashed var(--border-color);
        margin-left: 5px;
    }
    .sub-step {
        padding: 4px 15px;
        font-size: 0.85rem;
        color: var(--text-secondary);
        position: relative;
    }
    .sub-step.active {
        color: var(--ornl-green-light);
        font-weight: 600;
    }
    .sub-step.active::before {
        content: "→";
        position: absolute;
        left: 0;
        animation: bounceX 1s infinite alternate;
    }
    
    @keyframes bounceX {
        from { transform: translateX(0); }
        to { transform: translateX(3px); }
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES ---
def get_ram_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / (1024 * 1024) # MB
    return mem

# --- STATE INITIALIZATION ---
if 'run_active' not in st.session_state:
    st.session_state.run_active = False
if 'run_finished' not in st.session_state:
    st.session_state.run_finished = False
if 'run_dir' not in st.session_state:
    st.session_state.run_dir = None
if 'log_lines' not in st.session_state:
    st.session_state.log_lines = []
if 'pipeline_process' not in st.session_state:
    st.session_state.pipeline_process = None
if 'log_queue' not in st.session_state:
    st.session_state.log_queue = None
if 'funnel_data' not in st.session_state:
    st.session_state.funnel_data = {
        "Total Database": 0, "Elements": 0, "Spacegroup": 0, "Stability": 0
    }
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'status_msg' not in st.session_state:
    st.session_state.status_msg = "Ready"
if 'knee_logs' not in st.session_state:
    st.session_state.knee_logs = []

if 'pipeline_state' not in st.session_state:
    st.session_state.pipeline_state = {
        "global_stage_idx": -1,
        "global_stage_desc": "Ready",
        "current_pass": 0,
        "pass_stage": None,
        "stage0_status": "pending", # pending, running, complete, skipped
        "stages_complete": set()
    }

# --- LOG STATE CLEANUP (Recovery from previous formatting mistakes) ---
if 'log_lines' in st.session_state:
    # If any line contains a span tag, it shouldn't be there. Revert to raw.
    if any("<span" in str(line) for line in st.session_state.log_lines):
        st.session_state.log_lines = [re.sub(r'<[^>]+>', '', line) for line in st.session_state.log_lines]

if 'log_autoscroll' not in st.session_state:
    st.session_state.log_autoscroll = True

# --- HELPER FUNCTIONS ---
def format_log_line(line):
    """Simplified highlighting: Lavender for headers, Cyan for metrics."""
    # Escape HTML
    l = html.escape(line)
    
    # Check for Headers/Boundaries
    if any(k in l for k in ["STAGE", "PASS", "SUMMARY", "PROCESSING", "════", "────"]):
        return f'<span class="log-header">{l}</span>'
    
    # Check for Metrics
    if any(k in l for k in ["score", "cos", "alpha", "knee", "explained", "Rwp", "GOF", "Pearson"]):
        return f'<span class="log-metric">{l}</span>'
    
    return l

# --- STAGE TRACKING ---
GLOBAL_STAGES = [
    "Stage 0: Bootstrap (Find Main Phase)",
    "Stage 1: Main Phase Refinement",
    "Stage 2: Residual Extraction",
    "Sequential Discovery Passes",
    "Final Reporting"
]

PASS_STAGES = [
    ("screening", "ML Screening"),
    ("nudging", "Lattice Nudging"),
    ("pearson", "Pearson Refinement"),
    ("joint", "Joint Refinement"),
    ("polish", "Polishing"),
    ("summary", "Pass Summary")
]

def parse_pipeline_log_line(line, state):
    """Monotonic log parser using anchored markers."""
    l = line.strip()
    
    # Global Transitions
    l_up = l.upper()
    if "STAGE 0: BOOTSTRAP" in l_up:
        state["global_stage_idx"] = 0
        state["stage0_status"] = "running"
    elif "STAGE 1: MAIN PHASE REFINEMENT" in l_up:
        if state["global_stage_idx"] < 1:
            if state["stage0_status"] == "pending":
                state["stage0_status"] = "skipped"
            state["global_stage_idx"] = 1
    elif "STAGE 2: RESIDUAL EXTRACTION" in l_up:
        state["global_stage_idx"] = 2
    elif "SEQUENTIAL PHASES" in l_up or ("SEQUENTIAL PASS" in l_up and "discovery" in l):
        state["global_stage_idx"] = 3
        m = re.search(r"PASS (\d+)", l, re.I)
        if m:
            state["current_pass"] = int(m.group(1))
            state["pass_stage"] = "screening"
    elif "FINAL REPORTING" in l_up or "PIPELINE COMPLETED SUCCESSFULLY" in l_up:
        state["global_stage_idx"] = 4
        
    # Pass-level anchors (within Global Stage 3)
    if state["global_stage_idx"] == 3:
        if "COMPREHENSIVE CANDIDATE SCREENING" in l_up:
            state["pass_stage"] = "screening"
        elif "PROCESSING TOP" in l_up:
            state["pass_stage"] = "nudging"
        elif "[PEARSON]" in l_up:
            state["pass_stage"] = "pearson"
        elif "[CLONE]" in l_up and any(k in l_up for k in ["JOINT", "KEPT", "COMMIT"]):
            state["pass_stage"] = "joint"
        elif "[POLISH] STARTING" in l_up:
            state["pass_stage"] = "polish"
        elif "PASS" in l_up and "SUMMARY" in l_up:
            state["pass_stage"] = "summary"
            
    return state
def update_funnel_metrics(new_lines):
    """Incremental update of funnel metrics from new log lines."""
    data = st.session_state.funnel_data
    for line in new_lines:
        if "catalog size:" in line:
            m = re.search(r"catalog size:\s+(\d+)", line)
            if m: data["Total Database"] = int(m.group(1))
        if "matching elements" in line:
            m = re.search(r"elements:\s+(\d+)", line)
            if m: data["Elements"] = int(m.group(1))
        if "matching spacegroup" in line:
            m = re.search(r"spacegroup:\s+(\d+)", line)
            if m: data["Spacegroup"] = int(m.group(1))
        if "stable phases loaded" in line:
            m = re.search(r"loaded:\s+(\d+)", line)
            if m: data["Stability"] = int(m.group(1))
    
    # Heuristic fix for missing stability log
    if data["Stability"] == 0 and data["Spacegroup"] > 0:
        data["Stability"] = data["Spacegroup"]
        
    st.session_state.funnel_data = data

def render_file_explorer(path: Path, key_prefix: str, filter_exts=None, depth=0):
    """Recursive file explorer UI component with improved layout."""
    if not path.is_dir():
        return

    items = sorted(list(path.iterdir()), key=lambda x: (not x.is_dir(), x.name.lower()))
    
    for item in items:
        if item.name.startswith("."): continue
        
        # Indentation for depth
        margin_left = depth * 20
        
        
        # Recursive Folder
        if item.is_dir():
            unique_key = f"{key_prefix}_{item.name}"
            # Use a stable key for expanders to prevent closure during reruns
            # Removing 'key' because local Streamlit 1.53.1 signature lacks it
            with st.expander(f"📁 {item.name}", expanded=(depth < 1)):
                render_file_explorer(item, unique_key, filter_exts, depth + 1)
                
        # File Display
        else:
            if filter_exts and item.suffix.lower() not in filter_exts:
                continue
                
            # Render File Row with Download
            c1, c2 = st.columns([0.8, 0.2])
            with c1:
                st.markdown(f"""
                    <div style="margin-left: {margin_left}px; padding: 4px 0;">
                        <span class="file-tree-file">📄 **{item.name}**</span>
                        <span style="font-size: 0.8em; color: #718096; margin-left: 10px;">({item.stat().st_size / 1024:.1f} KB)</span>
                    </div>
                """, unsafe_allow_html=True)
            with c2:
                # Key must be unique per file
                with open(item, "rb") as f:
                    st.download_button("⬇", f, file_name=item.name, key=f"dl_{key_prefix}_{item.name}", width="stretch")
            
            # Preview for Artifacts (Lazy Load)
            if item.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                if item.stat().st_size > 0:
                    # Memory optimization: Only auto-expand in Plot folders, otherwise use a toggle
                    # AND: Only auto-expand if the run is NOT active to prevent UI churn
                    is_running = st.session_state.get("run_active", False)
                    should_preview = ("Plots" in str(item.parent) or "Diagnostics" in str(item.parent)) and not is_running
                    
                    if should_preview:
                        try:
                            from PIL import Image
                            st.image(str(item), width="stretch" if hasattr(st, "image") else None)
                        except Exception:
                            st.caption(f"⚠️ Image {item.name} is still being written.")
                    else:
                        if st.checkbox(f"👁️ Preview {item.name}", key=f"pv_{key_prefix}_{item.name}"):
                            from PIL import Image
                            st.image(str(item), width="stretch")

def update_ui_state():
    """Polls the runner queue and updates session state."""
    if st.session_state.pipeline_process and st.session_state.log_queue:
        q = st.session_state.log_queue
        process = st.session_state.pipeline_process
        new_lines = []
        
        try:
            while True:
                line = q.get_nowait()
                new_lines.append(line)
        except queue.Empty:
            pass
        
        if new_lines:
            st.session_state.log_lines.extend(new_lines)
            update_funnel_metrics(new_lines)
            
            # MEMORY OPTIMIZATION: Keep only last 500 lines
            if len(st.session_state.log_lines) > 500:
                st.session_state.log_lines = st.session_state.log_lines[-500:]
            
        # Update Progress State
        state = st.session_state.pipeline_state
        
        # 1. Fallback / Complementary: Parsing raw logs (Heuristic-based)
        if new_lines:
            KNEE_KEYS = ["[KNEE] hist/UNION", "[KNEE] nudge/score", "[KNEE] nudge/filter", "[KNEE] pearson/r", "[KNEE] pearson/filter"]
            for line in new_lines:
                state = parse_pipeline_log_line(line, state)
                # Populate Decision Logic
                if any(k in line for k in KNEE_KEYS):
                    st.session_state.knee_logs.append(line.strip())
                    if len(st.session_state.knee_logs) > 100:
                        st.session_state.knee_logs = st.session_state.knee_logs[-100:]

        # 2. Primary Source: Structured Events (JSONL) - Accurate and high-confidence
        if st.session_state.run_dir:
            run_path = Path(st.session_state.run_dir)
            evt_file = run_path / "Technical" / "Logs" / "run_events.jsonl"
            
            if evt_file.exists():
                try:
                    with open(evt_file, "r") as f:
                        lines = f.readlines()
                        if lines:
                            for line in lines[-5:]: # Look at recent events
                                evt = json.loads(line)
                                if "percent" in evt:
                                    st.session_state.progress = int(evt["percent"])
                                
                                stage = evt.get("stage", "")
                                metrics = evt.get("metrics", {})
                                
                                if "Stage 0" in stage:
                                    state["global_stage_idx"] = 0
                                    if "Bootstrap complete" in evt.get("message", ""):
                                        state["stage0_status"] = "complete"
                                    else:
                                        state["stage0_status"] = "running"
                                elif "Stage 1" in stage:
                                    state["global_stage_idx"] = 1
                                elif "Stage 2" in stage:
                                    state["global_stage_idx"] = 2
                                elif "Pass" in stage:
                                    state["global_stage_idx"] = 3
                                    state["current_pass"] = metrics.get("pass", state["current_pass"])
                                    event_type = metrics.get("event")
                                    if event_type in ["pass_start", "screening_start"]: state["pass_stage"] = "screening"
                                    elif event_type == "nudging_start": state["pass_stage"] = "nudging"
                                    elif event_type == "pearson_start": state["pass_stage"] = "pearson"
                                    elif event_type in ["joint_compare_start", "joint_refine_start"]: state["pass_stage"] = "joint"
                                    elif event_type == "polish_start": state["pass_stage"] = "polish"
                                    elif event_type == "pass_end": state["pass_stage"] = "summary"
                                elif "Final" in stage or "Complete" in stage:
                                    state["global_stage_idx"] = 4
                except:
                    pass
        
        st.session_state.pipeline_state = state

        # Check Process Status
        if process.poll() is not None and q.empty():
            st.session_state.run_active = False
            st.session_state.run_finished = True
            if process.returncode == 0:
                st.success("✅ Run Completed Successfully!")
                st.balloons()
            else:
                st.error(f"❌ Run Failed (Exit Code {process.returncode})")
            
            # Important: Trigger a full rerun to synchronize the "Results" and "Explorer" tabs
            # since they are outside this fragment.
            st.rerun()

# Note: run_monitor_fragment was consolidated into the log/sidebar fragments below 
# to ensure data synchronization and avoid queue race conditions.

# --- UI HEADER ---
st.title("🔬Impurity Phase Detection for NPD")
st.markdown("Automated crystallography impurity phase discovery using ML-guided refinement.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("🛠️ Configuration")
    
    # GSAS-II Health in Sidebar to reduce main UI churn
    c1, c2 = st.columns(2)
    with c1:
        if GSAS_READY: st.success("✅ GSAS-II: [OK]")
        else: st.error("❌ GSAS-II: [FAILED]")
    with c2:
        ram = get_ram_usage()
        # Adjusted threshold for HF Spaces (16GB)
        threshold = 14000 if IS_HF_SPACES else 800
        if ram > threshold: st.warning(f"🔋 RAM: {ram:.0f} MB")
        else: st.info(f"🔋 RAM: {ram:.0f} MB")
        
    if IS_HF_SPACES:
        st.caption("🚀 Running on Hugging Face Spaces (16GB RAM)")

    if not GSAS_READY:
        if st.button("🔄 Retry GSAS-II Check"):
            del st.session_state.gsas_ready
            st.rerun()

    # Database Status
    if not DB_EXISTS:
        st.warning("📊 Database missing or incomplete")
        with st.expander("🛠️ How to fix", expanded=True):
            st.markdown("""
                The 2.3GB database was excluded from Git. 
                **Download the ZIP archive** manually or provide a direct link.
            """)
            db_url = st.text_input("Direct Download URL (ZIP)", value="https://drive.google.com/uc?id=1BxPXjdbn7oYTXKfDeLct5-2PMkhcLVSH", placeholder="https://.../database_aug.zip")
            if st.button("📥 Download & Install Database"):
                if download_and_extract_db(db_url):
                    st.success("Database ready! Please refresh.")
                    st.rerun()
            st.info("💡 Locally, checks for JSON, CSVs, and profiles64.")
    else:
        st.success("📊 Database: [OK] (Ready for discovery)")

    with st.expander("📁 Mandatory Inputs", expanded=True):
        example_mode = st.checkbox("📖 Example Mode (TbSSL Demo)", value=False)
        if example_mode:
            st.info("Using bundled TbSSL dataset (4K neutrons). Parameters are pre-configured.")
            data_file, instprm_file, main_cif = None, None, None
            # Read-only display of what will be used
            st.code("Allowed: Tb, Be, Ge, O\nEnv: Al\nCIF: TbSSL.cif", language="text")
            allowed_elements_str = "Tb, Be, Ge, O"
            sample_env_elements_str = "Al"
        else:
            data_file = st.file_uploader("Diffraction Data", type=["dat", "xye", "gsa", "fxye"])
            instprm_file = st.file_uploader("Instrument Params (.instprm)", type=["instprm"])
            main_cif = st.file_uploader("Main CIF (Optional)", type=["cif"])
            allowed_elements_str = st.text_input("Allowed Elements", "Tb, Be, Ge, O")
            sample_env_elements_str = st.text_input("Sample Env", "")

    with st.expander("🧠 Advanced Params", expanded=False):
        run_name = st.text_input("Run Name", f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        min_impurity = st.slider("Min Wt%", 0.0, 5.0, 0.5)
        max_passes = st.number_input("Max Passes", 1, 10, 3)
        top_candidates = st.number_input("Top Candidates", 1, 100, 10)
        
    # START BUTTON
    if not st.session_state.run_active:
        if st.button("🚀 RUN PIPELINE"):
            # Validation
            if not example_mode and (not data_file or not instprm_file):
                st.error("Missing required files!")
            else:
                # Setup
                clean_name = run_name.replace(" ", "_")
                rdir = PROJECT_ROOT / Path("runs") / clean_name
                input_dir = rdir / "inputs"
                input_dir.mkdir(parents=True, exist_ok=True)
                
                # File Handling
                dpath, ipath, cpath = None, None, None
                if example_mode:
                    dpath = str((Path(PROJECT_ROOT) / "examples" / "tbssl" / "HB2A_TbSSL.dat").resolve())
                    ipath = str((Path(PROJECT_ROOT) / "examples" / "tbssl" / "hb2a_si_ge113.instprm").resolve())
                    cpath = str((Path(PROJECT_ROOT) / "examples" / "tbssl" / "TbSSL.cif").resolve())
                else:
                    # Save logic inline
                    if data_file:
                        with open(input_dir / data_file.name, "wb") as f: f.write(data_file.getbuffer())
                        dpath = str((input_dir / data_file.name).resolve())
                    if instprm_file:
                        with open(input_dir / instprm_file.name, "wb") as f: f.write(instprm_file.getbuffer())
                        ipath = str((input_dir / instprm_file.name).resolve())
                    if main_cif:
                        with open(input_dir / main_cif.name, "wb") as f: f.write(main_cif.getbuffer())
                        cpath = str((input_dir / main_cif.name).resolve())

                # Config
                els = [e.strip() for e in allowed_elements_str.split(",") if e.strip()]
                env = [e.strip() for e in sample_env_elements_str.split(",") if e.strip()]
                
                cfg = build_pipeline_config(
                    run_name=run_name, data_file=dpath, instprm_file=ipath,
                    allowed_elements=els, sample_env_elements=env, main_cif=cpath,
                    work_root=str(rdir), project_root=PROJECT_ROOT,
                    min_impurity_percent=min_impurity, max_passes=max_passes,
                    advanced_params={"top_candidates": top_candidates}
                )
                
                with open(rdir / "pipeline_config.yaml", "w") as f: f.write(cfg)
                
                # Start
                st.session_state.run_dir = str(rdir)
                st.session_state.run_name = clean_name
                st.session_state.log_lines = []
                st.session_state.funnel_data = {"Total Database": 0, "Elements": 0, "Spacegroup": 0, "Stability": 0}
                st.session_state.progress = 0
                st.session_state.status_msg = "Initializing..."
                
                lpath = str(rdir / "pipeline.log")
                process, q = PipelineRunner(PROJECT_ROOT, use_pixi=st.session_state.use_pixi).start_non_blocking(str(rdir/"pipeline_config.yaml"), clean_name, log_path=lpath)
                st.session_state.pipeline_process = process
                st.session_state.log_queue = q
                st.session_state.run_active = True
                st.rerun()

    # STOP BUTTON
    if st.session_state.run_active:
        if st.button("🛑 STOP PIPELINE"):
            if st.session_state.pipeline_process:
                st.session_state.pipeline_process.terminate()
            st.session_state.run_active = False
            st.session_state.run_finished = True
            st.warning("Pipeline Terminated")
            st.rerun()
            
    st.markdown("---")
    # Wrap status and progress in a fragment to update smoothly
    @st.fragment(run_every=2.0)
    def render_sidebar_progress():
        if st.session_state.run_active or st.session_state.run_finished:
            st.caption(f"Status: {st.session_state.get('current_stage_desc', 'Ready')}")
            st.progress(min(1.0, max(0.0, st.session_state.progress / 100)))
            
            st.markdown("### 🚦 Pipeline Progress")
            state = st.session_state.pipeline_state
            g_idx = state["global_stage_idx"]
            
            html_parts = ['<div class="timeline-container">']
            
            for i, stage_name in enumerate(GLOBAL_STAGES):
                is_last = (i == len(GLOBAL_STAGES) - 1)
                item_class = "timeline-item"
                if is_last: item_class += " last"
                
                # Determine status
                if i == 0:
                    status = state["stage0_status"]
                    if status == "complete": status_class = "complete"
                    elif status == "running": status_class = "active"
                    elif status == "skipped": status_class = "complete" # Or "skipped"
                    else: status_class = ""
                else:
                    if i < g_idx: status_class = "complete"
                    elif i == g_idx: status_class = "active"
                    else: status_class = ""
                
                html_parts.append(f'<div class="{item_class} {status_class}">')
                html_parts.append('<div class="timeline-dot"></div>')
                html_parts.append('<div class="timeline-content">')
                
                # Icon mapping for fun
                icons = ["🚀", "🔬", "🛰️", "🔄", "📊"]
                icon = icons[i] if i < len(icons) else "🔹"
                
                title = f"{icon} {stage_name}"
                html_parts.append(f'<div class="timeline-title">{title}</div>')
                
                # Sub-stages for Pass
                if i == 3 and i == g_idx:
                    curr_pass = state["current_pass"]
                    curr_p_stage = state["pass_stage"]
                    html_parts.append(f'<div class="timeline-subtitle">Pass {curr_pass} in progress</div>')
                    html_parts.append('<div class="sub-steps">')
                    for s_key, s_name in PASS_STAGES:
                        sub_class = "sub-step"
                        if s_key == curr_p_stage: sub_class += " active"
                        html_parts.append(f'<div class="{sub_class}">{s_name}</div>')
                    html_parts.append('</div>')
                
                html_parts.append('</div></div>')
            
            html_parts.append('</div>')
            st.markdown("".join(html_parts), unsafe_allow_html=True)
        else:
            st.caption("Status: Ready")
            st.progress(0)
            st.info("Start a run to see live progress.")

    render_sidebar_progress()

# --- TABS ---
t_run, t_res, t_exp = st.tabs(["🚀 Run & Progress", "📊 Results", "📂 Run File Browser"])

with t_run:
    c1, c2 = st.columns([2, 1])
    with c1:
        @st.fragment(run_every=2.0)
        def render_logs_and_monitor():
            # 1. Heartbeat: Update state for the entire app from this fragment
            if st.session_state.run_active:
                update_ui_state()
                # Occasional GC
                if random.random() < 0.05: gc.collect()

            c1_t, c1_a = st.columns([1, 1])
            c1_t.subheader("📜 Live Logs")
            st.session_state.log_autoscroll = c1_a.checkbox("🔄 Autoscroll", value=st.session_state.log_autoscroll, key="log_as_toggle")
            
            # Format logs with highlighting
            formatted_logs = "<br>".join([format_log_line(line.rstrip()) for line in st.session_state.log_lines])
            
            # Unique IDs to prevent collision and "gray out" jank
            run_slug = re.sub(r'[^a-zA-Z0-9]', '_', st.session_state.get('run_name', 'default'))
            container_id = f"log-container-{run_slug}"

            # Log viewer container
            st.markdown(f'<div id="{container_id}" class="log-viewer">{formatted_logs}</div>', unsafe_allow_html=True)
            
            # 2. MutationObserver Autoscroll Logic (Most Robust)
            if st.session_state.log_autoscroll:
                cache_buster = len(st.session_state.log_lines)
                st.markdown(f"""
                    <script data-cache="{cache_buster}">
                    (function() {{
                        const container = window.parent.document.getElementById('{container_id}');
                        if (!container) return;

                        // Immediate scroll to bottom
                        container.scrollTop = container.scrollHeight;

                        // Setup Observer to watch for new lines added by React/Streamlit
                        const observer = new MutationObserver(() => {{
                            container.scrollTop = container.scrollHeight;
                        }});

                        observer.observe(container, {{
                            childList: true,
                            subtree: true,
                            characterData: true
                        }});

                        // Fallback safety scrolls for fragments
                        setTimeout(() => {{ container.scrollTop = container.scrollHeight; }}, 100);
                        setTimeout(() => {{ container.scrollTop = container.scrollHeight; }}, 500);
                    }})();
                    </script>
                    """, unsafe_allow_html=True)
        
        render_logs_and_monitor()

        # Filter Logic Panel
        st.markdown("---")
        st.markdown("### 🧠 Filter Logic")
        st.markdown('<p style="font-size: 0.8rem; color: #718096; margin-top: -10px;">Automated reasoning and knee-filter analytics.</p>', unsafe_allow_html=True)
        
        @st.fragment(run_every=2.5)
        def render_decision_engine():
            if not st.session_state.knee_logs:
                st.markdown('<div class="decision-engine" style="display: flex; align-items: center; justify-content: center; color: #475569;">Waiting for intelligence events...</div>', unsafe_allow_html=True)
                return

            # Build HTML string manually to avoid markdown auto-formatting glitches
            items_html = ""
            for line in reversed(st.session_state.knee_logs):
                # Parse tag and message
                # Example: [KNEE] nudge/filter (pass 1): 10 → 4 ['pid1', 'pid2']
                parts = line.split(":", 1)
                tag = parts[0].replace("[KNEE]", "").strip() if len(parts) > 1 else "ANALYTICS"
                msg = parts[1].strip() if len(parts) > 1 else line
                
                # Use a single-line string to prevent Streamlit from interpreting it as a code block
                items_html += f'<div class="decision-item"><span class="decision-tag">⚡ {tag}</span><div>{html.escape(msg)}</div></div>'
            
            st.markdown(f'<div class="decision-engine">{items_html}</div>', unsafe_allow_html=True)
        
        render_decision_engine()
    
    with c2:
        st.subheader("🖼️ Artifacts")
        @st.fragment(run_every=30.0)
        def render_artifacts_fragment():
            if st.session_state.run_dir:
                rdir = Path(st.session_state.run_dir)
                p_dir_new = rdir / "Results" / "Plots"
                diag_dir = rdir / "Diagnostics"
                p_dir_old = rdir / "plots"
                sub_plots = rdir / st.session_state.get('run_name', '') / "plots"
                
                if p_dir_new.exists():
                    st.markdown("**Plots**")
                    render_file_explorer(p_dir_new, "art_new", [".png", ".jpg", ".pdf"])
                
                if diag_dir.exists():
                    st.markdown("**Diagnostics**")
                    render_file_explorer(diag_dir, "art_diag", [".png", ".jpg", ".pdf"])

                if not p_dir_new.exists() and not diag_dir.exists():
                    if p_dir_old.exists():
                        render_file_explorer(p_dir_old, "art_root", [".png", ".jpg", ".pdf"])
                    elif sub_plots.exists():
                        render_file_explorer(sub_plots, "art_sub", [".png", ".jpg", ".pdf"])
                    else:
                        st.info("No plots directory found yet.")
            else:
                st.info("Start a run to see artifacts.")
        
        render_artifacts_fragment()

with t_res:
    if st.session_state.run_dir:
        rdir = Path(st.session_state.run_dir)
        
        # Metrics Overview (Optional, maybe keep it simple)
        mpath = next(rdir.rglob("run_manifest.json"), None)
        if mpath and mpath.exists():
            try:
                with open(mpath) as f: m = json.load(f)
                mets = m.get("metrics", {})
                st.markdown(f"**Status:** {m.get('status', 'Processing')} | **Final Rwp:** {mets.get('final_rwp', 0):.2f}%")
            except: pass

        st.markdown("---")
        st.subheader("📊 Generated Data Sheets")
        
        # Find all CSV files recursively
        csv_files = sorted(list(rdir.rglob("*.csv")), key=lambda x: x.stat().st_mtime, reverse=True)
        
        if csv_files:
            for fcsv in csv_files:
                with st.expander(f"📄 {fcsv.name}", expanded=True):
                    try:
                        import pandas as pd
                        df = pd.read_csv(fcsv)
                        st.dataframe(df, width="stretch")
                        st.download_button(f"Download {fcsv.name}", open(fcsv, "rb"), file_name=fcsv.name, key=f"dl_res_{fcsv.name}")
                    except Exception as e:
                        st.error(f"Error loading {fcsv.name}: {e}")
        else:
            st.info("No CSV data files generated yet.")
    else:
        st.info("No run data available.")

with t_exp:
    st.subheader("📂 Run File Browser")
    if st.session_state.run_dir:
        # Show full file tree
        rdir = Path(st.session_state.run_dir)
        render_file_explorer(rdir, "exp_root", None) # No filter, show all
    else:
        st.info("No active run directory.")

