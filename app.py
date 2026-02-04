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
        overflow-y: scroll; 
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
    
    /* ============ Success/Warning/Error Boxes ============ */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 8px;
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
    ("screening", "Candidate Screening"),
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
    if "STAGE 0: BOOTSTRAP" in l:
        state["global_stage_idx"] = 0
        state["stage0_status"] = "running"
    elif "STAGE 1: MAIN PHASE REFINEMENT" in l:
        if state["global_stage_idx"] < 1:
            if state["stage0_status"] == "pending":
                state["stage0_status"] = "skipped"
            state["global_stage_idx"] = 1
    elif "STAGE 2: RESIDUAL EXTRACTION" in l:
        state["global_stage_idx"] = 2
    elif "SEQUENTIAL PASS" in l and "discovery" in l:
        state["global_stage_idx"] = 3
        m = re.search(r"PASS (\d+)", l)
        if m:
            state["current_pass"] = int(m.group(1))
            state["pass_stage"] = "screening"
    elif "STAGE 6: FINAL REPORTING" in l:
        state["global_stage_idx"] = 4
        
    # Pass-level anchors
    if state["global_stage_idx"] == 3:
        if "Comprehensive Candidate Screening" in l:
            state["pass_stage"] = "screening"
        elif "[INFO]" in l and "Processing top" in l:
            state["pass_stage"] = "nudging"
        elif "[PEARSON]" in l:
            state["pass_stage"] = "pearson"
        elif "[clone]" in l and "joint" in l:
            state["pass_stage"] = "joint"
        elif "[polish] Starting" in l:
            state["pass_stage"] = "polish"
        elif "PASS" in l and "SUMMARY" in l:
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
            # STABLE KEY: Use path hash to keep expander state stable during updates
            unique_key = f"{key_prefix}_{item.name}"
            # Use a stable key for expanders to prevent closure during reruns
            with st.expander(f"📁 {item.name}", expanded=(depth < 1), key=f"exp_{unique_key}"):
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
                    should_preview = "Plots" in str(item.parent) or "Diagnostics" in str(item.parent)
                    
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

# --- GAME LOOP: FRAGMENT-BASED UPDATE ---
@st.fragment(run_every=2.0)
def run_monitor_fragment():
    """Isolated fragment that updates state and logs periodically."""
    if st.session_state.run_active:
        update_ui_state()

# Invoke the fragment
run_monitor_fragment()
# Periodic Memory Cleanup
gc.collect()

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
    # Tracker
    st.caption(f"Status: {st.session_state.get('current_stage_desc', 'Ready')}")
    st.progress(st.session_state.progress / 100)
    
    # Detailed Stage Tracker
    if st.session_state.run_active or st.session_state.run_finished:
        st.markdown("### 🚦 Pipeline Progress")
        state = st.session_state.pipeline_state
        g_idx = state["global_stage_idx"]
        
        for i, stage_name in enumerate(GLOBAL_STAGES):
            if i == 0: # Special handling for Stage 0 (Skipped/Running/Done)
                status = state["stage0_status"]
                if status == "complete":
                    st.markdown(f"✅ <span style='color: #48bb78; text-decoration: line-through;'>{stage_name}</span>", unsafe_allow_html=True)
                elif status == "skipped":
                    st.markdown(f"⏩ <span style='color: #718096; text-decoration: line-through;'>{stage_name} (Skipped)</span>", unsafe_allow_html=True)
                elif status == "running":
                    st.markdown(f"**🔵 {stage_name}**")
                else:
                    st.markdown(f"⚪ <span style='color: #718096;'>{stage_name}</span>", unsafe_allow_html=True)
            
            elif i < g_idx:
                st.markdown(f"✅ <span style='color: #48bb78; text-decoration: line-through;'>{stage_name}</span>", unsafe_allow_html=True)
            elif i == g_idx:
                st.markdown(f"**🔵 {stage_name}**")
                # Nested Pass Progress
                if i == 3: # Sequential Passes
                    curr_pass = state["current_pass"]
                    curr_p_stage = state["pass_stage"]
                    st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;**Pass {curr_pass}**")
                    for s_key, s_name in PASS_STAGES:
                        # Simple monotonic check for pass stages
                        matched = False
                        for sk, _ in PASS_STAGES:
                            if sk == curr_p_stage: matched = True; break
                        
                        # Since we don't track historical pass stages perfectly yet, 
                        # just show current vs pending
                        if s_key == curr_p_stage:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;🔹 {s_name}")
                        else:
                            st.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;▫️ {s_name}")
            else:
                st.markdown(f"⚪ <span style='color: #718096;'>{stage_name}</span>", unsafe_allow_html=True)

# --- TABS ---
t_run, t_res, t_exp = st.tabs(["🚀 Run & Progress", "📊 Results", "📂 Run File Browser"])

with t_run:
    c1, c2 = st.columns([2, 1])
    with c1:
        c1_t, c1_a = st.columns([1, 1])
        c1_t.subheader("📜 Live Logs")
        st.session_state.log_autoscroll = c1_a.checkbox("🔄 Autoscroll", value=st.session_state.log_autoscroll)
        
        # Format logs with highlighting
        formatted_logs = "<br>".join([format_log_line(line.rstrip()) for line in st.session_state.log_lines])
        
        # Log viewer with anchor for robust JS-driven autoscroll
        st.markdown(f'''
            <div id="log-container" class="log-viewer">
                {formatted_logs}
                <div id="log-anchor"></div>
            </div>
        ''', unsafe_allow_html=True)
        
        # Robust Autoscroll JS using scrollIntoView with a small delay
        if st.session_state.log_autoscroll:
            st.markdown("""
                <script>
                setTimeout(function() {
                    var anchor = window.parent.document.getElementById('log-anchor');
                    if (anchor) {
                        anchor.scrollIntoView({behavior: 'smooth', block: 'end'});
                    }
                }, 100);
                </script>
                """, unsafe_allow_html=True)
    
    with c2:
        st.subheader("🖼️ Artifacts")
        if st.session_state.run_dir:
            rdir = Path(st.session_state.run_dir)
            
            # Primary: New Reorganized Path
            p_dir_new = rdir / "Results" / "Plots"
            diag_dir = rdir / "Diagnostics"
            
            # Legacy/Fallback paths
            p_dir_old = rdir / "plots"
            sub_plots = rdir / st.session_state.get('run_name', '') / "plots"
            
            if p_dir_new.exists():
                st.markdown("**Plots**")
                render_file_explorer(p_dir_new, "art_new", [".png", ".jpg", ".pdf"])
            
            if diag_dir.exists():
                st.markdown("**Diagnostics**")
                render_file_explorer(diag_dir, "art_diag", [".png", ".jpg", ".pdf"])

            if not p_dir_new.exists() and not diag_dir.exists():
                # Fallback check
                if p_dir_old.exists():
                    render_file_explorer(p_dir_old, "art_root", [".png", ".jpg", ".pdf"])
                elif sub_plots.exists():
                    render_file_explorer(sub_plots, "art_sub", [".png", ".jpg", ".pdf"])
                else:
                    st.info("No plots directory found yet.")
        else:
            st.info("Start a run to see artifacts.")

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

# --- GAME LOOP: RERUN TRIGGER ---
# Only trigger rerun when actively running AND not yet finished
# This prevents flickering after the run completes
if st.session_state.run_active and not st.session_state.run_finished:
    # 2.0s delay prevents the "glowing" effect caused by rapid React re-mounts
    time.sleep(2.0) 
    st.rerun()
