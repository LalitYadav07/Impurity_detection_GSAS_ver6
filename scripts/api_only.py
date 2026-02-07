"""
GSAS-II API-Only Server (v7.2)
Standalone FastAPI app optimized for Hugging Face Spaces Free Tier.
Writes all data to /tmp to avoid permission/persistence issues.
"""
import os
import json
import shutil
import asyncio
import datetime
import logging
import sys
from pathlib import Path
from typing import Optional

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger("api_only")
logger.info("Starting GSAS-II API-Only Server (v7.2)...")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="GSAS-II Impurity Detector API (Sole)")

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Hugging Face Free Tier only allows writing to /tmp
# We'll use /tmp/gsas_runs for job data
RUNS_DIR = Path("/tmp/gsas_runs")
RUNS_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Runs Directory: {RUNS_DIR}")

# Optional API Key
API_KEY = os.getenv("GSAS_API_KEY")

# --- AUTHENTICATION ---
async def verify_api_key(x_api_key: Optional[str] = Header(None)):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API Key")
    return x_api_key

# --- MODELS ---
class RunResponse(BaseModel):
    run_id: str
    message: str

class StatusResponse(BaseModel):
    run_id: str
    status: str
    progress: float
    last_event: Optional[str] = None

# --- UTILS ---
def datetime_run_id():
    return f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

async def execute_pipeline_task(run_id: str, config_path: str):
    """Bridge to existing PipelineRunner"""
    try:
        # We need to add scripts/ to path to find runner
        sys.path.append(str(PROJECT_ROOT / "scripts"))
        from runner import PipelineRunner
        
        # Initialize runner with original project root implies referencing 
        # static data (structure db) from the image, which is fine (read-only).
        # FORCE use_pixi=False because in Docker we are already in the env
        # and 'pixi' binary is not in PATH.
        runner = PipelineRunner(str(PROJECT_ROOT), use_pixi=False)
        
        logger.info(f"Task {run_id}: Starting pipeline...")
        for line in runner.run(config_path, run_id):
            # Log output to server console for debugging
            if line.strip():
                print(f"[Pipeline {run_id}] {line.strip()}", flush=True)
        
        # Create Zip Archive of results
        shutil.make_archive(str(RUNS_DIR / run_id / "results"), 'zip', str(RUNS_DIR / run_id))
        logger.info(f"Task {run_id}: Pipeline finished. Results zipped.")
        
    except Exception as e:
        logger.error(f"Task {run_id} failed: {e}")

# --- ENDPOINTS ---

@app.get("/")
async def root():
    return {
        "service": "GSAS-II Impurity Detection API",
        "version": "v7.4-api-only",
        "status": "online",
        "docs_url": "/docs"
    }

@app.get("/health")
async def health():
    return {"status": "ok", "storage": str(RUNS_DIR)}

@app.post("/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run_pipeline(
    data_file: UploadFile = File(...),
    instrument_file: UploadFile = File(...),
    cif_file: Optional[UploadFile] = File(None),
    allowed_elements: str = Form(""),
    min_phase_fraction: float = Form(0.01)
):
    run_id = datetime_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"New Run: {run_id} in {run_dir}")

    # Save data file
    data_path = run_dir / data_file.filename
    with open(data_path, "wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    # Save Instrument file
    inst_path = run_dir / instrument_file.filename
    with open(inst_path, "wb") as buffer:
        shutil.copyfileobj(instrument_file.file, buffer)

    # Save CIF if provided
    main_cif_path = None
    if cif_file:
        cif_path = run_dir / cif_file.filename
        with open(cif_path, "wb") as buffer:
            shutil.copyfileobj(cif_file.file, buffer)
        main_cif_path = str(cif_path)

    # Build Config
    # We must ensure config_builder is accessible
    sys.path.append(str(PROJECT_ROOT / "scripts"))
    from config_builder import build_pipeline_config
    
    # Generate YAML config
    config_yaml = build_pipeline_config(
        run_name=run_id,
        data_file=str(data_path),
        instprm_file=str(inst_path), 
        allowed_elements=allowed_elements.split(",") if allowed_elements else [],
        main_cif=main_cif_path,
        min_impurity_percent=min_phase_fraction * 100,
        work_root=str(run_dir),
        project_root=str(PROJECT_ROOT), 
        # Important: db_root must point to the read-only image data, not /tmp
        db_root=str(PROJECT_ROOT / "data" / "database_aug") 
    )

    config_path = run_dir / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        f.write(config_yaml)

    # Launch Background Task
    asyncio.create_task(execute_pipeline_task(run_id, str(config_path)))

    return RunResponse(run_id=run_id, message="Pipeline started successfully")

@app.get("/status/{run_id}", response_model=StatusResponse)
async def get_status(run_id: str):
    run_dir = RUNS_DIR / run_id
    event_file = run_dir / "run_events.jsonl"
    
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run ID not found")

    status = "running"
    progress = 0.0
    last_event = None

    if event_file.exists():
        try:
            with open(event_file, "r") as f:
                lines = f.readlines()
                if lines:
                    last_line = json.loads(lines[-1])
                    last_event = last_line.get("message")
                    progress = last_line.get("progress", 0.0)
                    if last_line.get("type") == "finish":
                        status = "completed"
                    elif last_line.get("type") == "error":
                        status = "error"
        except:
            pass
    
    return StatusResponse(run_id=run_id, status=status, progress=progress, last_event=last_event)

@app.get("/results/{run_id}", dependencies=[Depends(verify_api_key)])
async def get_results(run_id: str):
    run_dir = RUNS_DIR / run_id
    summary_file = run_dir / "pipeline_summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary not ready or run failed")

    with open(summary_file, "r") as f:
        data = json.load(f)

    # Add download links
    base_url = f"/download/{run_id}/"
    data["zip_link"] = base_url + "results.zip"

    for cand in data.get("candidates", []):
        cand["cif_link"] = base_url + "Refined_CIFs/" + cand["sample"] + ".cif"
        cand["plot_link"] = base_url + "Screening_Traces/" + cand["sample"] + "_trace.png"

    return data

@app.get("/download/{run_id}/{path:path}")
async def download_file(run_id: str, path: str):
    file_path = RUNS_DIR / run_id / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
