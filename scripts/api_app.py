"""
GSAS-II Impurity Detection API
Exposes the pipeline via REST for Custom GPTs and external integration.
"""
import os
import json
import uuid
import shutil
import asyncio
import datetime
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header, Depends
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

app = FastAPI(title="GSAS-II Impurity Detector API")

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_ROOT = PROJECT_ROOT / "data"
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

# --- ENDPOINTS ---

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/api/run", response_model=RunResponse, dependencies=[Depends(verify_api_key)])
async def run_pipeline(
    data_file: UploadFile = File(...),
    cif_file: Optional[UploadFile] = File(None),
    allowed_elements: str = Form(""),
    instrument_type: str = Form("auto"),
    min_phase_fraction: float = Form(0.01)
):
    """
    Starts an impurity detection run.
    Returns a run_id for status polling.
    """
    run_id = datetime_run_id()
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save data file
    data_path = run_dir / data_file.filename
    with open(data_path, "wb") as buffer:
        shutil.copyfileobj(data_file.file, buffer)

    # Save CIF if provided
    main_cif_path = None
    if cif_file:
        cif_path = run_dir / cif_file.filename
        with open(cif_path, "wb") as buffer:
            shutil.copyfileobj(cif_file.file, buffer)
        main_cif_path = str(cif_path)

    # Build internal config
    # We'll use the existing config_builder logic indirectly
    from config_builder import build_pipeline_config
    
    config = build_pipeline_config(
        data_file_path=str(data_path),
        instrument_params=instrument_type,
        main_phase_cif=main_cif_path,
        allowed_elements=allowed_elements.split(",") if allowed_elements else [],
        min_fraction=min_phase_fraction,
        output_dir=str(run_dir)
    )

    config_path = run_dir / "pipeline_config.yaml"
    with open(config_path, "w") as f:
        import yaml
        yaml.dump(config, f)

    # Start runner in background
    asyncio.create_task(execute_pipeline_task(run_id, str(config_path)))

    return RunResponse(run_id=run_id, message="Pipeline started successfully")

@app.get("/api/status/{run_id}", response_model=StatusResponse)
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

@app.get("/api/results/{run_id}", dependencies=[Depends(verify_api_key)])
async def get_results(run_id: str):
    run_dir = RUNS_DIR / run_id
    summary_file = run_dir / "pipeline_summary.json"

    if not summary_file.exists():
        raise HTTPException(status_code=404, detail="Summary not ready or run failed")

    with open(summary_file, "r") as f:
        data = json.load(f)

    # Add download links
    base_url = f"/api/download/{run_id}/"
    for cand in data.get("candidates", []):
        cand["cif_link"] = base_url + "Refined_CIFs/" + cand["sample"] + ".cif"
        cand["plot_link"] = base_url + "Screening_Traces/" + cand["sample"] + "_trace.png"

    return data

@app.get("/api/download/{run_id}/{path:path}")
async def download_file(run_id: str, path: str):
    file_path = RUNS_DIR / run_id / path
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)

# --- UTILS ---

def datetime_run_id():
    import datetime
    return f"api_run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

async def execute_pipeline_task(run_id: str, config_path: str):
    """Bridge to existing PipelineRunner"""
    from runner import PipelineRunner
    # Create a fresh runner
    runner = PipelineRunner(str(PROJECT_ROOT), use_pixi=os.path.exists(PROJECT_ROOT / "pixi.toml"))
    
    # We run it as a subprocess to keep GSAS-II isolated
    # We consume the generator to ensure it finishes
    for line in runner.run(config_path, run_id):
        # We don't necessarily need to capture every line here 
        # as the script itself writes to run_events.jsonl
        pass
