# Quick Start

## Web App First Run

1. Open `http://128.219.184.26:8502/` from a machine that can reach the hosted app.
2. Open or create a workspace. For documentation examples, use `docs_demo` and PIN `2468`.
3. Choose **Measurement Type** first: neutron or X-ray.
4. Choose a **Candidate Library**:
   - built-in MP/COD catalog,
   - built-in catalog plus your CIF files,
   - only your CIF files.
5. Upload diffraction data and an instrument parameter file, or reuse files from a previous run.
6. Optionally upload a known main phase CIF.
7. Enter sample elements and sample can/environment elements.
8. Review the run plan.
9. Choose Full RADAR-PD or Rapid Hypothesis Mode.
10. Start the run and monitor live logs, progress, artifacts, and results.

## Recommended First Choice

Use **Full RADAR-PD** when the result will be used for serious interpretation, reporting, or decision making. Use **Rapid Hypothesis Mode** when you want a fast map of plausible phase combinations before spending time on a longer refinement workflow.

## API First Run

The API is useful when data arrives from an automated experiment or when you want to submit multiple scans without using the browser.

```python
from pathlib import Path
import time
import requests

base = "http://128.219.184.26:8501"

files = {
    "config": open("radar_config.json", "rb"),
    "data": open("scan.xy", "rb"),
    "instrument": open("instrument.instprm", "rb"),
}

response = requests.post(f"{base}/api/v1/jobs", files=files, timeout=60)
response.raise_for_status()
job = response.json()["job_id"]

while True:
    status = requests.get(f"{base}/api/v1/jobs/{job}", timeout=20).json()
    progress = status.get("progress") or {}
    print(status["state"], progress.get("stage", "starting"), progress.get("percent", "-"))
    if status["state"] in {"complete", "failed", "cancelled"}:
        break
    time.sleep(10)

if status["state"] == "complete":
    result = requests.get(f"{base}/api/v1/jobs/{job}/results.zip", timeout=120)
    Path("results.zip").write_bytes(result.content)
```

The practical API chapter later in this manual explains how to create the config from the web app and how to poll partial outputs.




