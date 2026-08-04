# API Usage

The hosted API allows users to submit the same kind of job that can be configured in the web app. The recommended workflow is:

1. Build and review a configuration in the web app.
2. Download the API configuration.
3. Submit the config and files from a script.
4. Poll status.
5. Download results.

## Health Check

```python
import requests

base = "http://128.219.184.26:8501"
print(requests.get(f"{base}/health", timeout=10).json())
```

Expected result: a JSON response with service status, project root, jobs root, and time.

## Submit A Job

Export or download the API configuration from the web app after setting up the run. Then submit that config with the current data and instrument files.

```python
import requests

base = "http://128.219.184.26:8501"

files = {
    "config": open("radar_config.yaml", "rb"),
    "data": open("sample.xy", "rb"),
    "instrument": open("instrument.instprm", "rb"),
}

# Include this only when a known main phase is part of the run.
files["main_cif"] = open("main_phase.cif", "rb")

payload = {
    "mode": "rapid",          # rapid, full, or auto
    "dataset": "my_dataset", # dataset name inside the config, when needed
    "run_name": "sample_001",
}

response = requests.post(f"{base}/api/v1/jobs", files=files, data=payload, timeout=60)
response.raise_for_status()
print(response.json()["job_id"])
```

The API rewrites uploaded file paths into an isolated job folder before running. That means your local file paths do not need to exist on the VM.

## Poll Status

The status endpoint returns the job state plus the latest pipeline progress event when the run has started writing events.

```python
import time
import requests

base = "http://128.219.184.26:8501"
job_id = "replace-with-job-id"

while True:
    status = requests.get(f"{base}/api/v1/jobs/{job_id}", params={"tail": 20}, timeout=20).json()
    progress = status.get("progress") or {}
    print(
        status["state"],
        progress.get("stage", "starting"),
        progress.get("percent", "-"),
        "artifacts:", status.get("artifact_count", 0),
    )
    if status["state"] in {"complete", "failed", "cancelled"}:
        break
    time.sleep(10)
```

A verified LK-99 rapid API tutorial job reported `Complete Rapid`, `100%`, and 202 artifacts at completion.

## List Intermediate Artifacts

```python
import requests

base = "http://128.219.184.26:8501"
job_id = "replace-with-job-id"

artifacts = requests.get(f"{base}/api/v1/jobs/{job_id}/artifacts", timeout=30).json()["artifacts"]
for item in artifacts[:20]:
    print(item["path"], item["size"])
```

Use `GET /api/v1/jobs/{job_id}/artifact?path=relative/path` to download one intermediate file.

## Download Results

```python
from pathlib import Path
import requests

base = "http://128.219.184.26:8501"
job_id = "replace-with-job-id"

result = requests.get(f"{base}/api/v1/jobs/{job_id}/results.zip", timeout=120)
result.raise_for_status()
Path("radar_results.zip").write_bytes(result.content)
```

## Live Acquisition Pattern

For an experiment that writes new scans into a folder, use a small watcher script outside RADAR-PD:

```python
from pathlib import Path
import time

seen = set()
scan_dir = Path("/path/to/scans")

while True:
    for scan in sorted(scan_dir.glob("*.xy")):
        if scan in seen:
            continue
        seen.add(scan)
        print("submit", scan)
        # Submit config + scan + instrument to the API here.
    time.sleep(5)
```

The API should return a job ID quickly. The heavier diffraction analysis continues on the hosted VM.

## Data Retention

API jobs should provide downloadable outputs. If a job is intended to be temporary, download `results.zip` after completion and do not rely on the VM as long-term storage.




