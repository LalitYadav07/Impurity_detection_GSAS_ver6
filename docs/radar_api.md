# RADAR-PD API

The web UI runs on port `8502`. The script API runs on port `8501`.

Base URL on the ORNL network:

```text
http://128.219.184.26:8501
```

## Workflow

1. Configure a run in the web UI.
2. In **Review Run Plan**, download the API config.
3. From another machine, submit that config with the diffraction data and instrument file.
4. Poll the job status.
5. Download `results.zip`.

The API rewrites the uploaded data, instrument, and optional main CIF paths into an ephemeral job folder. The job folder is removed when `results.zip` is downloaded unless `cleanup=0` is used.

## Endpoints

```text
GET  /health
POST /api/v1/jobs
GET  /api/v1/jobs/{job_id}
GET  /api/v1/jobs/{job_id}/log?tail=120
GET  /api/v1/jobs/{job_id}/artifacts
GET  /api/v1/jobs/{job_id}/artifact?path=relative/output/path
GET  /api/v1/jobs/{job_id}/results.zip
DELETE /api/v1/jobs/{job_id}
```

## Submit With Python

```python
from pathlib import Path
import time
import requests

base = "http://128.219.184.26:8501"

with open("my_api_config.yaml", "rb") as config, \
     open("my_pattern.dat", "rb") as data, \
     open("my_instrument.instprm", "rb") as instrument:
    response = requests.post(
        f"{base}/api/v1/jobs",
        files={
            "config": config,
            "data": data,
            "instrument": instrument,
            # "main_cif": open("main_phase.cif", "rb"),
        },
        data={
            "mode": "auto",       # auto, rapid, or full
            "run_name": "scan_001",
        },
        timeout=60,
    )
response.raise_for_status()
job = response.json()
job_id = job["job_id"]
print("job_id:", job_id)

while True:
    status = requests.get(f"{base}/api/v1/jobs/{job_id}", params={"tail": 20}, timeout=30).json()
    progress = status.get("progress") or {}
    print(status["state"], progress.get("percent"), progress.get("stage"), progress.get("message"))
    if status["state"] in {"complete", "failed"}:
        break
    time.sleep(10)

if status["state"] == "complete":
    result = requests.get(f"{base}/api/v1/jobs/{job_id}/results.zip", timeout=120)
    result.raise_for_status()
    Path("radar_results.zip").write_bytes(result.content)
```

## Live Acquisition Pattern

For live experiments, keep the same config and instrument file, then submit each new data file as it appears:

```python
for data_file in sorted(Path("/SNS/IPTS-xxxxx/shared/autoreduce").glob("*.dat")):
    submit_one(data_file)
```

Use a unique `run_name` for each scan.
