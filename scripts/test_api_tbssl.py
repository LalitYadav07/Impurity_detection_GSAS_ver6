
import requests
import time
import sys
import os

# Configuration
# Configuration
BASE_URL = "https://lalityadav07-phase-detection.hf.space"
DATA_FILE = r"data\HB2A_TbSSL.dat"
CIF_FILE = r"data\cifs\TbSSL.cif"
INST_FILE = r"data\instrument_params\hb2a_si_ge113.instprm"

def test_pipeline():
    print(f"=== Testing API with TbSSL ===")
    print(f"Target: {BASE_URL}")
    
    # 1. Start Run
    print("\n[1] Submitting Run...")
    try:
        with open(DATA_FILE, "rb") as f_data, open(CIF_FILE, "rb") as f_cif, open(INST_FILE, "rb") as f_inst:
            files = {
                "data_file": ("HB2A_TbSSL.dat", f_data, "application/octet-stream"),
                "instrument_file": ("instrument.prm", f_inst, "application/octet-stream"),
                "cif_file": ("TbSSL.cif", f_cif, "application/octet-stream")
            }
            data = {
                "min_phase_fraction": 0.01,
                "allowed_elements": "", # No restriction
                "fast_mode": "True"
            }
            
            resp = requests.post(f"{BASE_URL}/run", files=files, data=data, timeout=30)
            
            if resp.status_code != 200:
                print(f"FAILED to submit run: {resp.text}")
                return
            
            run_data = resp.json()
            run_id = run_data["run_id"]
            print(f"[OK] Job Started! Run ID: {run_id}")

    except Exception as e:
        print(f"Error submitting job: {e}")
        return

    # 2. Poll Status
    print("\n[2] Polling Status...")
    while True:
        try:
            # Note: In API-Only mode (v7.2), the prefix is just /status, not /api/status.
            # But wait, api_only.py mounts directly at root.
            # Let's try both or just update to what api_only.py defines.
            # api_only.py defines: @app.post("/run"), @app.get("/status/{run_id}")
            
            r_status = requests.get(f"{BASE_URL}/status/{run_id}")
            if r_status.status_code != 200:
                print(f"Warning: Status check failed ({r_status.status_code})")
                time.sleep(2)
                continue
                
            info = r_status.json()
            status = info["status"]
            progress = info.get("progress", 0.0) * 100
            msg = info.get("last_event") or "Processing..."
            
            print(f"\rStatus: {status.upper()} | Progress: {progress:.1f}% | {msg[:50]}...", end="")
            
            if status in ["completed", "error"]:
                print("") # Newline
                break
            
            time.sleep(2)
            
        except KeyboardInterrupt:
            print("\nAborted by user.")
            return

    # 3. Get Results
    print("\n[3] Fetching Results...")
    if status == "completed":
        r_res = requests.get(f"{BASE_URL}/results/{run_id}")
        if r_res.status_code == 200:
            results = r_res.json()
            print("\n[SUCCESS] ANALYSIS COMPLETE")
            print(f"Sample: {results.get('sample_name')}")
            print(f"Candidates Found: {len(results.get('candidates', []))}")
            if results.get('candidates'):
                best = results['candidates'][0]
                print(f"Top Candidate: {best.get('phase_name', 'Unknown')} (Score: {best.get('score', 0):.2f})")
            
            print("\nDownload Links:")
            # Note: api_only.py defines @app.get("/download/{run_id}/{path:path}")
            # The JSON returns "cif_link" which we set in api_only.py as base_url + ...
            # api_only.py sets base_url = f"/download/{run_id}/" which already includes the prefix
            # So just ensuring we print correct full URLs
            for c in results.get('candidates', [])[:3]:
                 # relative link from JSON is like /download/... so just append to BASE_URL
                 link = c['cif_link']
                 if link.startswith("/"):
                     print(f"- {c['sample']}: {BASE_URL}{link}")
                 else:
                     print(f"- {c['sample']}: {BASE_URL}/{link}")
        if r_res.status_code == 200:
            results = r_res.json()
            print("\n[SUCCESS] ANALYSIS COMPLETE")
            print(f"Sample: {results.get('sample_name')}")
            print(f"Candidates Found: {len(results.get('candidates', []))}")
            if results.get('candidates'):
                best = results['candidates'][0]
                print(f"Top Candidate: {best.get('phase_name', 'Unknown')} (Score: {best.get('score', 0):.2f})")
            
            # Auto-verification of Zip Download
            zip_link = results.get("zip_link")
            if zip_link:
                # Handle relative URL
                if zip_link.startswith("/"):
                    zip_url = f"{BASE_URL}{zip_link}"
                else:
                    zip_url = f"{BASE_URL}/{zip_link}"
                
                print(f"\n[VERIFYING] Downloading results zip: {zip_url} ...")
                r_zip = requests.get(zip_url, stream=True)
                if r_zip.status_code == 200:
                    local_zip = f"results_{run_id}.zip"
                    with open(local_zip, "wb") as f:
                        for chunk in r_zip.iter_content(chunk_size=8192):
                            f.write(chunk)
                    size_mb = os.path.getsize(local_zip) / (1024 * 1024)
                    print(f"[PASS] Downloaded {local_zip} ({size_mb:.2f} MB)")
                else:
                    print(f"[FAIL] Failed to download zip: {r_zip.status_code}")
            else:
                print("[FAIL] No zip_link found in results.")

            print("\nDownload Links:")
    else:
        print(f"Run ended with status: {status}")

if __name__ == "__main__":
    if not os.path.exists(DATA_FILE):
        print(f"Error: Data file not found at {DATA_FILE}")
    else:
        test_pipeline()
