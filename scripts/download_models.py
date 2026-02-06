import os
import sys
import subprocess
from pathlib import Path

# Repository base URL for raw files
REPO_RAW_URL = "https://github.com/LalitYadav07/Impurity_detection_GSAS_ver6/raw/master"

MODELS = [
    "ML_components/two_phase_training.pt",
    "ML_components/residual_training.pt",
    "ML_ranker/mlp_ranker_for_phase_detection-main/mlp_ranker.pt"
]

def is_lfs_pointer(path):
    if not os.path.exists(path):
        return True
    if os.path.getsize(path) > 1024: # Actual models are > 500KB
        return False
    with open(path, "r") as f:
        content = f.read(100)
        return "version https://git-lfs.github.com/spec/v1" in content

def download_model(relative_path):
    target_path = Path(relative_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    
    url = f"{REPO_RAW_URL}/{relative_path}"
    print(f"Downloading {relative_path} from GitHub...")
    
    try:
        # Use curl for simplicity on Linux/macOS, falls back to requests if needed later
        subprocess.run(["curl", "-L", "-o", str(target_path), url], check=True)
        print(f"Successfully downloaded {relative_path}")
    except Exception as e:
        print(f"Error downloading {relative_path}: {e}")
        return False
    return True

def main():
    base_dir = Path(__file__).parent.parent
    os.chdir(base_dir)
    
    any_downloaded = False
    for model_path in MODELS:
        if is_lfs_pointer(model_path):
            print(f"Detected LFS pointer or missing file: {model_path}")
            if download_model(model_path):
                any_downloaded = True
        else:
            print(f"Model already present: {model_path}")

    if not any_downloaded:
        print("All ML models are already present.")
    else:
        print("ML model download process complete.")

if __name__ == "__main__":
    main()
