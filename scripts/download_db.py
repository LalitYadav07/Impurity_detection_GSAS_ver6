import os
import sys
import zipfile
import shutil
from pathlib import Path
import gdown

# Constants (Shared logic with app.py)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "data" / "database_aug"
DB_URL = "https://drive.google.com/uc?id=1BxPXjdbn7oYTXKfDeLct5-2PMkhcLVSH"

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

def main():
    if check_db_integrity():
        print("[INFO] Database already exists and is healthy. Skipping download.")
        return

    print(f"[INFO] Database missing or incomplete. Starting automated download...")
    DB_DIR.mkdir(parents=True, exist_ok=True)
    
    zip_path = PROJECT_ROOT / "temp_db.zip"
    gdrive_id = get_gdrive_id(DB_URL)
    
    try:
        if gdrive_id:
            print(f"[INFO] Downloading from Google Drive (ID: {gdrive_id})...")
            gdown.download(id=gdrive_id, output=str(zip_path), quiet=False)
        else:
            print(f"[ERROR] Could not extract GDrive ID from {DB_URL}")
            sys.exit(1)

        if not zip_path.exists() or not zipfile.is_zipfile(zip_path):
            print("[ERROR] Downloaded file is not a valid ZIP.")
            if zip_path.exists(): zip_path.unlink()
            sys.exit(1)

        print("[INFO] Extracting database archive...")
        temp_extract_dir = PROJECT_ROOT / "temp_extract"
        if temp_extract_dir.exists(): shutil.rmtree(temp_extract_dir)
        temp_extract_dir.mkdir()

        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_extract_dir)
        
        extracted_db_path = temp_extract_dir
        if (temp_extract_dir / "database_aug").exists():
            extracted_db_path = temp_extract_dir / "database_aug"
        
        print(f"[INFO] Moving files to {DB_DIR}...")
        for item in extracted_db_path.iterdir():
            dest = DB_DIR / item.name
            if dest.exists():
                if dest.is_dir(): shutil.rmtree(dest)
                else: dest.unlink()
            shutil.move(str(item), str(DB_DIR))
            
        # Cleanup
        zip_path.unlink()
        shutil.rmtree(temp_extract_dir)
        
        if check_db_integrity():
            print("[SUCCESS] Database installed and verified.")
        else:
            print("[ERROR] Database integrity check failed after extraction.")
            sys.exit(1)

    except Exception as e:
        print(f"[ERROR] Database installation failed: {e}")
        if zip_path.exists(): zip_path.unlink()
        sys.exit(1)

if __name__ == "__main__":
    main()
