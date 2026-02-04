import os
import shutil
import zipfile
from pathlib import Path

def main():
    # Fix: Resolve paths relative to where this script lives, not CWD
    script_dir = Path(__file__).parent.resolve()
    source_dir = script_dir / "data" / "database_aug"
    zip_name = script_dir / "database_aug.zip"
    
    # Essential items to include
    include_files = [
        "highsymm_metadata.json",
        "catalog_deduplicated.csv",
        "mp_experimental_stable.csv"
    ]
    include_dirs = [
        "profiles64"
    ]

    print(f"Zipping essential database files into {zip_name}...")
    
    if zip_name.exists():
        zip_name.unlink()

    with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as z:
        # Add files
        for fname in include_files:
            fpath = source_dir / fname
            if fpath.exists():
                print(f"   + Adding {fname}")
                z.write(fpath, arcname=f"database_aug/{fname}")
            else:
                print(f"   ! Warning: {fname} not found!")
        
        # Add directories
        for dname in include_dirs:
            dpath = source_dir / dname
            if dpath.exists():
                print(f"   + Adding directory {dname}...")
                for root, dirs, files in os.walk(dpath):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = f"database_aug/{dname}/{file_path.relative_to(dpath)}"
                        z.write(file_path, arcname=arcname)
            else:
                print(f"   ! Warning: directory {dname} not found!")

    print(f"Done! Created {zip_name} ({zip_name.stat().st_size / 1024 / 1024:.2f} MB)")
    print("> Upload this file to Google Drive or GitHub Releases.")

if __name__ == "__main__":
    main()
