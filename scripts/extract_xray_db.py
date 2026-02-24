"""
Extract the X-ray database ZIP (which may use Windows backslash path separators)
into data/database_xray/, stripping any top-level wrapper folder.

Usage:
    python3 scripts/extract_xray_db.py /tmp/database_xray.zip data/database_xray
"""
import sys
import zipfile
import os
import shutil

zip_path = sys.argv[1]
dest = sys.argv[2]
os.makedirs(dest, exist_ok=True)

STRIP_PREFIXES = ("database_xray", "database_aug")

with zipfile.ZipFile(zip_path) as z:
    for m in z.infolist():
        # Normalize Windows backslash separators
        name = m.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p]
        if not parts:
            continue

        # Strip top-level wrapper folder if it matches known names
        if parts[0] in STRIP_PREFIXES:
            parts = parts[1:]
        if not parts:
            continue

        rel = "/".join(parts)
        out = os.path.join(dest, rel)

        if name.endswith("/"):
            # Directory entry
            os.makedirs(out, exist_ok=True)
        else:
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)

print(f"Extraction complete. Files in {dest}:")
for f in sorted(os.listdir(dest)):
    print(f"  {f}")
