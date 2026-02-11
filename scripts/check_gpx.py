
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from GSASII import GSASIIscriptable as G2sc

def check_gpx_internals():
    gpx_path = r"C:\Users\ly6\Documents\My_coding_projects\scripts\crash_mp46.gpx"
    if not os.path.exists(gpx_path):
        print(f"File not found: {gpx_path}")
        return

    proj = G2sc.G2Project(gpxfile=gpx_path)
    print(f"--- GPX INTERNALS: {gpx_path} ---")
    
    # Check max cycles
    try:
        max_cyc = proj.data['Controls']['data']['max cyc']
        print(f"Max Cycles: {max_cyc}")
    except:
        print("Max Cycles: Could not find")

    hist = proj.histograms()[0]
    print(f"Histogram: {hist.name}")

    for p in proj.phases():
        print(f"\nPhase: {p.name}")
        # Scale Refine Flag
        try:
            hap_data = p.data.get('Histograms', {}).get(hist.name, {})
            refine_scale = hap_data.get('Scale', [1.0, False])[1]
            scale_val = hap_data.get('Scale', [1.0, False])[0]
            print(f"  Refine Scale: {refine_scale}")
            print(f"  Scale Value: {scale_val}")
        except:
            print("  Refine Scale: Error reading")

        # Reflection Count
        try:
            reflist = hap_data.get('RefList', [])
            print(f"  Reflection Count: {len(reflist)}")
            if len(reflist) == 0:
                print("  WARNING: This phase has 0 reflections. It will trigger index -1 in Least Squares.")
        except:
            print("  Reflection Count: Error reading")

if __name__ == "__main__":
    check_gpx_internals()
