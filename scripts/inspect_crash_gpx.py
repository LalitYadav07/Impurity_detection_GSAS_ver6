
import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from GSASII import GSASIIscriptable as G2sc

def inspect_phase():
    gpx_path = r"C:\Users\ly6\Documents\My_coding_projects\scripts\crash_mp46.gpx"
    if not os.path.exists(gpx_path):
        print(f"File not found: {gpx_path}")
        return

    proj = G2sc.G2Project(gpxfile=gpx_path)
    hists = proj.histograms()
    hist = hists[0]
    
    print(f"Project: {gpx_path}")
    print(f"Histogram: {hist.name}")
    
    for p in proj.phases():
        # Get reflections for this histogram
        # Note: In GSAS-II Scriptable, reflections are stored in p.data['Histograms'][hist.name]['RefList']
        # but let's see if we can get them via API
        try:
             # This might trigger the crash if it calculates reflections
             print(f"\nPhase: {p.name}")
             # We can't easily call internal GSAS-II methods without risk of crash
             # Let's just try to see if it's there
             reflist = p.data.get('Histograms', {}).get(hist.name, {}).get('RefList', [])
             print(f"  Existing reflections in list: {len(reflist)}")
        except Exception as e:
             print(f"  Error accessing phase {p.name}: {e}")

    # Try a 1-cycle refinement and capture it
    from io import StringIO
    capture = StringIO()
    old_stdout = sys.stdout
    print("\nAttempting 1-cycle refinement...")
    try:
        sys.stdout = capture
        proj.do_refinements([{'refine': True}])
    except Exception as e:
        print(f"Refinement Exception: {e}")
    finally:
        sys.stdout = old_stdout
    
    output = capture.getvalue()
    print("--- Capture Start ---")
    print(output)
    print("--- Capture End ---")

if __name__ == "__main__":
    inspect_phase()
