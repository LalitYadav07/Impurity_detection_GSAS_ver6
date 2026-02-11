
import sys
import os
import traceback
import shutil
from pathlib import Path

# Ensure GSAS II is in path if needed (though pixi env should have it)
try:
    import GSASII.GSASIIscriptable as G2sc
except ImportError:
    print("Could not import GSASIIscriptable. Make sure you run this in the GSAS-II environment.")
    sys.exit(1)

def test_refine(gpx_path):
    print(f"\n--- Testing Refinement on {gpx_path} ---")
    if not os.path.exists(gpx_path):
        print(f"File not found: {gpx_path}")
        return False

    try:
        proj = G2sc.G2Project(gpxfile=gpx_path)
        phases = proj.phases()
        print(f"Loaded project with {len(phases)} phases:")
        for p in phases:
            print(f"  - {p.name} (id: {p.id})")
        
        hists = proj.histograms()
        if not hists:
            print("No histograms found!")
            return False
        hist = hists[0]
        print(f"Histogram: {hist.name}")

        # Mimic joint_refine_add_phases setup:
        # Background refine ON
        hist.set_refinements({'Background': {'refine': True}})
        
        print("Setting initial scales...")
        main_phase_name = "Ce O2"
        for p in phases:
            # Print cell
            data = p.data
            cell = data['General']['Cell'][1:7]
            print(f"  Phase {p.name}: Cell={cell}")
            sys.stdout.flush()
            
            p.set_HAP_refinements({'Scale': True}, histograms=[hist])
            if p.name != main_phase_name:
                p.HAPvalue('Scale', 0.05, targethistlist=[hist])
            
        # Optional: Corrupt one phase to see if it reproduces the crash
        # (Uncomment to test manually)
        # if len(phases) > 1:
        #     phase_to_corrupt = phases[1]
        #     phase_to_corrupt.data['General']['Cell'][1] = 0.05 
        #     print(f"Propagating corruption test...")
        

        print("Starting refinement...")
        # We use a small number of cycles to fail fast
        # proj.data['Controls']['data']['max cyc'] = 1  <-- Let's stick with whatever is in GPX or force 1
        
        proj.do_refinements([
            {'set': {'Background': {'refine': True}}},
            {'refine': True},
        ])
        print("Refinement SUCCESS!")
        return True

    except Exception:
        print("Refinement FAILED with exception:")
        traceback.print_exc()
        return False

def isolate_bad_phase(gpx_path):
    print("\n=== Isolating Bad Phase ===")
    
    # Load original to get list of phases
    proj = G2sc.G2Project(gpxfile=gpx_path)
    all_phases = [p.name for p in proj.phases()]
    
    main_phase_name = "Ce O2" # Assumed from inspection
    candidates = [p for p in all_phases if p != main_phase_name]
    
    print(f"Main phase: {main_phase_name}")
    print(f"Candidates: {candidates}")

    # Test 1: Main Only
    print(f"\n[Test] Main Phase Only ({main_phase_name})")
    test_gpx = gpx_path.replace(".gpx", "_debug_main.gpx")
    shutil.copy2(gpx_path, test_gpx)
    
    # Open copy and delete others
    proj = G2sc.G2Project(gpxfile=test_gpx)
    
    # We must iterate over a copy of the list because delete_phase modifies it? 
    # Actually projected.phases() returns a fresh list of objects?
    # G2Project.phases() returns a list of G2Phase objects.
    # But delete_phase modifies the project internals.
    # Safer to iterate names.
    
    phases_to_delete = [p for p in proj.phases() if p.name != main_phase_name]
    for p in phases_to_delete:
        proj.delete_phase(p)
    proj.save() # save to disk
    
    if test_refine(test_gpx):
        print("-> Main phase is OK.")
    else:
        print("-> Main phase FAILED. The issue is likely in the main phase or global setup!")
        # If main fails, we don't need to test candidates individually necessarily, but we should check.
        # But let's proceed.

    # Test Each Candidate Individually (Main + Candidate) with FRESH copy each time
    for cand in candidates:
        print(f"\n[Test] Main + {cand}")
        test_gpx = gpx_path.replace(".gpx", f"_debug_{cand}.gpx")
        shutil.copy2(gpx_path, test_gpx)
        
        proj = G2sc.G2Project(gpxfile=test_gpx)
        
        # Delete all EXCEPT main and this candidate
        phases_to_delete = [p for p in proj.phases() if p.name != main_phase_name and p.name != cand]
        for p in phases_to_delete:
            proj.delete_phase(p)
        proj.save()
        
        if test_refine(test_gpx):
            print(f"-> Candidate {cand} is OK.")
        else:
            print(f"-> Candidate {cand} CAUSED CRASH.")

if __name__ == "__main__":
    # Use absolute path for reliability
    base_gpx = r"C:\Users\ly6\Documents\My_coding_projects\runs\four_phase_1\Technical\GSAS_Projects\seq_pass1_compare.gpx"
    
    if len(sys.argv) > 1:
        base_gpx = sys.argv[1]
    
    print(f"Using GPX: {base_gpx}")
    
    # First verify it crashes blindly
    print("Reproducing original crash...")
    if not test_refine(base_gpx):
        # If it crashes, proceed to isolate
        isolate_bad_phase(base_gpx)
    else:
        print("Original file did NOT crash. Maybe it requires specific starting values not saved in GPX?")
