
import sys
import os
import shutil
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.abspath("scripts"))
try:
    from GSASII import GSASIIscriptable as G2sc
except ImportError:
    print("Could not import GSASIIscriptable")
    sys.exit(1)

def reproduce_crash():
    work_dir = Path("reproduce_work")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir()
    
    # Paths
    base_gpx = r"c:\Users\ly6\Documents\My_coding_projects\runs\four_phase_4\Technical\GSAS_Projects\seq_pass1_compare.gpx"
    cand_cif = r"c:\Users\ly6\Documents\My_coding_projects\runs\four_phase_4\Models\Refined_CIFs\mp-19399_nudged.cif"
    target_gpx = work_dir / "repro.gpx"
    
    print(f"Copying base GPX {base_gpx} -> {target_gpx}")
    shutil.copy2(base_gpx, target_gpx)
    
    print(f"Opening project {target_gpx}")
    proj = G2sc.G2Project(gpxfile=str(target_gpx))
    
    # Get Histogram (should be present)
    hist = proj.histograms()[0]
    print(f"Found histogram: {hist.name}")
    
    # Get Main Phase (should be present)
    # pipeline uses get_hist_and_main_phase, usually "Ce O2"
    main_ph_name = "Ce O2"
    main_ph = None
    for p in proj.phases():
        if p.name == main_ph_name:
            main_ph = p
            break
    
    if not main_ph:
        print(f"Main phase {main_ph_name} not found in GPX")
        # List phases
        print("Phases in project:", [p.name for p in proj.phases()])
        return

    print(f"Found main phase: {main_ph.name}")
    
    # Add Candidate Phase
    print(f"Adding candidate phase: {cand_cif}")
    cand_ph = proj.add_phase(cand_cif, phasename="mp-19399", histograms=[hist])
    
    # Configure Refinement - EXACTLY like validation
    print("Configuring refinement...")
    
    def set_cell_fixed(ph):
        ph.set_refinements({'Cell': False})

    # Main Phase: Fixed Scale, Fixed Cell
    # Validation loop: 
    #   p_val = proj_val.add_phase(cif...
    #   p_val.set_HAP_refinements({'Scale': True}, ...)
    #   hist_val.set_refinements({'Background': {'refine': True}})
    #   set_phase_cell_refine(p_val, refine=False)
    #   for other_p: set_phase_cell_refine, set_HAP_Scale=False
    
    # So Main Phase (other_p):
    set_cell_fixed(main_ph)
    try:
        main_ph.set_HAP_refinements({'Scale': False}, histograms=[hist])
    except Exception as e:
        print(f"Warning setting main phase HAP: {e}")

    # Candidate Phase: Refine Scale, Fixed Cell
    set_cell_fixed(cand_ph)
    cand_ph.set_HAP_refinements({'Scale': True}, histograms=[hist])
    # Set initial scale (hap_init from pipeline default 0.01 or calculated?)
    # pipeline uses float(hap_init). In gsas_main_phase_refiner, hap_init defaults to 0.05 or computed.
    # Let's try 0.01
    cand_ph.HAPvalue('Scale', 0.01, targethistlist=[hist])
    
    # Background: Refine
    hist.set_refinements({'Background': {'refine': True}})
    
    # Print parameters
    # Print parameters by inspecting data dict
    print(f"Main Phase ({main_ph.name}) Data keys: {list(main_ph.data.keys())}")
    
    print(f"Candidate Phase ({cand_ph.name}) Data:")
    # Check 'Histograms' entry for our histogram
    hist_key = f"PWDR {hist.name}" # GSAS-II uses "PWDR " + name usually
    # Or just iterate
    for k, v in cand_ph.data.items():
        if k == 'Histograms':
            print("  Histograms:")
            for hk, hv in v.items():
                print(f"    {hk}: {hv.keys()}")
                if 'Instrument Parameters' in hv:
                     print(f"      Inst Params: {hv['Instrument Parameters']}")
                if 'Histogram Scale' in hv:
                     print(f"      Scale: {hv['Histogram Scale']}")
                # Profile parameters might be here or in main dict?
                # Size/Strain are in main dict usually
        elif k in ['General', 'Pawley ref', 'Atoms']:
            # limit output
            pass
        else:
            print(f"  {k}: {v}")
            
    # Check Histogram Instrument Parameters
    print(f"Histogram ({hist.name}) Instrument Parameters:")
    try:
        inst = hist.getHistEntryValue(['Instrument Parameters'])[0]
        for k, v in inst.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"  Error getting inst params: {e}")
    
    # Run Refinement with Safe Limits Check
    print("Applying Safe Limits from library...")
    try:
        sys.path.append(str(Path(__file__).parent / "scripts"))
        from gsas_safe_limits import apply_safe_limits
        applied = apply_safe_limits(proj)
        print(f"Safe limits applied: {applied}")
    except ImportError:
        print("Could not import gsas_safe_limits. Please ensure scripts/gsas_safe_limits.py exists.")
    except Exception as e:
        print(f"Error applying safe limits: {e}")

    print("Starting refinement (1 cycle) with library safe limits...")
    proj.data['Controls']['data']['max cyc'] = 1
    
    try:
        res = proj.do_refinements([{'refine': True}])
        print("Refinement finished without exception!")
    except Exception as e:
        print(f"Refinement crashed with exception: {e}")
        # In validation loop, we capture stdout. G2sc usually prints to stdout.
        
if __name__ == "__main__":
    reproduce_crash()
