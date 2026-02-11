
import os
import sys
from pathlib import Path

# Add project root to sys.path if needed
sys.path.append(str(Path(__file__).parent.parent))

from GSASII import GSASIIscriptable as G2sc
from scripts.gsas_main_phase_refiner import clone_gpx, get_hist_and_main_phase, set_phase_cell_refine

def generate_crash_gpx():
    base_gpx = r"C:\Users\ly6\Documents\My_coding_projects\runs\four_phase_2\Technical\GSAS_Projects\four_phase_2_project.gpx"
    out_gpx = r"C:\Users\ly6\Documents\My_coding_projects\scripts\crash_mp46.gpx"
    cif_path = r"C:\Users\ly6\Documents\My_coding_projects\runs\four_phase_2\Models\Refined_CIFs\mp-46_nudged.cif"
    main_phase_name = "Ce O2"

    if not os.path.exists(base_gpx):
        print(f"Error: Base GPX not found at {base_gpx}")
        return

    print(f"Cloning {base_gpx} -> {out_gpx}")
    clone_gpx(base_gpx, out_gpx)
    
    proj = G2sc.G2Project(gpxfile=out_gpx)
    hist, main_p = get_hist_and_main_phase(proj, main_phase_name)
    
    print(f"Adding problematic phase mp-46 from {cif_path}")
    p = proj.add_phase(cif_path, phasename="mp-46_CRASH_TEST", histograms=[hist])
    
    # Setup as it was during validation
    set_phase_cell_refine(p, refine=False)
    p.set_HAP_refinements({'Scale': True}, histograms=[hist])
    p.HAPvalue('Scale', 0.05, targethistlist=[hist])
    
    # Fix main phase
    set_phase_cell_refine(main_p, refine=False)
    main_p.set_HAP_refinements({'Scale': False}, histograms=[hist])
    
    hist.set_refinements({'Background': {'refine': True}})
    
    proj.save()
    print(f"Successfully created: {out_gpx}")
    print("You can now open this in GSAS-II and run 1-cycle refinement to see the crash.")

if __name__ == "__main__":
    generate_crash_gpx()
