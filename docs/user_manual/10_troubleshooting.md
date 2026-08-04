# Troubleshooting

## The Website Does Not Load

Check:

- whether you are on a network allowed by the OpenStack security group,
- whether the VM is powered on,
- whether the app service is running,
- whether you are using the right port: web app on `8502`, API on `8501`.

## Uploaded Files Disappear After Refresh

Use a persistent workspace and saved-run reuse. Browser upload widgets can lose their visible state after refresh, but a completed or staged run should keep copied files in the run folder.

## GSAS-II Is Not Ready

Open System Diagnostics and run the GSAS-II check. If it fails, the hosted app environment needs attention before refinement modes will work.

## A CIF Fails To Attach

Possible causes:

- invalid cell parameters,
- symmetry mismatch,
- missing atomic positions,
- malformed CIF syntax,
- lattice nudging produced a cell inconsistent with the space group.

Check whether the same CIF can be loaded in GSAS-II directly.

## Main Phase Fraction Collapses

This can happen if the main phase lattice is not anchored and another candidate reproduces the same strong peaks. Use main phase anchoring and main-shadow filtering. Then inspect whether the main phase strongest peaks are matched before impurity search.

## Rapid Mode Finds A Strange Representative

Open the solution inspector. If the representative is a less intuitive database variant, look for alternate family members with the same element set, similar stoichiometry, similar pattern, and compatible space group.

## X-ray Data Looks Poor

For PXRD, check:

- X-ray mode is selected,
- instrument profile matches the wavelength and geometry,
- K-alpha doublet behavior is expected,
- background is not overwhelming the signal,
- 2-theta range is correctly interpreted.

## Run Failed But Results Page Looks Old

The active view may still show an older completed run. Open the Run Monitor and check the live run name, status, and failure message. Then use Open Previous Run to load a specific completed run.

