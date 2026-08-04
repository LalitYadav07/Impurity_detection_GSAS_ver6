# GSAS-II Instrument Parameter Source Notes

This catalog should include sources that publish downloadable GSAS-II instrument
parameter assets (`.instprm`, ZIPs containing `.instprm`, or GSAS-II `.gpx`
calibration inputs).

Do not add a beamline only because it is a diffraction beamline. Add it only
when a public source exposes a GSAS-II parameter/calibration asset, or add it to
the review notes below until a downloadable file is verified.

## Active Sources

| Source code | Institution | Facility | Beamline/instrument | Status | Source |
| --- | --- | --- | --- | --- | --- |
| `HB2A` | Oak Ridge National Laboratory | HFIR | POWDER / HB-2A | Downloads and extracts `.instprm` ZIPs | https://neutrons.ornl.gov/powder/users |
| `HB2C` | Oak Ridge National Laboratory | HFIR | WAND2 / HB-2C | Downloads GSAS-II `.gpx` when `--include-gpx` is used | https://neutrons.ornl.gov/wand/users |
| `NOM` | Oak Ridge National Laboratory | SNS | NOMAD / BL-1B | Downloads and extracts `.instprm` ZIPs | https://neutrons.ornl.gov/nomad/users |
| `PG3` | Oak Ridge National Laboratory | SNS | POWGEN / BL-11A | Downloads and extracts `.instprm` ZIPs | https://neutrons.ornl.gov/powgen/users |
| `DESY_P61B` | Deutsches Elektronen-Synchrotron DESY | PETRA III | P61B Large Volume Press | Downloads and extracts `Ge-SSD_params_P61B.instprm` from the official `Instrm_GSAS2_eng.zip` | https://photon-science.desy.de/facilities/petra_iii/beamlines/p61_high_energy_wiggler_beamline_lvp/p61b_large_volume_press_desy/software_tools/index_eng.html |

## Advertised But Not Currently Downloadable Headlessly

| Source code | Institution | Facility | Beamline/instrument | Status | Source |
| --- | --- | --- | --- | --- | --- |
| `APS_11BM` | Argonne National Laboratory | APS | 11-BM | The standards page advertises `GSAS-II .instprm file for 11-BM`, but the target currently returns HTTP 403/Cloudflare to headless requests. Keep in manifest as an error until a stable public URL is found. | https://wiki-ext.aps.anl.gov/ug11bm/index.php/Standards_Data |
| `CLS_BXDS_WLE` | Canadian Light Source | BXDS/WLE | High-Resolution PXRD | The WLE page advertises `BXDS_Huber.instrprm` as a GSAS-II instrument parameter file, but the download is behind an Atlassian share whose direct file URL has not been validated yet. | https://brockhouse.lightsource.ca/about/low-energy-wiggler-beamline/ |

## Reviewed Without Public GSAS-II Asset Found Yet

| Institution/facility | Beamline/instrument | Notes |
| --- | --- | --- |
| NIST NCNR | BT-1 | Public pages describe BT-1 data and GSAS/EXPGUI workflows, but no public GSAS-II `.instprm` download was found in the initial search. |
| BNL NSLS-II | XPD 28-ID-2, PDF 28-ID-1 | Public pages describe diffraction/PDF capabilities, but no public GSAS-II `.instprm` download was found in the initial search. |
| Europe/Japan broader search | ILL, ISIS, ESRF, Diamond, J-PARC, SPring-8 | Initial web search found GSAS-II documentation/tutorial references but no verified public beamline `.instprm` source. Needs deeper facility-by-facility search. |
