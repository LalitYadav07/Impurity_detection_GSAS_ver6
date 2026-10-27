# HB-2A published CW calibrations

These 52 profiles are unchanged files from the HB-2A instrument team's public
[calibration table](https://neutrons-user-facilities.ornl.gov/instruments/powder/),
retrieved on 2026-10-27. `catalog.json` records the source page hash, ZIP URL/hash,
archive member, file hash and explicit published validity dates/configuration.

Included cycles: 501, 502, 504–511 and 513. Coverage is discontinuous. Not every
cycle publishes every Ge/collimation combination. Cycle 503 has reversed dates
on the source page; cycle 512 has no published GSAS-II download. Neither is
mapped. Dates outside the published windows, including 2026 acquisitions, are
not covered. No adjacent cycle or different collimation is substituted.

Automatic matching establishes the identity of a facility-published calibration.
It does not independently establish accuracy for every specimen or accept its
absorption, displacement, magnetic scattering or sample-environment corrections.
The Live form retains the experiment review and supports a supplied profile for
unmapped setups. The profiles are packaged locally; the application does not
download a changing calibration at launch time.

Before adding a mapping, verify the current official table and exact asset,
retain provenance/checksums, check finite CW coefficients and positive variance
through the measured angular range, test the native GSAS-II importer, and review
an appropriate standard when available. Do not repair a contradictory date range
by guessing.
