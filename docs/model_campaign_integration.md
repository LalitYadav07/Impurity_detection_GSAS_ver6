# Reviewed model campaigns

Model exploration lives inside the existing Refinement workspace. The original
Results and Transition tracking tabs are retained. A user inspects a completed
RADAR-PD GPX or a verified recipe checkpoint, discovers a finite candidate set,
reviews the models and starts, and explicitly approves and starts a campaign.
Approval binds the source revision, full scientific settings and acquisition
scope. Changing any of these requires a new review.

## Scope and new scans

Choose the selected acquisition or the already published acquisitions for the
same sample and physical bank. Optionally follow future matching GPXs while the
NOVA session is open. Existing unselected acquisitions are not silently added.
Original RADAR-PD outputs and each exact recipe/stage are separate streams.
Identical GPX bytes belonging to different acquisitions retain separate trial
identities.

Each additional GPX must have compatible parent symmetry, atom labels,
chemistry, occupancy, histogram type and native runtime. Structural parent
positions must also match the reviewed mode basis. A changed basis is held for
review and candidate regeneration. Cell changes and per-scan instrument
calibration remain in their own source GPX; they are not copied from the first
scan. The worker independently repeats native compatibility checks before
refining. Incompatible scans remain visible in the acquisition queue.

Pause stops enrollment and new dispatch, preserving the requested follow policy.
Continue explicitly recovers existing jobs before submitting remaining starts.
Ending or clearing a viewer stops its enrollment; already submitted Galaxy jobs
continue independently. Local scientific processes are stopped on session close.
A fresh viewer recovers saved campaign ledgers and inputs from the same Galaxy
History. It does not automatically resume dispatch. Unacknowledged submissions
are held until their stable dispatch identity can be resolved; they are not
blindly retried. A single Galaxy chunk contains several starts, so cancelling
that job cancels all starts still active within that chunk.

Use one active dispatcher viewer for a campaign. History capacity checks and
local campaign locks do not provide a server-side distributed lease between
simultaneously active NOVA containers. Close or pause the older viewer before
continuing in another one. Concurrent-viewer coordination is a separate NDIP
release gate, not an exactly-once guarantee of this client.

## Resource use

The dedicated `neutrons_radar_pd_model_campaign_prototype` worker consumes a
data-only ZIP, an exact GPX and a closed JSON request. It receives no Galaxy,
Bilbao or model credential. Discovery and optional intent drafting run in NOVA.
The worker image contains the reviewed implementation and validated native
stack; the request cannot supply Python code or choose arbitrary modules.

Common GPX/catalog/CIF inputs are captured once per acquisition/protocol.
Independent starts use small manifests and fresh native processes, with one
BLAS/OpenMP thread each. The work-conserving pool respects `GALAXY_SLOTS`, CPU
quota and memory. Missing slot information permits one worker, regardless of
visible CPU count. Initial memory guidance reserves 512 MiB plus 1 GiB per
worker; larger supercells can need a larger resource class and must be measured.
The actual allocation is recorded in the result and displayed in the workspace.

The default permits one active model batch per History, with an explicit option
for two. The same limit includes native preparation jobs. A small pilot interleaves models before
later chunk sizes adapt to measured duration and GPX size. Acquisitions receive
round-robin dispatch. A soft chunk deadline stops new starts; every unstarted
trial remains pending. Trial deadlines, failures and incomplete scientific
coverage stay in the original denominator. Galaxy still controls site-wide
fairness and quotas. The existing Analyze and recipe resource policies are
unchanged.

Structural child mode inventories are batched in groups of up to 32, preserving
a fresh native process per child. The k survey similarly batches native
templates, and excludes Galaxy queue/transfer time from its scientific budget.
Completed evidence is cached for responsive browsing; changed inode, ctime,
mtime or size invalidates the cache immediately, and unchanged evidence is
cryptographically rechecked at least every five minutes and on a new viewer.

## Scientific limits

Structural exploration currently supports commensurate ISODISTORT displacement
modes. The OPD tier and construction budget are explicit. Deferred and failed
children are retained in coverage. Each selected mode gets signed starts, with
at most six modes per child and 2,000 starts per acquisition. Fifty children with
two starts each are supported; the old global 64-seed restriction no longer
incorrectly rejects candidate-specific structural starts.

Magnetic exploration supports reviewed single-k collinear models and
site-symmetry-allowed moment components, including commensurate canting. The
moment model is part of the approved protocol. The latter uses four component
sign combinations for each selected dominant parent axis, at 25% and 75% of
the declared moment ceiling. These are finite starts, not an exhaustive
continuous-orientation search. Operators, final site-symmetry subspaces,
observation arrays, Al/reference masks and fixed nuclear/nuisance parameters
are independently checked. Legacy sealed protocols keep their collinear
policy. Landé g and the moment ceiling require explicit review.

An optional, explicitly reviewed **common histogram intensity scale** may vary
with the magnetic moments. A low-temperature nuclear-only fit can absorb
unmodelled magnetic intensity into its scale; freezing that value can bias a
later magnetic search. This option applies identically to all candidates and
keeps nuclear/magnetic phase amounts linked. Nuclear geometry, calibration,
occupancies, displacement parameters and broadening remain fixed.
It is off for legacy protocols and unrequested LLM plans. The independent audit
allows exactly the selected histogram scale and requires a positive result.
It never releases separate nuclear and magnetic phase scales.

The existing background coefficients may also vary when explicitly reviewed.
The basis, coefficient count, added peak terms and Debye terms remain unchanged;
the worker and independent audit permit only the existing selected histogram's
polynomial coefficients. All candidates use the same declared scope. It is off
for legacy protocols and unrequested plans. A nuclear-only background can absorb
unmodelled magnetic scattering, so a fixed-background screen is not a final joint
refinement. The model-complexity view includes these additional free parameters.

Rwp is independently recalculated using the native GSAS-II convention. Excluded
residuals are omitted, while their observed intensity remains in GSAS-II's
normalization. A separate fit-points-only metric is retained with its own label;
it must not be compared as if it were the native displayed Rwp.

Inside a selected fit, **Compare fit quality and search repeatability** shows
one row per exact candidate embedding in that source protocol. It retains the
terminal and qualified start counts, independent variable count, qualified Rwp
range and lowest unresolved checkpoint. Equal BNS numbers can still identify
different embeddings. Results from another scan, GPX revision or fit protocol
are excluded from this table. Counts across initial configurations describe
optimizer behavior; they are not independent measurements or a confidence in
the proposed structure. The table does not relabel receipts, choose a physical
winner or hide unresolved competition. Compare complexity and conditional
uncertainties alongside fit quality and evidence across temperature.

The setup planner may return an unknown k only as a preparation for a native
survey. No executable protocol accepts an unknown vector. After reviewing and
using a survey candidate, the user discovers and approves a finite catalog.
The survey tests bounded commensurate high-symmetry vectors and fits an
intensity amplitude; it is not a full refinement or proof of magnetic order.
Incommensurate, multi-k, arbitrary cross-component constraints, occupancy and
tensor-ADP searches are outside this executor.

A qualified fit requires checked inputs, the pinned native stack, independent
parameter/observation audits and the user-declared fit criterion. Lower Rwp
alone does not establish a transition or unique structure. Structural evidence
still needs a matched parent/nuisance comparison; that scientific control is not
automatically generated by this integration. Downloaded magnetic GPXs also need
hosted-desktop compatibility validation before release.

## Release procedure

NOVA 0.5.0 and the dedicated worker were deployed to NDIP-test on 2026-10-30.
Further source changes remain local until their image builds, Galaxy pins and
fresh-session validation complete.
`publish-models-to-ghcr.yaml` builds a separate `models-<full SHA>` Linux image
from the pinned native base and gates publication on runtime, adapter and
protocol checks. Build and validate this image before changing Galaxy XML.

The Galaxy repository retains a template and renderer for the active worker.
Render it with the successful full-SHA image or registry digest, run the Galaxy
contract checks, then deploy through the normal prototype process. Publish and
pin both NOVA and its matching worker when the scientific package changes;
their exact implementation signatures must agree. Existing running viewers keep
their previous image and need a fresh session to receive the update.

## NDIP credentials

The NOVA tool declares optional `ornl_illumine` and `bilbao` vault groups. Each
uses a selected per-user `api_key` secret. Galaxy injects `AZURE_ILLUMINE_API_KEY`
and `BCS_API_KEY` into NOVA only. Recheck selection for each new Galaxy tool
version. The headless launcher accepts `--with-illumine --with-bilbao` and submits
only selected group references. Numerical model jobs receive neither key.

## Ending a viewer

End session saves its recovery checkpoint before leaving the page. In the
supervised NDIP image, a detached helper then shuts down that container's own
supervisor, releasing both the application and web proxy. Exiting only the
application child can leave the interactive job running. The helper inherits
no API credentials and is enabled only in the supervised image; local runs
retain their normal application exit. If supervisor control is unavailable,
the page stays open with a warning instead of reporting a successful shutdown.
