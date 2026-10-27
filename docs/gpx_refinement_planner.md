# GPX refinement planner prototype

This branch provides a separate, optional continuation of trusted RADAR-PD
powder GPX projects. It has not been deployed to NDIP or wired into the NOVA
Live page. The local interface is a review/test harness, not a replacement for
the existing RADAR-PD interface.

## Components

- `nova/src/gpx_recipe/snapshot.py`: native GSAS-II extraction of phases,
  atom identities, site symmetry and positional degrees of freedom, cells,
  histograms, calibration/profile values, background, HAP models, parameter
  states, constraints/restraints/bounds and covariance summaries. Diffraction,
  calculated-pattern and full covariance arrays are not sent to the planner.
- `studio_state.py`: the MIT-licensed, read-only effective-state extractor
  copied from Diffraction Studio; exact source hashes/revision are recorded in
  `studio-provenance.json`. The Studio repository is not a runtime dependency.
- `knowledge.json`: a distilled vocabulary, actual API forms, flag-reset and
  constraint rules, examples, and explicit version-one limitations.
- `planner.py`: the fixed ORNL SambaNova `gpt-oss-120b` endpoint, one planning
  request plus at most one JSON/contract repair. Network/provider errors do not
  trigger model substitution or execution. Revisions include the previous plan.
- `contracts.py`: validated complete active sets, resolved phase/atom targets,
  background edits, occupancy relations, and free/newly-fixed tables.
- `execution.py`: clears ordinary refinement flags before each stage, preserves
  original constraints/restraints and data, performs native preflight, records
  effective and actual independent variables, and branches stage GPXs.
- `compiler.py`: exact-plan approval receipts and deterministic Python wrapper
  generation. Model text is never evaluated as Python.
- `jobs.py`: persistent per-scan jobs, stable run/bank identity plus input/plan
  hashes, duplicate suppression, bounded native execution, restart handling and
  explicit retries. `StableArrivals` requires two unchanged file observations.
- `web.py` / `web.html`: local review, revision, preflight, approval, downloads,
  and execution on an explicitly supplied set of source scans.

## Supported scientific scope

Version one accepts ordinary nuclear powder projects. A stage can release the
polynomial background, linked phase scales, cells and explicitly selected atom
positions (`X`), isotropic displacement parameters (`U`) and occupancies (`F`).
Background functions/counts and retention/removal of existing support points
are explicit. New constraints are explained linear occupancy sums using exact
observed variable identities. Occupancy bounds are 0–1; released Uiso is bounded
below by zero. Existing project constraints/restraints are preserved.

Magnetic/modulated and mixed single-crystal projects, rigid-body operations,
instrument calibration changes, anisotropic ADP release, new phases/atoms,
symmetry transformations and general arbitrary Python are not supported here.
Requests for them must remain unresolved/unsupported, never be silently omitted.
Some unsupported retained model terms will be caught by native effective-state
preflight; an arbitrary GPX is not promised to be compatible.

Each stage's `free` list is complete, not cumulative. A lower Rwp does not imply
scientific acceptance. Jobs require review of values, convergence, covariance,
correlations, constraints and residuals. A native error, missing fresh covariance,
unexpected vary list or variable frozen at a bound stops execution. The original
GPX is immutable; intermediate and failed-attempt artifacts remain available.

## Model context and approval

A GPX is a native binary/pickle project, not JSON. Only the backend scientific
runtime loads it. Supply trusted pipeline-produced GPXs, not arbitrary files to
the HTTP process. Context is normalized supported-model state, not a lossless
export of every possible GSAS-II tree extension. Payloads above 350 kB are
rejected, not silently truncated. Coordinates/cells and scan values are retained
in context but excluded from the compatibility signature, so an approved recipe
can run on later compatible scans. Phase/atom topology, labels, site symmetry,
instrument values and initial background type/count are bound strictly.

For NDIP integration, also bind the existing RADAR-PD stable phase/CIF provenance
to the approval. Matching GPX labels alone cannot certify that arbitrary same-named
structural variants have the same scientific identity. Every worker independently
re-extracts its actual input and checks compatibility before refinement.

Preflight configures every stage without refining and reconstructs independent
variables through GSAS-II. Approval binds the exact plan and successfully
preflighted source. A user revision invalidates prior preflight/approval, even if
the model request fails. Compilation uses tested operations and requires no new
model call. Runtime execution requires no model service or credential.

## CLI

Run from the source checkout, using the installed scientific interpreter for
`inspect`, `preview` and `run`:

```sh
python scripts/ndip_recipe.py inspect starting.gpx context.json
python scripts/ndip_recipe.py context context.json request.txt payload.json
python scripts/ndip_recipe.py plan context.json request.txt planning.json
python scripts/ndip_recipe.py plan context.json changes.txt revised.json --previous planning.json
python scripts/ndip_recipe.py preview starting.gpx revised.plan.json preview.gpx
python scripts/ndip_recipe.py approve context.json revised.plan.json preview.execution.json approval.json --approver USER
python scripts/ndip_recipe.py compile revised.plan.json approval.json refine_project.py
PYTHONPATH=scripts python refine_project.py --source-gpx starting.gpx --output-gpx polished.gpx
```

`GSASII_ROOT` optionally locates a checkout; a pip-installed `GSASII` package
works without it. Never hard-code a Mac path in the recipe/script. The scientific
worker needs the existing NumPy/GSAS-II runtime; the planner's HTTPS client uses
the standard library. The context/contract modules use NumPy through the shared
extractor, which is already present in NOVA and the batch image.

The planner accepts `SAMBANOVA_API_KEY` only from its backend environment. Its
fixed endpoint is `https://sambanova.cades.ornl.gov/v1`; it refuses redirects or
another host. No key belongs in a plan, GPX, notebook, repository, browser request
or job receipt. The local validation harness may load the already configured
Studio provider through its existing configuration helper in memory. Production
must use NDIP-managed backend secret injection; it must not depend on Studio,
Mac Keychain, a laptop tunnel or laptop paths.

## Local interface

```sh
PYTHONPATH=scripts python -m gpx_recipe.web \
  --gpx starting.gpx --gpx next_scan.gpx \
  --workdir /work/recipe-review \
  --python /path/to/scientific/python \
  --port 8891
```

The optional `--fixture-plan` explicitly labels a deterministic execution test
recipe as **not model generated**. It allows preflight/execution/UI validation
while the external model is unavailable. It is not evidence of one-shot model
accuracy. The server binds loopback and is a local harness. Integrate its service
operations into the authenticated NOVA backend, not a public standalone HTTP
server. Neither core calculations nor workers require the local web harness.

## NDIP deployment boundary

The shared core is under `nova/src/gpx_recipe` and ships in the NOVA wheel.
The batch image copies the same source and uses a thin `scripts/gpx_recipe`
bridge. Both image workflows include the shared core in their path rules. The compiled wrapper imports
`gpx_recipe` from the batch image's existing `/opt/radar-pd/scripts` PYTHONPATH.
See `galaxy/radar_pd_refine_recipe.xml.in` for the worker contract; replace its
immutable image pin only after publication/testing. It is a template, not a live
Galaxy registration. The prototype does not push any repository or deploy tools.

Production wiring still needs: authenticated NOVA plan/revision/approval actions,
secure model configuration, Galaxy
worker registration with a tested immutable image, and dispatch of accepted final
GPX datasets as they arrive. Pass a stable run/bank/revision or HDA ID as `scan_id`;
local paths are only a local fallback. Keep preliminary RADAR-PD results visible
while these optional continuation jobs run. New incompatible phases/models
require review, not positional phase-index guesses.

## Validation scope

Validation artifacts are outside git in `audit/2026-10-27/gpx-refinement-planner`.
They distinguish deterministic test recipes, real native refinements, constructed
mixed-site API fixtures, Linux parity and actual provider responses/timeouts.
See the generated validation JSON there for current counts and remaining gaps.

On 2026-10-27, 25 contract/job/session tests passed. A four-stage deterministic
recipe completed on two distinct real POWGEN GPXs through the review interface.
The same warm-scan recipe passed in native macOS and Linux/amd64 GSAS-II, including
an arbitrary non-root container user. All stage independent-variable lists matched
their preflights (10, 13, 6 and 6 variables); the largest Mac/Linux Rwp difference
was 0.00000103 percentage points. Five-phase NOMAD context extraction and a
constructed mixed-site occupancy preflight also passed; these are not claimed as
new scientifically validated NOMAD refinements.

Live 120B planning remains unvalidated: the model catalog responds, but planning
and minimal completion requests timed out, including the existing Studio client.
The displayed local recipe is explicitly an execution fixture. These results
validate execution mechanics and portability, not model planning accuracy or
general scientific correctness of arbitrary user recipes.
