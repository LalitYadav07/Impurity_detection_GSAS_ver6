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
- `planner.py` / `providers.py`: one planning request plus at most one
  JSON/contract repair. Development explicitly selects Illumine `gpt-6-luna`
  with reasoning `none`; the default NDIP target remains SambaNova
  `gpt-oss-120b`. Network/provider errors never trigger model substitution or
  execution. Revisions include the previous plan. Per-atom capability summaries
  expose symmetry-fixed positions and supported ADP flags before the full context.
- `contracts.py` / `schema.py`: validated complete active sets, resolved phase/atom
  targets, a closed provider-neutral JSON schema, and free/newly-fixed tables.
- `advanced.py`: calibrated fit ranges/exclusions, explicit occupancy/Uiso bounds,
  scale normalization, linear occupancy relations and phase-specific texture.
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
polynomial background, linked phase scales, global histogram scale, cells,
March-Dollase or spherical-harmonic preferred orientation, and selected atom
positions (`X`), isotropic displacement parameters (`U`) and occupancies (`F`).
Background functions/counts and retention/removal of existing support points
are explicit. New constraints are explained linear occupancy equations using
exact observed variable identities, including weighted coefficients and coupled
equations. Occupancy bounds lie within 0–1; held co-occupants reduce a shared
site's available capacity. Released Uiso is bounded below by zero. Explicit bounds
that permit overfilling are rejected rather than silently tightened. Existing
project constraints/restraints and tighter existing bounds are preserved.

A requested normalized multiphase refinement rescales starting HAP phase scales
to sum to one and compensates the global histogram scale so starting calculated
intensities are preserved. It then constrains the HAP scale sum while releasing
global scale and the linked phase scales. A single phase is held at one, with
only global scale released. This is a parameterization convention: raw phase
scales are not weight percentages. The alternative convention of fixed global
scale and free phase scales remains supported. A fixed weight-percent constraint
while atomic occupancies change is nonlinear and is not implemented.

Fit limits and excluded intervals accept explicit native, TOF, d, Q or two-theta
units where the histogram permits them. Conversion uses the installed GSAS-II
calibration functions, including higher-order TOF coefficients and zero offset.
Ranges outside measured coverage and ambiguous units are rejected. Exclusions
must explicitly add, replace or clear intervals. Texture is optional and local
to a selected phase/histogram: nonzero hkl for March-Dollase, or a supported even
spherical-harmonics order from the actual phase symmetry.

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
The vary audit resolves generated constraint variables through their actual
native inverse equations; a variable family or `::constr` prefix is insufficient
authority. Every refinement must produce fresh covariance, with no unexpected
atom/phase/histogram variables. A new occupancy equation that leaves exactly
one released term and holds its peers fixed is rejected because it removes that
term's refinement freedom.

## Model context and approval

A GPX is a native binary/pickle project, not JSON. Only the backend scientific
runtime loads it. Supply trusted pipeline-produced GPXs, not arbitrary files to
the HTTP process. Context is normalized supported-model state, not a lossless
export of every possible GSAS-II tree extension. Repetitive parameter records
are encoded as lossless column/row tables; no values are removed. Qualified
phase/atom capability summaries distinguish identical labels in different phases.
Payloads above 350 kB are
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

The default planner accepts `SAMBANOVA_API_KEY` from its backend environment;
its endpoint is restricted to `https://sambanova.cades.ornl.gov/v1`. Explicit
`--provider illumine-luna` selects the Illumine Responses API at
`https://openai-eus-illumine.openai.azure.com/openai/v1`, using only
`AZURE_ILLUMINE_API_KEY`, reasoning `none` and `store:false`. Both clients refuse
redirects and other resources. HTTP 429 reports safe provider retry guidance;
requests are not retried automatically. Never copy a key between these resources.
Luna uses strict JSON-schema output. Its initial output allowance is 2048 tokens;
one incomplete response can receive a bounded repair with a doubled allowance
(maximum 8192). Total inference remains at most two calls per revision. Explicit
stage counts are checked so joint stages cannot silently become sequential ones.
Clarifications receive at most one capability recheck; a network timeout remains
a provider failure, not evidence that the scientific request is unsupported.

For local development only, `--key-file /outside/repository/private-key-file`
can read an existing small owner-only Illumine key file into backend memory. It
rejects symlinks and public permissions. No key belongs in a plan, GPX, notebook,
repository, browser request or job receipt. Scientific subprocesses receive an
allowlisted environment without either model credential. Production must use
NDIP-managed backend secret injection. No Studio import, client, service,
configuration helper or Python environment is required. Studio supplies only
the vendored knowledge/extractor provenance recorded above.

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

For Studio-independent local development, the existing RADAR-PD Linux/amd64
image can provide the scientific interpreter:

```sh
export RADAR_GPX_DOCKER_IMAGE=radar-pd-ndip:validated-runtime-fix
export RADAR_GPX_DOCKER_WORKDIR=/absolute/validation/root
PYTHONPATH=scripts python -m gpx_recipe.web \
  --provider illumine-luna --key-file /outside/repository/private-key-file \
  --gpx "$RADAR_GPX_DOCKER_WORKDIR/starting.gpx" \
  --workdir "$RADAR_GPX_DOCKER_WORKDIR/review" \
  --python "$PWD/scripts/recipe_docker_python.py" --port 8891
```

The Docker shim uses an explicitly present image (`--pull never`), runs with the
host user's numeric UID/GID and mounts only shared code plus the chosen validation
directory. The model key is not passed to Docker. This is a local test adapter;
the NDIP worker runs directly in its published scientific image. At deployment,
select `--provider sambanova` (or `RADAR_GPX_PLANNER_PROVIDER=sambanova`) and inject
only the SambaNova credential. Revalidate planning with OSS 120B before release;
Luna success does not establish OSS planning accuracy.

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

Subsequent development uses Illumine Luna with reasoning disabled. Initial and
revised real POWGEN recipes were each produced in one call with zero reported
reasoning tokens, independently checked against the requested parameter sets,
and executed on two real scans through the browser harness using the RADAR-PD
container. The UI identifies Luna development and OSS deployment separately.
One synthetic symmetry-fixed request was rejected by the validator, prompting
the explicit capability summary; additional model testing can be delayed by
provider rate limits. Exact results, limitations and follow-up checks
are in `audit/2026-10-27/gpx-refinement-planner/luna-development`.

The subsequent user-authorized campaign is in
`audit/2026-10-27/gpx-recipe-benchmark`. It has 100 explicit natural-language
courses on 11 copied real GPXs, with independently specified expected operations.
There are 87 executable requests and 13 requests requiring clarification or
rejection. The latter include three original bounds that would overfill mixed
sites; their original requests and rejected recipes remain available. Categories
cover occupancies, Uiso, positions plus Uiso, phase-specific sequences, scales,
limits/exclusions, real mixed sites, texture and negative requests. These are
controlled operation tests, not an exhaustive evaluation of arbitrary language.

`benchmarks/gpx_recipes/catalog.py` generates the cases from private local
contexts. `run_campaign.py` uses immutable code snapshots and preserves every
attempt. A model plan must match the independent operation oracle before it
can execute. `native_worker.py` runs the compiled wrapper in the RADAR-PD
Linux/amd64 image and checks actual flags, vary lists, fixed values, bounds,
data/calibration preservation, occupancy equations and scale normalization.
The independent oracle belongs only to the benchmark; it is never supplied to
the model and is not a substitute for user review in production.

`report.py` produces a filterable per-case report and links to exact requests,
model replies, execution receipts and GPXs. `figures.py` renders representative
native fits with residuals and Bragg ticks. Input seals and original source
hashes distinguish real files from constructed tests. Scientific stops, model
intent errors, provider failures and successful mechanics are separate outcomes.
Read the generated report for final counts rather than treating an earlier
attempt as the final implementation. Follow-up regressions have their own
receipts and do not erase the complete campaign's failures.

Site-capacity inference currently recognizes exact co-located coordinates with
matching multiplicity, not all symmetry-equivalent or split-site descriptions.
Mass fractions and their uncertainty need scientific review when occupancies
or structural models change. A correct executed recipe does not validate the
starting phase identities, extinction/absorption models or the resulting structure.

The completed full Luna run on 2026-10-27 recorded 69 verified executions,
16 stops at parameter bounds, 12 correct clarifications/rejections, two intent
errors blocked by the independent oracle, and one HTTPS timeout during repair.
The independent native reference run recorded 70 verified executions and 17
bound stops across the 87 executable requests. Neither recorded a runtime defect.
After improvements to constraint validation and phase-qualified capability
checks, a separate 17-case targeted regression recorded 11 verified executions,
two scientific stops and four correct blocks, including all three outstanding
full-run cases. This is not a second complete 100-case pass. All 139 successful
saved reference/model GPXs passed 355 additional fixed-cell/background and
inverse-calibration checks. The 11 original GPXs and all 23 sealed campaign
inputs were unchanged. The latest focused unit suite has 56 passing tests.
The built NOVA wheel contains the new modules/assets and no GPX inputs; all
100 contexts fit the payload guard (largest 313709 bytes).
