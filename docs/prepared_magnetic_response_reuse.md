# Prepared magnetic response reuse

Model Campaign and Live Snapshot use the same native magnetic response
compiler. A selected preparation can supply both reflection-level scattering
factors and a fully rendered, unweighted magnetic quadratic response. Neither
artifact contains observations, variances, likelihoods or fitted results.

The rendered response avoids reconstructing every moment cross term and
repeating native peak-profile capture at a matching physical context. The
consumer still calculates its current baseline, binds its own observations,
weights, exclusions and normalization, and performs its existing independent
native forward, derivative and fitted-state checks. Model starts and scientific
candidate coverage are unchanged.

## Compatibility

The response key pins the native sources, functions and binaries, numerical
runtime versions/architecture and reported BLAS/LAPACK build metadata, exact grid
and fit range, physical parameters, phase/site roles, complete moment map and
reflection inventory. Changing a cell, coordinate, ADP, instrument coefficient
or reduction grid invalidates the rendered response. Compatible scattering
factors can still be used while the affected response is rendered again.

Common histogram scale is applied to the unit-scale response by the consumer.
Observed values, weights, exclusions, ordinary background coefficients and
refinement cycle count do not enter the response key. They remain fresh inputs
to the calculation that owns them. The correction-wrapper active flag is
ignored only for the pinned installed build where that wrapper replaces no
functions; actual function/source/binary identities remain required.

## Portable provenance

An intact ZIP, checksum or uploaded qualification claim is insufficient. The
installed native producer publishes a certificate tied to its exact response
bytes and physical scope. The launcher verifies that certificate against the
creating Galaxy job before a numerical child receives an ephemeral origin
reference. A missing, inaccessible or incompatible origin uses the original
checked native compilation path.

The reader hashes the exact bounded bytes that it decodes. A file replacement
cannot substitute a different response between checksum verification and
loading. Invalid optional response files retain compatible factors and the
complete scientific search. Independent coefficient-qualification calls
explicitly bypass rendered-response reuse, so they cannot verify a response
against itself.

## Transport and receipts

One dependency-closure helper enumerates factors and responses for Model and
Live packets. The existing bounded packet budget still applies. Older
factor-only entries remain usable. Omitted or unavailable optional assets do
not remove scientific candidates.

Native compiler receipts record `prepared_rendered_response_reused`, the
exact rendered identity and component hashes, originating certificate, native
construction call counts and fallback notes. Preparation attachments separately
record how many rendered responses actually reached the worker. A transferred
file alone is not reported as reused computation.

## Repeated representative checks in one worker

After the complete magnetic response passes its native compilation probes,
the owning process can reuse that qualification for another freshly opened
context with the exact same decoded coefficients and physical dependencies.
The bounded ledger contains hashes and prior check errors, not model arrays.
It binds observations, weights, masks, fit limits, fixed parameters, complete
constraint-map contents, moment maps, active native functions, source/binary
identity and arithmetic controls. Changed or unsupported contexts use the
original complete probe set. A stronger requested probe count also runs in full.
Passing those probes on an imported cache does not create an owned coefficient
qualification; only this process's native compilation can establish its origin.

Each reuse still calculates the current native offset and one nonzero native
probe. Independent checks at the actual fitted solution, its fresh serialized
Jacobian/stationarity and final GPX remain unchanged. Receipts distinguish
prior complete probes from current probes; they do not count skipped work as
new native calculations. No qualification is loaded from a file or inherited
across a process boundary. Parallel representatives in separate workers each
establish their own qualification.

## Acceptance

The mandatory native publication gate includes complete Model/Live handoffs
and fresh-process portable controls for PNC/PNT profiles, three- and
nine-dimensional moment bases, changed scale, masks, grid, cell, ADPs, profile
widths, unverified origins and damaged files. Saved GPXs and actual fitted
native residuals remain checked. Synthetic controls establish numerical and
integration behavior; real-material identification, uncertainty calibration,
whole-catalog throughput and NDIP latency require their own acceptance.

Numerical build metadata is a conservative reuse boundary; it does not replace
native checks or promise bitwise equality across different hardware. An older
response without this runtime identity is rebuilt by the existing checked path.

The pinned native profile driver also receives a plain, zero-copy axis view
when its standard masked array contains no excluded bins. This avoids masked
temporaries without changing the values or native profile calculation. Actual
masks and unsupported array layouts retain the original path. A counterbalanced
local nine-moment, 18-start synthetic portfolio improved from 13.35 to 10.78
seconds (median, 19.2%), with identical residuals and the same 25 native forward
and one native derivative calls. This is a bounded local measurement, not an
NDIP latency or general experiment speed guarantee.
