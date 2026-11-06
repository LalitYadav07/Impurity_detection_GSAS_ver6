The numerical kernels here were extracted from the structural-transition
research branch. The installed entry point is `structural_live.service`, not
an experiment runner. `profiled_native_fit` is the main fitter;
`automatic_fit_recovery` retains a complementary native search. Both evaluate
the same original-bin objective and preserve unresolved outcomes.

The main fitter retains separate mode-block and cell-block LM damping within
each start, and resets both for the next independent start. An unsuccessful
step can increase damping and retry; that bookkeeping step is not evidence of
small physical progress. Exhausted native descent remains an unresolved
attempt. Block receipts record the start and initial/final damping. These
changes preserve the native acceptance, stationarity-retry and export checks;
they do not establish global convergence or a complete-fit speedup.

The empirical contact policy uses aggregate Materials Project and
Crystallography Open Database bond statistics, with source hashes preserved in
the policy JSON. It is an experimental, source-screened assumption; neither
the pooled moments nor minimum distances define a calibrated probability.
Missing credible pairs have no fabricated minimum. The CSV files contain
element-pair statistics and public COD identifiers, not facility experiments,
proposal metadata, diffraction arrays, credentials or private CIF libraries.

Native exact-cell response caching is optional within the fitter. It retains
the full GSAS-II path for unqualified contexts. No custom peak convolution or
global structure-uniqueness guarantee is introduced.
