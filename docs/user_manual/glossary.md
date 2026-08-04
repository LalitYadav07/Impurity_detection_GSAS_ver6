# Glossary

## Candidate Library

The searchable set of candidate crystalline phases. It can be built-in, built-in plus user CIFs, or only user CIFs.

## CIF

Crystallographic Information File. RADAR-PD uses CIFs to define candidate or main phase structures.

## Full RADAR-PD

The rigorous residual-aware phase detection workflow with deeper refinement and polishing stages.

## GSAS-II

The crystallographic refinement engine used by RADAR-PD for Rietveld-style project creation and refinement.

## Lattice Nudging

A symmetry-respecting search over small changes in lattice parameters to improve candidate peak alignment before expensive refinement.

## Main Phase

The known or dominant phase in the sample, usually supplied as a CIF. The main phase is refined or anchored before searching for impurities.

## Main-Shadow Candidate

A candidate whose strongest peaks mostly coincide with the main phase strongest peaks. Such a candidate can falsely steal intensity from a poorly anchored main phase.

## Pattern Scoring

The rapid-stage comparison of observed signal or residual signal with calculated candidate profiles.

## Phase Family

A group of candidate entries that are nearly duplicate scientific explanations, often sharing element set, space group, similar stoichiometry, and similar pattern.

## Rapid Hypothesis Mode

A faster workflow that searches and ranks phase-combination hypotheses before focused final refinements.

## Rwp

Weighted profile R-factor. Lower values often indicate a better fit, but Rwp alone is not enough; inspect residuals and phase evidence.

## Sample Can / Environment

Elements or phases associated with holders, cans, cryostats, furnaces, substrates, or other experimental environment contributions.

