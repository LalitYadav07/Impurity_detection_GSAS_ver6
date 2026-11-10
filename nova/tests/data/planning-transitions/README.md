# Planner transition test fixtures

These inputs exercise software and native forward-calculation invariants. They
do not validate agreement with measured magnetic or structural transitions.
`provenance.json` inventories every CIF with its content hash and source type;
the regression test requires a corresponding review when that inventory changes.
Content hashes do not establish redistribution authorization.

## Synthetic magnetic parent

`magnetic-parent.cif` was constructed for these regression tests. Its cell is the
arbitrary tetragonal cell a = b = 8 Å, c = 5.4 Å. Its fractional coordinates were
chosen independently as simple fractions, subject only to the listed SG 113
(`P -4 21 m`) site multiplicities:

| Site | Fractional coordinates | Multiplicity |
| --- | --- | --- |
| Er1 | (3/10, 4/5, 3/5) | 4 |
| Ge1 | (0, 0, 0) | 2 |
| Be1 | (1/8, 5/8, 7/8) | 4 |
| O1 | (0, 1/2, 7/10) | 2 |
| O2 | (1/5, 7/10, 3/10) | 4 |
| O3 | (1/10, 7/8, 3/4) | 8 |

All occupancies are one and all Uiso values are the arbitrary value 0.003 Å².
The element labels and stoichiometry exercise the Er magnetic-ion path and the
same SG 113 subgroup controls as before. The geometry is neither a measured nor
a published Er2Be2GeO7 structure, and is not intended as a plausible specimen
model. No cell, free coordinate, or displacement value was fitted to data.

`magnetic-candidates.json` retains four previously captured subgroup settings
for SG 113 at k = (0, 0, 0). Its matrices, origin shifts, and candidate IDs depend
on the space-group setting, not on a specimen's cell or atomic coordinates.
It is a bounded replay catalog, not an exhaustive discovery result. The native
controls check zero moments and squared-moment intensity scaling against the
synthetic parent.

## Public-source structural parent

`structural-parent.cif` transcribes the 1303 K geometry in Table II of Alexandra
S. Gibbs, Kevin S. Knight, and Philip Lightfoot, *The High Temperature Phase
Transitions of Hexagonal YMnO3*, [arXiv:1008.3042v1](https://arxiv.org/abs/1008.3042v1),
published as [Physical Review B 83, 094111](https://doi.org/10.1103/PhysRevB.83.094111).
The cell is a = b = 3.618961 Å, c = 11.34090 Å; the free O2 z coordinate is
0.08557. Full occupancy is assumed. The reported anisotropic ADPs are omitted:
this is a geometry fixture, not a complete refinement model. Its standard SG
194 operators use the Hall setting `-P 6c 2c`, as documented by
[COD entry 9008496](https://www.crystallography.net/cod/9008496.html); that entry
supplies the symmetry operators, not the YMnO3 specimen geometry.

## Generated structural child and replay catalog

`structural-child.cif` is a captured [ISODISTORT](https://iso.byu.edu/iso/isodistort.php)
6.12.2 output derived from the above parent. The file retains the generator's
header and mode definitions. It uses k = (1/3, 1/3, 0), K1, the P1 (a, 0)
direction, SG 193 `P 63/m c m`, basis {(-1,-2,0), (2,1,0), (0,0,1)}, and zero
origin shift. `combined-k0-catalog.json` captures subgroup settings for this
constructed child. `manifest.json` selects the native regression controls and
binds the child CIF by SHA256.

This constructed K1 path is an implementation control, not a claim that YMnO3
undergoes this transition; the cited paper reports a different observed path.
For the distortion method, see B. J. Campbell, H. T. Stokes, D. E. Tanner, and
D. M. Hatch, *ISODISPLACE: an Internet tool for exploring structural distortions*,
[Journal of Applied Crystallography 39, 607–614 (2006)](https://doi.org/10.1107/S0021889806014075).

The replay fixtures allow tests to run without network access. Genuine local
benchmark inputs and measured data are not part of these test fixtures.
