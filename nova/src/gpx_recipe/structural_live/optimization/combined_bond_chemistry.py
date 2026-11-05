"""Source-screened MP/COD contact priors with explicit unconstrained pairs.

Screening thresholds diagnose suspect *observations*. They are never substituted
for a measured minimum. Aggregate means/SDs do not define a probability density.
"""
from collections import Counter
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import numpy as np

from .chemical_geometry import ChemicalPotential, ContactRule, ChemistryUnqualified, fingerprint
from .physical_prior_geometry import metric_matrix
from .materials_project_chemistry import digest, element_label


SOURCE_NAMES = ('statistics', 'mp_statistics', 'cod_statistics', 'radii')
COMBINED_FIELDS = {
    'element_pair', 'source_coverage', 'mp_bond_observations', 'cod_bond_observations',
    'combined_bond_observations', 'mp_average_angstrom', 'cod_average_angstrom',
    'combined_average_angstrom', 'mp_stddev_angstrom', 'cod_stddev_angstrom',
    'combined_stddev_angstrom', 'mp_minimum_angstrom', 'cod_minimum_angstrom',
    'combined_minimum_angstrom', 'combined_minimum_source', 'combined_minimum_cod_id',
    'minimum_under_0p5_angstrom', 'mp_minus_cod_average_angstrom',
}


def _pair(value):
    pair = tuple(value.split('-'))
    if len(pair) != 2 or tuple(sorted(map(element_label, pair))) != pair:
        raise ValueError('Noncanonical elemental pair: '+value)
    return pair


def _read(path, fields):
    rows = {}
    with Path(path).open(newline='') as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or ()) != fields:
            raise ValueError('Unrecognized bond-statistics schema: '+str(path))
        for row in reader:
            key = _pair(row['element_pair'])
            if key in rows: raise ValueError('Duplicate elemental pair')
            rows[key] = row
    if not rows: raise ValueError('Empty empirical chemistry catalog')
    return rows


def _stats(row, prefix=''):
    n = int(row[prefix+'bond_observations'])
    minimum, mean, sd = (float(row[prefix+k]) for k in
                         ('minimum_angstrom', 'average_angstrom', 'stddev_angstrom'))
    if n < 1 or not np.isfinite([minimum, mean, sd]).all() or minimum <= 0 or mean < minimum-1e-12 or sd < 0:
        raise ValueError('Invalid bond statistics')
    return dict(observations=n, minimum=minimum, mean=mean, stddev=sd)


def _close(actual, expected):
    # CSV serialization may round the pooled moments; this is not a distance tolerance.
    if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=2e-12, abs_tol=1e-12):
        raise ValueError('Combined statistics do not reconcile with source catalogs')


@dataclass(frozen=True)
class NoMinimum:
    """An explicit zero kernel, not a fabricated ContactRule or measured zero."""
    species: tuple
    source: str
    hard_min: float = 0.
    soft_min: float = 0.


class ScreenedContactPotential(ChemicalPotential):
    def __init__(self, rules, unconstrained, *, partial_occupancy='unknown', catalog_fingerprint):
        if partial_occupancy not in ('unknown', 'independent_static_sites'):
            raise ValueError('Explicit occupancy correlation policy required')
        self.unconstrained = dict(unconstrained)
        self.catalog_fingerprint = catalog_fingerprint
        if rules:
            super().__init__(rules, partial_occupancy=partial_occupancy)
        else:
            self.rules = (); self.by_pair = {}; self.context = {}
            self.partial_occupancy = partial_occupancy; self.max_images = 2000000
        for pair, reason in self.unconstrained.items():
            if pair in self.by_pair: raise ValueError('Pair cannot be constrained and unconstrained')
            # The bounded-domain implementation expects an explicit entry for
            # each pair. Zero imposes no bound, including at coincident sites.
            self.by_pair[pair] = NoMinimum(pair, reason)
        self.fingerprint = fingerprint(self.manifest())

    def manifest(self):
        return dict(**super().manifest(),
            unconstrained_pairs={'-'.join(k): v for k, v in sorted(self.unconstrained.items())},
            chemistry_catalog_fingerprint=self.catalog_fingerprint,
            screened_minima_are_not_universal_laws=True)

    def evaluate(self, coordinates, direct_metric, species, occupancy, *, joint_occupancy=None):
        types = sorted(set(species))
        if any((a, b) not in self.by_pair for i, a in enumerate(types) for b in types[i:]):
            raise ValueError('Chemical species differ from the declared contact policy')
        if self.rules:
            result = super().evaluate(coordinates, direct_metric, species, occupancy, joint_occupancy=joint_occupancy)
        else:
            xyz = np.asarray(coordinates, float); occ = np.asarray(occupancy, float)
            metric_matrix(direct_metric)
            if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all()
                    or occ.shape != (len(xyz),) or len(species) != len(xyz) or not np.isfinite(occ).all()
                    or np.any(occ < 0) or np.any(occ > 1) or occ.sum() <= 0 or any(not s for s in species)):
                raise ValueError('Finite geometry and per-site occupancy in [0,1] required')
            result = dict(feasible=True, soft_penalty=0., log_kernel=0., hard_violations=[],
                          active_contacts=[], periodic_contacts_examined=0,
                          potential_fingerprint=self.fingerprint, scope=self.manifest()['scope'])
        result['unconstrained_pairs'] = ['-'.join(k) for k in sorted(self.unconstrained)]
        result['complete_minimum_coverage'] = not self.unconstrained
        return result


class CombinedBondCatalog:
    def __init__(self, sources, quality):
        if set(sources) != set(SOURCE_NAMES): raise ValueError('Four pinned MP/COD sources required')
        self.paths = {k: Path(v) for k, v in sources.items()}
        self.hashes = {k: digest(v) for k, v in self.paths.items()}
        self.statistics = self.paths['statistics']; self.statistics_sha256 = self.hashes['statistics']
        expected_quality = {'schema_version', 'absolute_screen_angstrom', 'covalent_radius_fraction',
                            'no_credible_source', 'selection'}
        if (not isinstance(quality, dict) or set(quality) != expected_quality
                or type(quality['schema_version']) is not int or quality['schema_version'] != 1
                or quality['no_credible_source'] != 'no_minimum'
                or quality['selection'] != 'minimum_of_credible_sources'):
            raise ValueError('Unsupported source-quality policy')
        for key in ('absolute_screen_angstrom', 'covalent_radius_fraction'):
            if type(quality[key]) not in (int, float) or not np.isfinite(quality[key]) or not 0 < quality[key] < 1:
                raise ValueError('Finite explicit source-screening thresholds required')
        self.quality = dict(quality)
        radii = json.loads(self.paths['radii'].read_text())
        if radii.get('schema_version') != 1 or not isinstance(radii.get('radius_angstrom'), dict):
            raise ValueError('Versioned covalent-radius data required')
        self.radii = radii['radius_angstrom']
        if any(element_label(k) != k or type(v) not in (int, float) or not math.isfinite(v) or v <= 0
               for k, v in self.radii.items()): raise ValueError('Invalid radius table')
        base_fields = {'element_pair', 'bond_observations', 'minimum_angstrom', 'average_angstrom', 'stddev_angstrom'}
        source_rows = dict(mp=_read(self.paths['mp_statistics'], base_fields),
                           cod=_read(self.paths['cod_statistics'], base_fields | {'minimum_cod_id'}))
        rows = _read(self.statistics, COMBINED_FIELDS)
        if set(rows) != set(source_rows['mp']) | set(source_rows['cod']):
            raise ValueError('Combined pair coverage differs from the union of its sources')
        self.pairs = {}; self.reviews = {}
        for pair, row in rows.items():
            stats = {}; assessments = {}
            for source in ('mp', 'cod'):
                original = source_rows[source].get(pair)
                if original is None:
                    if (row[source+'_bond_observations'] != '0' or any(row[source+'_'+field] for field in
                            ('minimum_angstrom', 'average_angstrom', 'stddev_angstrom'))):
                        raise ValueError('Absent source contains values in combined table')
                    assessments[source] = dict(state='absent', credible=False)
                    continue
                value = _stats(original); stats[source] = value
                if _stats(row, source+'_') != value:
                    raise ValueError('Combined source values differ from the source-only CSV')
                if source == 'cod' and re.fullmatch(r'[0-9]{7}', original['minimum_cod_id']) is None:
                    raise ValueError('COD minimum requires its source identifier')
                reasons = []
                if value['minimum'] < quality['absolute_screen_angstrom']: reasons.append('severe_absolute_shortness')
                radius_sum = sum(self.radii[e] for e in pair) if all(e in self.radii for e in pair) else None
                if radius_sum is None: reasons.append('no_radius_screen_available')
                elif value['minimum'] < quality['covalent_radius_fraction']*radius_sum:
                    reasons.append('severe_shortness_relative_to_covalent_radii')
                assessments[source] = dict(state='suspect' if reasons else 'passes_screen', credible=not reasons,
                    reasons=reasons, **value, radius_sum_angstrom=radius_sum,
                    minimum_cod_id=original.get('minimum_cod_id'))
            coverage = 'both' if len(stats) == 2 else next(iter(stats))+'_only'
            if row['source_coverage'] != coverage: raise ValueError('Source coverage label disagrees')
            pooled = _stats(row, 'combined_'); n = sum(s['observations'] for s in stats.values())
            mean = math.fsum(s['observations']*s['mean'] for s in stats.values())/n
            variance = math.fsum(s['observations']*(s['stddev']**2+(s['mean']-mean)**2) for s in stats.values())/n
            if pooled['observations'] != n: raise ValueError('Combined observation counts do not reconcile')
            _close(pooled['mean'], mean); _close(pooled['stddev'], math.sqrt(variance))
            minimum = min(s['minimum'] for s in stats.values()); _close(pooled['minimum'], minimum)
            owners = [k for k, s in stats.items() if s['minimum'] == minimum]
            owner = 'both' if len(owners) == 2 else owners[0]
            cod_id = source_rows['cod'][pair]['minimum_cod_id'] if 'cod' in owners else ''
            if (row['combined_minimum_source'] != owner or row['combined_minimum_cod_id'] != cod_id
                    or row['minimum_under_0p5_angstrom'] != str(minimum < .5).lower()):
                raise ValueError('Combined minimum provenance or short-distance flag differs')
            if coverage == 'both': _close(float(row['mp_minus_cod_average_angstrom']), stats['mp']['mean']-stats['cod']['mean'])
            elif row['mp_minus_cod_average_angstrom']: raise ValueError('Absent source has a mean difference')
            accepted = {k: s['minimum'] for k, s in assessments.items() if s['credible']}
            selected = min(accepted.values()) if accepted else None
            selected_sources = [k for k, v in accepted.items() if v == selected]
            self.pairs[pair] = dict(pooled, raw_minimum=minimum, minimum=selected)
            self.reviews[pair] = dict(pair='-'.join(pair), source_coverage=coverage, sources=assessments,
                selected_minimum_angstrom=selected, selected_sources=selected_sources,
                decision='no_minimum' if selected is None else 'source_fallback' if any(s['state']=='suspect' for s in assessments.values()) else 'screened_minimum',
                validation='Heuristic screen only; not independent validation of the minimum structure')
        self.fingerprint = fingerprint(self.manifest())

    def manifest(self):
        return dict(schema_version=2, catalog='mp_cod_ordered', source_hashes=self.hashes,
            statistics_sha256=self.statistics_sha256,
            quality=self.quality, pairs=len(self.pairs), bond_observations=sum(v['observations'] for v in self.pairs.values()),
            decisions=dict(Counter(v['decision'] for v in self.reviews.values())),
            unconstrained_pairs=['-'.join(k) for k, v in sorted(self.pairs.items()) if v['minimum'] is None],
            decisions_sha256=fingerprint(self.reviews_as_rows()),
            source='User-provided MP and ordered-site COD CrystalNN aggregates', units='angstrom',
            counts='Pooled bond observations; overlapping structures and different edge counts; not independent materials',
            distribution_scope='Summary statistics only; no empirical CDF or Gaussian prior inferred',
            universal_physical_law=False, production_qualified=False)

    def reviews_as_rows(self): return [self.reviews[k] for k in sorted(self.reviews)]

    def distribution(self, pair, distance):
        key = tuple(sorted(map(element_label, pair)))
        if key not in self.pairs: raise ChemistryUnqualified('No empirical observations for '+'-'.join(key))
        if not np.isfinite(distance) or distance < 0: raise ValueError('Finite nonnegative bond length required')
        return dict(pair=key, distance_angstrom=float(distance), **self.pairs[key], source_review=self.reviews[key],
            empirical_cdf_interval=None, interpretation='Pooled summary statistics; no distribution shape or structural probability available')

    def potential(self, species, *, partial_occupancy='unknown'):
        labels = tuple(map(element_label, species)); types = sorted(set(labels)); rules = []; unconstrained = {}
        for i, a in enumerate(types):
            for b in types[i:]:
                pair = (a, b)
                if pair not in self.pairs:
                    unconstrained[pair] = 'No source observations; no empirical minimum imposed'
                elif self.pairs[pair]['minimum'] is None:
                    unconstrained[pair] = 'No source minimum passed screening; no empirical minimum imposed'
                else:
                    review = self.reviews[pair]
                    rules.append(ContactRule(pair, 'Screened MP/COD minimum from '+','.join(review['selected_sources'])+
                        '; catalog '+self.fingerprint, hard_min=self.pairs[pair]['minimum']))
        return ScreenedContactPotential(rules, unconstrained, partial_occupancy=partial_occupancy,
                                        catalog_fingerprint=self.fingerprint), labels


def load_combined_catalog(policy_path):
    path = Path(policy_path); policy = json.loads(path.read_text())
    if (not isinstance(policy, dict) or type(policy.get('schema_version')) is not int
            or policy['schema_version'] != 2 or policy.get('catalog') != 'mp_cod_ordered'
            or policy.get('mode') != 'screened_empirical_minimum'):
        raise ChemistryUnqualified('Unsupported empirical chemistry policy schema')
    sources = {}
    for name in SOURCE_NAMES:
        expected = policy.get(name+'_sha256'); value = policy.get(name)
        if not isinstance(expected, str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
            raise ChemistryUnqualified('Required pinned SHA-256 is invalid: '+name)
        if not isinstance(value, str) or not value.strip():
            raise ChemistryUnqualified('Required empirical chemistry source path is invalid: '+name)
        source = Path(value); source = source if source.is_absolute() else path.parent/source
        if digest(source) != expected: raise ValueError('Empirical chemistry source changed; a new protocol is required')
        sources[name] = source
    catalog = CombinedBondCatalog(sources, policy.get('quality'))
    if any(catalog.hashes[name] != policy[name+'_sha256'] for name in SOURCE_NAMES):
        raise ValueError('Empirical chemistry source changed during loading')
    return catalog
