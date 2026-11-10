"""Independent trajectory calibration of conditional mode-amplitude ranges.

All aliases and observation variants of a physical trajectory share one worst
error. A condition with too few independent trajectories stays unresolved.
These are simulation diagnostics, never evidence of experimental absence or
permission to discard a structural candidate. Targets remain float64.
"""
import copy
import json
import math

import numpy as np

from ..symmetry.contracts import fields, digest
from . import structural_joint_modes as joint
from .structural_mode_readout import MAX_QUERIES
from .structural_model_files import MAX_METADATA_BYTES
from .tof_kernel_arrays import checksum

SCHEMA = 'radar-conditional-mode-group-calibration/v1'
METHOD = 'per-condition-whole-trajectory-interval-expansion/v1'
MAX_GROUPS = 4096
MAX_CELLS = 32768
IDENTITIES = ['weights_sha256', 'parent_sha256', 'instrument_sha256', 'dataset_sha256']


def _groups(values):
    if (not isinstance(values, list) or len(values) > MAX_GROUPS
            or any(not isinstance(v, str) or not 1 <= len(v) <= 500 for v in values)):
        raise ValueError('Complete bounded physical trajectory identities required')
    return values


def _identities(value):
    d = fields(value, IDENTITIES)
    if any(not checksum(v) for v in d.values()):
        raise ValueError('Mode calibration needs exact model and physical identities')
    return d


def _batch(predictions, target, groups, spec):
    groups = _groups(groups)
    p = np.asarray(predictions)
    d = fields(target, ['row', 'space_group', 'family', 'lower', 'upper'])
    r, c, f, lo, hi = (np.asarray(d[k]) for k in ['row', 'space_group', 'family', 'lower', 'upper'])
    if (not 1 <= len(groups) <= 256 or p.dtype.kind != 'f' or p.ndim != 2
            or p.shape != (len(r), 2) or len(p) > MAX_QUERIES
            or not np.isfinite(p).all() or np.any(p[:, 0] < 0) or np.any(p[:, 0] > p[:, 1])
            or any(v.shape != (len(p),) for v in [r, c, f, lo, hi])
            or any(v.dtype.kind not in 'iu' for v in [r, c, f])
            or np.any(r < 0) or np.any(r >= len(groups))
            or np.any(c < 0) or np.any(c >= len(spec['classes']))
            or np.any(f < 0) or np.any(f >= len(spec['families']))
            or lo.dtype != np.float64 or hi.dtype != np.float64
            or not np.isfinite(lo).all() or not np.isfinite(hi).all()
            or np.any(lo < 0) or np.any(hi < lo)):
        raise ValueError('Complete finite predictions and float64 native target intervals required')
    if len(set(zip(map(int, r), map(int, c), map(int, f)))) != len(r):
        raise ValueError('A native row can have only one complete interval per conditional family')
    return p.astype(np.float64), r, c, f, lo, hi


def _rank(n, rate):
    return math.ceil((n + 1) * (1 - rate))


class Calibration:
    def __init__(self, spec, *, identities, groups, used_groups, error_rate=.05):
        self.spec = copy.deepcopy(joint.validate(spec))
        self.identities = dict(_identities(identities))
        if self.identities['parent_sha256'] != spec['families'][0]['parent_sha256']:
            raise ValueError('Mode calibration belongs to a different native parent structure')
        declared = _groups(groups)
        used = _groups(used_groups)
        if (not declared or declared != sorted(set(declared)) or used != sorted(set(used))
                or set(declared).intersection(used)):
            raise ValueError('Mode calibration trajectories must be separate from fitting, selection and test')
        if type(error_rate) not in (int, float) or not math.isfinite(error_rate) or not 0 < error_rate < 1:
            raise ValueError('A finite declared calibration error rate is required')
        self.groups = declared.copy()
        self.used_sha = digest(used)
        self.rate = float(error_rate)
        self.scores = {}
        self.queries = {}
        self.rows = 0
        self.observed = set()

    def add(self, predictions, target, groups):
        p, r, c, f, lo, hi = _batch(predictions, target, groups, self.spec)
        if not set(groups).issubset(self.groups):
            raise ValueError('Calibration received an undeclared physical trajectory')
        # Preserve every native ambiguity endpoint, including sub-float32
        # amplitudes; a score of zero means containment, not mode absence.
        with np.errstate(over='ignore'):
            errors = np.maximum.reduce([p[:, 0] - lo, hi - p[:, 1], np.zeros(len(p))])
            errors = np.where(errors > 0, np.nextafter(errors, np.inf), errors)
        if not np.isfinite(errors).all():
            raise ValueError('Mode calibration errors exceed finite arithmetic')
        keys = [(int(ci), int(fi), groups[int(ri)]) for ri, ci, fi in zip(r, c, f)]
        if len(self.scores.keys() | set(keys)) > MAX_CELLS:
            raise ValueError('Shard complete mode calibration within its declared memory bound')
        for (ci, fi, group), error in zip(keys, errors):
            key = (ci, fi, group)
            self.scores[key] = max(self.scores.get(key, 0.), float(error))
            condition = (ci, fi)
            self.queries[condition] = self.queries.get(condition, 0) + 1
        self.rows += len(groups)
        self.observed.update(groups)

    def finish(self):
        if self.observed != set(self.groups):
            raise ValueError('Every declared calibration trajectory must be processed, including unknown targets')
        conditions = []
        for c, f in sorted(self.queries):
            scores = [[self.groups.index(g), error] for (ci, fi, g), error in sorted(self.scores.items())
                      if (ci, fi) == (c, f)]
            n = len(scores)
            rank = _rank(n, self.rate)
            supported = self.spec['known_intervals'][c][f] > 0
            radius = sorted(v for _, v in scores)[rank - 1] if supported and rank <= n else None
            conditions.append(dict(space_group_index=c, family_index=f, queries=self.queries[(c, f)],
                group_scores=scores, quantile_rank=rank, expansion=radius))
        return validate(dict(schema=SCHEMA, method=METHOD, identities=self.identities,
            mode_readout_identity=digest(self.spec), groups=self.groups, used_groups_sha256=self.used_sha,
            error_rate=self.rate, rows=self.rows, conditions=conditions,
            experimental_coverage_qualified=False, experimental_mode_absence_qualified=False,
            candidate_pruning_authorized=False), self.spec, identities=self.identities)


def validate(value, spec, *, identities):
    joint.validate(spec)
    d = fields(value, ['schema', 'method', 'identities', 'mode_readout_identity', 'groups',
        'used_groups_sha256', 'error_rate', 'rows', 'conditions', 'experimental_coverage_qualified',
        'experimental_mode_absence_qualified', 'candidate_pruning_authorized'])
    groups = _groups(d['groups'])
    rate = d['error_rate']
    if (d['schema'] != SCHEMA or d['method'] != METHOD or not groups or groups != sorted(set(groups))
            or _identities(d['identities']) != _identities(identities)
            or d['identities']['parent_sha256'] != spec['families'][0]['parent_sha256']
            or d['mode_readout_identity'] != digest(spec) or not checksum(d['used_groups_sha256'])
            or type(rate) not in (int, float) or not math.isfinite(rate) or not 0 < rate < 1
            or type(d['rows']) is not int or not len(groups) <= d['rows'] <= 1_000_000
            or not isinstance(d['conditions'], list) or len(d['conditions']) > MAX_CELLS
            or any(d[k] is not False for k in ['experimental_coverage_qualified',
                'experimental_mode_absence_qualified', 'candidate_pruning_authorized'])):
        raise ValueError('Mode calibration changed its physical identity, denominator or authority')
    conditions = []
    cells = 0
    for supplied in d['conditions']:
        row = fields(supplied, ['space_group_index', 'family_index', 'queries', 'group_scores',
            'quantile_rank', 'expansion'])
        c, f = row['space_group_index'], row['family_index']
        if (type(c) is not int or not 0 <= c < len(spec['classes'])
                or type(f) is not int or not 0 <= f < len(spec['families'])
                or not isinstance(row['group_scores'], list) or not row['group_scores']
                or type(row['queries']) is not int or not len(row['group_scores']) <= row['queries'] <= d['rows']):
            raise ValueError('Mode calibration needs complete per-condition native denominators')
        indices = []
        errors = []
        for score in row['group_scores']:
            if (not isinstance(score, list) or len(score) != 2 or type(score[0]) is not int
                    or not 0 <= score[0] < len(groups) or type(score[1]) is not float
                    or not math.isfinite(score[1]) or score[1] < 0):
                raise ValueError('Every independent trajectory needs its finite worst-case mode error')
            indices.append(score[0]); errors.append(score[1])
        n = len(indices)
        rank = _rank(n, rate)
        expected = sorted(errors)[rank - 1] if spec['known_intervals'][c][f] and rank <= n else None
        if (indices != sorted(set(indices)) or type(row['quantile_rank']) is not int
                or row['quantile_rank'] != rank or row['expansion'] != expected
                or row['expansion'] is not None and type(row['expansion']) is not float):
            raise ValueError('Mode calibration cannot manufacture independent groups or narrow its expansion')
        conditions.append((c, f)); cells += n
    if conditions != sorted(set(conditions)) or cells > MAX_CELLS:
        raise ValueError('Mode calibration changed or exceeded its complete condition inventory')
    if len(json.dumps(d, separators=(',', ':'), allow_nan=False).encode()) > MAX_METADATA_BYTES:
        raise ValueError('Mode calibration exceeds its portable metadata bound')
    return d


def expanded(predictions, target, groups, calibration, spec, *, identities, within_support=True):
    """Expand held-out query predictions; unknowns never become zeros.

    The bounds have the same asinh/fitting-scale units as the shared loss.
    This evaluation helper uses native targets, not observations from a user.
    Decode only available rows using the existing float64 physical transform.
    """
    d = validate(calibration, spec, identities=identities)
    p, r, c, f, lo, hi = _batch(predictions, target, groups, spec)
    if type(within_support) is not bool:
        raise ValueError('Mode interval use needs an explicit prepared-domain check')
    radii = {(v['space_group_index'], v['family_index']): v['expansion'] for v in d['conditions']}
    return _expanded(p, c, f, radii, within_support=within_support)


def _expanded(p, c, f, radii, *, within_support=True):
    """Apply already owned, validated expansions without re-auditing metadata."""
    # These are internal numerical arrays, not a JSON artifact. NaN plus the
    # explicit mask prevents an unsupported condition looking like [0, 0].
    bounds = np.full_like(p, np.nan)
    available = np.zeros(len(p), bool)
    for i, (ci, fi) in enumerate(zip(c, f)):
        radius = radii.get((int(ci), int(fi))) if within_support else None
        if radius is not None:
            bounds[i] = [max(0., np.nextafter(p[i, 0] - radius, -np.inf)),
                         np.nextafter(p[i, 1] + radius, np.inf)]
            available[i] = True
    if not np.isfinite(bounds[available]).all():
        raise ValueError('Expanded mode intervals exceed finite arithmetic')
    return dict(bounds=bounds, available=available,
        scope='Per-condition simulation trajectory calibration; not an experimental detectability or absence claim',
        experimental_coverage_qualified=False, experimental_mode_absence_qualified=False,
        candidate_pruning_authorized=False)


class Evaluation:
    """Report both conditional coverage and unqualified/unsupported cases.

    The nominal error rate is per condition and whole trajectory. It is not
    simultaneous coverage across all modes, nor an experimental guarantee.
    Calibration groups cannot be scored again as independent test groups.
    """
    def __init__(self, calibration, spec, *, identities):
        self.calibration = copy.deepcopy(validate(calibration, spec, identities=identities))
        self.spec = copy.deepcopy(spec)
        self.identities = dict(identities)
        self._radii = {(v['space_group_index'], v['family_index']): v['expansion']
                      for v in self.calibration['conditions']}
        self.counts = {}
        self.groups = {}
        self.rows = 0
        self.test_groups = set()

    def add(self, predictions, target, groups):
        if set(_groups(groups)).intersection(self.calibration['groups']):
            raise ValueError('Mode evaluation cannot score its own calibration trajectories')
        p, r, c, f, lo, hi = _batch(predictions, target, groups, self.spec)
        result = _expanded(p, c, f, self._radii)
        known = result['available']
        b = result['bounds']
        contained = known & (b[:, 0] <= lo) & (b[:, 1] >= hi)
        keys = [(int(ci), int(fi), groups[int(ri)]) for ri, ci, fi in zip(r, c, f)]
        if len(self.groups.keys() | set(keys)) > MAX_CELLS:
            raise ValueError('Shard complete mode evaluation within its declared memory bound')
        for (ci, fi, group), available, hit in zip(keys, known, contained):
            totals = self.counts.setdefault((ci, fi), [0, 0, 0])
            totals[0] += 1; totals[1] += int(available); totals[2] += int(hit)
            before = self.groups.setdefault((ci, fi, group), [True, True])
            before[0] &= bool(available); before[1] &= bool(available and hit)
        self.rows += len(groups)
        self.test_groups.update(groups)

    def finish(self):
        if not self.rows or self.rows > 1_000_000 or len(self.test_groups) > MAX_GROUPS:
            raise ValueError('A bounded complete independent test population is required')
        conditions = []
        for (c, f), (queries, available, contained) in sorted(self.counts.items()):
            statuses = [v for (ci, fi, _), v in self.groups.items() if (ci, fi) == (c, f)]
            qualified_groups = sum(v[0] for v in statuses)
            contained_groups = sum(v[1] for v in statuses)
            conditions.append(dict(space_group_index=c, family_index=f, queries=queries,
                interval_available_queries=available, interval_contained_queries=contained,
                physical_groups=len(statuses), interval_available_groups=qualified_groups,
                wholly_contained_groups=contained_groups,
                target_interval_containment=contained/available if available else None,
                whole_group_containment=contained_groups/qualified_groups if qualified_groups else None))
        return dict(schema=SCHEMA, method=METHOD, identities=self.identities,
            mode_readout_identity=digest(self.spec), calibration_sha256=digest(self.calibration),
            rows=self.rows, physical_groups=len(self.test_groups),
            known_queries=sum(v[0] for v in self.counts.values()),
            interval_available_queries=sum(v[1] for v in self.counts.values()), conditions=conditions,
            scope='Per-condition simulated held-out trajectory coverage; not joint family coverage or experimental detectability',
            experimental_coverage_qualified=False, experimental_mode_absence_qualified=False,
            candidate_pruning_authorized=False)


def validate_evaluation(value, calibration, spec, *, identities, native_metrics, rows, groups):
    """Bind every test condition to the unchanged native mode metric counts."""
    cal = validate(calibration, spec, identities=identities)
    native = joint.validate_metrics(native_metrics)
    d = fields(value, ['schema', 'method', 'identities', 'mode_readout_identity', 'calibration_sha256',
        'rows', 'physical_groups', 'known_queries', 'interval_available_queries', 'conditions', 'scope',
        'experimental_coverage_qualified', 'experimental_mode_absence_qualified', 'candidate_pruning_authorized'])
    if (d['schema'] != SCHEMA or d['method'] != METHOD or _identities(d['identities']) != identities
            or d['mode_readout_identity'] != digest(spec) or d['calibration_sha256'] != digest(cal)
            or type(d['rows']) is not int or d['rows'] != rows or rows != native['rows']
            or type(d['physical_groups']) is not int or d['physical_groups'] != groups or not 1 <= groups <= rows
            or native['classes'] != spec['classes'] or native['families'] != [f['id'] for f in spec['families']]
            or type(d['known_queries']) is not int or d['known_queries'] != native['known_queries']
            or type(d['interval_available_queries']) is not int or not 0 <= d['interval_available_queries'] <= d['known_queries']
            or not isinstance(d['conditions'], list) or len(d['conditions']) > MAX_CELLS
            or not isinstance(d['scope'], str) or not 1 <= len(d['scope']) <= 1000
            or any(d[k] is not False for k in ['experimental_coverage_qualified',
                'experimental_mode_absence_qualified', 'candidate_pruning_authorized'])):
        raise ValueError('Mode interval evaluation differs from its frozen model or native test population')
    expected = {(v['space_group_index'], v['family_index']): v['queries'] for v in native['conditions']}
    radii = {(v['space_group_index'], v['family_index']): v['expansion'] for v in cal['conditions']}
    observed = []
    available_total = 0
    for supplied in d['conditions']:
        row = fields(supplied, ['space_group_index', 'family_index', 'queries', 'interval_available_queries',
            'interval_contained_queries', 'physical_groups', 'interval_available_groups', 'wholly_contained_groups',
            'target_interval_containment', 'whole_group_containment'])
        c, f = row['space_group_index'], row['family_index']
        count, av, hit, ng, ag, hg = (row[k] for k in ['queries', 'interval_available_queries',
            'interval_contained_queries', 'physical_groups', 'interval_available_groups', 'wholly_contained_groups'])
        if (type(c) is not int or type(f) is not int or (c, f) not in expected
                or any(type(v) is not int for v in [count, av, hit, ng, ag, hg])
                or count != expected[(c, f)] or not 1 <= ng <= min(groups, count)
                or not 0 <= hit <= av <= count or not 0 <= hg <= ag <= ng
                or av != (count if radii.get((c, f)) is not None else 0)
                or ag != (ng if av else 0) or hg > hit
                or (av > 0 and (hit == av) != (hg == ag))):
            raise ValueError('Mode interval evaluation lost a native condition or changed availability')
        for key, numerator, denominator in [('target_interval_containment', hit, av), ('whole_group_containment', hg, ag)]:
            rate = row[key]
            if ((not denominator and rate is not None) or (denominator and
                    (type(rate) is not float or not math.isfinite(rate) or rate != numerator / denominator))):
                raise ValueError('Mode interval coverage must retain its actual available denominator')
        observed.append((c, f)); available_total += av
    if observed != sorted(expected) or available_total != d['interval_available_queries']:
        raise ValueError('Mode interval evaluation must retain the complete native test condition inventory')
    return d
