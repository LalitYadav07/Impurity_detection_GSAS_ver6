"""Balanced parent-range strata, without changing any physical group identity.

Widening the prepared metric domain need not throw away existing native pairs.
Different ranges may share one training run only with identical seeds, physical
grid, complete candidate definitions and complete per-candidate trajectories.
The original group strings remain authoritative for alias closure/holdouts.
"""
from ..symmetry.contracts import digest, fields, text, validate_candidate
from .batch import relative
from .structural_training_blocks import parent_context, REPRESENTATION
from .reference import validate_grid

MAX_BLOCKS = 5000
SAMPLING = ['seed', 'parent_seed', 'parent_relative_metric_range', 'trajectories', 'steps']


def _sampling(context):
    seed, parent_seed, low, high, trajectories, steps = context
    return dict(seed=seed, parent_seed=parent_seed,
                parent_relative_metric_range=[low, high], trajectories=trajectories, steps=steps)


def reference_range(strata):
    """A widening needs one retained range contained in every added range.

    This identifies the original narrow-range reference in a complete range
    expansion. Unrelated/disjoint sampling campaigns need explicit prior-fold
    provenance; their historical holdouts cannot be inferred from filenames.
    """
    candidates=[row for row in strata if all(
        other['parent_relative_metric_range'][0]<=row['parent_relative_metric_range'][0]
        and row['parent_relative_metric_range'][1]<=other['parent_relative_metric_range'][1]
        for other in strata)]
    if len(candidates)!=1:
        raise ValueError('A range expansion needs one retained reference interval contained in every added range')
    return candidates[0]


def validate(value, block_names):
    """Inspect the closed plan inventory before downloading numerical parts."""
    if (not isinstance(value, list) or not 2 <= len(value) <= len(block_names) <= MAX_BLOCKS
            or len(set(block_names)) != len(block_names)):
        raise ValueError('A stratified plan needs a bounded complete parent-range inventory')
    contexts = []; catalogs = []; names = []
    for supplied in value:
        row = fields(supplied, SAMPLING+['candidate_ids', 'blocks'])
        context = parent_context(dict(row, representation=REPRESENTATION))
        ids = row['candidate_ids']; blocks = row['blocks']
        if (not isinstance(ids, list) or not 1 <= len(ids) <= MAX_BLOCKS
                or any(text(i, pattern=r'[a-zA-Z0-9:_-]{1,120}') != i for i in ids)
                or ids != sorted(set(ids)) or not isinstance(blocks, list) or not blocks
                or len(blocks) > MAX_BLOCKS or any(relative(p) != p for p in blocks)
                or blocks != sorted(set(blocks))):
            raise ValueError('Parent-range strata need complete sorted candidate and block identities')
        contexts.append(context); catalogs.append(ids); names.extend(blocks)
    if contexts != sorted(set(contexts)) or len({c[:2]+c[4:] for c in contexts}) != 1:
        raise ValueError('Range strata must retain shared parent trajectories, seeds and step counts')
    if any(catalog != catalogs[0] for catalog in catalogs):
        raise ValueError('Every parent-range stratum needs the same complete candidate catalog')
    if len(names) != len(set(names)) or sorted(names) != sorted(block_names):
        raise ValueError('Parent-range strata must partition every captured training block exactly once')
    reference_range(value)
    return value


class Inventory:
    """Keep small metadata summaries, never every expanded raw block in RAM."""
    def __init__(self):
        self.groups = {}; self.definitions = {}; self.grids = set(); self.names = set()

    def add(self, name, document):
        relative(name)
        if name in self.names or len(self.names) >= MAX_BLOCKS:
            raise ValueError('Parent-range inputs require bounded unique logical blocks')
        context = parent_context(dict(document, representation=REPRESENTATION))
        candidate = validate_candidate(document.get('candidate')); key = candidate['id']
        definition = digest(candidate)
        if key in self.definitions and self.definitions[key] != definition:
            raise ValueError('A structural candidate identity cannot change between parent-range strata')
        interval = document.get('trajectory_slice')
        if (not isinstance(interval, list) or len(interval) != 2
                or any(type(i) is not int for i in interval)
                or not 0 <= interval[0] < interval[1] <= context[4]):
            raise ValueError('A parent-range block needs its declared whole-trajectory slice')
        self.names.add(name); self.definitions[key] = definition
        grid = document.get('grid')
        if grid is not None:
            grid = validate_grid(**fields(grid, ['kind', 'lower', 'upper', 'bins', 'bank']))
        self.grids.add(digest(grid))
        group = self.groups.setdefault(context, dict(blocks=[], candidates={}))
        group['blocks'].append(name)
        group['candidates'].setdefault(key, []).append(tuple(interval))

    def describe(self):
        """Single contexts keep the old schema and old partial-scope semantics."""
        if len(self.groups) <= 1:
            return None
        contexts = sorted(self.groups)
        if len({c[:2]+c[4:] for c in contexts}) != 1:
            raise ValueError('Training blocks must share one representation and parent trajectory context; '
                             'range strata must retain shared parent trajectories and seeds')
        if len(self.grids) != 1:
            raise ValueError('Parent-range strata must share one exact physical bank grid')
        catalogs = [set(self.groups[c]['candidates']) for c in contexts]
        if any(catalog != catalogs[0] for catalog in catalogs):
            raise ValueError('Every parent-range stratum needs the same complete candidate catalog')
        result = []
        for context in contexts:
            group = self.groups[context]
            for intervals in group['candidates'].values():
                # Overlapping *different* portfolios retain every original
                # block. Shared group IDs still close their correlated rows.
                stop = 0
                for lower, upper in sorted(intervals):
                    if lower > stop:
                        raise ValueError('Every candidate in each parent-range stratum needs all declared trajectories')
                    stop = max(stop, upper)
                if stop != context[4]:
                    raise ValueError('Every candidate in each parent-range stratum needs all declared trajectories')
            result.append(dict(_sampling(context), candidate_ids=sorted(group['candidates']),
                               blocks=sorted(group['blocks'])))
        return validate(result, self.names)
