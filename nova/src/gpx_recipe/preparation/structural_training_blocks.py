"""One typed boundary for old saved blocks and new three-curve preparation.

The app writes the paired format by default. Existing saved libraries retain
their exact settings and identities; a training job cannot mix representations
or change the shared parent sampling underneath a candidate family.
"""
import json
from pathlib import Path

from ..symmetry.contracts import number
from .structural_attention_weights import REPRESENTATION
from .structural_cohort import read_block
from .structural_paired_blocks import read as read_pairs, SCHEMA as PILOT_SCHEMA
from .structural_pair_arrays import SCHEMA as PAIR_SCHEMA
from .structural_mode_layout import SCHEMAS as NAMED_SCHEMAS
from .structural_mode_projection import PAIR_SCHEMAS as PROJECTION_SCHEMAS
from .tof_kernel_arrays import read_bounded
from .structural_domain import DEFAULT_PARENT_METRIC_RANGE

LEGACY_SCHEMA='radar-structural-ranking-examples/v3'
PARENT_RANGE=DEFAULT_PARENT_METRIC_RANGE
PAIRED_FIELDS=('representation','parent_seed','parent_relative_metric_range')


def representation(document):
    schema=document.get('schema') if isinstance(document,dict) else None
    if schema==LEGACY_SCHEMA:return None
    if schema in {PILOT_SCHEMA,PAIR_SCHEMA}|NAMED_SCHEMAS|PROJECTION_SCHEMAS:return REPRESENTATION
    raise ValueError('Unsupported structural training block representation')


def validate_representation(settings):
    if not isinstance(settings,dict) or settings.get('representation')!=REPRESENTATION:
        raise ValueError('New structural training requires the three-curve representation')
    return REPRESENTATION


def parent_context(settings):
    """All child aliases use the same physical parent trajectories and groups."""
    validate_representation(settings)
    if not {'seed','parent_seed','parent_relative_metric_range','trajectories','steps'}<=set(settings):
        raise ValueError('Retain every declared shared parent sampling parameter')
    seed=number(settings['seed'],0,2**32-1,integer=True)
    parent_seed=number(settings['parent_seed'],0,2**32-1,integer=True)
    interval=settings['parent_relative_metric_range']
    if (not isinstance(interval,(list,tuple)) or len(interval)!=2
            or any(type(v) not in (int,float) for v in interval)):
        raise ValueError('Declare a bounded positive parent metric interval')
    low=number(interval[0],0,.02,positive=True)
    high=number(interval[1],low,.02)
    return (seed,parent_seed,low,high,
        number(settings['trajectories'],1,4096,integer=True),
        number(settings['steps'],1,32,integer=True))


def context(document):
    if representation(document) is None:return None
    return parent_context(dict(document,representation=REPRESENTATION))


def metadata(directory):
    document=json.loads(read_bounded(Path(directory)/'examples.json',8*1024**2))
    representation(document)
    context(document)
    return document


def read(directory):
    document=metadata(directory)
    return (read_block(directory) if representation(document) is None
            else read_pairs(directory,names='examples'))
