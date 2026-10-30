"""Beamline registration contract for shared Live UI and orchestration.

Each adapter owns scientific discovery, metadata, preflight and bank staging;
the controller lifecycle remains shared. Adding a beamline is an explicit
registration, never a filename-prefix or nominal-wavelength guess.
"""
from dataclasses import dataclass, replace
from typing import Callable

from .powgen_controller import (PowgenExperimentSettings, PowgenWatchController,
    discover_powgen_samples, preflight_powgen_experiment)
from .powgen_samples import PowgenSampleIndex
from .nomad_controller import (NomadExperimentSettings, NomadWatchController,
    discover_nomad_samples, preflight_nomad_experiment)
from .nomad_metadata import NomadSampleIndex
from .nomad import MODERN_FOLDER
from .hb2a_controller import (Hb2aExperimentSettings, Hb2aWatchController,
    Hb2aSampleIndex, discover_hb2a_samples, preflight_hb2a_experiment)
from .snap import DEFAULT_FOLDER as SNAP_FOLDER
from .snap_metadata import SnapSampleIndex
from .snap_controller import (SnapExperimentSettings, SnapWatchController,
    discover_snap_samples, preflight_snap_experiment)


@dataclass(frozen=True)
class BeamlineAdapter:
    instrument: str
    name: str
    default_folder: str
    sample_index: type
    settings: type
    controller: type
    discover_samples: Callable
    preflight: Callable
    automatic_profiles: bool
    facility: str = 'SNS'
    instrument_mode: str = 'tof'
    preview: bool = False
    preview_runs: Callable | None = None
    preview_setup: Callable | None = None


def _tof_runs(definition, policy):
    from . import nomad
    from .powgen_controller import bounded_directory_listing
    from .powgen_watch import discover_from_listing
    listing = nomad.directory_listing if definition.instrument == 'NOM' else bounded_directory_listing
    return discover_from_listing(definition, listing(definition.source_directory))


def _tof_setup(path, definition, policy):
    from . import nomad, powgen_setup
    setup = (nomad.inspect_reduced_setup(path, inventory_only=True) if definition.instrument == 'NOM'
             else powgen_setup.inspect_reduced_setup(path))
    return setup, [{'title':f'Bank {b.bank}', 'value':b.bank} for b in setup.banks]


def _hb2a_runs(definition, policy):
    from . import hb2a
    normalization = policy.get('normalization', 'time')
    # Both readers enforce the selected product family. Keep their policies
    # identical without mutating the caller's immutable watch definition.
    preview = replace(definition, config_refs={**definition.config_refs,
                                              'hb2a_normalization': normalization})
    return hb2a.discover_listing(preview, hb2a.directory_listing(preview.source_directory,
        normalization=normalization))


def _hb2a_setup(path, definition, policy):
    from . import hb2a
    setup = hb2a.inspect_reduced_setup(path, ipts=definition.ipts,
        condition_bindings=policy.get('condition_bindings'),
        collimator_override=policy.get('collimator_override', ''), allow_stopped=policy.get('allow_stopped', False))
    return setup, [{'title':'Combined detector-bank pattern · 2θ (degrees)', 'value':1}]


def _snap_runs(definition, policy):
    from . import snap
    from .powgen_watch import discover_from_listing
    rows = snap.directory_listing(definition.source_directory)
    families = snap.product_families(rows)
    chosen = policy.get('product_family') or (families[0]['value'] if len(families) == 1 else '')
    if len(families) > 1 and not chosen:
        raise ValueError('Choose one SNAP export variant before inspecting samples; re-reductions are kept separate.')
    return discover_from_listing(definition, [r for r in rows if r['snap_product_family'] == chosen])


def _snap_setup(path, definition, policy):
    from . import snap
    setup = snap.inspect_reduced_setup(path, inventory_only=True)
    return setup, [{'title':f'Focused histogram {b.bank}', 'value':b.bank} for b in setup.banks]


BEAMLINES = {
    'PG3':BeamlineAdapter('PG3','POWGEN','shared/autoreduce',PowgenSampleIndex,
        PowgenExperimentSettings,PowgenWatchController,discover_powgen_samples,preflight_powgen_experiment,True,
        preview_runs=_tof_runs, preview_setup=_tof_setup),
    'NOM':BeamlineAdapter('NOM','NOMAD',MODERN_FOLDER,NomadSampleIndex,
        NomadExperimentSettings,NomadWatchController,discover_nomad_samples,preflight_nomad_experiment,False,
        preview_runs=_tof_runs, preview_setup=_tof_setup),
    'HB2A':BeamlineAdapter('HB2A','HB-2A','shared/autoreduce',Hb2aSampleIndex,
        Hb2aExperimentSettings,Hb2aWatchController,discover_hb2a_samples,preflight_hb2a_experiment,
        True,facility='HFIR',instrument_mode='cw',preview=True, preview_runs=_hb2a_runs, preview_setup=_hb2a_setup),
    'SNAP':BeamlineAdapter('SNAP','SNAP',SNAP_FOLDER,SnapSampleIndex,
        SnapExperimentSettings,SnapWatchController,discover_snap_samples,preflight_snap_experiment,False,preview=True,
        preview_runs=_snap_runs, preview_setup=_snap_setup),
}


def beamline_adapter(instrument):
    try:return BEAMLINES[instrument]
    except KeyError:raise ValueError('Choose a supported Live beamline') from None
