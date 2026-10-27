"""Beamline registration contract for shared Live UI and orchestration.

Each adapter owns scientific discovery, metadata, preflight and bank staging;
the controller lifecycle remains shared. Adding a beamline is an explicit
registration, never a filename-prefix or nominal-wavelength guess.
"""
from dataclasses import dataclass
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


BEAMLINES = {
    'PG3':BeamlineAdapter('PG3','POWGEN','shared/autoreduce',PowgenSampleIndex,
        PowgenExperimentSettings,PowgenWatchController,discover_powgen_samples,preflight_powgen_experiment,True),
    'NOM':BeamlineAdapter('NOM','NOMAD',MODERN_FOLDER,NomadSampleIndex,
        NomadExperimentSettings,NomadWatchController,discover_nomad_samples,preflight_nomad_experiment,False),
    'HB2A':BeamlineAdapter('HB2A','HB-2A','shared/autoreduce',Hb2aSampleIndex,
        Hb2aExperimentSettings,Hb2aWatchController,discover_hb2a_samples,preflight_hb2a_experiment,
        False,facility='HFIR',instrument_mode='cw',preview=True),
    'SNAP':BeamlineAdapter('SNAP','SNAP',SNAP_FOLDER,SnapSampleIndex,
        SnapExperimentSettings,SnapWatchController,discover_snap_samples,preflight_snap_experiment,False,preview=True),
}


def beamline_adapter(instrument):
    try:return BEAMLINES[instrument]
    except KeyError:raise ValueError('Choose a supported Live beamline') from None
