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


BEAMLINES = {
    'PG3':BeamlineAdapter('PG3','POWGEN','shared/autoreduce',PowgenSampleIndex,
        PowgenExperimentSettings,PowgenWatchController,discover_powgen_samples,preflight_powgen_experiment,True),
    'NOM':BeamlineAdapter('NOM','NOMAD',MODERN_FOLDER,NomadSampleIndex,
        NomadExperimentSettings,NomadWatchController,discover_nomad_samples,preflight_nomad_experiment,False),
}


def beamline_adapter(instrument):
    try:return BEAMLINES[instrument]
    except KeyError:raise ValueError('Choose a supported Live beamline') from None
