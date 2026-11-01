"""Reviewed, deterministic starting recipes for explicitly prepared models."""
from copy import deepcopy

from ..contracts import validate_plan

BASELINE_REQUEST = (
    "Battery baseline: refine each active bank background and linked phase HAP scale, "
    "then add the shared cell of each active phase. Keep histogram scales, instrument "
    "profiles, sample widths, texture, atomic positions, occupancies and displacement "
    "parameters fixed. This trial does not measure Li or establish model acceptance."
)


def baseline_recipe(context):
    histograms = context['histograms']
    phases = [phase for phase in context['phases'] if phase['histograms']]
    if not histograms or not phases:
        raise ValueError('The baseline recipe requires linked powder histograms and phases')
    free = [dict(kind='background',histogram=histogram['key']) for histogram in histograms]
    free += [dict(kind='phase_scale',phase=phase['key'],histogram=histogram['key'])
             for phase in phases for histogram in histograms if histogram['name'] in phase['histograms']]
    stages = [dict(id='scales',title='Bank backgrounds and linked phase scales',cycles=6,
                   free=deepcopy(free),constraints=[])]
    free += [dict(kind='cell',phase=phase['key']) for phase in phases]
    stages.append(dict(id='cells',title='Shared phase cells with backgrounds and phase scales',
                       cycles=6,free=free,constraints=[]))
    return validate_plan(dict(schema='radar-gpx-recipe/v1',model_signature=context['model_signature'],
        request=BASELINE_REQUEST,stages=stages,questions=[],unsupported=[],assumptions=[
            'Use only the explicit phase-to-bank links and initial values in the prepared model.',
            'HAP scales are intensity coefficients; they are not mass fractions.',
            'All occupancies including Li remain fixed. Chemistry, calibration and model adequacy are unqualified.',
            'Review the native parameter preflight and scan scope before running this trial.'
        ]),context)
