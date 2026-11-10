"""Deployment-owned Galaxy identities, shared by every NOVA workflow.

User configurations and saved scientific bundles cannot choose a deployment.
Existing test installations keep their established IDs by default.
"""
import os
import re

_INTERACTIVE = frozenset({'interactive','gsasii_interactive','result_explorer'})
_ROLES = _INTERACTIVE | frozenset({
    'analyze','resolve_sns_input','library_builder','gpx_handoff','compare_series',
    'configure','facility_access_probe','refine_recipe','model_campaign',
    'live_snapshot','preparation_cuda_training','preparation_cuda_acceptance',
})


def deployment(environ=None):
    value=(os.environ if environ is None else environ).get('RADAR_PD_DEPLOYMENT','prototype')
    if value not in {'prototype','production'}:
        raise ValueError('RADAR_PD_DEPLOYMENT must be prototype or production')
    return value


def tool_id(role,*,environ=None):
    if role not in _ROLES:raise ValueError('Unknown RADAR-PD tool role')
    env=os.environ if environ is None else environ
    mode=deployment(env)
    override=env.get('RADAR_PD_'+role.upper()+'_TOOL_ID')
    if override is not None:
        if not isinstance(override,str) or not re.fullmatch(r'[A-Za-z0-9_.-]+',override):
            raise ValueError('Invalid configured Galaxy tool ID for '+role)
        return override
    name='neutrons_radar_pd_'+role
    return (name+'_prototype' if mode=='prototype'
            else ('nova-'+name if role in _INTERACTIVE else name))


def is_interactive_tool(value):
    return value in {tool_id(role) for role in _INTERACTIVE}


def is_scientific_tool(value):
    return isinstance(value,str) and not is_interactive_tool(value) and (
        value.startswith('neutrons_radar_pd_') or
        value in {tool_id(role) for role in _ROLES-_INTERACTIVE})
