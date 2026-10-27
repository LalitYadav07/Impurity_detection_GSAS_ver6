"""SNAP run identities and qualified, bounded sample-environment observations."""
from pathlib import Path
from hashlib import sha256
import json

from .facility_metadata import ExperimentMetadataIndex, resolve_nexus_path, resolve_identity, title_role
from .powgen_metadata import read_powgen_sample_identity, _text_value, _first_snapshot, _numeric_log, _duration_seconds
from .powgen_samples import PowgenSampleIndex
from .scan_conditions import attach_conditions


def snap_nexus_path(ipts, run):
    return resolve_nexus_path('SNAP', ipts, run)


def read_snap_sample_identity(ipts, run, *, nexus_path=None):
    path = Path(nexus_path) if nexus_path else snap_nexus_path(ipts,run)
    result = read_powgen_sample_identity(ipts,run,nexus_path=path,instrument='SNAP')
    # SNS's unsigned representation of an unset ID is common in real SNAP
    # calibration runs. It is not a shared specimen with an enormous ID.
    sentinels = {'4294967295','18446744073709551615'}
    sources = result.get('identity_sources') or []
    if any(str(source.get('value')) in sentinels for source in sources):
        result['withheld_identity_sources'] = [source for source in sources if str(source.get('value')) in sentinels]
        result.update(resolve_identity(source for source in sources if str(source.get('value')) not in sentinels))
        result.setdefault('metadata_notices',[]).append('Unset unsigned SNAP sample identifier was withheld; no identity was inferred from the title')
    result['run_title'] = _text_value(path,('/entry/title',))
    return result


class SnapMetadataIndex(ExperimentMetadataIndex):
    def __init__(self, ipts, *, root=None):
        super().__init__('SNAP',ipts,root=root)

    def enrich(self, run, raw_identity, *, include_xml=True):
        return {**raw_identity, **super().enrich(run,raw_identity,include_xml=include_xml)}


class SnapSampleIndex(PowgenSampleIndex):
    def __init__(self, ipts, **kwargs):
        super().__init__(ipts,instrument='SNAP',nexus_path=kwargs.pop('nexus_path',snap_nexus_path),
                         identity_reader=kwargs.pop('identity_reader',read_snap_sample_identity),
                         metadata_index=kwargs.pop('metadata_index',SnapMetadataIndex(ipts)),**kwargs)

    def inventory(self, runs, *, catalog=False):
        runs = list(runs)
        result = super().inventory(runs,catalog=catalog)
        by_number = {run.run_number:run for run in runs}
        for row in result['run_options']:
            number = row['value']
            cached = self._preview_cache.get(number) if catalog else self._cache.get(number)
            metadata = cached[1] if cached else {}
            title = metadata.get('run_title') or (metadata.get('table_fields') or {}).get('run_title') or ''
            row.update(run_title=title,run_role=title_role(title),
                       title=f"{by_number[number].run_id} — {title_role(title)}: {title or 'Title not reported'}")
        return result


def read_snap_scan_metadata(ipts, run, *, nexus_path=None, condition_bindings=None):
    path = Path(nexus_path) if nexus_path else snap_nexus_path(ipts,run)
    result = {'schema_version':2,'instrument':'SNAP','ipts':ipts,'nexus_path':str(path),'available':path.is_file()}
    if not result['available']:
        return result
    result.update(read_snap_sample_identity(ipts,run,nexus_path=path))
    for key,field in [('sample_formula','/entry/sample/chemical_formula'),('start_time','/entry/start_time'),
                      ('end_time','/entry/end_time')]:
        result[key] = _text_value(path,(field,))
    result['duration_seconds'] = _duration_seconds(result['start_time'],result['end_time'])
    result['sample_environment_selector'] = _text_value(path,('/entry/DASlogs/BL3:SE:IM:SelectSampleEnv/value_strings',))
    result['metadata_notices'] = [*result.get('metadata_notices', []),
        'SNAP temperature sensors need a reviewed specimen binding; inactive sensors are not substituted',
        'DAC sample pressure is not inferred from pump/load pressure, run titles or equation-of-state constants; import a reviewed pressure mapping when needed']
    geometry = {}
    for name in ['BL3:Mot:vdet_arc1','BL3:Mot:vdet_arc2','BL3:Mot:det_lin1','BL3:Mot:det_lin2']:
        value = _numeric_log(_first_snapshot(path,(f'/entry/DASlogs/{name}/average_value',)))
        if value:
            geometry[name] = value
    result['instrument_geometry_observations'] = geometry
    result = attach_conditions(result,path,'SNAP')
    if condition_bindings:
        from .snap_conditions import binding_scope
        scope = binding_scope(result)
        checked = {}
        for quantity, binding in condition_bindings.items():
            valid = (bool(result.get('sample_id')) and binding.get('scope_sha256') == scope['scope_sha256']
                and binding.get('product_window_reviewed') is True)
            checked[quantity] = {**binding, 'verified': binding.get('verified') is True and valid}
            if not valid:
                result['metadata_notices'].append('Specimen identity is unavailable or the specimen, environment selector or sensor schema changed; review a new SNAP condition mapping')
        result = attach_conditions(result,path,'SNAP',bindings=checked)
        if all(b.get('verified') is True for b in checked.values()):
            packet = result['conditions']
            packet['source_metadata_sha256'] = packet.pop('metadata_sha256')
            packet.update(window_scope='reviewed_full_parent_run',
                product_window_basis='User-reviewed individual GSAS export covers the complete named NeXus acquisition; not a time slice')
            packet['metadata_sha256'] = sha256(json.dumps(packet,sort_keys=True,allow_nan=False).encode()).hexdigest()
    return result
