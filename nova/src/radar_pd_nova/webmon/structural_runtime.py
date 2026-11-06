"""Structural point-fit transport under the existing WebMon source owner.

No acquisition polling or independent job queue lives here. Model Campaign
owns dispatch, uncertain-ack recovery, allocation and native job collection.
"""
from pathlib import Path
import json

from gpx_recipe.structural_live.acquisition import _mapping
from gpx_recipe.structural_live.job import reviewed_job
from gpx_recipe.symmetry.compiler import confined,file_sha
from gpx_recipe.symmetry.contracts import digest


class StructuralSnapshotRuntime:
    def __init__(self,model_runtime):
        self.runtime=model_runtime

    def submit(self,payload,root,*,source,timeout=600,cancel=None,on_intent=None):
        return self.runtime.submit_native('structural-snapshot',payload,root,source=source,
            timeout=timeout,cancel=cancel,on_intent=on_intent)

    def poll(self,handle):
        return self.runtime.poll_native(handle)

    def collect(self,handle,root):
        if handle.get('request',{}).get('operation')!='structural-snapshot':
            raise ValueError('This handle is not a structural acquisition job')
        root=Path(root).resolve()
        # Viewer loss cannot turn an uploaded completion flag into source
        # authority. Recover exact authenticated inputs through Model runtime.
        self.runtime.restore_native_inputs(handle,root)
        job,protocol,candidate=reviewed_job(confined(root,handle['request']['payload']['directory']))
        payload=handle['request']['payload']
        result,seconds=self.runtime.collect_native(handle,root)
        policy=protocol['policy']
        settings=dict(mode_radius=policy['structural_mode_radius'],cell_radius=policy['structural_cell_radius'],
            seconds=policy['structural_seconds'],starts=policy['structural_starts'],
            recovery_seconds=policy['structural_recovery_seconds'],background_scale=None)
        expected=dict(schema='radar-structural-snapshot-result/v1',kind='structural_point_fit',
            state='verified_point_fit',optimization_status='optimization_unresolved',
            protocol_sha256=job['protocol_sha256'],trial_id=job['id'],candidate_id=candidate['id'],
            implementation_sha256=handle['request']['implementation_sha256'],
            source_sha256=handle['source']['source_sha256'],child_sha256=candidate['child']['sha256'],
            reference_sha256=protocol.get('structural_reference',{}).get('sha256'),
            acquisition=digest([_mapping(payload['mapping']),payload['source_scope']]),
            anchor_sha256=payload['mapping']['snapshot_sha256'],snapshot_sha256=payload['receipt']['content_sha256'],
            revision=payload['revision'],forecast_enabled=False,model_exclusion_authorized=False,
            statistical_policy_qualified=False,settings=settings)
        if not isinstance(result,dict) or any(result.get(k)!=v for k,v in expected.items()):
            raise ValueError('Structural result belongs to another source, model or snapshot revision')
        fitted=result.get('result',{})
        if (fitted.get('snapshot_sha256')!=expected['snapshot_sha256']
                or fitted.get('snapshot_revision')!=payload['revision']
                or fitted.get('previous_likelihood_accumulated') is not False
                or fitted.get('all_physical_modes_retained') is not True
                or fitted.get('optimization_qualified') is not False or fitted.get('settings')!=settings):
            raise ValueError('Structural result changed its observation or scientific scope')
        output=confined(root,str(Path(result['output']).relative_to(root)))
        if output!=confined(root,payload['destination']):
            raise ValueError('Structural publication escaped its captured destination')
        for role in ('frozen','continuation'):
            artifact=fitted[role];path=confined(output,str(Path(artifact['path']).relative_to(output)))
            if file_sha(path)!=artifact['sha256']:
                raise ValueError('An independently checked structural GPX changed')
        view=confined(output,str(Path(result['view']).relative_to(output)))
        if view.stat().st_size>32*1024*1024:raise ValueError('Structural view exceeds its display budget')
        display=json.loads(view.read_text())
        if display.get('schema')!='radar-symmetry-view/v1' or display.get('histogram')!=protocol['histogram']:
            raise ValueError('Structural display belongs to another native histogram')
        return result,seconds
