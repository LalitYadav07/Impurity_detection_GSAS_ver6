"""Structural point-fit transport under the existing WebMon source owner.

No acquisition polling or independent job queue lives here. Model Campaign
owns dispatch, uncertain-ack recovery, allocation and native job collection.
"""
from pathlib import Path
import json
import math

from gpx_recipe.structural_live.acquisition import _mapping
from gpx_recipe.structural_live.job import reviewed_job
from gpx_recipe.symmetry.compiler import confined,file_sha,job_inputs
from gpx_recipe.symmetry.contracts import digest


class StructuralSnapshotRuntime:
    def __init__(self,model_runtime):
        self.runtime=model_runtime

    def submit(self,payload,root,*,source,timeout=600,cancel=None,on_intent=None):
        return self.runtime.submit_native('structural-snapshot',payload,root,source=source,
            timeout=timeout,cancel=cancel,on_intent=on_intent)

    def has_prepared_ranking(self,payload):
        job,protocol,_=reviewed_job(payload['directory'])
        from gpx_recipe.preparation.structural_hints import read
        try:document=read(job_inputs(Path(payload['directory']),job),protocol)
        except (OSError,ValueError,EOFError):return False
        return bool(document and document.get('ranking_models',{}).get('entries'))

    def submit_ranking(self,payload,root,*,source,timeout=120,cancel=None,on_intent=None):
        return self.runtime.submit_native('structural-snapshot-ranking',payload,root,source=source,
            timeout=timeout,cancel=cancel,on_intent=on_intent)

    def poll(self,handle):
        return self.runtime.poll_native(handle)

    def collect_ranking(self,handle,root):
        if handle.get('request',{}).get('operation')!='structural-snapshot-ranking':
            raise ValueError('This handle is not a structural snapshot ranking job')
        root=Path(root).resolve();self.runtime.restore_native_inputs(handle,root)
        payload=handle['request']['payload'];job,protocol,_=reviewed_job(confined(root,payload['directory']))
        result,seconds=self.runtime.collect_native(handle,root)
        keys={'schema','kind','protocol_sha256','implementation_sha256','source_sha256','observation','ranking',
              'ranking_sha256','output','artifacts','seconds','candidate_pruning_authorized','scientific_result'}
        expected=dict(schema='radar-structural-snapshot-ranking-result/v1',kind='structural_ranking',
            protocol_sha256=job['protocol_sha256'],implementation_sha256=handle['request']['implementation_sha256'],
            source_sha256=handle['source']['source_sha256'],candidate_pruning_authorized=False,scientific_result=False,
            artifacts=['ranking/ranking.json','transfer.json','result.json'])
        if (not isinstance(result,dict) or set(result)!=keys or any(result.get(k)!=v for k,v in expected.items())
                or type(result['seconds']) not in (int,float) or not math.isfinite(result['seconds'])
                or not 0<=result['seconds']<=3600):
            raise ValueError('Structural ranking result differs from its captured source and scope')
        from gpx_recipe.preparation.structural_ranking_result import observation_binding,validate
        observation=observation_binding(result['observation'])
        if (observation['snapshot_sha256']!=payload['receipt']['content_sha256']
                or observation['revision']!=payload['revision']
                or observation['acquisition_sha256']!=digest([_mapping(payload['mapping']),payload['source_scope']])):
            raise ValueError('Structural ranking result belongs to another snapshot revision')
        prepared=payload.get('prepared_scattering')
        validate(result['ranking'],protocol,snapshot=prepared['snapshot_sha256'] if prepared else None,observation=observation)
        output=confined(root,str(Path(result['output']).relative_to(root)))
        if output!=confined(root,payload['destination']):raise ValueError('Structural ranking publication escaped its destination')
        path=confined(output,'ranking/ranking.json')
        if path.stat().st_size>4*1024*1024 or file_sha(path)!=result['ranking_sha256'] or json.loads(path.read_text())!=result['ranking']:
            raise ValueError('Structural ranking evidence changed after collection')
        return result,seconds

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
