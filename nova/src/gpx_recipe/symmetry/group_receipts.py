"""Verify a screened alternative's link to its native profile representative."""
import json
from pathlib import Path
import re

import numpy as np

from .compiler import confined, file_sha
from .contracts import digest
from .profiled_cache import bounded_npz
from .profiled_groups import same_profile


def native_rwp_for_profile(reference_rwp,own,tested):
    """Use the audited native display convention, including its mask policy.

    The optimizer's normalization is retained for optimization diagnostics.
    It must not silently replace the independently audited GPX Rwp displayed
    beside native representatives, particularly for explicitly masked data.
    """
    value=float(reference_rwp)
    if not np.isfinite(value) or value<0:raise ValueError('Invalid native representative Rwp')
    own_cost=float(own@own);tested_cost=float(tested@tested)
    if tested_cost==0:
        if own_cost!=0:raise ValueError('A zero-residual reference cannot normalize a different profile')
        return value
    return value*np.sqrt(own_cost/tested_cost)


def reference_paths(directory,result):
    """Dependencies that must invalidate a cached equivalence receipt."""
    if result.get('status')!='profile_equivalent':return []
    directory=Path(directory);link=result.get('native_reference') or {};kernel=result.get('group_kernel') or {}
    for value in [link.get('id'),kernel.get('owner')]:
        if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{24}',value):
            raise ValueError('Invalid grouped native-reference identity')
    native=confined(directory.parent,link['id'])
    raw=json.loads((native/'result.json').read_text())
    group=confined(directory.parent,kernel['owner']+'/group-data')
    return [(native,p) for p in ['job.json','result.json',*raw.get('files',{})]]+[
        (group.parent,'job.json'),(group,'kernel.json'),(group,'kernel.npz')]


def verify_equivalent(directory,result,job,verify_reference):
    directory=Path(directory)
    link=result.get('native_reference') or {}
    kernel=result.get('group_kernel') or {}
    for value in [link.get('id'),kernel.get('owner')]:
        if not isinstance(value,str) or not re.fullmatch('[a-f0-9]{24}',value):
            raise ValueError('Invalid grouped native-reference identity')
    if link['id']==job['id'] or result.get('effects_verified') is not False:
        raise ValueError('A screened start cannot claim its own native verification')
    reference=confined(directory.parent,link['id'])
    reference_job=json.loads((reference/'job.json').read_text())
    if reference_job.get('candidate_id')!=job['candidate_id']:
        raise ValueError('A screened start cannot borrow a different model')
    preliminary=json.loads((reference/'result.json').read_text())
    if preliminary.get('status') not in {'valid','inadequate_fit'}:
        raise ValueError('Screened equivalence needs a native-checked representative')
    native=verify_reference(reference,dict(id=link['id'],protocol_sha256=job['protocol_sha256']))
    if digest(native)!=link.get('receipt_sha256') or not native.get('audit',{}).get('passed'):
        raise ValueError('Native representative changed or lost its independent audit')
    data=confined(directory.parent,kernel['owner']+'/group-data')
    owner_job=json.loads((data.parent/'job.json').read_text())
    if (owner_job.get('protocol_sha256')!=job['protocol_sha256'] or
            owner_job.get('candidate_id')!=job['candidate_id']):
        raise ValueError('Shared kernel owner belongs to a different model or protocol')
    metadata=json.loads((data/'kernel.json').read_text())
    if (metadata.get('schema')!='radar-profiled-group-kernel/v1'
            or metadata.get('protocol_sha256')!=job['protocol_sha256']
            or metadata.get('candidate_id')!=job['candidate_id']
            or metadata.get('kernel')!={k:v for k,v in kernel.items() if k!='metadata_sha256'}
            or not isinstance(kernel.get('metadata_sha256'),str)
            or not re.fullmatch('[a-f0-9]{64}',kernel['metadata_sha256'])
            or file_sha(data/'kernel.json')!=kernel['metadata_sha256']
            or file_sha(data/'kernel.npz')!=kernel.get('sha256')
            or not bounded_npz(data/'kernel.npz',kernel.get('identity'))):
        raise ValueError('Screened equivalence has no matching checked kernel')
    proof_name='results/optimizer-receipt.json'
    if proof_name not in native.get('files',{}):
        raise ValueError('Native representative lacks an optimizer receipt')
    proof=json.loads(confined(reference,proof_name).read_text())
    if (proof.get('objective_identity',{}).get('cache_identity')!=kernel['identity']
            or not proof.get('qualification',{}).get('passed')):
        raise ValueError('Native representative and screened kernel differ')
    screen=json.loads((directory/'results/surrogate-screening.json').read_text())
    if screen.get('id')!=job['id'] or screen.get('native_verified') is not False:
        raise ValueError('Screened result identity changed')
    with np.load(data/'kernel.npz',allow_pickle=False) as values:
        def residual(fit):
            coefficients=np.asarray(fit['coefficients'],float)
            background=np.asarray(fit['background'],float)
            scale=float(fit['scale'])
            if (coefficients.shape!=(values['quadratic'].shape[1],)
                    or background.shape!=(values['background'].shape[1],)
                    or not np.isfinite(coefficients).all() or not np.isfinite(background).all()
                    or not np.isfinite(scale) or scale<=0):
                raise ValueError('Invalid screened moment/nuisance vector')
            maximum=float(np.max(np.linalg.norm(values['maps']@coefficients,axis=1)))
            if maximum>job['protocol']['policy']['maximum_moment']*(1+1e-8):
                raise ValueError('Screened moments exceed their declared bounds')
            return (values['offset']+scale*(values['nuclear']+
                np.einsum('i,nij,j->n',coefficients,values['quadratic'],coefficients))+
                values['background']@background)
        own=residual(screen['fit']);tested=residual(proof['raw_solution'])
        coefficients=np.asarray(screen['fit']['coefficients'],float)
        components=np.asarray(metadata['component_maps'],float)@coefficients
        magnitudes=np.linalg.norm(values['maps']@coefficients,axis=1)
        moments=result.get('moments')
        if (not isinstance(moments,list) or not moments or
            [m['label'] for m in moments]!=metadata['labels'] or
            not np.allclose([m['moment'] for m in moments],components,rtol=1e-9,atol=1e-10) or
            not np.allclose([m['magnitude'] for m in moments],magnitudes,rtol=1e-9,atol=1e-10)):
            raise ValueError('Screened magnetic parameters differ from the recorded basis solution')
    if not same_profile(own,tested):
        raise ValueError('Screened start does not reproduce the native-tested profile')
    expected=native_rwp_for_profile(native['rwp'],own,tested)
    if not np.isclose(expected,result.get('rwp'),rtol=1e-9,atol=1e-9):
        raise ValueError('Screened Rwp differs from its complete weighted residual')
