"""Read-only, full-resolution presentation of archived inference."""
import json
import hashlib
from pathlib import Path
import numpy as np

STATUS = {'conditional_posterior':'Replica screen passed', 'sampling_unresolved':'Sampling unresolved',
          'constraint_incompatible':'Excluded by assumption', 'policy_omitted':'Omitted by policy',
          'pending':'Waiting', 'construction_unresolved':'Construction unresolved'}

def full_posterior(path):
    path=Path(path);p=json.loads(path.read_text())
    with np.load(path.parent/'kernel.npz',allow_pickle=False) as a:
        x,y,sigma,base,q=(a[k].copy() for k in ('x','y','sigma','base','quadratic'))
    with np.load(path.parent/'posterior.npz',allow_pickle=False) as a:
        samples=a['samples'];best=samples[np.argmax(a['loglike'])]
        draws=samples
    # Bound temporary curve memory without thinning bins or posterior draws.
    band=np.empty((3,len(base)))
    for start in range(0,len(base),256):
        stop=min(start+256,len(base));curves=base[start:stop]+np.einsum('ni,ijb,nj->nb',draws,q[:,:,start:stop],draws,optimize=True)
        band[:,start:stop]=np.quantile(curves,[.025,.5,.975],axis=0)
    p.update(x=x,y=y,sigma=sigma,base=base,fit=base+np.einsum('i,ijb,j->b',best,q,best,optimize=True),mean_band=band)
    request=path.parent/'inference-request.json'
    if request.exists():p['particles']=json.loads(request.read_text()).get('particles')
    ticks=path.parent/'reflections.json';p['reflections']=[]
    p['reflection_status']='Native reflection tables unavailable for this saved result'
    if ticks.exists():
        metadata=json.loads(ticks.read_text())
        valid=all(metadata.get(key)==hashlib.sha256((path.parent/name).read_bytes()).hexdigest()
                  for key,name in [('kernel_sha256','kernel.npz'),('posterior_sha256','posterior.npz')])
        if valid:
            p['reflections']=metadata['tables'];p['reflection_status']=metadata['scope']
        else:p['reflection_status']='Reflection tables do not match this fit; ticks withheld'
    p['fitted_gpx'] = None
    receipt = path.parent/'posterior-gpx.json'
    if receipt.is_file():
        export = json.loads(receipt.read_text())
        candidate = path.parent/'posterior-maximum.gpx'
        valid = candidate.is_file() and export.get('fitted_gpx') == candidate.name
        for key, artifact in [('fitted_gpx_sha256', candidate), ('kernel_sha256', path.parent/'kernel.npz'), ('posterior_sha256', path.parent/'posterior.npz')]:
            valid = valid and artifact.is_file() and export.get(key) == hashlib.sha256(artifact.read_bytes()).hexdigest()
        if valid:
            p['fitted_gpx'] = str(candidate)
            p['gpx_export'] = export
    return p

def moment_rows(post,lower=None,upper=None):
    rows=[]
    for label,(lo,med,hi),width in zip(post['labels'],post['intervals'],post['relative_half_width']):
        edge=lower is not None and upper is not None and (lo<lower+.01*(upper-lower) or hi>upper-.01*(upper-lower))
        rows.append(dict(site=label,median=f'{med:.8g}',interval=f'{lo:.8g} – {hi:.8g}',precision=f'{100*width:.3g}%',warning='Touches imposed bound' if edge else 'Conditional only'))
    return rows


def sampler_summary(post):
    history=post.get('sampler_history',[])
    if not history:return 'No sampling history was saved.'
    last=history[-1]
    text=f"Final recorded acceptance: {100*last['acceptance']:.1f}%. "
    if 'local_acceptance' in last:
        text+=f"Local-move acceptance: {100*last['local_acceptance']:.1f}%; sign-reversal acceptance: {100*last['sign_acceptance']:.1f}%. Sign reversals do not explore a different powder pattern."
    else:
        text+='This archive combines local moves with global sign reversals (proposed 5% of the time). Sign reversals leave powder intensities unchanged, so acceptance near 5% may indicate very little useful movement. Local acceptance was not saved separately.'
    health=post.get('health',{})
    if 'predictive_agreement' in health and 'physical_second_moment_agreement' in health:
        disagreements=[]
        if health.get('median_discrepancy_in_interval_widths',0.)>=.5:disagreements.append('moment magnitudes')
        if not health['predictive_agreement']['passed']:disagreements.append('calculated patterns')
        if not health['physical_second_moment_agreement']['passed']:disagreements.append('moment directions or relative signs')
        if disagreements:
            text+=' Independent sampling runs disagree on '+', '.join(disagreements)+'. Sampling remains unresolved.'
        else:text+=' Independent runs agree on the tested magnitudes, patterns and moment relationships. This does not establish a unique structure or global convergence.'
    elif health:text+=' The saved replica comparison covered moment magnitudes only; pattern and directional agreement were not checked.'
    return text
