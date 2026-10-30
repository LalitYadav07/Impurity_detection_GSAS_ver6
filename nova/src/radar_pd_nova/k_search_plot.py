"""Calibrated evidence overlay for a selected k-survey template."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gpx_recipe.symmetry.compiler import file_sha
from .symmetry_plots import empty,figure_json


def evidence(result,key,*,axis='q',reset=0):
    family=next((r for r in result['families'] if ','.join(r['k'])==key),None)
    if not family or not family.get('best_template'):return empty('No native simulation is available for this k family')
    template=result['templates'][family['best_template']]
    if file_sha(result['baseline'])!=result['baseline_sha256'] or file_sha(template['path'])!=template['sha256']:
        raise ValueError('A saved survey profile changed; run the search again')
    with np.load(result['baseline'],allow_pickle=False) as data:
        x=data['x'].copy();q=data['q'].copy();obs=data['observed'].copy();baseline=data['nuclear'].copy();mask=data['mask'].copy()
    with np.load(template['path'],allow_pickle=False) as data:magnetic=data['intensity'].copy()
    amplitude=next(r['amplitude'] for r in result['ranking']['templates'] if r['id']==family['best_template'])
    if axis=='q':coordinate=q;label='Q (Å⁻¹)'
    elif axis=='d':coordinate=np.divide(2*np.pi,q,out=np.full_like(q,np.nan),where=q>0);label='d (Å)'
    else:coordinate=x;label='Time of flight (µs)' if result['coordinate']=='tof' else '2θ (°)'
    order=np.argsort(coordinate);coordinate=coordinate[order];obs=obs[order];baseline=baseline[order];mask=mask[order];magnetic=magnetic[order]*amplitude
    # Retain extrema and mask boundaries when a native histogram is very large.
    indices=np.arange(len(coordinate))
    if len(indices)>6000:
        keep=set(np.flatnonzero(np.r_[True,mask[1:]!=mask[:-1]]).tolist())
        keep.update(np.flatnonzero(np.r_[mask[:-1]!=mask[1:],True]).tolist())
        for chunk in np.array_split(indices,750):
            keep.update([int(chunk[0]),int(chunk[-1])])
            for values in (obs,obs-baseline,magnetic):
                finite=np.where(np.isfinite(values[chunk]),values[chunk],0)
                keep.update([int(chunk[np.argmax(finite)]),int(chunk[np.argmin(finite)])])
        indices=np.array(sorted(keep))
    xx=coordinate[indices]
    def display(values):return np.where(mask[indices],values[indices],np.nan).tolist()
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.58,.42],vertical_spacing=.05)
    xx=xx.tolist()
    for name,values,color,mode in [('Observed',obs,'#263e35','markers'),('Fixed nuclear baseline',baseline,'#8b9590','lines'),('Nuclear + scaled magnetic template',baseline+magnetic,'#087d56','lines')]:
        fig.add_trace(go.Scattergl(x=xx,y=display(values),name=name,mode=mode,marker=dict(size=3,color=color),line=dict(color=color,width=1.3),connectgaps=False),row=1,col=1)
    for name,values,color in [('Observed − nuclear',obs-baseline,'#8b9590'),('Added magnetic template',magnetic,'#087d56')]:
        fig.add_trace(go.Scattergl(x=xx,y=display(values),name=name,mode='lines',line=dict(color=color,width=1.3),connectgaps=False),row=2,col=1)
    fig.update_yaxes(title_text='Intensity',row=1,col=1);fig.update_yaxes(title_text='Excess intensity',row=2,col=1)
    fig.update_xaxes(title_text=label,row=2,col=1)
    fig.update_layout(template='plotly_white',height=440,margin=dict(l=65,r=20,t=40,b=50),
        legend=dict(orientation='h',y=1.16,font=dict(size=11)),uirevision=result['source_sha256']+axis+str(reset))
    return figure_json(fig)
