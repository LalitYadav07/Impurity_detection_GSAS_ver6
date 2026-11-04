"""Calibrated evidence overlay for a selected k-survey template."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from gpx_recipe.symmetry.compiler import file_sha
from .symmetry_plots import empty,figure_json


def evidence(result,key,*,axis='q',reset=0,display_mode='full'):
    family=next((r for r in result['families'] if ','.join(r['k'])==key),None)
    if not family or not family.get('best_template'):return empty('No native simulation is available for this k family')
    template=result['templates'][family['best_template']]
    if file_sha(result['baseline'])!=result['baseline_sha256'] or file_sha(template['path'])!=template['sha256']:
        raise ValueError('A saved survey profile changed; run the search again')
    with np.load(result['baseline'],allow_pickle=False) as data:
        x=data['x'].copy();q=data['q'].copy();obs=data['observed'].copy();baseline=data['nuclear'].copy();mask=data['mask'].copy()
    with np.load(template['path'],allow_pickle=False) as data:
        magnetic=data['intensity'].copy()
        ticks={name:data[name].copy() for name in ('tick_kind','tick_phase','tick_hkl','tick_d','tick_native') if name in data}
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
    residual_only=display_mode=='residual'
    rows=2 if residual_only else 3
    fig=make_subplots(rows=rows,cols=1,shared_xaxes=True,row_heights=[.82,.18] if residual_only else [.5,.35,.15],vertical_spacing=.05)
    xx=xx.tolist()
    if not residual_only:
        for name,values,color,mode in [('Observed',obs,'#263e35','markers'),('Fixed nuclear baseline',baseline,'#8b9590','lines'),('Nuclear + scaled magnetic template',baseline+magnetic,'#087d56','lines')]:
            fig.add_trace(go.Scattergl(x=xx,y=display(values),name=name,mode=mode,marker=dict(size=3,color=color),line=dict(color=color,width=1.3),connectgaps=False),row=1,col=1)
    for name,values,color in [('Observed − nuclear',obs-baseline,'#65716b'),('Added magnetic template',magnetic,'#a45016')]:
        fig.add_trace(go.Scattergl(x=xx,y=display(values),name=name,mode='lines',line=dict(color=color,width=1.3),connectgaps=False),row=1 if residual_only else 2,col=1)
    fig.update_yaxes(title_text='Excess intensity' if residual_only else 'Intensity',row=1,col=1)
    if not residual_only:fig.update_yaxes(title_text='Excess intensity',row=2,col=1)
    if len(ticks)==5:
        positions=ticks['tick_native'] if axis not in {'q','d'} else ticks['tick_d'] if axis=='d' else 2*np.pi/ticks['tick_d']
        finite_x=np.asarray(xx)[np.isfinite(xx)]
        lower,upper=(float(finite_x.min()),float(finite_x.max())) if len(finite_x) else (np.inf,-np.inf)
        for kind,y,color,title,basis in [('nuclear',1,'#65716b','Nuclear Bragg positions','parent'),('magnetic',0,'#8b3eb0','Magnetic template Bragg positions','candidate child')]:
            selected=(ticks['tick_kind']==kind)&np.isfinite(positions)&(positions>=lower)&(positions<=upper)
            texts=[f"{phase}<br>hkl ({basis} basis): {int(h)}, {int(k)}, {int(l)}" for phase,(h,k,l) in zip(ticks['tick_phase'][selected],ticks['tick_hkl'][selected])]
            fig.add_trace(go.Scattergl(x=positions[selected].tolist(),y=[y]*int(selected.sum()),mode='markers',name=title,
                marker=dict(symbol='line-ns',size=9,color=color),text=texts,hovertemplate='%{text}<br>%{x:.6g}<extra></extra>'),row=rows,col=1)
    else:
        fig.add_annotation(text='Exact Bragg annotations are unavailable in this older survey.',xref='paper',yref='paper',x=0,y=.03,showarrow=False)
    fig.update_yaxes(tickvals=[0,1],ticktext=['Magnetic','Nuclear'],range=[-.7,1.7],row=rows,col=1)
    fig.update_xaxes(title_text=label,row=rows,col=1)
    fig.update_layout(template='plotly_white',height=520,margin=dict(l=80,r=20,t=70,b=50),
        legend=dict(orientation='h',y=1.18,font=dict(size=11)),uirevision=result['source_sha256']+axis+str(reset))
    return figure_json(fig)
