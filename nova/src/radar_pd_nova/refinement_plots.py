"""Scientific plots for immutable original/stage/final refinement artifacts."""
from __future__ import annotations

import copy
import math

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .results import experiment_axis_options, _experiment_axis

COLORS=['#147d64','#b56016','#8064a2','#1d82a3','#c35069','#6c7d28','#5064a2']


def empty_figure(message='Select a completed refinement to inspect its fit'):
    figure=go.Figure();figure.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False,font=dict(color='#64776c',size=14))
    figure.update_layout(template='plotly_white',height=450,margin=dict(l=60,r=25,t=35,b=45),xaxis=dict(visible=False),yaxis=dict(visible=False))
    return figure


def _coordinate(hist,axis):
    if axis=='native':return np.asarray(hist['x'],dtype=float),'Time of flight (µs)' if hist['coordinate']=='tof' else '2θ (degrees)'
    d=np.asarray([np.nan if x is None else x for x in hist['d']],dtype=float)
    if axis=='d':return d,'d-spacing (Å)'
    if axis=='Q':return 2*np.pi/d,'Q (Å⁻¹)'
    raise ValueError('Choose native coordinates, d-spacing or Q')


def fit_figure(baseline,polished,*,histogram=None,axis='d',display='compare',reset=0):
    if not baseline or not polished:return empty_figure(),{}
    name=histogram or polished['histograms'][0]['name']
    current=next((h for h in polished['histograms'] if h['name']==name),None)
    original=next((h for h in baseline['histograms'] if h['name']==name),None)
    if current is None or original is None:return empty_figure('The selected histogram is not in both projects'),{}
    if current['x']!=original['x'] or current['observed']!=original['observed']:
        raise ValueError('Original and polished views do not have the same measured observations')
    figure=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.63,.22,.15])
    xx,title=_coordinate(current,axis);order=np.argsort(xx);valid=np.isfinite(xx)
    x=xx[order]
    y=np.asarray([np.nan if value is None else value for value in current['observed']],dtype=float)[order]
    y=np.where(valid[order],y,np.nan)
    figure.add_trace(go.Scattergl(x=x,y=y,mode='markers',marker=dict(color='#202c32',size=2.2),name='Observed',hovertemplate='%{x:.6g}<br>Observed %{y:.6g}<extra></extra>'),row=1,col=1)
    traces=[]
    if display in {'original','compare'}:traces.append(('Original RADAR-PD',original,'#81908b','dash'))
    if display in {'polished','compare'}:traces.append(('Selected refinement',current,'#2368c4','solid'))
    for label,h,color,dash in traces:
        calc=np.asarray([np.nan if value is None else value for value in h['calculated']],dtype=float)[order]
        fitted=np.asarray(h['fitted'],dtype=bool)[order]&valid[order]
        calc=np.where(fitted,calc,np.nan)
        figure.add_trace(go.Scattergl(x=x,y=calc,mode='lines',line=dict(color=color,width=1.25,dash=dash),name=label),row=1,col=1)
        figure.add_trace(go.Scattergl(x=x,y=np.where(fitted,y-calc,np.nan),mode='lines',line=dict(color=color,width=.9,dash=dash),name=label+' residual',showlegend=False),row=2,col=1)
    active=original if display=='original' else current
    for index,ticks in enumerate(active.get('ticks',[])):
        dd=np.asarray(ticks['d'],dtype=float)
        if axis=='d':positions=dd
        elif axis=='Q':positions=2*np.pi/dd
        else:
            positions=np.asarray(ticks['native'],dtype=float)
        label=ticks['phase'].split(' - ')[0]
        figure.add_trace(go.Scattergl(x=positions,y=[index]*len(positions),mode='markers',marker=dict(symbol='line-ns',size=8,color=COLORS[index%len(COLORS)]),name=label,legendgroup='phase-'+str(index),hovertemplate=label+'<br>%{x:.6g}<extra></extra>'),row=3,col=1)
    figure.update_layout(template='plotly_white',height=620,margin=dict(l=75,r=22,t=32,b=55),
        legend=dict(orientation='h',y=1.13,x=0,font=dict(size=11)),hovermode='closest',
        uirevision=f'refinement-fit/{name}/{axis}/{reset}')
    figure.update_yaxes(title_text='Intensity',row=1,col=1)
    figure.update_yaxes(title_text='Obs − calc',zeroline=True,zerolinecolor='#b7c4bd',row=2,col=1)
    figure.update_yaxes(title_text='Bragg ticks',showticklabels=False,showgrid=False,zeroline=False,row=3,col=1)
    figure.update_xaxes(title_text=title,row=3,col=1)
    comparable=original['fitted']==current['fitted'] and original['limits']==current['limits'] and original['exclusions']==current['exclusions']
    return figure,dict(original_rwp=original.get('rwp'),polished_rwp=current.get('rwp'),rwp_comparable=comparable,
        notice='Identical fitted observations and exclusions' if comparable else 'Fit limits or exclusions changed; the two Rwp values are not directly comparable')


def _group(job):
    source=job['source'];return (source['sample_key'],str(source.get('bank','')),job['recipe_id'])


def same_fit_domain(a,b):
    if a.get('fit_domain_sha256') and b.get('fit_domain_sha256'):return a['fit_domain_sha256']==b['fit_domain_sha256']
    return all(a.get(k)==b.get(k) for k in ['fitted','limits','exclusions'])


def histogram_map(view):
    return {h.get('key') or f'h{i}':h for i,h in enumerate(sorted(view['histograms'],key=lambda h:h['name']))}


def trend_figure(rows,*,metric='rwp',x_key='run_number',display='absolute'):
    """Rows contain source metadata and native baseline/final views, never guessed conditions."""
    if not rows:return empty_figure('Completed refinements will populate this trend')
    figure=go.Figure();groups={};changed_domain=False
    for row in sorted(rows,key=lambda r:(r['source'].get('run_number') or 0,r['id'])):groups.setdefault(_group(row),[]).append(row)
    title='Rwp (%)' if metric=='rwp' else 'Modeled crystalline weight (%)' if metric=='phase_fractions' else ''
    for index,group in enumerate(groups.values()):
        scans=[dict(run_number=j['source'].get('run_number'),metadata=j['source'].get('metadata') or {},
                    axis_overrides=j['source'].get('axis_overrides') or {}) for j in group]
        x,x_title,x_type=_experiment_axis(scans,x_key)
        label=group[0]['source']['sample_label']+(f" · bank {group[0]['source']['bank']}" if group[0]['source'].get('bank') else '')
        if group[0].get('recipe_label'):label+=' · '+group[0]['recipe_label']
        color=COLORS[index%len(COLORS)]
        if metric=='rwp':
            # A multi-histogram project is plotted one histogram per trace.
            histograms=histogram_map(group[0]['final'])
            for hkey in histograms:
                for row in group:
                    before=histogram_map(row['baseline']).get(hkey)
                    after=histogram_map(row['final']).get(hkey)
                    if before and after and not same_fit_domain(before,after):changed_domain=True
                for source_key,line,marker in [('baseline','dash','circle-open'),('final','solid','circle')]:
                    values=[histogram_map(row[source_key]).get(hkey,{}).get('rwp') for row in group]
                    suffix=' · histogram '+str(list(histograms).index(hkey)+1) if len(histograms)>1 else ''
                    figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=label+suffix+(' · original' if source_key=='baseline' else ' · polished'),line=dict(color=color,dash=line,width=1.3),marker=dict(symbol=marker,size=6),customdata=[j['id'] for j in group],connectgaps=False))
            continue
        if metric=='phase_fractions':
            phases={p['key']:p['phase'] for row in group for p in row['final'].get('parameters',[]) if p['kind']=='weight'}
            for phase_index,(key,phase) in enumerate(phases.items()):
                values=[next((p['value'] for p in row['final'].get('parameters',[]) if p['key']==key),None) for row in group]
                figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=phase.split(' - ')[0]+' · '+label,line=dict(color=COLORS[(index+phase_index)%len(COLORS)],width=1.3),customdata=[j['id'] for j in group],connectgaps=False))
            continue
        values=[];errors=[];statuses=[];unit='';parameter=None
        for row in group:
            p=next((p for p in row['final'].get('parameters',[]) if p['key']==metric),None)
            parameter=parameter or p;values.append(p['value'] if p else None);errors.append(p.get('sigma') if p else None);statuses.append(p.get('refined_stage') if p else None)
            if p:unit=p.get('unit','');title=p['label']+(f' ({unit})' if unit else '')
        valid=[value for value in values if value is not None]
        if not valid:continue
        reference=valid[0]
        if display=='relative':
            if not parameter or parameter['kind']!='cell' or parameter['parameter'] in {'alpha','beta','gamma'} or reference==0:
                raise ValueError('Relative percent is available for nonzero cell lengths and volume; choose absolute values or differences for this parameter')
            values=[100*(v/reference-1) if v is not None else None for v in values];errors=[100*e/abs(reference) if e is not None else None for e in errors];title='Change from first scan (%)'
        elif display=='difference':
            fractional=parameter and parameter['kind']=='atom' and parameter['parameter'] in {'x','y','z'}
            values=[((v-reference+.5)%1-.5 if fractional else v-reference) if v is not None else None for v in values];title='Change from first scan'+(f' ({unit})' if unit else '')
            # Reference covariance/cross-scan correlations are unknown; do not manufacture difference errors.
            errors=[None]*len(errors)
        figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=label,line=dict(color=color,width=1.3),marker=dict(size=6),
            error_y=dict(type='data',array=errors,visible=display=='absolute' and any(e is not None for e in errors)),
            customdata=[j['id'] for j in group],connectgaps=False))
    figure.update_layout(template='plotly_white',height=430,margin=dict(l=75,r=25,t=55,b=60),legend=dict(orientation='h',y=1.18),
        xaxis=dict(title=x_title if groups else 'Scan number',type=x_type if groups else 'linear'),yaxis=dict(title=title),
        uirevision='refinement-trend/'+metric+'/'+x_key+'/'+display)
    if changed_domain:figure.add_annotation(text='Some Rwp pairs use different fit limits or exclusions and are not directly comparable.',xref='paper',yref='paper',x=0,y=1.03,xanchor='left',showarrow=False,font=dict(size=11,color='#9a601b'))
    return figure
