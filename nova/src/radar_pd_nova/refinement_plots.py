"""Scientific plots for immutable original/stage/final refinement artifacts."""
from __future__ import annotations

import copy
import math
import re
from html import escape

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .results import experiment_axis_options, _experiment_axis, _display_metric_record
from .refinement_messages import refinement_review_status
from gpx_recipe.fit_domain import DOMAIN_SCHEMAS, NATIVE_FIT_CONVENTION, histogram_domain_schema, canonical_observations, fit_domain_digest
from gpx_recipe.phase_labels import phase_label

COLORS=['#147d64','#b56016','#8064a2','#1d82a3','#c35069','#6c7d28','#5064a2']


def electrical_exposure(source):
    metadata=source.get('metadata') or {}
    evidence=metadata.get('electrochemistry') or {}
    exposure=evidence.get('exposure') or {}
    if (evidence.get('schema')!='radar-battery-model-electrochemistry/v1' or
        exposure.get('sample_id')!=metadata.get('sample_id') or
        exposure.get('cell_id')!=metadata.get('cell_id')):
        return {}
    return exposure


def refinement_axis_options(sources):
    options=experiment_axis_options(sources)
    if any(electrical_exposure(source).get('voltage_v') for source in sources):
        options.append(dict(title='Mean voltage over covered exposure (V)',value='battery_voltage'))
    return options


def refinement_parameter_label(parameter, source):
    label=parameter['label']
    histogram=parameter.get('histogram')
    if histogram:
        bank=source.get('metadata',{}).get('physical_banks',{}).get(histogram)
        context=f'physical bank {bank}' if bank is not None else histogram
        if context not in label.split(' · '):
            label+=' · '+context
    return label


def empty_figure(message='Select a completed refinement to inspect its fit'):
    figure=go.Figure();figure.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False,font=dict(color='#64776c',size=14))
    figure.update_layout(template='plotly_white',height=450,margin=dict(l=60,r=25,t=35,b=45),xaxis=dict(visible=False),yaxis=dict(visible=False))
    return figure


def li_profile_figure(diagnostic, trials, *, revision=''):
    """Discrete conditional scores; no interpolated intervals or hidden failures."""
    points = diagnostic['points']
    if not any(p['minimum_wss_per_point'] is not None for p in points):
        return empty_figure('Held-Li results will appear as the declared trials finish')
    figure = go.Figure()
    for passed, label, color, symbol in ((True, 'Starts passing optimization screens', '#147d64', 'circle'),
                                         (False, 'Unqualified completed starts', '#b56016', 'x')):
        rows = [r for r in trials if r['wss_per_point'] is not None and
                ((r.get('convergence', {}).get('status') == 'passed' and r.get('adequacy') != 'threshold_exceeded') == passed)]
        if rows:
            figure.add_trace(go.Scatter(x=[r['li_occupancy'] for r in rows], y=[r['wss_per_point'] for r in rows],
                mode='markers', name=label, marker=dict(color=color, symbol=symbol, size=9),
                hovertemplate='Held Li occupancy %{x:.6g}<br>Weighted residual / point %{y:.6g}<extra></extra>'))
    figure.add_trace(go.Scatter(x=[p['occupancy'] for p in points], y=[p['minimum_wss_per_point'] for p in points],
        mode='lines', name='Grid minima (guide)', line=dict(color='#78857e', dash='dot'), connectgaps=False))
    values=[r['wss_per_point'] for r in trials if r['wss_per_point'] is not None]
    lower,upper=min(values),max(values)
    # Do not magnify floating-point roundoff into an apparent profile minimum.
    # Numerical values remain exact in the table/export; users can zoom the plot.
    padding=max((upper-lower)*.1, diagnostic.get('screening_band',0.)/2, abs(upper)*.01, 1e-6)
    figure.update_layout(template='plotly_white', height=350, margin=dict(l=65, r=20, t=30, b=55),
        xaxis_title='Held Li site occupancy', yaxis_title='Weighted residual / fitted point',
        yaxis=dict(range=[max(0.,lower-padding),upper+padding]),
        legend=dict(orientation='h', y=1.2, font=dict(size=11)), uirevision='held-li-profile/'+revision)
    return figure


def _coordinate(hist,axis):
    if axis=='native':return np.asarray(hist['x'],dtype=float),'Time of flight (µs)' if hist['coordinate']=='tof' else '2θ (degrees)'
    d=np.asarray([np.nan if x is None else x for x in hist['d']],dtype=float)
    if axis=='d':return d,'d-spacing (Å)'
    if axis=='Q':return 2*np.pi/d,'Q (Å⁻¹)'
    raise ValueError('Choose native coordinates, d-spacing or Q')


def fit_figure(baseline,polished,*,histogram=None,axis='d',display='compare',reset=0,histogram_label=None,
               starting_label='Starting model'):
    if not baseline or not polished:return empty_figure(),{}
    name=histogram or polished['histograms'][0]['name']
    current=next((h for h in polished['histograms'] if h['name']==name),None)
    original=next((h for h in baseline['histograms'] if h['name']==name),None)
    if current is None or original is None:return empty_figure('The selected histogram is not in both projects'),{}
    observed=canonical_observations(current)
    if current['x']!=original['x'] or observed!=canonical_observations(original):
        raise ValueError('Original and polished views do not have the same measured observations')
    figure=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.035,row_heights=[.63,.22,.15])
    xx,title=_coordinate(current,axis);order=np.argsort(xx);valid=np.isfinite(xx)
    x=xx[order]
    y=np.asarray([np.nan if value is None else value for value in observed],dtype=float)[order]
    y=np.where(valid[order],y,np.nan)
    figure.add_trace(go.Scattergl(x=x,y=y,mode='markers',marker=dict(color='#202c32',size=2.2),name='Observed',hovertemplate='%{x:.6g}<br>Observed %{y:.6g}<extra></extra>'),row=1,col=1)
    traces=[]
    if display in {'original','compare'}:traces.append((starting_label,original,'#81908b','dash'))
    if display in {'polished','compare'}:traces.append(('Selected refinement',current,'#2368c4','solid'))
    evaluated=[h for _,h,_,_ in traces if h.get('fit_statistics',{}).get('calculation_status')!='not_yet_evaluated']
    standardized=bool(evaluated) and all(h.get('weights') is not None and isinstance(h.get('weight_factor'),(int,float)) and h['weight_factor']>0 for h in evaluated)
    for label,h,color,dash in traces:
        if h.get('fit_statistics',{}).get('calculation_status')=='not_yet_evaluated':
            continue
        calc=np.asarray([np.nan if value is None else value for value in h['calculated']],dtype=float)[order]
        fitted=np.asarray(h['fitted'],dtype=bool)[order]&valid[order]
        calc=np.where(fitted,calc,np.nan)
        figure.add_trace(go.Scattergl(x=x,y=calc,mode='lines',line=dict(color=color,width=1.25,dash=dash),name=label),row=1,col=1)
        residual=y-calc
        if standardized:
            w=np.asarray([np.nan if value is None else value for value in h['weights']],dtype=float)[order]
            residual=residual*np.sqrt(w*h['weight_factor'])
        figure.add_trace(go.Scattergl(x=x,y=np.where(fitted,residual,np.nan),mode='lines',line=dict(color=color,width=.9,dash=dash),name=label+' residual',showlegend=False),row=2,col=1)
    active=original if display=='original' else current
    active_view=baseline if display=='original' else polished
    phases={p['name']:p for p in active_view.get('phases',[]) if p.get('name')}
    for index,ticks in enumerate(active.get('ticks',[])):
        dd=np.asarray(ticks['d'],dtype=float)
        if axis=='d':positions=dd
        elif axis=='Q':positions=2*np.pi/dd
        else:
            positions=np.asarray(ticks['native'],dtype=float)
        label=ticks['phase']
        label=phase_label(phases.get(label,dict(name=label)))
        color=COLORS[index%len(COLORS)]
        figure.add_trace(go.Scattergl(x=positions,y=[index]*len(positions),mode='markers',marker=dict(symbol='line-ns',size=11,color=color,line=dict(color=color,width=2)),name=label,legendgroup='phase-'+str(index),hovertemplate=label+'<br>%{x:.6g}<extra></extra>'),row=3,col=1)
    figure.update_layout(template='plotly_white',height=620,margin=dict(l=75,r=22,t=32,b=55),
        legend=dict(orientation='h',y=1.13,x=0,font=dict(size=11)),hovermode='closest',
        uirevision=f'refinement-fit/{name}/{axis}/{reset}',
        meta=dict(refinement_histogram=dict(name=name,title=histogram_label or name)))
    figure.update_yaxes(title_text='Intensity',row=1,col=1)
    figure.update_yaxes(title_text='Weighted residual' if standardized else 'Obs − calc',zeroline=True,zerolinecolor='#b7c4bd',row=2,col=1)
    figure.update_yaxes(title_text='Bragg ticks',showticklabels=False,showgrid=False,zeroline=False,row=3,col=1)
    figure.update_xaxes(title_text=title,row=3,col=1)
    comparable=same_fit_domain(original,current)
    statistics=active.get('fit_statistics') or {}
    sse,points=statistics.get('weighted_residual_sum'),statistics.get('point_count')
    diagnostic=(f" Weighted residual sum per fitted point: {sse/points:.4g} ({points} points). This is not a degrees-of-freedom-corrected chi-square or a calibration acceptance test."
                if isinstance(sse,(int,float)) and math.isfinite(sse) and isinstance(points,int) and points>0 else '')
    unevaluated=original.get('fit_statistics',{}).get('calculation_status')=='not_yet_evaluated'
    comparison=('Starting model has not been calculated; no starting-model curve or residual is shown.' if unevaluated else
        'Identical observations, weights, masks and fit limits.' if comparable else
        'Observations, weights, masks or fit limits differ or lack provenance; the two Rwp values are not directly comparable.')
    return figure,dict(original_rwp=original.get('rwp'),polished_rwp=current.get('rwp'),rwp_comparable=comparable,
        starting_label=starting_label,
        notice=comparison+diagnostic+
        (' Historical residual report: the first bin above the upper limit was omitted. A new battery comparison re-exports corrected metrics from the verified GPX.'
         if active.get('fit_range_convention')!=NATIVE_FIT_CONVENTION else '')+
        (' Residuals use sqrt(weight × histogram weight factor) × (observed − calculated).' if standardized else ''))


def _group(job):
    source=job['source'];return (source['sample_key'],str(source.get('bank','')),job['recipe_id'])


def same_fit_domain(a,b):
    def identity(histogram):
        schema=histogram.get('fit_domain_schema')
        if schema in DOMAIN_SCHEMAS:
            sha=histogram.get('fit_domain_sha256')
            return (schema,sha) if sha else None
        sha=fit_domain_digest(histogram)
        return (histogram_domain_schema(histogram),sha) if sha else None
    left,right=identity(a),identity(b)
    return bool(left and right and left==right)


def histogram_map(view):
    return {h.get('key') or f'h{i}':h for i,h in enumerate(sorted(view['histograms'],key=lambda h:h['name']))}


def histogram_series_label(view, source, key):
    histograms=histogram_map(view);histogram=histograms.get(key,{})
    bank=(source.get('metadata') or {}).get('physical_banks',{}).get(histogram.get('name'))
    if bank is not None:return 'physical bank '+str(bank)
    return 'histogram '+str(list(histograms).index(key)+1) if key in histograms else str(key)


def _reference_caption(references):
    """Bound the static caption; full identities remain in each trace's hover."""
    def short(value, limit):
        text=' '.join(str(value).split())
        return escape(text if len(text)<=limit else text[:limit-1]+'…')

    def number(value):
        return f'{value:.8g}' if isinstance(value,(int,float)) else str(value)

    shown=references if len(references)<=3 else references[:2]
    lines=[]
    for reference in shown:
        run=reference['run_number']
        identity=f'scan {run}' if run is not None else 'result '+short(reference['result_id'],24)
        parts=[identity]
        if reference['coordinate_key']!='run_number':
            value=reference['coordinate_value'];label=reference['coordinate_label']
            if value is None:
                condition=short(label,30)+': unavailable'
            elif label.endswith(')') and ' (' in label:
                unit=label.rsplit(' (',1)[1][:-1]
                condition=short(number(value),28)+' '+escape(unit)
            else:
                condition=short(label,22)+': '+short(number(value),28)
            parts.append(condition)
        parts.append('value '+number(reference['reference_value'])+
                     (' '+escape(reference['unit']) if reference['unit'] else ''))
        prefix='Reference' if len(references)==1 else 'Reference · '+short(reference['group'],36)
        lines.append(prefix+': '+' · '.join(parts))
    if len(shown)<len(references):
        lines.append(f'{len(references)-len(shown)} more reference groups; hover each series for its full reference.')
    return '<br>'.join(lines),len(lines)


def _temperature_span(scans, coordinates):
    """Measured acquisition ranges, never a temperature standard uncertainty."""
    minus, plus = [], []
    for scan, coordinate in zip(scans, coordinates):
        record = _display_metric_record(scan, 'temperature') or {}
        if not isinstance(record, dict):
            record = {}
        unit = str(record.get('unit') or '').strip().lower()
        low, high = record.get('minimum'), record.get('maximum')
        valid = (unit in {'k', 'kelvin', 'c', '°c', 'degc', 'celsius'}
                 and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                         and math.isfinite(v) for v in (coordinate, low, high)))
        if valid:
            offset = 273.15 if unit in {'c', '°c', 'degc', 'celsius'} else 0.
            low, high = low + offset, high + offset
            valid = low <= coordinate <= high
        minus.append(coordinate - low if valid else None)
        plus.append(high - coordinate if valid else None)
    return dict(type='data', symmetric=False, array=plus, arrayminus=minus,
                visible=any(v is not None for v in plus), color='#82918a', thickness=1, width=3)


def trend_figure(rows,*,metric='rwp',x_key='run_number',display='absolute',show_temperature_span=True):
    """Rows contain source metadata and native baseline/final views, never guessed conditions."""
    if not rows:return empty_figure('Completed refinements will populate this trend')
    figure=go.Figure();groups={};changed_domain=False;fixed_inputs=False;references=[];temperature_spans=False
    for row in sorted(rows,key=lambda r:(r['source'].get('run_number') or 0,r['id'])):groups.setdefault(_group(row),[]).append(row)
    title='Rwp (%)' if metric=='rwp' else 'Modeled crystalline weight (%)' if metric=='phase_fractions' else ''
    for index,group in enumerate(groups.values()):
        reviews=[row.get('review') or refinement_review_status({}) for row in group]
        needs_review=[review['needs_review'] for review in reviews]
        review_hover=[escape(review['message']) for review in reviews]
        review_suffix=' · × needs review' if any(needs_review) else ''
        scans=[dict(run_number=j['source'].get('run_number'),metadata=j['source'].get('metadata') or {},
                    axis_overrides=j['source'].get('axis_overrides') or {}) for j in group]
        if x_key=='battery_voltage':
            x=[(electrical_exposure(j['source']).get('voltage_v') or {}).get('mean') for j in group]
            x_title,x_type='Mean voltage over covered exposure (V)','linear'
        else:x,x_title,x_type=_experiment_axis(scans,x_key)
        error_x = (_temperature_span(scans, x) if x_key == 'temperature' and show_temperature_span
                   else dict(visible=False))
        temperature_spans = temperature_spans or error_x['visible']
        if x_key=='battery_voltage' and all(value is None for value in x):continue
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
                    if before and after and before.get('rwp') is not None and after.get('rwp') is not None and not same_fit_domain(before,after):changed_domain=True
                for source_key,line,marker in [('baseline','dash','circle-open'),('final','solid','circle')]:
                    values=[histogram_map(row[source_key]).get(hkey,{}).get('rwp') for row in group]
                    if all(value is None for value in values):continue
                    suffix=' · '+histogram_series_label(group[0]['final'],group[0]['source'],hkey) if len(histograms)>1 else ''
                    review_options=(dict(hovertext=review_hover,hovertemplate='%{x}<br>%{y}<br>%{hovertext}<extra>%{fullData.name}</extra>') if source_key=='final' else {})
                    symbols=['x' if value else marker for value in needs_review] if source_key=='final' and any(needs_review) else marker
                    figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=label+suffix+(' · original' if source_key=='baseline' else ' · polished'+review_suffix),line=dict(color=color,dash=line,width=1.3),marker=dict(symbol=symbols,size=6),error_x=error_x,customdata=[j.get('trend_point',j['id']) for j in group],connectgaps=False,**review_options))
            continue
        if metric=='phase_fractions':
            phases={p['key']:p for row in group for p in row['final'].get('parameters',[]) if p['kind']=='weight'}
            for phase_index,(key,parameter) in enumerate(phases.items()):
                values=[next((p['value'] for p in row['final'].get('parameters',[]) if p['key']==key),None) for row in group]
                phase=parameter['phase']
                if len(histogram_map(group[0]['final']))>1:
                    phase+=' · '+histogram_series_label(group[0]['final'],group[0]['source'],parameter.get('histogram_key') or key.rsplit('|',1)[-1])
                figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=phase+' · '+label+review_suffix,line=dict(color=COLORS[(index+phase_index)%len(COLORS)],width=1.3),
                    marker=dict(symbol=['x' if value else 'circle' for value in needs_review],size=6),
                    hovertext=review_hover,hovertemplate='%{x}<br>%{y}<br>%{hovertext}<extra>%{fullData.name}</extra>',
                    error_x=error_x,customdata=[j.get('trend_point',j['id']) for j in group],connectgaps=False))
            continue
        values=[];errors=[];statuses=[];unit='';parameter=None
        for row in group:
            p=next((p for p in row['final'].get('parameters',[]) if p['key']==metric),None)
            if p and p.get('status') in {'fixed','fixed_by_model','configured','unknown'}:fixed_inputs=True
            parameter=parameter or p;values.append(p['value'] if p else None);errors.append(p.get('sigma') if p else None);statuses.append(p.get('refined_stage') if p else None)
            if p:unit=p.get('unit','');title=refinement_parameter_label(p,row['source'])+(f' ({unit})' if unit else '')
        valid=[value for value in values if value is not None]
        if not valid:continue
        reference=valid[0]
        reference_description=None
        if display in {'relative','difference'}:
            reference_index=next(i for i,value in enumerate(values) if value is not None)
            reference_row=group[reference_index]
            run=reference_row['source'].get('run_number')
            scan=f'scan {run}' if run is not None else 'result '+str(reference_row['id'])
            coordinate=x[reference_index]
            condition=(f'{x_title}: {coordinate:.8g}' if isinstance(coordinate,(int,float))
                       else f'{x_title}: {coordinate}' if coordinate is not None else f'{x_title}: unavailable')
            reference_description=f'{label} → {scan}, {condition}, reference {reference:.8g} {unit}'.strip()
            references.append(dict(group=label,result_id=reference_row['id'],run_number=run,
                coordinate_key=x_key,coordinate_label=x_title,coordinate_value=coordinate,
                reference_value=reference,unit=unit,description=reference_description))
        if display=='relative':
            if not parameter or parameter['kind']!='cell' or parameter['parameter'] in {'alpha','beta','gamma'} or reference==0:
                raise ValueError('Relative percent is available for nonzero cell lengths and volume; choose absolute values or differences for this parameter')
            values=[100*(v/reference-1) if v is not None else None for v in values];errors=[100*e/abs(reference) if e is not None else None for e in errors];title='Change from reference (%)'
        elif display=='difference':
            fractional=parameter and parameter['kind']=='atom' and parameter['parameter'] in {'x','y','z'}
            values=[((v-reference+.5)%1-.5 if fractional else v-reference) if v is not None else None for v in values];title='Change from reference'+(f' ({unit})' if unit else '')
            # Reference covariance/cross-scan correlations are unknown; do not manufacture difference errors.
            errors=[None]*len(errors)
        hover=[('Reference: '+escape(reference_description)+'<br>' if reference_description else '')+review for review in review_hover]
        figure.add_trace(go.Scatter(x=x,y=values,mode='lines+markers',name=label+review_suffix,line=dict(color=color,width=1.3),
            marker=dict(size=6,symbol=['x' if value else 'circle' for value in needs_review]),
            error_y=dict(type='data',array=errors,visible=display=='absolute' and any(e is not None for e in errors)),
            error_x=error_x,
            customdata=[j.get('trend_point',j['id']) for j in group],connectgaps=False,
            hovertext=hover,hovertemplate='%{x}<br>%{y}<br>%{hovertext}<extra>%{fullData.name}</extra>'))
    figure.update_layout(template='plotly_white',height=430,margin=dict(l=75,r=25,t=55,b=60),legend=dict(orientation='h',y=1.18),
        xaxis=dict(title=x_title if groups else 'Scan number',type=x_type if groups else 'linear'),yaxis=dict(title=title),
        uirevision='refinement-trend/'+metric+'/'+x_key+'/'+display,
        meta=dict(references=references,temperature_spans=temperature_spans))
    if changed_domain:figure.add_annotation(text='Some Rwp pairs use different fit limits or exclusions and are not directly comparable.',xref='paper',yref='paper',x=0,y=1.03,xanchor='left',showarrow=False,font=dict(size=11,color='#9a601b'))
    if fixed_inputs:figure.add_annotation(text='Some plotted values are fixed or unestimated model inputs; they are not independent measurements.',
        xref='paper',yref='paper',x=0,y=1.03,xanchor='left',showarrow=False,font=dict(size=11,color='#9a601b'))
    if x_key=='battery_voltage':
        figure.update_traces(mode='markers')
        figure.add_annotation(text='Voltage means describe covered intervals only; inspect exposure coverage and clock uncertainty in each result.',
            xref='paper',yref='paper',x=0,y=-.22,xanchor='left',showarrow=False,font=dict(size=11,color='#64776c'))
        figure.update_layout(margin=dict(b=90))
    if references:
        caption,lines=_reference_caption(references)
        offset=92 if x_key=='battery_voltage' else 58
        figure.add_annotation(text=caption,xref='paper',yref='paper',x=0,y=0,
            xanchor='left',yanchor='top',yshift=-offset,showarrow=False,align='left',
            font=dict(size=10,color='#52665b'),name='reference-caption')
        figure.update_layout(margin=dict(b=offset+15*lines+12))
    return figure
