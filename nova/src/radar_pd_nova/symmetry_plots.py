"""Linked, instrument-derived native/d/Q plots for a finite model portfolio."""
from __future__ import annotations
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .symmetry_workspace import view_for

COLORS={'valid':'#16744f','optimization_unresolved':'#b26b12','inadequate_fit':'#8d4961','physical_rejection':'#a1483a'}


def figure_json(fig):return __import__('json').loads(fig.to_json())


def empty(message):
    fig=go.Figure();fig.add_annotation(text=message,x=.5,y=.5,xref='paper',yref='paper',showarrow=False)
    fig.update_layout(template='plotly_white',height=280,xaxis=dict(visible=False),yaxis=dict(visible=False),margin=dict(l=40,r=20,t=35,b=35))
    return figure_json(fig)


def overview(rows,selected=''):
    fig=go.Figure()
    for status in ['valid','inadequate_fit','optimization_unresolved','physical_rejection']:
        points=[r for r in rows if r['status']==status and r['rwp'] is not None and r['nvars'] is not None]
        if not points:continue
        fig.add_trace(go.Scatter(x=[p['nvars'] for p in points],y=[p['rwp'] for p in points],mode='markers',name=status.replace('_',' '),
            customdata=[p['id'] for p in points],text=[p['name']+' · start '+p['seed'] for p in points],
            marker=dict(color=COLORS[status],size=[13 if p['id']==selected else 8 for p in points],symbol='circle' if status=='valid' else 'diamond-open'),
            hovertemplate='%{text}<br>Rwp %{y:.3f}% · %{x} free variables<extra>%{fullData.name}</extra>'))
    if not fig.data:return empty('No audited finite fit is available yet')
    fig.update_layout(template='plotly_white',height=300,margin=dict(l=60,r=25,t=15,b=60),legend=dict(orientation='h',y=1.2),xaxis_title='Independent refined variables',yaxis_title='Rwp (%)',clickmode='event',uirevision='symmetry-overview')
    return figure_json(fig)


def coordinate(view,axis):
    if axis=='native':return np.asarray(view['x'],dtype=float),'Time of flight (µs)' if view['coordinate']=='tof' else '2θ (°)'
    d=np.asarray([v if v is not None else np.nan for v in view['d']],dtype=float)
    d=np.where(d>0,d,np.nan)
    if axis=='d':return d,'d (Å)'
    return np.divide(2*np.pi,d,out=np.full_like(d,np.nan),where=d>0),'Q (Å⁻¹)'


def fit(selected,pins=(),*,axis='d',reset=0):
    if not selected or not selected['has_view']:return empty('Select a model with a native fitted checkpoint')
    v=view_for(selected);x,label=coordinate(v,axis);obs=np.asarray(v['observed'],dtype=float);calc=np.asarray(v['calculated'],dtype=float)
    native=np.asarray(v['x'],dtype=float);mask=np.isfinite(x)&(native>=v['limits'][0])&(native<=v['limits'][1])
    if not mask.any():return empty('This instrument calibration provides no usable positive d spacing here; select the native coordinate')
    order=np.argsort(x[mask]);xx=x[mask][order];fitted=np.asarray(v['fitted'],dtype=bool)[mask][order]
    def display(values):return np.where(fitted,np.asarray(values,dtype=float)[mask][order],np.nan)
    fig=make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.025,row_heights=[.65,.14,.21])
    fig.add_trace(go.Scattergl(x=xx,y=display(obs),mode='markers',marker=dict(color='#293e38',size=3),name='Observed'),row=1,col=1)
    fig.add_trace(go.Scattergl(x=xx,y=display(calc),mode='lines',connectgaps=False,line=dict(color='#2377be',width=1.5),name='Selected calculated'),row=1,col=1)
    for index,pin in enumerate(pins[:2]):
        if pin['source']!=selected['source'] or not pin['has_view'] or pin['id']==selected['id']:continue
        pv=view_for(pin)
        if pv['x']!=v['x'] or pv['fitted']!=v['fitted'] or pv['observed']!=v['observed']:continue
        fig.add_trace(go.Scattergl(x=xx,y=display(pv['calculated']),mode='lines',connectgaps=False,line=dict(color=['#c2701b','#884fba'][index],dash='dot'),name='Pinned · '+pin['name']),row=1,col=1)
    for index,tick in enumerate(v['ticks']):
        native=np.asarray(tick['native'],dtype=float);d=np.asarray(tick['d'],dtype=float)
        tx=native if axis=='native' else d if axis=='d' else 2*np.pi/d
        keep=np.isfinite(tx)&(native>=v['limits'][0])&(native<=v['limits'][1])
        fig.add_trace(go.Scatter(x=tx[keep],y=np.full(keep.sum(),index),mode='markers',marker=dict(symbol='line-ns-open',size=8,color=['#157347','#ae4e4e','#815ba1'][index%3]),name=tick['phase']+' Bragg',hovertemplate=tick['phase']+'<br>%{x:.5g}<extra></extra>'),row=2,col=1)
    fig.add_trace(go.Scattergl(x=xx,y=display(obs-calc),mode='lines',connectgaps=False,line=dict(color='#6b7974',width=1),name='Observed − calculated'),row=3,col=1)
    fig.add_hline(y=0,line_width=1,line_color='#b5c8bc',row=3,col=1)
    fig.update_layout(template='plotly_white',height=570,margin=dict(l=65,r=25,t=25,b=55),legend=dict(orientation='h',y=1.14),uirevision=selected['source']+axis+str(reset))
    fig.update_yaxes(title_text='Intensity',row=1,col=1);fig.update_yaxes(title_text='Bragg',showticklabels=False,row=2,col=1)
    fig.update_yaxes(title_text='Difference',row=3,col=1);fig.update_xaxes(title_text=label,row=3,col=1)
    return figure_json(fig)
