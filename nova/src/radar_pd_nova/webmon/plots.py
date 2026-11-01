"""Coordinated scientific views; presentation never changes inference arrays."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = ['#1769aa', '#cc6b16', '#7b4da6', '#168779', '#b44366']

def layout(fig, title='', height=420, revision='webmon'):
    fig.update_layout(template='plotly_white', height=height,
        font=dict(family='Arial, sans-serif', size=14, color='#263849'),
        margin=dict(l=76,r=28,t=28,b=60),
        legend=dict(orientation='h', y=-.16, x=0, font=dict(size=12)),
        hovermode='x unified', uirevision=revision,
        paper_bgcolor='white', plot_bgcolor='white')
    if title: fig.update_layout(title=dict(text=title, font=dict(size=16)), margin_t=58)
    fig.update_xaxes(showline=True, linecolor='#a9b4bf', gridcolor='#edf1f5', zeroline=False)
    fig.update_yaxes(showline=False, gridcolor='#edf1f5', zeroline=False)
    return fig

def observed(pattern, axis='d', difc=None, errors=False):
    use_tof=axis=='tof' and difc is not None
    x=pattern.x*difc if use_tof else pattern.x
    f=go.Figure(go.Scattergl(x=x,y=pattern.y,mode='markers',marker=dict(size=3,color='#4c6074',opacity=.65),name='Stored observations',error_y=dict(type='data',array=pattern.sigma,visible=errors,thickness=.6,width=0)))
    layout(f,height=300,revision=f'observed-{axis}')
    f.update_xaxes(title_text='TOF (µs)' if use_tof else 'd (Å)')
    f.update_yaxes(title_text='Reduced density (file units)')
    return f

def coordinates(post,axis,difc):
    x=np.asarray(post['x'])
    if axis in {'d','q'}:
        if difc is None or difc<=0: raise ValueError('Verified export DIFC is required for d-spacing display')
        d=x/difc
        if np.any(d<=0):raise ValueError('Positive d spacing required')
        return (2*np.pi/d,'Q (Å⁻¹)') if axis=='q' else (d,'d (Å)')
    return x,'TOF (µs)'

def checkpoint_review(data, *, axis='d', difc=None, revision=''):
    """Published GPX against its own observations; never a target-run fit."""
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.76,.24],vertical_spacing=.05)
    mask=np.asarray(data['valid'],dtype=bool)
    x,label=coordinates(data,axis,difc);order=np.argsort(x[mask]);x=x[mask][order]
    y=np.asarray(data['y'])[mask][order];calc=np.asarray(data['calc'])[mask][order]
    sigma=np.asarray(data['sigma'])[mask][order]
    f.add_trace(go.Scattergl(x=x,y=y,mode='markers',marker=dict(size=3,color='#334155'),name='Original autoreduced observations'),row=1,col=1)
    f.add_trace(go.Scattergl(x=x,y=calc,mode='lines',line=dict(color='#dd6b20',width=1.6),name='Published GPX calculation'),row=1,col=1)
    lx,_=coordinates({'x':data['live_x']},axis,difc);lo=np.argsort(lx)
    f.add_trace(go.Scattergl(x=lx[lo],y=np.asarray(data['live_scaled'])[lo],mode='lines',line=dict(color='#008577',width=1),name=('Target live snapshot · reviewed unit conversion' if 'reviewed_live_conversion' in data else 'Same-run WebMON · affine scaled for shape comparison')),row=1,col=1)
    f.add_trace(go.Scattergl(x=x,y=(y-calc)/sigma,mode='lines',line=dict(color='#66788a',width=1),name='Original fit residual',showlegend=False),row=2,col=1)
    f.add_hline(y=0,line_color='#7c8894',row=2,col=1)
    layout(f,height=510,revision=f'checkpoint-{revision}-{axis}')
    f.update_layout(margin_b=115,legend=dict(y=-.30,font=dict(size=11)))
    f.update_xaxes(title_text=label,row=2,col=1)
    f.update_xaxes(range=[float(x[0]),float(x[-1])],autorange=False)
    f.update_yaxes(title_text='Original GPX units',row=1,col=1)
    f.update_yaxes(title_text='Residual / σ',row=2,col=1)
    return f

def comparison(models,axis='tof',difc=None,errors=False,ensemble=False,component=False,region='full',ticks=True):
    """Same observations and masks; Q is sorted before plotting filled bands."""
    f=make_subplots(rows=4,cols=1,shared_xaxes=True,row_heights=[.52,.14,.21,.13],vertical_spacing=.035)
    if not models:return layout(f,height=700)
    first=models[0];p=first['post'];rawx,xlabel=coordinates(p,axis,difc);order=np.argsort(rawx);x=rawx[order]
    y=np.asarray(p['y'])[order];sigma=np.asarray(p['sigma'])[order];scale=1e6
    for m in models[1:]:
        for key in ('x','y','sigma'):
            if not np.array_equal(p[key],m['post'][key]):raise ValueError('Compared fits must use the same observations and mask')
    ref=np.asarray(p['fit'])[order]
    tick_values=[];tick_labels=[];nuclear_done=False
    if component and 'base' in p:
        f.add_trace(go.Scattergl(x=x,y=np.asarray(p['base'])[order]*scale,line=dict(color='#75808a',dash='dash',width=1.3),name='Fixed nuclear + background'),row=1,col=1)
    for index,m in enumerate(models):
        a=m['post'];color=m['color'];fit=np.asarray(a['fit'])[order];name=m['label']
        if ensemble:
            band=np.asarray(a['mean_band'])[:,order];rgb=tuple(int(color[i:i+2],16) for i in (1,3,5))
            f.add_trace(go.Scatter(x=x,y=band[0]*scale,line=dict(width=0),showlegend=False,hoverinfo='skip',legendgroup=m['id']),row=1,col=1)
            f.add_trace(go.Scatter(x=x,y=band[2]*scale,fill='tonexty',fillcolor=f'rgba({rgb[0]},{rgb[1]},{rgb[2]},0.16)',line=dict(width=0),name=name+' · 95% mean envelope',showlegend=False,legendgroup=m['id'],hoverinfo='skip'),row=1,col=1)
        label=name+(f" · Rwp {a['rwp']:.2f}%" if 'rwp' in a else '')
        f.add_trace(go.Scattergl(x=x,y=fit*scale,line=dict(color=color,width=1.8),name=label,legendgroup=m['id']),row=1,col=1)
        f.add_trace(go.Scattergl(x=x,y=(y-fit)/sigma,mode='lines',line=dict(color=color,width=1),showlegend=False,legendgroup=m['id'],name=name+' residual'),row=3,col=1)
        f.add_trace(go.Scattergl(x=x,y=(fit-ref)/sigma,mode='lines',line=dict(color=color,width=1.2),showlegend=False,legendgroup=m['id'],name=name+' minus reference'),row=4,col=1)
        if ticks:
            for table in a.get('reflections',[]):
                magnetic=table['kind']=='magnetic'
                if not magnetic and nuclear_done:continue
                level=-(index+1) if magnetic else 0
                tx,_=coordinates({'x':table['tof']},axis,difc)
                tick_values.append(level);tick_labels.append(f'M{index+1}' if magnetic else 'N')
                texts=[f"{table['phase']} · ({h} {k} {l}) · F²={v:.4g}" for (h,k,l),v in zip(table['hkl'],table['f2'])]
                f.add_trace(go.Scattergl(x=tx,y=[level]*len(tx),mode='markers',marker=dict(symbol='line-ns-open',size=9,color=color if magnetic else '#425466',line=dict(width=1)),text=texts,hovertemplate='%{text}<extra></extra>',showlegend=False,legendgroup=m['id'] if magnetic else 'nuclear'),row=2,col=1)
            if a.get('reflections'):nuclear_done=True
    f.add_trace(go.Scattergl(x=x,y=y*scale,mode='markers',marker=dict(size=3,color='#2e3f50',opacity=.65),name='Measured',error_y=dict(type='data',array=sigma*scale,visible=errors,thickness=.5,width=0)),row=1,col=1)
    if not tick_values:
        f.add_annotation(text='Reflection ticks unavailable' if ticks else 'Reflection ticks hidden',xref='x2 domain',yref='y2 domain',x=.5,y=.5,showarrow=False,font=dict(size=11,color='#647589'))
    f.update_yaxes(tickmode='array',tickvals=tick_values,ticktext=tick_labels,range=[-len(models)-.6,.6],showgrid=False,title_text='Reflections',row=2,col=1)
    f.add_hrect(y0=-3,y1=3,fillcolor='#e7edf3',opacity=.7,line_width=0,row=3,col=1)
    for row in (3,4):f.add_hline(y=0,line_color='#7c8894',line_width=1,row=row,col=1)
    layout(f,height=700,revision=f"{first.get('revision','')}-{axis}-{region}")
    f.update_layout(margin_b=115,legend=dict(y=-.13,font=dict(size=11)))
    f.update_yaxes(title_text='Intensity × 10⁶',row=1,col=1)
    f.update_yaxes(title_text='Residual / σ',row=3,col=1)
    f.update_yaxes(title_text='Δ fit / σ',row=4,col=1)
    f.update_xaxes(title_text=xlabel,row=4,col=1)
    if region=='difference' and len(models)>1:
        score=np.max([abs((np.asarray(m['post']['fit'])[order]-ref)/sigma) for m in models[1:]],axis=0)
        peak=int(np.argmax(score));width=max((x[-1]-x[0])*.04,1e-9)
        limits=[max(x[0],x[peak]-width),min(x[-1],x[peak]+width)]
    elif region=='large_d':
        limits=[x[0],x[0]+.45*(x[-1]-x[0])] if axis=='q' else [x[0]+.55*(x[-1]-x[0]),x[-1]]
    else:limits=[float(x[0]),float(x[-1])]
    f.update_xaxes(range=limits,autorange=False)
    return f

def moments(post,lower=None,upper=None):
    labels=post['labels'];interval=np.asarray(post['intervals']);f=go.Figure()
    if 'prior_moment_draws' in post:
        q=np.quantile(np.asarray(post['prior_moment_draws']),[.025,.5,.975],axis=0).T
        f.add_trace(go.Scatter(x=q[:,1],y=labels,mode='markers',name='Prior 95%',marker=dict(color='#9ca8b3',size=7),error_x=dict(type='data',symmetric=False,array=q[:,2]-q[:,1],arrayminus=q[:,1]-q[:,0],color='#c5cdd5',thickness=6,width=0)))
    f.add_trace(go.Scatter(x=interval[:,1],y=labels,mode='markers',name='Posterior 95%',marker=dict(color='#1769aa',size=9,symbol='diamond'),error_x=dict(type='data',symmetric=False,array=interval[:,2]-interval[:,1],arrayminus=interval[:,1]-interval[:,0],color='#1769aa',thickness=2.5,width=7)))
    for bound in (lower,upper):
        if bound is not None:f.add_vline(x=bound,line_dash='dot',line_color='#b66a1c')
    layout(f,height=max(290,100+len(labels)*42));f.update_layout(legend=dict(y=1.16),margin_t=45,margin_b=60);f.update_xaxes(title_text='Ordered moment (µB)');f.update_yaxes(autorange='reversed')
    return f

def correlation(post,pair=(0,1)):
    f=go.Figure();m=np.asarray(post['moment_draws']);a,b=map(int,pair)
    if m.shape[1]>1 and a!=b and 0<=a<m.shape[1] and 0<=b<m.shape[1]:
        f.add_trace(go.Scattergl(x=m[:,a],y=m[:,b],mode='markers',marker=dict(size=4,color='#1769aa',opacity=.4),name='Posterior draws'))
        f.update_xaxes(title_text=post['labels'][a]+' (µB)');f.update_yaxes(title_text=post['labels'][b]+' (µB)')
    else:f.add_annotation(text=('One independent moment magnitude: no inter-site correlation to plot.' if m.shape[1]==1 else 'Select two different sites to inspect their moment correlation.'),xref='paper',yref='paper',x=.5,y=.5,showarrow=False)
    return layout(f,height=320)

def sampler(post):
    histories=post.get('sampler_histories',[post['sampler_history']]);n=post.get('particles')
    f=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.22)
    for j,h in enumerate(histories):
        ess=[v['ess']/n*100 if n else v['ess'] for v in h]
        f.add_trace(go.Scatter(x=[v['beta'] for v in h],y=ess,name=f'Replica {j+1}: weight ESS',line=dict(color=COLORS[j],width=2)),row=1,col=1)
        key='local_acceptance' if h and 'local_acceptance' in h[0] else 'acceptance'
        label='shape-changing proposals' if key=='local_acceptance' else 'all proposals, incl. sign flips'
        f.add_trace(go.Scatter(x=[v['beta'] for v in h],y=[100*v[key] for v in h],name=f'Replica {j+1}: {label}',line=dict(color=COLORS[j],width=2)),row=2,col=1)
    layout(f,height=420);f.update_layout(legend=dict(y=-.32,font=dict(size=10)),margin_b=120)
    if n:f.add_hline(y=80,line_dash='dot',line_color='#8090a0',row=1,col=1)
    f.update_yaxes(title_text='Weight ESS (%)' if n else 'Weight ESS',range=[0,100] if n else None,row=1,col=1)
    f.update_yaxes(title_text='Accepted (%)',range=[0,100],row=2,col=1)
    f.update_xaxes(title_text='Data influence β (0 = prior; 1 = full data)',row=2,col=1)
    return f
