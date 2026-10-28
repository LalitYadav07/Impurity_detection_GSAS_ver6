"""Native GSAS-II Rietveld plots for representative completed benchmark recipes."""
import argparse,json,math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gpx_recipe.snapshot import gsas
from gpx_recipe.advanced import lattice


def main():
    parser=argparse.ArgumentParser();parser.add_argument('campaign');args=parser.parse_args();root=Path(args.campaign)
    state=json.loads((root/'model-status.json').read_text());catalog={c['id']:c for c in json.loads((root/'catalog.json').read_text())};seen=set();(root/'figures').mkdir(exist_ok=True)
    for result in sorted(state['results'].values(),key=lambda r:r['case']):
        if result.get('source_sha256')!=state['current_code'] or result['status']!='verified_execution':continue
        case=catalog[result['case']]
        if case['category'] in seen:continue
        seen.add(case['category']);work=Path(result['artifact_directory']);g=gsas().G2Project(gpxfile=str(work/'refined.gpx'));h=g.histograms()[0]
        arrays=h.data['data'][1];x,y,w,calc=[np.asarray(np.ma.getdata(a),dtype=float) for a in arrays[:4]];lower,upper=h.data['Limits'][1];valid=(x>=lower)&(x<=upper)&np.isfinite(y)&np.isfinite(calc)
        d=np.asarray(lattice(g).Pos2dsp(h.InstrumentParameters,x),dtype=float);valid &= np.isfinite(d)&(d>0)
        ids=np.flatnonzero(valid);ids=ids[np.argsort(d[ids])];xx=d[ids];yy=y[ids];yc=calc[ids]
        fitmask=np.ones(len(ids),dtype=bool)
        excluded=[]
        for lo,hi in h.Excluded():
            fitmask &= ~((x[ids]>=lo)&(x[ids]<=hi));excluded.append(sorted(float(lattice(g).Pos2dsp(h.InstrumentParameters,a)) for a in [lo,hi]))
        fig,(ax,delta)=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw={'height_ratios':[4,1]},layout='constrained')
        ax.plot(xx,yy,'.',color='#202626',markersize=1.6,label='Observed',rasterized=True);ax.plot(xx,yc,color='#2566bd',lw=.85,label='GSAS-II calculated')
        residual=np.where(fitmask,yy-yc,np.nan);delta.plot(xx,residual,color='#6d7d75',lw=.65);delta.axhline(0,color='#87988f',lw=.5)
        span=max(np.percentile(yy,99.9)-np.percentile(yy,.1),1.);base=min(float(np.min(yy)),float(np.min(yc)))-.07*span
        reflections=h.reflections()
        for i,(name,record) in enumerate(reflections.items()):
            refs=np.asarray(record.get('RefList',[]))
            if refs.size:
                peaks=refs[:,4];peaks=peaks[(peaks>=xx[0])&(peaks<=xx[-1])];ax.plot(peaks,np.full(len(peaks),base-i*.045*span),'|',color=plt.get_cmap('tab10')(i%10),ms=5,mew=.7,label=name.split(' - ')[0][:38])
        for lo,hi in excluded:
            ax.axvspan(lo,hi,color='#c9944b',alpha=.16);delta.axvspan(lo,hi,color='#c9944b',alpha=.16)
        ax.set_title(f"{case['id']} · {case['project']}\n{case['scenario']} · final Rwp {h.get_wR():.3f}%",loc='left',fontsize=11)
        ax.set_ylabel('Intensity');delta.set_ylabel('Obs − calc');delta.set_xlabel('d-spacing (Å)');ax.legend(fontsize=7,ncol=2,loc='upper right',framealpha=.9)
        for a in [ax,delta]:a.grid(alpha=.16);a.tick_params(labelsize=9)
        fig.savefig(root/'figures'/(case['id']+'.png'),dpi=170);plt.close(fig)
    print('Figures generated for',len(seen),'recipe categories')


if __name__=='__main__':main()
