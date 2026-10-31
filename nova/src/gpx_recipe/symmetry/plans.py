"""Build an explicit finite protocol from reviewed native discovery choices."""
import copy
import itertools
from pathlib import Path
from .compiler import file_sha,package_signature
from .contracts import validate_protocol
from .chemistry import validate_sites


def build_plan(context,catalog,root,settings):
    root=Path(root);parent=next(p for p in context['phases'] if p['name']==settings['parent'])
    if settings['histogram'] not in parent['histograms']:raise ValueError('The selected parent is not linked to this histogram / bank')
    histogram=next(h for h in context['histograms'] if h['name']==settings['histogram'])
    if catalog['parent_setting_sha256']!=parent['setting_sha256'] or catalog['k']!=settings['k']:raise ValueError('Discover candidates again for this exact parent setting and k vector')
    selected=set(settings['candidates'])
    candidates=[copy.deepcopy(c) for c in catalog['candidates'] if c['id'] in selected]
    if not selected or len(candidates)!=len(selected):raise ValueError('Choose constructed candidates from this catalog')
    kind=settings['kind'];seeds=[];sites=[]
    if kind=='magnetic':
        if not str(histogram['instrument']['Type'][0]).startswith('PN'):raise ValueError('This magnetic executor requires neutron powder diffraction')
        sites=settings['sites'];validate_sites(parent['atoms'],sites)
        if not settings['directions']:raise ValueError('Select at least one finite moment direction')
        seeds=[dict(id=axis,axis=axis,magnitude=min(2.,settings['maximum_moment'])) for axis in settings['directions']]
        if settings.get('relative_sign_starts'):
            if len(sites)>5:raise ValueError('Exhaustive parent-site relative signs are bounded to five selected sites')
            seeds=[]
            # Reversal of every moment is powder-equivalent. Fix the first
            # parent site's sign and enumerate relative parent-site signs.
            for axis in settings['directions']:
                for index,tail in enumerate(itertools.product([-1,1],repeat=len(sites)-1)):
                    signs=dict(zip([site['label'] for site in sites],(1,*tail)))
                    seeds.append(dict(id=axis+'-signs-'+str(index+1),axis=axis,magnitude=min(2.,settings['maximum_moment']),site_signs=signs))
    elif kind=='structural':
        amplitude=float(settings['start_mode'])
        if not 0<amplitude<=settings['maximum_mode']:raise ValueError('A nonzero mode start must be within its declared bound')
        for index,candidate in enumerate(candidates):
            chosen=[mode for mode in candidate['modes'] if candidate['id']+'|'+mode in settings['modes']]
            if not chosen:raise ValueError('Select at least one native mode for every chosen child')
            if len(chosen)>6:raise ValueError('The exhaustive signed-start budget requires selecting at most six modes per child')
            candidate['modes']=chosen
            for number,signs in enumerate(itertools.product([-1,1],repeat=len(chosen))):
                seeds.append(dict(id=f'child-{index+1}-signs-{number+1}',candidate_ids=[candidate['id']],mode_values=dict(zip(chosen,[amplitude*s for s in signs]))))
    else:raise ValueError('Unknown exploration kind')
    if kind=='magnetic' and settings.get('moment_model')=='symmetry_allowed':
        starts=[]
        for seed in seeds:
            other=[axis for axis in 'xyz' if axis!=seed['axis']]
            for index,signs in enumerate(itertools.product([-1,1],repeat=2)):
                direction={seed['axis']:1.,**{axis:.5*sign for axis,sign in zip(other,signs)}}
                for level,fraction in enumerate([.25,.75]):
                    starts.append(dict(seed,id=seed['id']+'-cant-'+str(index+1)+'-amp-'+str(level+1),independent_mode_seed='xyz'.index(seed['axis'])*8+index*2+level+1,
                        magnitude=settings['maximum_moment']*fraction,parent_direction=[direction[a] for a in 'xyz']))
        seeds=starts
    policy=dict(cycles=settings['cycles'],timeout_seconds=120,maximum_trials=2000,maximum_cell_multiplier=8,maximum_moment=settings['maximum_moment'],fit_band_percent=.25,maximum_mode=settings['maximum_mode'],adequate_rwp_percent=settings['adequate_rwp'],
        refine_cell=kind=='structural',refine_background=kind=='structural',refine_histogram_scale=kind=='structural')
    if kind=='magnetic' and 'magnetic_d_min' in settings:policy['magnetic_d_min']=settings['magnetic_d_min']
    if kind=='magnetic' and 'moment_model' in settings:policy['moment_model']=settings['moment_model']
    if kind=='magnetic':
        policy['refine_histogram_scale']=settings.get('refine_histogram_scale',False)
        policy['refine_background']=settings.get('refine_background',False)
    protocol=dict(kind=kind,source=dict(path='source.gpx',sha256=file_sha(root/'source.gpx')),parent_phase=parent['name'],histogram=settings['histogram'],parent_setting_sha256=parent['setting_sha256'],magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],implementation_sha256=package_signature(),catalog=dict(path='catalog.json',sha256=file_sha(root/'catalog.json')),k=settings['k'],candidates=candidates,seeds=seeds,sites=sites,policy=policy)
    if settings.get('acquisition_id'):protocol['acquisition_id']=settings['acquisition_id']
    return validate_protocol(protocol)
