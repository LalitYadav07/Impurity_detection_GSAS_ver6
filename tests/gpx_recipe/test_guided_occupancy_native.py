"""Known artificial Li occupancy: native CW/TOF recovery and weak precision.

This is an inverse-workflow check using public synthetic structures, not an
experimental calibration or a chemical claim about vacancy-bearing Li2O.
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

if os.environ.get('RADAR_REQUIRE_STRUCTURAL_NATIVE') == '1':
    from GSASII import GSASIIscriptable as G, GSASIIstrMain as M
else:
    G=pytest.importorskip('GSASII.GSASIIscriptable')
    from GSASII import GSASIIstrMain as M

from gpx_recipe import guided
from gpx_recipe.compiler import approve
from gpx_recipe.execution import freeze
from gpx_recipe.snapshot import snapshot, json_safe
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.worker import work


def occupancy_case(root, kind, method, exposure):
    root=Path(root);root.mkdir()
    profile=root/'profile.instprm'
    values=('Lam:1.54\nZero:0\nU:.12\nV:0\nW:.08\nX:0\nY:0\nZ:0\nSH/L:.002\nPolariz.:0\nAzimuth:0\n'
        if kind=='PNC' else 'difC:6200\ndifA:2\ndifB:.4\nZero:1\n2-theta:105\nfltPath:15.5\nalpha:.99\nbeta-0:.02\nbeta-1:.03\nbeta-q:0\nsig-0:1\nsig-1:66\nsig-2:0\nsig-q:0\nX:0\nY:0\nZ:0\nAzimuth:0\n')
    profile.write_text('#GSAS-II instrument parameter file\nType:'+kind+'\nBank:1\n'+values)
    source=root/'source.gpx';project=G.G2Project(newgpx=str(source))
    phase=project.add_phase(phasename='Artificial Li-O occupancy control',spacegroup='F m -3 m',cell=[4.6,4.6,4.6,90.,90.,90.])
    phase.add_atom(.25,.25,.25,'Li','Li1',occ=.72,uiso=.01)
    phase.add_atom(0,0,0,'O','O1',occ=1.,uiso=.01)
    limits=(12.,125.) if kind=='PNC' else (3.,40.)
    h=project.add_simulated_powder_histogram('synthetic '+kind,str(profile),*limits,Npoints=1601,scale=1000.,phases=[phase])
    h.data['Background'][0][2]=3;h.data['Background'][0][3:]=[50.*exposure,0.,0.]
    phase.data['Histograms'][h.name]['Scale'][0]=float(exposure)
    freeze(project);h.data['data'][1][1][:]=1.;project.save()
    calculated=np.asarray(M.DoNoFit(str(source),h.name),float)
    sigma=np.sqrt(np.maximum(calculated,1.))
    h.data['data'][1][1][:]=calculated+np.random.default_rng(20261110).normal(size=len(calculated))*sigma
    h.data['data'][1][2][:]=1./sigma**2
    phase.atom('Li1').occupancy=.9
    phase.data['Histograms'][h.name]['Scale'][0]=.75*exposure
    h.data['Background'][0][3]=60.*exposure
    project.data['Controls']['data']['deriv type']=method;project.save()
    before=file_sha(source);context=snapshot(source);state=guided.start(context)
    answers={'recipe':'vacancy','model_review':'reviewed','cell_phases':[],'solver_review':'accepted',
        'phase':'p0','sites':'Li1','positions':[],
        'reference':'Known artificial diffraction generator with fixed ADPs and instrument; vacancies explicitly permitted for this numerical control.',
        'conditional':'accepted','reference_scope':'single'}
    while (question:=guided.next_question(context,state)) is not None:
        state=guided.answer(context,state,question['id'],answers[question['id']])
    plan=guided.build_plan(context,state)
    preview=work('preview',source,root/'preview',plan=plan)
    assert preview['status']=='preview_passed',preview
    approval=approve(plan,context,preview['preview'],approver='Public synthetic occupancy regression')
    result=work('run',source,root/'run',plan=plan,approval=approval)
    assert result['status']=='completed',result
    final=G.G2Project(gpxfile=str(root/'run'/result['final_gpx']))
    view=json.loads((root/'run'/result['final_view']).read_text())
    parameter=next(row for row in view['parameters'] if row['label'].endswith('Li1 occupancy'))
    for old,new in zip(project.phases()[0].atoms(),final.phases()[0].atoms()):
        np.testing.assert_allclose(old.coordinates,new.coordinates,atol=1e-12,rtol=0)
        assert json_safe(old.ADP)==json_safe(new.ADP)
    assert final.phases()[0].atom('O1').occupancy==1.
    assert json_safe(final.histograms()[0].data['Instrument Parameters'])==json_safe(h.data['Instrument Parameters'])
    assert json_safe(final.histograms()[0].data['Limits'])==json_safe(h.data['Limits'])
    for old,new in zip(h.data['data'][1][:3],final.histograms()[0].data['data'][1][:3]):
        np.testing.assert_array_equal(np.ma.getdata(old),np.ma.getdata(new))
        np.testing.assert_array_equal(np.ma.getmaskarray(old),np.ma.getmaskarray(new))
    assert file_sha(source)==before
    assert parameter['status']=='refined' and 0<parameter['value']<1 and parameter['sigma']>0
    return parameter,result


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('method',['analytic Hessian','numeric'])
def test_guided_occupancy_recovers_known_high_count_control(tmp_path,kind,method):
    value,result=occupancy_case(tmp_path/'case',kind,method,100)
    assert abs(value['value']-.72)<.02
    assert abs(value['value']-.72)<3*value['sigma']
    assert value['sigma']<.01
    if method=='numeric':
        assert all(s['native_evaluation_budget']['evaluations']<=1000 for s in result['stage_outcomes'])
    else:
        assert all(s['convergence_screen']['status']=='passed' for s in result['stage_outcomes'])


def test_weak_tof_control_reports_low_precision_instead_of_a_precise_composition(tmp_path):
    weak,_=occupancy_case(tmp_path/'weak','PNT','numeric',1)
    strong,_=occupancy_case(tmp_path/'strong','PNT','numeric',100)
    assert weak['sigma']>.03
    assert weak['sigma']>5*strong['sigma']
    assert abs(weak['value']-.72)<3*weak['sigma']
