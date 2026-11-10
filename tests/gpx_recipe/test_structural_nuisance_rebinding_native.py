"""Native CW/TOF full-portfolio controls after nuclear baseline changes."""
import json
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation.structural_nuisance import affine_context
from gpx_recipe.structural_live.native import prepare
from gpx_recipe.structural_live.service import make_source,SearchSettings
from gpx_recipe.structural_live.optimization.offline_native_start_bank import build,OfflineNativeStartBank


@pytest.mark.parametrize('kind',['PNC','PNT'])
@pytest.mark.parametrize('change',['scale','phase_scale','background','both'])
def test_full_native_bank_rebinds_to_changed_scale_and_background(tmp_path,kind,change):
    original,source=native_source(tmp_path/'original',kind)
    original_context=affine_context(source)
    path=tmp_path/'starts.npz';receipt=build(source,path,policy='joint_cell_v1',reuse_fixed_cell_profiles=True)
    assert receipt['affine_nuisance']==original_context
    project=G.G2Project(gpxfile=str(original));hist=project.histograms()[0]
    if change in {'scale','both'}:hist.data['Sample Parameters']['Scale'][0]*=1.7
    if change in {'phase_scale','both'}:project.phase(0).data['Histograms'][hist.name]['Scale'][0]*=.73
    if change in {'background','both'}:hist.data['Background'][0][3:]=[-2.,.9,-.5]
    target=tmp_path/'changed.gpx';project.save(str(target))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    prepare(target,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],tmp_path/'prepared')
    current=make_source(tmp_path/'prepared',tmp_path/'evaluation',SearchSettings())
    current_context=affine_context(current)
    (tmp_path/'contexts.json').write_text(json.dumps(dict(original=original_context,current=current_context),indent=2))
    assert current_context['shape_sha256']==original_context['shape_sha256']
    bank=OfflineNativeStartBank(current,path)
    assert bank.nuisance_binding is not None
    assert len(bank.points)==receipt['native_curves'] and receipt['proposed_points']==185
    errors=[]
    for point,curve in zip(bank.points,bank.curves):
        native=current.native(point*current.radii)
        errors.append(float(np.linalg.norm(native-curve)/max(np.linalg.norm(native),1e-30)))
    assert max(errors)<1e-11
    (tmp_path/'rebound-native-errors.json').write_text(json.dumps(dict(kind=kind,change=change,
        checked_curves=len(errors),maximum_relative_error=max(errors),binding=bank.nuisance_binding),indent=2))
