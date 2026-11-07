"""Portable native kernels follow actual cells without freezing a fit domain."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation import kernel_geometry as geometry
from gpx_recipe.preparation import native_tof_kernels as kernels
from gpx_recipe.preparation.kernel_store import KernelStore
from gpx_recipe.structural_live.native import prepare
from gpx_recipe.structural_live.service import make_source,SearchSettings
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward
from gpx_recipe.symmetry.compiler import file_sha


def bank(root,source):
    points=[np.zeros(len(source.radii)),np.linspace(-.06,.08,len(source.radii))*source.radii]
    return geometry.prepare(source,points+[points[0]],root,model_sha256='a'*64,bank='PNT bank 1')


def shifted_source(root):
    original,_=native_source(root/'initial','PNT')
    path=root/'new-scan.gpx';project=G.G2Project(gpxfile=str(original),newgpx=str(path))
    phase=project.phases()[0]
    phase.data['General']['Cell'][1]*=1.00002
    from GSASII import GSASIIlattice as lattice
    phase.data['General']['Cell'][7]=float(lattice.calc_V(lattice.cell2A(phase.data['General']['Cell'][1:7])))
    project.save()
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    prepare(path,child,[[-1.,0,0],[0,-1.,0],[0,0,1.]],[0,0,0],root/'prepared')
    return path,make_source(root/'prepared',root/'evaluation',SearchSettings())


def test_renderer_consumes_a_resumable_absolute_cell_store_and_current_intensities(tmp_path):
    original,s=native_source(tmp_path/'inputs','PNT');before=file_sha(original)
    store,report=bank(tmp_path/'store',s)
    assert report['proposal_to_cell']==[0,1,0] and store.coverage()['stored']==2
    assert report['observations_included'] is False and report['scientific_result'] is False
    _,resumed=bank(store.root,s)
    assert resumed['prepared_cells']==[] and resumed['requalified_cells']==[0,1]
    reader=geometry.PreparedKernelBank(store,s);engine=NativeTOFForward(s)
    point=np.zeros(len(s.radii))
    for shift in [0.,.04,-.08]:
        point[:s.mode_dimensions]=np.linspace(-shift,shift,s.mode_dimensions)*s.radii[:s.mode_dimensions]
        expected=s.native(point).copy()
        np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
    assert not engine.disabled and engine.kernel_hits>0 and reader.hits>0
    assert reader.diagnostics()['resident_bytes']<=reader.maximum_bytes
    assert engine.diagnostics()['prepared_kernel_bank']['loads']==1
    assert file_sha(original)==before


def test_changed_nuclear_cell_uses_the_same_absolute_metric_with_fresh_atomic_intensities(tmp_path):
    _,first=native_source(tmp_path/'before','PNT');store,_=bank(tmp_path/'store',first)
    old_a=np.array(store.document['specification']['geometry']['cells'][0])
    current=tmp_path/'current';current.mkdir();original,s=shifted_source(current)
    assert not np.array_equal(s.fast.basis['baseline_a'],old_a)
    point=geometry.point_for_cell(s,old_a)
    assert np.any(point[s.mode_dimensions:]) and np.all(abs(point)<=s.radii)
    reader=geometry.PreparedKernelBank(store,s);engine=NativeTOFForward(s)
    previous=None
    for adp in [.01,.025]:
        s.fast.native.parms[f'{s.fast.slow.phase.id}::AUiso:0']=adp
        expected=s.native(point).copy()
        if previous is not None:assert not np.array_equal(previous,expected)
        np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
        previous=expected
    assert not engine.disabled and engine.kernel_hits>0
    assert reader.loads==1 and reader.hits>0


def test_actual_xyz_edit_rebuilds_kernels_without_rejecting_the_scientific_model(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');store,_=bank(tmp_path/'store',s)
    reader=geometry.PreparedKernelBank(store,s);engine=NativeTOFForward(s);point=np.zeros(len(s.radii))
    engine.evaluate(point,kernel=reader);prior=engine.peak_calls
    s.fast.native.parms[f':{s.fast.slow.hist.id}:X']+=.04
    expected=s.native(point).copy()
    np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
    assert not engine.disabled and engine.peak_calls>prior
    assert reader.bypasses['Current instrument/profile needs original native kernels']==1
    s.fast.native.parms[f':{s.fast.slow.hist.id}:X']-=.04
    # Returning to a compatible point does not reload its resident kernel.
    engine.evaluate(point,kernel=reader)
    assert reader.loads==1 and reader.hits>0


@pytest.mark.parametrize('condition',sorted(geometry.ACQUISITION_PARAMETERS))
def test_recorded_acquisition_condition_does_not_invalidate_raw_peak_physics(tmp_path,condition):
    _,s=native_source(tmp_path/'inputs','PNT');store,_=bank(tmp_path/'store',s)
    reader=geometry.PreparedKernelBank(store,s);engine=NativeTOFForward(s);point=np.zeros(len(s.radii))
    original=s.native(point).copy()
    key=f':{s.fast.slow.hist.id}:'+condition
    s.fast.native.parms[key]=float(s.fast.native.parms.get(key,0.))+117.5
    expected=s.native(point).copy()
    # These are recorded conditions; the fixed atomic/profile model has not
    # been transformed into a temperature/pressure-dependent constitutive law.
    np.testing.assert_array_equal(expected,original)
    np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
    assert not engine.disabled and engine.kernel_hits>0 and reader.loads==1
    assert not reader.bypasses


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_structural_forward_and_affine_identity_keep_recorded_conditions_separate(tmp_path,kind):
    from gpx_recipe.preparation.identity import ACQUISITION_PARAMETERS
    from gpx_recipe.preparation.structural_identity import structural_physics_identity
    from gpx_recipe.preparation.structural_nuisance import affine_context
    _,s=native_source(tmp_path/'inputs',kind);point=np.zeros(len(s.radii))
    expected=s.native(point).copy();before=structural_physics_identity(s);affine=affine_context(s)
    n=s.fast.native;histogram=n.histo_phases[0][s.fast.slow.hist.name]
    hfx=f':{s.fast.slow.hist.id}:'
    for index,name in enumerate(sorted(ACQUISITION_PARAMETERS)):
        n.parms[hfx+name]=118.+index;histogram['Sample Parameters'][name]=118.+index
    np.testing.assert_array_equal(s.native(point),expected)
    assert structural_physics_identity(s)==before and affine_context(s)==affine
    assert histogram['Sample Parameters']['Temperature']==n.parms[hfx+'Temperature']


@pytest.mark.parametrize('failure',['missing','corrupt','budget','setting'])
def test_unusable_optional_data_keeps_the_complete_original_native_calculation(tmp_path,failure):
    _,s=native_source(tmp_path/'inputs','PNT');store,_=bank(tmp_path/'store',s)
    if failure=='missing':
        spec=store.document['specification'];store=KernelStore.create(tmp_path/'empty',spec)
    elif failure=='corrupt':
        path=next((store.root/'chunks/0000').glob('*.npz'));path.write_bytes(b'changed')
    elif failure=='setting':
        spec=json.loads(json.dumps(store.document['specification']));spec['geometry']['setting_sha256']='f'*64
        changed=KernelStore.create(tmp_path/'other-setting',spec);changed.put(0,store.get(0));store=changed
    reader=geometry.PreparedKernelBank(store,s,maximum_bytes=1 if failure=='budget' else geometry.MAX_BYTES)
    engine=NativeTOFForward(s);point=np.zeros(len(s.radii));expected=s.native(point).copy()
    np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
    assert not engine.disabled and engine.peak_calls>0
    assert reader.diagnostics()['bypasses'] and reader.loads==0


def test_cell_mapping_does_not_clip_expand_or_freeze_the_reviewed_domain(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');point=np.zeros(len(s.radii))
    baseline=geometry.cell_for_point(s,point)
    matrix=s.fast.slow.a_scale*s.fast.basis['cell_basis']
    outside=baseline+matrix@np.full(s.cell_dimensions,20*s.radii[-1])
    with pytest.raises(geometry.GeometryUnavailable,match='reviewed bounds'):
        geometry.point_for_cell(s,outside)
    modes=point.copy();modes[0]=.02*s.radii[0]
    mapped=geometry.point_for_cell(s,baseline,mode_point=modes)
    np.testing.assert_array_equal(mapped[:s.mode_dimensions],modes[:s.mode_dimensions])


def test_cancellation_on_a_resident_hit_and_replaced_native_owner_remain_errors(tmp_path):
    _,s=native_source(tmp_path/'inputs','PNT');store,_=bank(tmp_path/'store',s)
    cancelled=[False]
    def check():
        if cancelled[0]:raise InterruptedError('cancel')
    reader=geometry.PreparedKernelBank(store,s,check_cancel=check);point=np.zeros(len(s.radii))
    assert reader.resolve(point) is not None
    cancelled[0]=True
    with pytest.raises(InterruptedError):reader.resolve(point)
    cancelled[0]=False
    native_source(tmp_path/'replaced','PNT')
    with pytest.raises(RuntimeError,match='replaced'):reader.resolve(point)


PRODUCER='''import sys,json,numpy as np
from pathlib import Path
from test_native_profile_capture import native_source
from test_native_kernel_bank import bank
root=Path(sys.argv[1]);root.mkdir()
_,s=native_source(root/'inputs','PNT');store,report=bank(root/'store',s)
export=store.export_parts(root/'parts')
(root/'result.json').write_text(json.dumps(dict(report=report,export=export)))
'''
CONSUMER='''import sys,json,numpy as np
from pathlib import Path
from test_native_kernel_bank import shifted_source
from gpx_recipe.preparation.kernel_store import KernelStore
from gpx_recipe.preparation.kernel_geometry import PreparedKernelBank,point_for_cell
from gpx_recipe.structural_live.optimization.native_forward import NativeTOFForward
root=Path(sys.argv[2]);root.mkdir();producer=Path(sys.argv[1])
report=json.loads((producer/'result.json').read_text());store=None
for part in reversed(report['export']['parts']):store=KernelStore.import_part(producer/'parts'/part['path'],root/'restored')
current=root/'current';current.mkdir();_,s=shifted_source(current)
reader=PreparedKernelBank(store,s);point=point_for_cell(s,store.document['specification']['geometry']['cells'][0])
engine=NativeTOFForward(s)
for scale in [0.,.04]:
 point[:s.mode_dimensions]=np.linspace(-scale,scale,s.mode_dimensions)*s.radii[:s.mode_dimensions]
 expected=s.native(point).copy();np.testing.assert_array_equal(engine.evaluate(point,kernel=reader),expected)
assert not engine.disabled and engine.kernel_hits>0
assert reader.native_peak_checks>0 and reader.loads==1
(root/'result.json').write_text(json.dumps(dict(bank=reader.diagnostics(),forward=engine.diagnostics(),producer_exited=True)))
'''


def test_producer_exits_then_a_cold_worker_imports_and_adapts_to_a_new_nuclear_cell(tmp_path):
    producer=tmp_path/'producer';consumer=tmp_path/'consumer'
    with (tmp_path/'producer.log').open('w') as log:
        subprocess.run([sys.executable,'-c',PRODUCER,str(producer)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT)
    with (tmp_path/'consumer.log').open('w') as log:
        subprocess.run([sys.executable,'-c',CONSUMER,str(producer),str(consumer)],check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT)
    result=json.loads((consumer/'result.json').read_text())
    assert result['producer_exited'] and result['bank']['native_peak_checks']>0
    assert result['forward']['kernel_hits']>0
