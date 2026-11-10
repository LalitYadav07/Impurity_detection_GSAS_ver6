"""Real native admission/rebase with synthetic structure and authored weights.

Random weights exercise complete model loading, not scientific ranking quality.
No refinement, network request or experimental observation is used.
"""
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
pytest.importorskip('torch')
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.isodistort_native import modes
from gpx_recipe.preparation.structural_domain import envelope
from gpx_recipe.preparation.structural_ranking_native import observed_features
from radar_pd_nova.model_admission import execute, validate_receipt
from test_native_profile_capture import native_source
from test_structural_attention_portable_native import artifact


@pytest.fixture(scope='module')
def control(tmp_path_factory):
    root=tmp_path_factory.mktemp('native-library-admission')
    source,_=native_source(root/'native','PNT')
    project=G.G2Project(gpxfile=str(source));parent=root/'parent.cif'
    project.phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    inventory=modes(str(child),str(root/'mode-check.gpx'))
    _,_,_,_,arrays,metadata=artifact(root/'model')
    q=observed_features(source,project.histogram(0))['q']
    arrays['q_edges']=np.linspace(q[2],q[-3],129)
    np.savez_compressed(root/'model/weights.npz',**arrays)
    for candidate in metadata['candidates']:
        candidate.update(modes=inventory['modes'],transformation_matrix=[[-1,0,0],[0,-1,0],[0,0,1]])
        candidate['child']['sha256']=file_sha(child)
    metadata.update(parent_sha256=file_sha(parent),instrument_sha256=file_sha(root/'native/profile.instprm'),
        weights_sha256=file_sha(root/'model/weights.npz'),training_domain=envelope(.001,parent_references=True))
    (root/'model/model.json').write_text(json.dumps(metadata))
    return dict(source=str(source),parent=str(parent),instrument=str(root/'native/profile.instprm'),
        model=str(root/'model/model.json'),weights=str(root/'model/weights.npz'),children=[str(child),str(child)],
        phase=project.phase(0).name,histogram=project.histogram(0).name,k=['0','0','0'],
        grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),library_id='a'*24)


def request(control,tmp_path,change=None):
    payload=copy.deepcopy(control)
    # Identical child content may occur under distinct, captured candidate paths.
    for index,original in enumerate(payload['children']):
        target=tmp_path/f'child-{index}.cif';shutil.copyfile(original,target);payload['children'][index]=str(target)
    source=tmp_path/'a-new-filename.gpx';shutil.copyfile(payload['source'],source)
    payload['source']=str(source);payload['destination']=str(tmp_path/'admitted')
    project=G.G2Project(gpxfile=str(source));phase=project.phase(0);hist=project.histogram(0)
    if change=='cell':phase.data['General']['Cell'][1]*=1.0001
    elif change=='outside':phase.data['General']['Cell'][1]*=1.1
    elif change=='profile':hist.InstrumentParameters['X'][1]+=.1
    elif change=='bank_label':hist.InstrumentParameters['Bank']=[3.,3.,False]
    elif change=='bank_profile':
        hist.InstrumentParameters['Bank']=[3.,3.,False];hist.InstrumentParameters['difC'][1]+=.1
    elif change=='lookup':hist.data['Instrument Parameters'][1]['Pdabc']={'d':[.1,3.],'sig':[.1,.2]}
    elif change=='atom':phase.data['Atoms'][0][phase.data['General']['AtomPtrs'][0]+1]+=.001
    elif change=='sample':hist.data['Sample Parameters']['Absorption'][0]+=.1
    elif change=='hap':phase.data['Histograms'][hist.name]['Mustrain'][1][0]+=.1
    elif change=='isotope':phase.data['General'].setdefault('Isotope',{})['Er']='unprepared isotope'
    elif change=='adp':phase.data['Atoms'][0][phase.data['General']['AtomPtrs'][3]]='A'
    elif change=='constraint':project.data['Constraints']['data']['Hist']=[['unprepared mapping']]
    elif change=='k':payload['k']=['0','0','1/2']
    elif change=='bytes':Path(payload['children'][0]).write_text('damaged child')
    if change=='name':
        old=phase.name;new='Freshly named phase'
        project.data['Phases'][new]=project.data['Phases'].pop(old)
        for row in project.names:
            if row and row[0]=='Phases':row[row.index(old)]=new
        hist.data.get('Reflection Lists',{}).pop(old,None)
        payload['phase']=new
    project.save()
    return payload


@pytest.mark.parametrize('change',[None,'cell','name','bank_label'])
def test_admission_accepts_new_gpx_identity_name_and_covered_cell(control,tmp_path,change):
    payload=request(control,tmp_path,change);before=file_sha(payload['source'])
    result=execute(payload);validate_receipt(result,payload)
    assert result['compatible'] and result['no_refinement'] and result['inputs_unchanged']
    assert file_sha(payload['source'])==before
    assert result['identity']['files']['source']!=file_sha(control['source']) or change is None
    assert len(result['candidate_checks'])==2


@pytest.mark.parametrize('change',['outside','profile','bank_profile','lookup','atom','sample','hap','isotope','adp','constraint','k','bytes'])
def test_admission_rejects_incompatible_context_and_tampered_child(control,tmp_path,change):
    payload=request(control,tmp_path,change);result=execute(payload)
    assert not result['compatible'] and result['no_refinement'] and result['reason']
    with pytest.raises(ValueError,match='incompatible'):validate_receipt(result,payload)


def test_stored_child_rebases_to_current_cell_before_zero_control(control,tmp_path):
    from gpx_recipe.structural_live.native import prepare
    payload=request(control,tmp_path,'cell')
    before=file_sha(payload['source']);child_before=file_sha(payload['children'][0])
    result=execute(payload);validate_receipt(result,payload)
    metadata=prepare(payload['source'],payload['children'][0],[[-1,0,0],[0,-1,0],[0,0,1]],
        [0,0,0],tmp_path/'rebased',phase_name=payload['phase'],histogram_name=payload['histogram'])
    assert metadata['zero_profile_error']<=1e-5
    original=G.G2Project(gpxfile=payload['source']).phase(0).data['General']['Cell'][1:7]
    child=G.G2Project(gpxfile=str(tmp_path/'rebased/prepared.gpx')).phase(0).data['General']['Cell'][1:7]
    np.testing.assert_allclose(child,original,rtol=0,atol=1e-8)
    assert file_sha(payload['source'])==before and file_sha(payload['children'][0])==child_before


def test_outer_admission_worker_exports_only_receipt_and_preserves_source(control,tmp_path):
    import zipfile
    from gpx_recipe.symmetry.bundle import pack
    from gpx_recipe.symmetry.compiler import package_signature
    from ndip_model_campaign import work
    payload=request(control,tmp_path)
    packet=tmp_path/'packet';packet.mkdir();relative={}
    for key in ('source','parent','instrument','model','weights'):
        target=packet/Path(payload[key]).name;shutil.copyfile(payload[key],target)
        relative[key]=target.name
    relative['children']=[]
    for path in payload['children']:
        target=packet/Path(path).name;shutil.copyfile(path,target);relative['children'].append(target.name)
    relative.update({k:payload[k] for k in ('phase','histogram','k','grid','library_id')},destination='admitted')
    hashes={p.name:file_sha(p) for p in packet.iterdir()}
    archive=tmp_path/'inputs.zip';pack(packet,archive,list(hashes))
    request_path=tmp_path/'request.json';request_path.write_text(json.dumps(dict(
        schema='radar-model-native/v1',operation='structural-library-admission',payload=relative,
        implementation_sha256=package_signature(),expected_source_sha256=hashes[relative['source']],
        dispatch_key='b'*24,timeout=300,input_hashes=hashes)))
    source_before=file_sha(payload['source']);result_zip=tmp_path/'result.zip'
    result=work('admission',payload['source'],archive,request_path,tmp_path/'worker',result_zip)
    assert result['result']['compatible'] and result['result']['no_refinement']
    assert file_sha(payload['source'])==source_before
    with zipfile.ZipFile(result_zip) as bundle:
        assert not any(name.endswith(('.gpx','.npz','.cif')) for name in bundle.namelist())
        assert set(bundle.namelist())=={'native/native-response.json','response.json'}


def test_fixed_reference_model_requires_the_current_before_scan(control,tmp_path):
    from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
    payload=request(control,tmp_path)
    metadata=json.loads(Path(payload['model']).read_text());metadata['reference_kind']=FIXED_REFERENCE_KIND
    model=tmp_path/'fixed-model.json';model.write_text(json.dumps(metadata));payload['model']=str(model)
    result=execute(payload)
    assert not result['compatible'] and 'before-transition reference' in result['reason']


def test_reference_profile_is_checked_independently(control,tmp_path):
    payload=request(control,tmp_path)
    before=tmp_path/'before.gpx';shutil.copyfile(payload['source'],before)
    project=G.G2Project(gpxfile=str(before));project.histogram(0).InstrumentParameters['X'][1]+=.1;project.save()
    payload['reference']=str(before)
    result=execute(payload)
    assert not result['compatible'] and 'profile' in result['reason']


@pytest.mark.parametrize('ambiguous',[False,True])
def test_reference_reader_labels_do_not_select_between_identical_profiles(control,tmp_path,ambiguous):
    payload=request(control,tmp_path)
    before=tmp_path/'before.gpx';shutil.copyfile(payload['source'],before)
    project=G.G2Project(gpxfile=str(before));hist=project.histogram(0)
    hist.InstrumentParameters['Bank']=[3.,3.,False]
    if ambiguous:
        project.add_simulated_powder_histogram('same profile with another reader label',payload['instrument'],
            3.,40.,Npoints=1501,scale=120.,phases=[project.phase(0)])
    project.save();payload['reference']=str(before)
    original=file_sha(before);result=execute(payload)
    assert result['compatible'] is (not ambiguous)
    if ambiguous:assert 'unique compatible native bank' in result['reason']
    assert file_sha(before)==original


def test_no_selected_k_candidates_reports_all_exclusions(control,tmp_path):
    payload=request(control,tmp_path,'k');result=execute(payload)
    assert not result['compatible'] and result['k_verified'] and result['mode_inventory_verified']
    assert result['eligible_candidate_ids']==[] and len(result['excluded_candidate_ids'])==2
    assert all(row['eligibility']=='excluded' and 'Gamma-only' in row['eligibility_reason']
               for row in result['candidate_checks'])


def test_out_of_scope_child_still_gets_full_integrity_validation(control,tmp_path):
    payload=request(control,tmp_path,'bytes');payload['k']=['0','0','1/2']
    result=execute(payload)
    assert not result['compatible'] and not result['mode_inventory_verified']
    assert 'differs from its exact trained candidate' in result['reason']
