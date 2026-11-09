"""Native admission and observed-feature fidelity; no classifier-quality claim."""
import json
from pathlib import Path
import shutil
import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from gpx_recipe.preparation.reference import create_reference
from gpx_recipe.preparation.structural_ranking_native import require_compatible,observed_features
from gpx_recipe.symmetry.compiler import file_sha
from test_native_profile_capture import native_source


def pair(root,kind):
    root.mkdir()
    source,_=native_source(root/'input',kind)
    project=G.G2Project(gpxfile=str(source));cif=root/'parent.cif'
    project.phase(0).export_CIF(str(cif))
    reference=create_reference(cif,root/'input/profile.instprm',root/'reference',kind=kind,
        lower=12. if kind=='PNC' else 3000.,upper=125. if kind=='PNC' else 40000.,bins=701,bank=1)
    other=G.G2Project(gpxfile=reference['path'])
    return source,project,other


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_native_features_recompute_background_and_preserve_real_masks(tmp_path,kind):
    source,p,_=pair(tmp_path/'control',kind);h=p.histogram(0)
    arrays=h.data['data'][1];x=np.asarray(arrays[0]).copy()
    h.data['Limits'][1]=[float(x[20]),float(x[-20])]
    h.data['Limits'].append([float(x[100]),float(x[110])])
    arrays[2][200]=0.
    arrays[1]=np.ma.array(arrays[1],mask=np.zeros(len(x),bool));arrays[1].mask[300]=True
    # Saved calculated/background arrays deliberately do not carry the actual
    # coefficients. Inference must evaluate the native background afresh.
    arrays[4][:]=-9876.
    p.save();before=file_sha(source)
    feature=observed_features(source,h)
    assert file_sha(source)==before
    assert np.all(np.diff(feature['q'])>0) and np.isfinite(feature['background']).all()
    assert not np.any(feature['background']==-9876.)
    expected=((x>=x[20])&(x<=x[-20])&~((x>=x[100])&(x<=x[110])))
    expected[200]=False;expected[300]=False
    selected=np.flatnonzero((x>=x[20])&(x<=x[-20]))
    if kind=='PNT':selected=selected[::-1]
    np.testing.assert_array_equal(feature['valid'],expected[selected])
    # Independent full native calculations with/without background agree with
    # the cheap background-only extraction on the exact observed fit interval.
    original=np.asarray(MAIN.DoNoFit(str(source),h.name),float)
    h.data['Background'][0][3:]=[0.]*h.data['Background'][0][2]
    zero=tmp_path/'zero-background.gpx';p.save(str(zero))
    no_background=np.asarray(MAIN.DoNoFit(str(zero),h.name),float)
    np.testing.assert_allclose(feature['background'],(original-no_background)[selected],rtol=1e-9,atol=1e-7)


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_three_curve_parent_is_fresh_native_profile_on_actual_acquisition_grid(tmp_path,kind):
    source,p,_=pair(tmp_path/'control',kind);h=p.histogram(0)
    data=h.data['data'][1];data[3][:]=-123456.;data[4][:]=-98765.
    # A covered thermal cell change must be present in the reference, not an
    # old nominal-cell curve resampled from the preparation library.
    p.phase(0).data['General']['Cell'][1]*=1.0001
    p.save();before=file_sha(source);observed=np.asarray(data[1]).copy()
    feature=observed_features(source,h,include_reference=True);ref=feature['reference']
    assert file_sha(source)==before
    np.testing.assert_array_equal(data[1],observed)
    np.testing.assert_array_equal(ref['q'],feature['q'])
    np.testing.assert_array_equal(ref['valid'],feature['valid'])
    assert np.all(ref['intensity']!=-123456.) and np.all(ref['background']!=-98765.)
    original=np.asarray(MAIN.DoNoFit(str(source),h.name),float)
    x=np.asarray(data[0]);limits=h.data['Limits'][1]
    selected=np.flatnonzero((x>=limits[0])&(x<=limits[1]))
    if kind=='PNT':selected=selected[::-1]
    np.testing.assert_allclose(ref['intensity'],original[selected],rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('change',['none','cell_within','cell_outside','profile','position','adp','occupancy','sample','sample_contrast','legacy_contrast_zero','isotope'])
def test_only_covered_native_contexts_enter_the_prepared_ranker(tmp_path,change):
    _,p,r=pair(tmp_path/'control','PNT');parent=p.phase(0);h=p.histogram(0)
    cx,_,_,cia=parent.data['General']['AtomPtrs']
    if change=='cell_within':parent.data['General']['Cell'][1]*=1.0001
    elif change=='cell_outside':parent.data['General']['Cell'][1]*=1.1
    elif change=='profile':h.InstrumentParameters['X'][1]+=.1
    elif change=='position':parent.data['Atoms'][0][cx+1]+=.001
    elif change=='adp':parent.data['Atoms'][0][cia+1]+=.001
    elif change=='occupancy':parent.data['Atoms'][0][cx+3]=.9
    elif change=='sample':h.data['Sample Parameters']['Absorption'][0]+=.1
    elif change=='sample_contrast':h.data['Sample Parameters']['Contrast'][1]=.1
    elif change=='legacy_contrast_zero':h.data['Sample Parameters']['Contrast'][1]=False
    elif change=='isotope':parent.data['General'].setdefault('Isotope',{})['Er']='unprepared isotope selection'
    if change in {'none','cell_within','legacy_contrast_zero'}:
        distance=require_compatible(parent,h,r.phase(0),r.histogram(0),maximum_metric=.001)
        assert 0<=distance<=.001
    else:
        with pytest.raises(ValueError):require_compatible(parent,h,r.phase(0),r.histogram(0),maximum_metric=.001)


@pytest.mark.parametrize('bound',[float('nan'),float('inf'),-.001,True,'0.001'])
def test_invalid_training_bound_cannot_bypass_native_admission(tmp_path,bound):
    _,p,r=pair(tmp_path/'control','PNT')
    # A NaN bound used to make the outside-envelope comparison false.
    p.phase(0).data['General']['Cell'][1]*=1.1
    with pytest.raises(ValueError,match='finite nonnegative training-only'):
        require_compatible(p.phase(0),p.histogram(0),r.phase(0),r.histogram(0),maximum_metric=bound)


def test_outside_cell_message_explains_current_and_trained_ranges(tmp_path):
    _,p,r=pair(tmp_path/'control','PNT')
    p.phase(0).data['General']['Cell'][1]*=1.001
    with pytest.raises(ValueError,match='current change .*trained maximum 0.001') as caught:
        require_compatible(p.phase(0),p.histogram(0),r.phase(0),r.histogram(0),maximum_metric=.001)
    assert 'Generate training examples over the wider parent-cell range' in str(caught.value)


@pytest.mark.parametrize('change',['same','profile','no_model','constraint','paired_parent_within','paired_parent_outside','paired_legacy_domain','live_counts','paired_adaptive','live_adaptive','paired_bad_adaptive','paired_recover_adaptive',
    'fixed_before','fixed_missing','fixed_profile','fixed_positions','fixed_constraint','live_fixed_before',
    'fixed_renamed','fixed_ambiguous'])
def test_actual_gpx_reaches_source_bound_receipt_through_native_admission(tmp_path,monkeypatch,change):
    """Inference is a controlled stub here; native input admission is real."""
    from gpx_recipe.preparation import structural_hints,structural_ranking_hints as hints
    from gpx_recipe.preparation.structural_domain import envelope
    from gpx_recipe.preparation.structural_ranking_native import rank_gpx
    from gpx_recipe.symmetry.compiler import package_signature
    from gpx_recipe.symmetry.contracts import validate_protocol,validate_candidate
    from gpx_recipe.symmetry.native_audit import inspect_source
    from gpx_recipe.execution import write_json
    source,p,_=pair(tmp_path/'control','PNT');root=source.parent
    if change=='profile':p.histogram(0).InstrumentParameters['X'][1]+=.1;p.save()
    elif change=='constraint':p.data['Constraints']['data']['Hist']=[['opaque unprepared mapping']];p.save()
    elif change.startswith('paired_'):
        p.phase(0).data['General']['Cell'][1]*=1.01 if change=='paired_parent_outside' else 1.0001
        p.save()
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    shutil.copyfile(child,root/'child.cif')
    candidate=validate_candidate(dict(id='public-ersi',modes=['fixture-admission-only'],
        child=dict(path='child.cif',sha256=file_sha(root/'child.cif')),
        transformation_matrix=[[-1,0,0],[0,-1,0],[0,0,1]],origin_shift=[0,0,0]))
    context=inspect_source(source);parent=context['phases'][0]
    write_json(root/'catalog.json',dict(schema='radar-symmetry-catalog/v1',
        parent_setting_sha256=parent['setting_sha256'],k=['0','0','0'],candidates=[candidate]))
    protocol=validate_protocol(dict(kind='structural',source=dict(path='source.gpx',sha256=file_sha(source)),
        catalog=dict(path='catalog.json',sha256=file_sha(root/'catalog.json')),parent_phase=parent['name'],
        histogram=p.histogram(0).name,parent_setting_sha256=parent['setting_sha256'],k=['0','0','0'],
        implementation_sha256=package_signature(),magnetic_function_sha256=context['runtime']['magnetic_function_sha256'],
        candidates=[candidate],seeds=[dict(id='admission-control')],sites=[],policy=dict(cycles=3,timeout_seconds=60,
            maximum_trials=1,maximum_cell_multiplier=8,maximum_moment=6.,fit_band_percent=.25,
            maximum_mode=.5,adequate_rwp_percent=12.)))
    fixed=change.startswith('fixed_') or change=='live_fixed_before'
    if fixed:
        from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
        protocol['policy'].update(structural_optimizer='prepared_nonlinear',refine_cell=True,
            refine_background=True,refine_histogram_scale=True)
        if change!='fixed_missing':
            before=root/'before.gpx'
            before_source=source
            if change=='fixed_renamed':
                before_source,_=native_source(tmp_path/'independent-before','PNT',histogram_name='before run 64146')
            before_project=G.G2Project(gpxfile=str(before_source),newgpx=str(before))
            bh=before_project.histogram(0);ba=bh.data['data'][1]
            if change=='fixed_ambiguous':
                before_project.add_simulated_powder_histogram('another compatible reference bank',str(root/'profile.instprm'),
                    3.,40.,Npoints=351,phases=[before_project.phase(protocol['parent_phase'])])
            # Deliberately distinct from the target. Stale calculated arrays
            # cannot substitute for this reference's observed values or mask.
            ba[1][50:60]=-17.;ba[2][51:54]=0.;ba[3][:]=-98765.
            if change=='fixed_profile':bh.InstrumentParameters['X'][1]+=.1
            elif change=='fixed_positions':
                bp=before_project.phase(protocol['parent_phase']);cx=bp.data['General']['AtomPtrs'][0]
                bp.data['Atoms'][0][cx]+=.001
            elif change=='fixed_constraint':before_project.data['Constraints']['data']['Hist']=[['opaque unprepared mapping']]
            before_project.save()
            protocol['structural_reference']=dict(path='before.gpx',sha256=file_sha(before))
        protocol=validate_protocol(protocol)
    metadata=dict(training_domain=envelope(.001))
    paired=change.startswith('paired_') or change in {'live_counts','live_adaptive'} or fixed
    if paired:
        from gpx_recipe.preparation.structural_attention_weights import SCHEMA
        metadata['schema']=SCHEMA
        metadata['training_domain']=envelope(.001,parent_references=change!='paired_legacy_domain')
        if fixed:metadata['reference_kind']=FIXED_REFERENCE_KIND
    model=tmp_path/'model.json';write_json(model,metadata)
    row=dict(entry='d'*24,bank='bank 1',grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1),
        files=dict(parent=dict(sha256=file_sha(tmp_path/'control/parent.cif')),
                   instrument=dict(sha256=file_sha(root/'profile.instprm'))))
    document=dict(snapshot_sha256='e'*64,ranking_models=dict(entries=[] if change=='no_model' else [row]))
    if change=='paired_recover_adaptive':document['ranking_models']['entries'].append(dict(row,entry='f'*24))
    monkeypatch.setattr(structural_hints,'read',lambda *a:document)
    monkeypatch.setattr(hints,'checked_files',lambda *a:dict(parent=tmp_path/'control/parent.cif',
        instrument=root/'profile.instprm',model=model))
    calls=[]
    observation=None;override=None;before_source=file_sha(source)
    if change in {'live_counts','live_adaptive','live_fixed_before'}:
        override=tmp_path/'current-ranking.gpx'
        live=G.G2Project(gpxfile=str(source),newgpx=str(override))
        arrays=live.histogram(0).data['data'][1]
        arrays[1][30:40]=-7.;arrays[2][31:34]=0.
        live.save()
        observation=dict(snapshot_sha256='0'*64,calculation_source_sha256=file_sha(override),
            acquisition_sha256='1'*64,revision=3)
        expected=observed_features(override,live.histogram(0),include_reference=not fixed)
    def inference(*args,**kwargs):
        calls.append(kwargs)
        assert ('reference' in kwargs)==paired
        if change in {'live_counts','live_adaptive','live_fixed_before'}:
            for key in ('q','intensity','valid','background'):
                np.testing.assert_array_equal(kwargs[key],expected[key])
            assert np.count_nonzero(kwargs['intensity']<0)>0 and np.count_nonzero(~kwargs['valid'])>=3
        if fixed:
            ref=kwargs['reference'];expected_before=observed_features(before,bh)
            for key in ('q','intensity','valid','background'):
                np.testing.assert_array_equal(ref[key],expected_before[key])
            assert ref['kind']=='measured_before_transition'
            assert ref['source_sha256']==file_sha(before)
            assert ref['parent_sha256']==row['files']['parent']['sha256']
            assert ref['instrument_sha256']==row['files']['instrument']['sha256']
            assert np.count_nonzero(ref['intensity']<0)>=10
            assert np.count_nonzero(~ref['valid'])>=3
        result=dict(ranking_available=True,candidate_order=['public-ersi'],
            families=[dict(space_group=63,model_score=1.,candidate_ids=['public-ersi'])],
            support=dict(status='within_empirical_synthetic_support',nearest_training_rms=.1,
                         validation_maximum_rms=.2,statistical_coverage_qualified=False))
        if 'adaptive' in change:
            from test_structural_ranking_result import work_set
            bad=(change=='paired_bad_adaptive' or change=='paired_recover_adaptive' and kwargs['entry_id']=='d'*24)
            result['adaptive_work_set']=work_set([166] if bad else [63])
        return result
    monkeypatch.setattr(hints,'rank',inference)
    destination=tmp_path/'ranking';result=rank_gpx(root,protocol,destination,source_override=override,observation=observation)
    assert result==json.loads((destination/'ranking.json').read_text())
    available=change in {'same','paired_parent_within','live_counts','paired_adaptive','live_adaptive','paired_recover_adaptive',
        'fixed_before','live_fixed_before','fixed_renamed'}
    assert result['ranking_available']==available
    assert len(calls)==(2 if change=='paired_recover_adaptive' else int(available or change=='paired_bad_adaptive'))
    if change=='paired_recover_adaptive':
        assert [r['status'] for r in result['checks']]==['incompatible','compatible']
        assert result['entry_id']=='f'*24
    assert ('adaptive_work_set' in result)==(available and 'adaptive' in change)
    assert result['candidate_order']==['public-ersi'] and result['candidate_pruning_authorized'] is False
    assert result['preparation_snapshot_sha256']=='e'*64
    assert file_sha(source)==before_source
    if change=='fixed_missing':assert 'before-transition reference GPX' in result['checks'][0]['reason']
    if change=='fixed_ambiguous':assert 'multiple compatible histograms' in result['checks'][0]['reason']
    if fixed and change!='fixed_missing':assert file_sha(before)==protocol['structural_reference']['sha256']
    if observation is not None:
        from gpx_recipe.preparation.structural_ranking_result import validate
        validate(result,protocol,observation=observation)
        with pytest.raises(ValueError):validate(result,protocol)
