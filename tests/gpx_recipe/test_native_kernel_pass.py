"""Reuse original kernel outputs from the mandatory independent native pass."""
import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
from test_native_profile_capture import native_source
from gpx_recipe.preparation.native_tof_kernels import recorder_for_source
from gpx_recipe.structural_live.optimization.captured_cell_derivative import CapturedCellDerivative
from gpx_recipe.structural_live.optimization.native_profile_capture import capture_rietveld_response


def record(source, recorder, point, maximum_bytes=64*1024*1024):
    n=source.fast.native;histogram=n.histo_phases[0][source.fast.slow.hist.name]
    return capture_rietveld_response(n.math.getPowderProfile,n.math.G2pwd,n.parms,n.histdict,
        np.asarray(source.fast.x[source.fast.slice],float),histogram,n.histo_phases[1],
        n.controls,n.pawley,histogram_name=source.fast.slow.hist.name,
        phase_name=source.fast.slow.phase.name,evaluate=lambda:source.native(point),
        maximum_bytes=maximum_bytes,observe_peak=recorder.observe)


def test_required_native_pass_produces_exact_kernels_without_another_peak_construction(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT');original=source.native;calls=[]
    def counted(point):
        calls.append(np.asarray(point).copy());return original(point)
    source.native=counted
    evaluator=CapturedCellDerivative(source,max_cells=2)
    p=np.zeros(len(source.radii));expected=original(p).copy()
    np.testing.assert_array_equal(evaluator.native_evaluate(p),expected)
    assert len(calls)==1
    entry=evaluator._entry(p[source.mode_dimensions:]);kernel=entry['kernel']
    proof=kernel.diagnostics()
    assert proof['method']=='already-required original native Rietveld pass'
    assert proof['native_peak_calls']==0 and proof['recorded_native_peak_calls']>0
    assert proof['verified_all_peaks'] and not proof['qualification_persisted']
    for scale in [.02,-.04,.07]:
        p[:source.mode_dimensions]=np.linspace(-scale,scale,source.mode_dimensions)*source.radii[:source.mode_dimensions]
        expected=original(p).copy()
        np.testing.assert_array_equal(evaluator.native_evaluate(p),expected)
    # The batched forward's one independent original anchor remains required.
    assert len(calls)==2
    assert evaluator.native_forward.peak_calls==0
    assert evaluator.native_forward.kernel_hits>0
    assert evaluator.numeric_bytes<=evaluator.maximum_bytes
    # The derivative decomposition still has its separate original-native check.
    evaluator.verify(p)
    assert len(calls)==3


def test_each_changed_cell_retains_its_own_primitive_context_and_bounded_cache(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT')
    evaluator=CapturedCellDerivative(source,max_cells=1)
    for shift in [0.,.02,-.03]:
        p=np.zeros(len(source.radii));p[source.mode_dimensions:]=shift*source.radii[source.mode_dimensions:]
        expected=source.native(p).copy()
        np.testing.assert_array_equal(evaluator.native_evaluate(p),expected)
        p[:source.mode_dimensions]=.01*source.radii[:source.mode_dimensions]
        expected=source.native(p).copy()
        np.testing.assert_array_equal(evaluator.native_evaluate(p),expected)
        assert len(evaluator.cache)==1
        assert evaluator.numeric_bytes<=evaluator.maximum_bytes
    assert evaluator.builds==3 and evaluator.kernel_native_evaluations==3


def test_recorder_memory_exhaustion_keeps_the_full_original_native_pass(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT');recorder=recorder_for_source(source)
    p=np.zeros(len(source.radii));expected=source.native(p).copy()
    original=recorder.observe
    recorder.observe=lambda h,params,raw,lo,hi,remaining:original(h,params,raw,lo,hi,1)
    _,_,_,actual=record(source,recorder,p)
    np.testing.assert_array_equal(actual,expected)
    assert recorder.reason and not recorder.records
    assert recorder.finish(p,maximum_bytes=1) is None


def test_changed_scalar_inputs_cannot_qualify_a_recorded_native_pass(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT');recorder=recorder_for_source(source)
    p=np.zeros(len(source.radii));record(source,recorder,p)
    source.fast.native.parms[f':{source.fast.slow.hist.id}:X']+=.1
    with pytest.raises(ValueError,match='arguments differ|inventory differs'):
        recorder.finish(p,maximum_bytes=64*1024*1024)


def test_equivalent_compiled_shape_inputs_need_no_false_float64_identity(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT');recorder=recorder_for_source(source)
    p=np.zeros(len(source.radii));record(source,recorder,p)
    for params,_,_,_ in recorder.records.values():
        params[1]=np.nextafter(params[1],np.inf)
    assert recorder.finish(p,maximum_bytes=64*1024*1024) is not None


def test_portable_format_limits_cannot_reject_an_otherwise_valid_native_pass(tmp_path,monkeypatch):
    from gpx_recipe.preparation import tof_kernel_arrays
    _,source=native_source(tmp_path/'inputs','PNT');recorder=recorder_for_source(source)
    p=np.zeros(len(source.radii));expected=source.native(p).copy()
    _,_,_,actual=record(source,recorder,p)
    np.testing.assert_array_equal(actual,expected)
    monkeypatch.setattr(tof_kernel_arrays,'MAX_REFLECTIONS',1)
    assert recorder.finish(p,maximum_bytes=64*1024*1024) is None
    assert 'admission domain' in recorder.reason and not recorder.records


def test_failed_recorder_restores_native_peak_functions_and_owned_capture(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNT');recorder=recorder_for_source(source)
    powder=source.fast.native.math.G2pwd;function=powder.getEpsVoigt
    def fail(*args):raise RuntimeError('induced recorder failure')
    recorder.observe=fail
    p=np.zeros(len(source.radii))
    with pytest.raises(RuntimeError,match='recorder failure'):record(source,recorder,p)
    assert powder.getEpsVoigt is function
    replacement=recorder_for_source(source);record(source,replacement,p)
    assert replacement.finish(p,maximum_bytes=64*1024*1024) is not None


def test_cw_response_stays_on_its_existing_native_engine(tmp_path):
    _,source=native_source(tmp_path/'inputs','PNC')
    assert recorder_for_source(source) is None
    evaluator=CapturedCellDerivative(source)
    p=np.zeros(len(source.radii));expected=source.native(p).copy()
    np.testing.assert_array_equal(evaluator.native_evaluate(p),expected)
    assert evaluator.native_forward is None
    assert 'kernel' not in evaluator._entry(p[source.mode_dimensions:])
