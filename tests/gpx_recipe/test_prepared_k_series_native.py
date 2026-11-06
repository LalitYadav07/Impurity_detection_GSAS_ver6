"""Prospective public numerical control: two k families and five new scans.

Nominal CIF/profile preparation is sealed before observations are generated.
Every family and declared start is fitted with and without portable factors;
each fitted prediction is checked against the ordinary native residual. This
is a synthetic search-integrity control, not an experimental benchmark or a
claim of global optimization, uncertainty calibration or NDIP latency.
"""
import copy
import json
from pathlib import Path
import time

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')
from test_preparation_theory_native import inputs
from gpx_recipe.execution import write_json
from gpx_recipe.native_forward import forward_profile
from gpx_recipe.preparation.consume import restore_magnetic_entry
from gpx_recipe.preparation.library import import_library
from gpx_recipe.preparation.operations import create_library,stage_entry,publish_entry
from gpx_recipe.preparation.worker import execute
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.native_magnetic import construct_theory
from gpx_recipe.symmetry.native_profiled import NativeMagneticStartingPoint
from gpx_recipe.symmetry.worker import Artifacts


def fit_portfolio(path,cache):
    began=time.monotonic()
    native=NativeMagneticStartingPoint(path,refine_scale=False,refine_background=False,maximum_moment=6.)
    model,compilation=native.build(factor_cache=cache)
    seeds=[np.eye(model.dimension)[i]*amplitude for amplitude in (.4,1.8) for i in range(model.dimension)]
    rows=[]
    for index,seed in enumerate(seeds):
        fit=model.fit(seed,max_iterations=250)
        residual=native.residual(fit.coefficients,fit.scale,fit.background)
        error=model.heldout_error(fit.coefficients,fit.scale,fit.background,residual)
        assert error<1e-8
        native_cost=float(.5*(residual@residual))
        assert native_cost==pytest.approx(fit.weighted_cost,rel=1e-8,abs=1e-7)
        rows.append(dict(start=index,cost=native_cost,forward_error=error,
            optimizer_qualified=fit.success,maximum_moment=fit.maximum_moment,
            residual=residual.copy()))
    best=min(rows,key=lambda row:row['cost'])
    return dict(rows=rows,best=best,compilation=compilation,seconds=time.monotonic()-began)


def test_portable_library_preserves_complete_k_search_across_a_new_series(tmp_path):
    nominal=tmp_path/'before-measurement';nominal.mkdir()
    reference,construction,ferro,_=inputs(nominal,'PNT')
    anti=copy.deepcopy(ferro)
    anti.update(id='public:antiferro-p1',magnetic_space_group='P 1',bns_number='1.3',
        bns_lattice='S',transformation_matrix=np.diag([1.,1.,2.]).tolist())
    families={'ferro':(ferro,['0','0','0']),'anti':(anti,['0','0','1/2'])}
    grid=dict(kind='PNT',lower=3000.,upper=40000.,bins=701,bank=1)
    library=create_library(tmp_path/'library',name='Prospective public two-k control',source=nominal/'parent.cif',
        banks={'bank 1':dict(instrument=nominal/'bank.instprm',grid=grid)},
        models=[dict(kind='magnetic',bank='bank 1',settings=dict(candidate=candidate,k=k,
            sites=construction['sites'],maximum_moment=6.,refine_histogram_scale=False,refine_background=False))
            for candidate,k in families.values()])
    began=time.monotonic()
    for entry in library.document['entries']:
        identifier=entry['id'];attempt=library.start(identifier)
        request=stage_entry(library,identifier,attempt,tmp_path/'preparation'/identifier)
        result=execute('magnetic',request)
        publish_entry(library,identifier,attempt,request,result)
    preparation_seconds=time.monotonic()-began
    archive=tmp_path/'portable.zip';library.export(archive)
    frozen=file_sha(archive)
    restored=import_library(archive,tmp_path/'new-session')
    caches={}
    for entry in restored.document['entries']:
        files=restored.prepared_assets(entry['id'])
        manifest=next(path for path in files.values() if path.name=='entry.json')
        response,cache,receipt=restore_magnetic_entry(manifest.parent)
        assert response._native_seal is None and receipt['native_qualification_required']
        label=next(name for name,(candidate,_) in families.items() if candidate['id']==entry['model']['settings']['candidate']['id'])
        caches[label]=cache
    # No observation, winning k or refined geometry was used to make the ZIP.
    # The low-precision weak scan is deliberately not separable on the
    # supplied uncertainty scale. Repeat it at higher precision instead of
    # interpreting a numerically leading k as detected magnetic order.
    scans=[('null',0.,'same',250000.,False),('anti',.25,'width',250000.,False),
        ('anti',.25,'width',2500000000.,True),('anti',1.6,'cell',250000.,True),
        ('ferro',1.6,'adp',250000.,True)]
    records=[]
    for revision,(truth,magnitude,change,weight,separated) in enumerate(scans,1):
        root=tmp_path/f'scan-{revision}';root.mkdir()
        parent=G.G2Project(gpxfile=reference['path'])
        if change=='cell':
            parent.phase(0).data['General']['Cell'][1]*=1.003
            parent.phase(0).data['General']['Cell'][2]*=.998
        if change=='adp':
            phase=parent.phase(0);cia=phase.data['General']['AtomPtrs'][3]
            for atom in phase.data['Atoms']:atom[cia+1]+=.01
        if change=='width':
            profile=parent.histograms()[0].data['Instrument Parameters'][0]
            for name,value in [('X',.1),('Y',.08),('Z',.03)]:profile[name][0]=profile[name][1]=value
        source=root/'nuclear-context.gpx';parent.save(str(source));models={}
        for label,(candidate,k) in families.items():
            protocol=copy.deepcopy(construction);protocol['source']['sha256']=file_sha(source);protocol['k']=k
            protocol['policy'].update(refine_histogram_scale=False,refine_background=False)
            output=root/label;output.mkdir()
            result=construct_theory(source,Artifacts(output),protocol,candidate,
                dict(axis='x',magnitude=1.,parent_direction=None,independent_mode_seed=17,site_signs={}))
            models[label]=output/result['gpx']
        truth_path=models['ferro' if truth=='null' else truth]
        project=G.G2Project(gpxfile=str(truth_path))
        for phase in project.phases():
            if phase.data['General']['Type']=='magnetic':
                cx=phase.data['General']['AtomPtrs'][0]
                phase.data['Atoms'][0][cx+4:cx+7]=(magnitude*np.array([.8,.5,.3])).tolist()
        project.save();observed,_=forward_profile(truth_path,project.histograms()[0].name)
        for path in models.values():
            measured=G.G2Project(gpxfile=str(path));histogram=measured.histograms()[0]
            arrays=histogram.data['data'][1];arrays[1]=observed.copy();arrays[2]=np.full(len(observed),weight)
            histogram.data['data'][0]['Dummy']=False;measured.save()
        results={}
        for label,path in models.items():
            prepared=fit_portfolio(path,caches[label])
            cold=fit_portfolio(path,None)
            assert len(prepared['rows'])==len(cold['rows'])==6
            for left,right in zip(prepared['rows'],cold['rows']):
                assert left['start']==right['start']
                assert left['cost']==pytest.approx(right['cost'],rel=1e-6,abs=1e-6)
                np.testing.assert_allclose(left['residual'],right['residual'],rtol=1e-6,atol=2e-5)
            proof=prepared['compilation']['compiler_diagnostics']
            assert proof['scattering_factors_reused']==(change in {'same','width'})
            results[label]=dict(best_cost=prepared['best']['cost'],cold_best_cost=cold['best']['cost'],
                prepared_seconds=prepared['seconds'],cold_seconds=cold['seconds'],
                starts=len(prepared['rows']),qualified=sum(row['optimizer_qualified'] for row in prepared['rows']),
                maximum_native_error=max(row['forward_error'] for row in prepared['rows']),
                factors_reused=proof['scattering_factors_reused'])
            write_json(tmp_path/'progress.json',dict(revision=revision,truth=truth,change=change,
                weight=weight,completed_scans=records,current_families=results))
        if truth=='null':assert all(row['best_cost']<1e-5 for row in results.values())
        else:
            assert min(results,key=lambda label:results[label]['best_cost'])==truth
            assert results[truth]['best_cost']<1e-5
            gap=results['anti' if truth=='ferro' else 'ferro']['best_cost']-results[truth]['best_cost']
            if separated:assert gap>1.
            else:assert 0.<gap<1.
        records.append(dict(revision=revision,truth=truth,change=change,weight=weight,
            declared_objective_separation=separated,families=results))
        assert file_sha(archive)==frozen
    write_json(tmp_path/'series-acceptance.json',dict(schema='radar-public-prepared-k-series-control/v1',
        preparation_seconds=preparation_seconds,archive_sha256=frozen,scans=records,
        complete_family_denominator=2,starts_per_family=6,measured_scan_before_preparation=False,
        native_checked_fitted_profiles=len(scans)*2*6*2,
        scope='Public noiseless synthetic numerical/search-integrity control; no experimental identification, calibrated uncertainty, global-search guarantee or NDIP latency claim'))
