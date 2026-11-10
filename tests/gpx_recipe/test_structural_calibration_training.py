import copy,json
from pathlib import Path

import numpy as np
import pytest

from gpx_recipe.preparation.structural_calibration_training import Logits
from gpx_recipe.preparation import structural_adaptive_sets as adaptive


def test_existing_inference_capture_checks_every_row_once_in_order():
    capture=Logits(np.arange(5),3)
    capture.add(np.array([0,1]),np.zeros((2,3),np.float32))
    with pytest.raises(ValueError,match='every'):capture.finish()
    for indices,values in [(np.array([3,2]),np.zeros((2,3),np.float32)),
        (np.array([2]),np.zeros((1,3),np.float64)),(np.array([2]),np.full((1,3),np.nan,np.float32))]:
        with pytest.raises(ValueError):capture.add(indices,values)
    capture.add(np.array([2,3,4]),np.ones((3,3),np.float32))
    assert capture.finish().shape==(5,3)


def test_vectorized_adaptive_metrics_match_individual_arriving_patterns():
    classes=[1,2,3];identities={k:'a'*64 for k in ['weights_sha256','parent_sha256','instrument_sha256','dataset_sha256']}
    cal=adaptive.calibrate(np.tile([.5,0.,-1.],(20,1)),np.ones((20,3),bool),
        [f'cal-{i}' for i in range(20)],classes=classes,identities=identities,used_groups=['train'])
    logits=np.random.default_rng(7).normal(size=(101,3));labels=np.ones((101,3),bool)
    sets=[set(adaptive.select(row[None],cal,identities=identities)['retained_space_groups']) for row in logits]
    result=adaptive.evaluate(logits,labels,[f'test-{i//3}' for i in range(101)],cal,identities=identities)
    assert result['all_compatible_retention']==pytest.approx(sum(len(v)==3 for v in sets)/len(sets))
    assert result['excluded_compatible_labels']==sum(3-len(v) for v in sets)
    adaptive.validate_evaluation(result,classes,rows=101,groups=34,class_rows=[101]*3)
    changed=copy.deepcopy(result);changed['excluded_compatible_labels']+=1
    with pytest.raises(ValueError):adaptive.validate_evaluation(changed,classes,rows=101,groups=34,class_rows=[101]*3)


def packet_fixture(root):
    """Artificial closed-contract pairs, solely for the >5000-row handoff gate.

    These authored curves test transport, group reservation and model receipts;
    they are not native diffraction calculations or scientific accuracy data.
    """
    from test_structural_paired_preparation import inputs
    from gpx_recipe.preparation.structural_paired_blocks import read,SCHEMA
    from gpx_recipe.preparation.structural_jobs import create_plan
    from gpx_recipe.symmetry.compiler import file_sha
    request,example=inputs(root);document=read(example,names='examples').manifest
    paths=[];q=np.linspace(.5,4.,65);background=np.full(len(q),3.)
    trajectories=np.r_[-1,np.repeat(np.arange(64),33)].astype(np.int64)
    roles=np.r_[0,np.tile(np.r_[1,np.full(32,2)],64)].astype(np.int8)
    parent_a=np.tile([.04,.04,.01,0.,0.,0.],(len(roles),1))
    parents=np.empty((len(roles),len(q)));parents[0]=background+np.exp(-((q-1.)/.05)**2)
    for t in range(64):
        chosen=trajectories==t
        parents[chosen]=background+np.exp(-((q-1.-(t+1)*.0001)/.05)**2)
        parent_a[chosen,:3]*=1.+(t+1)*1e-5
    for variant in range(7):
        path=root/f'cohort-{variant}';path.mkdir();paths.append(path)
        curves=parents.copy();parameters=np.zeros((len(roles),3))
        for t in range(64):
            selected=np.flatnonzero((trajectories==t)&(roles==2))
            amplitude=np.linspace(.005,.15,32)
            curves[selected]+=amplitude[:,None]*np.exp(-((q-2.-.13*variant)/.06)**2)
            parameters[selected,0]=amplitude*.01
        labels=np.where(roles==2,12,166).astype(np.int32)
        np.savez_compressed(path/'examples.npz',q=q,theory_intensity=curves,background=background,
            parent_intensity=parents,parent_background=np.broadcast_to(background,parents.shape),
            parent_a=parent_a,parameters=parameters,trajectory=trajectories,role=roles,
            geometry_groups=np.repeat(labels[:,None],2,axis=1),ambiguous_geometry=np.zeros(len(roles),bool),
            displacement_rms=np.linalg.norm(parameters[:,:1],axis=1),relative_metric_norm=np.zeros(len(roles)))
        d=copy.deepcopy(document);d.update(schema=SCHEMA,pairs_sha256=file_sha(path/'examples.npz'),
            examples=len(roles),bins=len(q),trajectories=64,steps=32,trajectory_slice=[0,64])
        d.pop('array_encoding',None);d.pop('original_pairs_sha256',None)
        d['candidate']['id']=f'contract-child-{variant}'
        source=d['source_sha256'];prefix=f'parent:{source}:{d["seed"]}:{d["parent_seed"]}:trajectory-'
        d['groups']=['parent:'+source]+[prefix+str(i) for i in trajectories[1:]]
        (path/'examples.json').write_text(json.dumps(d));read(path,names='examples')
    plan=root/'plan.json';create_plan(request['source'],paths,plan)
    return request,plan
