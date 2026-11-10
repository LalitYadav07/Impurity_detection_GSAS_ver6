"""Actual native isostructural controls; not a mode-accuracy qualification."""
import json
import time

import numpy as np
import pytest

pytest.importorskip('GSASII.GSASIIscriptable')
torch=pytest.importorskip('torch')


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_shared_attention_mode_readout_can_optimize_single_sg_native_targets(tmp_path,kind,record_property):
    from test_structural_family_targets_native import native_cohort
    from gpx_recipe.preparation.structural_features import ThreeCurveProjector
    from gpx_recipe.preparation import structural_mode_readout as modes
    from gpx_recipe.preparation.structural_attention import network
    from gpx_recipe.symmetry.compiler import file_sha
    began=time.monotonic();path,paired,cohort,table,before=native_cohort(tmp_path,kind)
    assert len(cohort['classes'])==1
    norm=modes.fit(table,cohort);a=paired.arrays;q=a['q']
    projector=ThreeCurveProjector(q,np.linspace(q.min(),q.max(),129))
    values=np.array([projector.features(a['theory_intensity'][i],a['parent_intensity'][i],
        background=a['background'],reference_background=a['parent_background'][i])['features']
        for i in range(len(table))],np.float32)
    fitting=np.flatnonzero(cohort['split']==0);heldout=np.flatnonzero(cohort['split']!=0)
    assert len(fitting) and len(heldout) and len(fitting)<=256 and len(heldout)<=256
    targets=norm.targets(table,fitting);other=norm.targets(table,heldout)
    assert len(targets['upper']) and np.any(targets['upper']>0) and np.any(targets['upper']==0)
    queries=[torch.from_numpy(targets[k]) for k in ('row','space_group','family')]
    other_queries=[torch.from_numpy(other[k]) for k in ('row','space_group','family')]
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(8911319);encoder=network(1);head=modes.network(1,len(norm.scales))
            optimizer=torch.optim.AdamW([*encoder.parameters(),*head.parameters()],lr=.002)
            inputs=torch.from_numpy(values[fitting]);encoder.eval();head.eval()
            with torch.no_grad():
                initial=float(modes.loss(head(encoder.embedding(inputs),*queries),targets,rows=len(fitting),class_count=1))
            for _ in range(100):
                encoder.train();head.train();optimizer.zero_grad(set_to_none=True)
                prediction=head(encoder.embedding(inputs),*queries)
                objective=modes.loss(prediction,targets,rows=len(fitting),class_count=1)
                objective.backward();torch.nn.utils.clip_grad_norm_([*encoder.parameters(),*head.parameters()],1.,error_if_nonfinite=True)
                optimizer.step()
            encoder.eval();head.eval()
            with torch.no_grad():
                final=float(modes.loss(head(encoder.embedding(inputs),*queries),targets,rows=len(fitting),class_count=1))
                prediction=head(encoder.embedding(torch.from_numpy(values[heldout])),*other_queries)
                validation=float(modes.loss(prediction,other,rows=len(heldout),class_count=1))
                decoded=norm.decode(prediction.numpy(),other['family'])
    finally:torch.set_num_threads(previous)
    assert np.isfinite([initial,final,validation]).all() and final<initial*.95
    assert decoded.shape==(len(other['row']),2) and np.isfinite(decoded).all()
    assert file_sha(path/'examples.npz')==before
    record_property('mode_control',json.dumps(dict(kind=kind,sg=cohort['classes'].tolist(),
        mode_families=len(norm.scales),fitting_rows=len(fitting),heldout_rows=len(heldout),
        initial_fitting_loss=initial,final_fitting_loss=final,heldout_interval_loss=validation,
        seconds=time.monotonic()-began,independent_mode_accuracy_qualified=False,
        experimental_mode_absence_qualified=False,candidate_pruning_authorized=False)))
