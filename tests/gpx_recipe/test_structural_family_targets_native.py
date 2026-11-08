"""Real paired native output reaches class-conditional family supervision."""
from pathlib import Path

import numpy as np
import pytest

G=pytest.importorskip('GSASII.GSASIIscriptable')


def native_cohort(tmp_path,kind):
    from test_native_profile_capture import native_source
    from gpx_recipe.preparation.worker import execute
    from gpx_recipe.preparation.structural_training_blocks import REPRESENTATION
    from gpx_recipe.preparation.structural_paired_blocks import read
    from gpx_recipe.preparation.structural_family_targets import build
    from gpx_recipe.preparation.structural_mode_projection import descriptor
    from gpx_recipe.symmetry.compiler import file_sha
    source,_=native_source(tmp_path/'input',kind)
    parent=tmp_path/'parent.cif';G.G2Project(gpxfile=str(source)).phase(0).export_CIF(str(parent))
    child=Path(__file__).with_name('data')/'synthetic-ersi-cmcm-isodistort.cif'
    request=dict(source=str(parent),instrument=str(tmp_path/'input/profile.instprm'),child=str(child),
        grid=dict(kind=kind,lower=12. if kind=='PNC' else 3000.,upper=125. if kind=='PNC' else 40000.,bins=701,bank=1),
        settings=dict(candidate=dict(id='public-control',transformation_matrix=[[-1.,0,0],[0,-1.,0],[0,0,1.]],
            origin_shift=[0,0,0],child=dict(path='child.cif',sha256=file_sha(child))),mode_radius=.05,cell_radius=.01,
            seed=1947,parent_seed=8341,parent_relative_metric_range=[.0001,.001],representation=REPRESENTATION,
            trajectories=4,steps=2,trajectory_slice=[0,4]),destination=str(tmp_path/'generated'))
    execute('structural-examples',request)
    path=tmp_path/'generated';before=file_sha(path/'examples.npz');paired=read(path,names='examples')
    # This public control is isostructural: its native rows have one geometric
    # SG. Test mode supervision directly, without inventing a second class to
    # satisfy the separate classifier's class-supported holdout contract.
    d=paired.manifest;a=paired.arrays;classes=np.unique(a['geometry_groups']).astype(np.int32)
    labels=np.array([[c in values for c in classes] for values in a['geometry_groups']],bool)
    groups=sorted(set(d['groups']));folds={g:i%3 for i,g in enumerate(groups)}
    cohort=dict(classes=classes,compatible_labels=labels,groups=list(d['groups']),
        split=np.array([folds[g] for g in d['groups']],np.int8),
        references=[[(0,i)] for i in range(d['examples'])],input_sha256=[d['pairs_sha256']],
        parent_sha256=d['source_sha256'],instrument_sha256=d['instrument_sha256'],
        mode_projection_references=[descriptor(paired.mode_projection)['identity']])
    table=build(lambda:iter([read(path,names='examples')]),cohort);table.verify(cohort)
    return path,paired,cohort,table,before


@pytest.mark.parametrize('kind',['PNC','PNT'])
def test_native_paired_data_keep_every_geometric_family_amplitude_and_origin(tmp_path,kind):
    from gpx_recipe.preparation.structural_mode_families import FamilyProjection
    from gpx_recipe.symmetry.compiler import file_sha
    path,paired,cohort,table,before=native_cohort(tmp_path,kind)
    assert table.metadata['native_points']==paired.manifest['examples']
    assert table.metadata['intervals']>0 and table.metadata['opaque_native_axes']==0
    model=FamilyProjection(paired.mode_projection)
    for row in range(min(20,len(table))):
        batch=table.batch(np.array([row],np.int64))
        for column,family,low,high in zip(batch['space_group'],batch['family'],batch['lower'],batch['upper']):
            identifier=table.metadata['families'][family]['id'];values=[]
            for _,index in cohort['references'][row]:
                if table.metadata['classes'][column] not in paired.arrays['geometry_groups'][index]:continue
                actual=model.project(paired.arrays['parameters'][index:index+1,:model.mode_dimensions])
                values.append(actual['amplitudes'][0,actual['families'].index(identifier)])
            assert low==min(values) and high==max(values)
    assert file_sha(path/'examples.npz')==before
    assert not table.metadata['candidate_pruning_authorized'] and not table.metadata['experimental_mode_absence_qualified']
