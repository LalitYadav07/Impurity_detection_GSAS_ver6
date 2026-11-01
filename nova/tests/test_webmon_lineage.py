from types import SimpleNamespace

from radar_pd_nova.webmon.lineage import posterior_source
from radar_pd_nova.refinement_sources import RefinementScope
from radar_pd_nova.refinement_workspace import RefinementWorkspace


def test_posterior_handoff_uses_target_acquisition_and_exact_single_scan_scope(tmp_path):
    parent=dict(id='parent',experiment='PG3:7',source_sha256='a'*64,scan_key='run-1',
        run_id='PG3_1',run_number=1,scan_label='Run 1',metadata={'temperature':300},
        axis_overrides={'temperature':300},bank='2',sample_key='id:old',sample_label='Reference',
        source_path='reference.gpx',dataset_id='old-dataset',context={'old':True})
    derived=posterior_source(parent,revision='PG3_2:'+'b'*64,path='fitted.gpx',checksum='c'*64,rwp=5.,
                             scope={'run':'2','sample_id':'new','sample':'Target'})
    assert derived['run_id']=='PG3_2' and derived['run_number']==2
    assert 'Run 2' in derived['scan_label'] and derived['sample_label']=='Target'
    assert 'temperature' not in derived['metadata'] and not derived['axis_overrides']
    assert derived['reference_ancestry']['metadata']=={'temperature':300}
    assert 'dataset_id' not in derived and 'context' not in derived
    workspace=RefinementWorkspace(tmp_path,history_id='history',runtime=SimpleNamespace(kind='galaxy'))
    workspace.observe([parent,derived])
    selected,_=workspace.selection(RefinementScope('PG3:7',mode='scans',scans=(derived['scan_key'],)))
    assert [r['id'] for r in selected['baseline']]==[derived['id']]
    assert parent['metadata']=={'temperature':300}
