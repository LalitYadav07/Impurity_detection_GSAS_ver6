"""Metric metadata locates optional physics but grants no native authority."""
import copy
import json

import numpy as np
import pytest

from gpx_recipe.preparation.kernel_geometry import SCHEMA,validate_geometry
from gpx_recipe.preparation.kernel_store import KernelStore,GEOMETRY_SCHEMA
if __package__:
    from .test_tof_kernel_storage import data
else:
    from test_tof_kernel_storage import data


def geometry():
    return dict(schema=SCHEMA,setting_sha256='a'*64,profile_sha256='b'*64,
        runtime=dict(kernel_sha256='c'*64,binary_sha256='d'*64),
        cells=[[.1,.2,.3,0.,0.,0.],[.11,.21,.31,0.,0.,0.]])


@pytest.mark.parametrize('kind',['extra','count','hash','runtime','nan','negative','duplicate','bool','shape'])
def test_invalid_metric_metadata_cannot_enter_a_store(tmp_path,kind):
    g=geometry()
    if kind=='extra':g['native_verified']=True
    elif kind=='count':g['cells'].pop()
    elif kind=='hash':g['setting_sha256']='not-a-checksum'
    elif kind=='runtime':g['runtime']['trusted']=True
    elif kind=='nan':g['cells'][0][0]=float('nan')
    elif kind=='negative':g['cells'][0][0]=-.1
    elif kind=='duplicate':g['cells'][1]=g['cells'][0].copy()
    elif kind=='bool':g['cells'][0][0]=True
    else:g['cells'][0].pop()
    with pytest.raises(ValueError):
        KernelStore.create(tmp_path/'store',dict(model_sha256='e'*64,bank='bank 2',cell_count=2,geometry=g))


def test_metric_parts_roundtrip_without_becoming_native_permissions(tmp_path):
    spec=dict(model_sha256='e'*64,bank='bank 2',cell_count=2,geometry=geometry())
    store=KernelStore.create(tmp_path/'store',spec);store.put(0,data())
    assert store.document['schema']==GEOMETRY_SCHEMA
    part=store.export_parts(tmp_path/'parts')['parts'][0]
    restored=KernelStore.import_part(tmp_path/'parts'/part['path'],tmp_path/'imported')
    assert restored.document['specification']==spec
    assert restored.coverage()['missing']==[1]
    assert restored.coverage()['scientific_qualification'] is False
    np.testing.assert_array_equal(restored.get(0)._arrays['values'],data()._arrays['values'])
    assert not hasattr(restored.get(0),'lookup')


def test_metric_table_changes_the_store_identity_and_cannot_replace_resume_context(tmp_path):
    spec=dict(model_sha256='e'*64,bank='bank 2',cell_count=2,geometry=geometry())
    store=KernelStore.create(tmp_path/'store',spec);changed=copy.deepcopy(spec)
    changed['geometry']['cells'][0][0]+=.001
    with pytest.raises(ValueError,match='specification differs'):
        KernelStore.open_or_create(store.root,changed)
    assert KernelStore(store.root).document['specification']==spec


def test_geometry_cannot_hide_in_the_legacy_format(tmp_path):
    store=KernelStore.create(tmp_path/'store',dict(model_sha256='e'*64,bank='B2',cell_count=2,geometry=geometry()))
    doc=json.loads((store.root/'index.json').read_text());doc['schema']='radar-tof-kernel-store/v1'
    (store.root/'index.json').write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='declared store format'):KernelStore(store.root)
