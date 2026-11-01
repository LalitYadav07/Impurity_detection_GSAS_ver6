"""Public artificial TOF control for the portable native preparation action."""
import json
from pathlib import Path
import numpy as np
import pytest
G=pytest.importorskip('GSASII.GSASIIscriptable')
from GSASII import GSASIIstrMain as MAIN
from gpx_recipe.execution import freeze
from gpx_recipe.live.prepare_source import prepare
from gpx_recipe.live.validated_backend import corrected_backend
from gpx_recipe.live.contracts import file_hash
from gsas_magnetic_preflight import setting_fingerprint


def test_preparation_preserves_reference_and_uses_actual_overlap(tmp_path):
    profile=Path(__file__).parents[1]/'src/radar_pd_nova/data/powgen_calibrations/2026A_HighRes_60HzB2_CWL1p5.instprm'
    source=tmp_path/'original.gpx';g=G.G2Project(newgpx=str(source))
    p=g.add_phase(phasename='Artificial',cell=[4.7,5.6,6.8,90.,90.,90.]);p.add_atom(.1,.2,.3,'Fe','Fe1')
    h=g.add_simulated_powder_histogram('artificial',str(profile),10.,80.,Npoints=401,scale=120.,phases=[p])
    freeze(g);h.data['data'][1][1][:]=1.;h.data['data'][1][2][:]=4.;g.save()
    with corrected_backend():
        calc=np.asarray(MAIN.DoNoFit(str(source),h.name));h.data['data'][1][1]=calc+.01;g.save()
        checksum=file_hash(source);instrument=h.data['Instrument Parameters'].copy()
        factor=22000.;d=np.linspace(.8,3.,221);y=np.interp(d*factor,h.data['data'][1][0],calc)*factor
        live=tmp_path/'live.dat';live.write_text('# X , Y , E Distribution=true\n0\n'+'\n'.join(f'{xx},{yy},100' for xx,yy in zip(d,y))+'\n')
        catalog=tmp_path/'catalog.json';catalog.write_text(json.dumps(dict(parent_setting_sha256=setting_fingerprint(p.data['General']['SGData']),k=['0','0','0'],candidates=[dict(id='p1',magnetic_space_group='P1',bns_number='1.1',bns_lattice='',transformation_matrix=np.eye(3).tolist(),origin_shift=[0,0,0])])) )
        request=dict(source_gpx=str(source),source_sha256=checksum,catalog=str(catalog),catalog_sha256=file_hash(catalog),
            observation_source=str(live),observation_source_sha256=file_hash(live),parent_phase=p.name,k=['0','0','0'],sites=[dict(label='Fe1',form_factor='Fe+3',lande_g=2)],
            mapping=dict(mapping_kind='user_reviewed',snapshot_sha256=file_hash(live),source_gpx_sha256=checksum,producer_config_sha256='a'*64,export_geometry={'microseconds_per_angstrom':factor}),
            source_revision='PG3_1:'+file_hash(live),reference_revision='PG3_2:'+checksum,refine_transfer=False)
        result=prepare(request,tmp_path)
        assert result['transfer']['performed'] is False
        assert result['transfer_limits_tof']==[max(h.data['Limits'][1][0],d[0]*factor),min(h.data['Limits'][1][1],d[-1]*factor)]
        assert result['scientifically_qualified'] is False
        assert file_hash(source)==checksum
        exported=G.G2Project(str(tmp_path/result['source_gpx']))
        assert exported.histogram(0).data['Instrument Parameters']==instrument
        np.testing.assert_array_equal(exported.phase(0).data['Atoms'],p.data['Atoms'])
        assert exported.data['Covariance']['data']=={}
