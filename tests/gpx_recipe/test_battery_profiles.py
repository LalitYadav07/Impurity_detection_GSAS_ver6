from pathlib import Path
from types import SimpleNamespace

import pytest

from gpx_recipe.battery.profiles import legacy_profile_metadata,native_profile


def profile(path):
    rows={'INS   HTYPE ':'PNTR','INS   FPATH1':'19.5','INS  2 ICONS':'2900 0 0',
          'INS  2BNKPAR':'2 31','INS  2PRCF1 ':'3 21 .002',
          'INS  2PRCF11':'1 2 3 4','INS  2PRCF12':'5 6 7 8',
          'INS  2PRCF13':'9 0 0 0','INS  2PRCF14':'0 0 0 0',
          'INS  2PRCF15':'0 0 0 0','INS  2PRCF16':'0','INS  2I ITYP':'0 .4 19 1'}
    path.write_text('ID    Synthetic profile fixture\n'+''.join(k+v+'\n' for k,v in rows.items()))
    return path


def test_legacy_profile_uses_bank_label_and_preserves_title(tmp_path):
    path=profile(tmp_path/'profile.prm');metadata=legacy_profile_metadata(path,2)
    assert metadata['profile_bank']==2 and metadata['profile_function']==3
    assert metadata['legacy_title']=='Synthetic profile fixture'
    assert metadata['calibration_status']=='requires_scientific_validation'
    with pytest.raises(ValueError,match='Missing legacy'):legacy_profile_metadata(path,1)


@pytest.mark.parametrize('old,new,message',[
    ('3 21 .002','2 21 .002','unsupported'),
    ('3 21 .002','-3 21 .002','table based'),
    ('3 21 .002','4 21 .002','unsupported'),
    ('3 21 .002','5 21 .002','unsupported'),
    ('3 21 .002','3 20 .002','coefficient count'),
    ('INS  2PRCF160','INS  2PRCF16nan','nonfinite'),
    ('0 .4 19 1','1 .4 19 1','incident-spectrum'),
    ('2900 0 0','nan 0 0','Nonfinite'),
    ('2900 0 0','broken 0 0','Invalid numeric'),
    ('PNTR','PXCR','neutron TOF'),
])
def test_unsupported_or_ambiguous_legacy_conversion_fails_closed(tmp_path,old,new,message):
    path=profile(tmp_path/'profile.prm');path.write_text(path.read_text().replace(old,new))
    with pytest.raises(ValueError,match=message):legacy_profile_metadata(path,2)


def test_duplicate_bank_record_is_not_silently_overwritten(tmp_path):
    path=profile(tmp_path/'profile.prm');path.write_text(path.read_text()+'INS  2 ICONS3000 0 0\n')
    with pytest.raises(ValueError,match='Duplicate'):legacy_profile_metadata(path,2)


def test_conversion_checks_native_roundtrip_and_keeps_source_bytes(tmp_path):
    path=profile(tmp_path/'profile.prm');before=path.read_bytes()
    values={'Bank':1.,'difC':2900.,'2-theta':31.,'alpha':1.,'beta-0':2.,'beta-1':3.,'sig-0':0.,'sig-1':5.,'sig-2':6.}
    def load(path,reader,bank):
        if Path(path).suffix=='.prm':return [{'Type':['PNT','PNT',False],**{k:[v,v,False] for k,v in values.items()}},{}]
        from radar_pd_nova.nomad import profile_blocks
        return [{'Type':['PNT','PNT',False],**{k:[v,v,False] for k,v in profile_blocks(path)[bank][1].items()}},{}]
    output,receipt=native_profile(path,2,tmp_path/'converted.instprm',SimpleNamespace(load_iprms=load))
    assert path.read_bytes()==before and output.is_file()
    assert receipt['converted_parameters']['Bank']==2.
    assert receipt['source_sha256']!=receipt['converted_sha256']
    # Independent GSAS function-3 definitions: sig²=4+5*d²+6*d⁴;
    # gamma=7+8*d+9*d². The legacy native reader omitted the constants/gamma.
    converted=receipt['converted_parameters']
    assert converted['sig-0']+4*converted['sig-1']+16*converted['sig-2']==120
    assert converted['Z']+2*converted['X']+4*converted['Y']==59
    assert receipt['native_loader_differences']['sig-0']==dict(native_loader_value=0.,source_value=4.)
    assert receipt['native_loader_differences']['Y']['source_value']==9.
    assert receipt['adapter']=='gsas-isotropic-tof-coefficients/v2'


def test_native_incident_corrections_are_not_discarded(tmp_path):
    path=profile(tmp_path/'profile.prm')
    with pytest.raises(ValueError,match='incident-spectrum'):
        native_profile(path,2,tmp_path/'converted.instprm',SimpleNamespace(load_iprms=lambda *args,**kw:[{}, {'Itype':'unknown'}]))


@pytest.mark.parametrize('index',range(9,21))
def test_nonzero_unmapped_function_three_term_is_never_dropped(tmp_path,index):
    path=profile(tmp_path/'profile.prm')
    rows=path.read_text().splitlines()
    key='INS  2PRCF1'+str(index//4+1)
    line=next(i for i,row in enumerate(rows) if row.startswith(key))
    values=rows[line][12:].split();values[index%4]='0.1'
    rows[line]=key+' '.join(values);path.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError,match='Unsupported nonzero'):
        legacy_profile_metadata(path,2)


def test_full_coefficient_count_is_required_not_only_native_imported_lines(tmp_path):
    path=profile(tmp_path/'profile.prm')
    path.write_text('\n'.join(row for row in path.read_text().splitlines() if not row.startswith('INS  2PRCF16'))+'\n')
    with pytest.raises(ValueError,match='Missing legacy profile coefficient'):
        legacy_profile_metadata(path,2)


def test_fixed_width_header_touching_cutoff_and_old_record_style(tmp_path):
    path=profile(tmp_path/'profile.prm')
    path.write_text(path.read_text().replace('3 21 .002',f'{3:5d}{21:5d}2.000E-004').replace('PRCF1','PRCF '))
    record=legacy_profile_metadata(path,2)
    assert record['coefficient_count']==21 and record['legacy_peak_cutoff']==.0002
    assert record['coefficient_mapping']['Y']==9.


def test_type_one_mapping_and_unsupported_constant_exponential(tmp_path):
    path=profile(tmp_path/'profile.prm')
    rows=[row for row in path.read_text().splitlines() if not row.startswith('INS  2PRCF1')]
    rows += ['INS  2PRCF1 1 12 .002','INS  2PRCF110 1 2 3','INS  2PRCF124 5 6 0','INS  2PRCF130 0 0 0']
    path.write_text('\n'.join(rows)+'\n')
    values=legacy_profile_metadata(path,2)['coefficient_mapping']
    assert values['sig-0']==4 and values['sig-1']==5 and values['sig-2']==6
    assert values['alpha']==1 and values['Z']==values['X']==values['Y']==0
    path.write_text(path.read_text().replace('PRCF110 1 2 3','PRCF111 1 2 3'))
    with pytest.raises(ValueError,match='Unsupported nonzero'):
        legacy_profile_metadata(path,2)
