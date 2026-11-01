import pytest

from radar_pd_nova import model_navigation as module


@pytest.mark.parametrize('scan,bank,expected',[
    ('220654 · Bank 2',2,'220654 · Bank 2 · 311.242 K'),
    ('Run 1 · Bank 12',2,'Run 1 · Bank 12 · 311.242 K · bank 2'),
    ('Run 1 · BANK2','2','Run 1 · BANK2 · 311.242 K'),
    ('Run 1',0,'Run 1 · 311.242 K · bank 0'),
    ('Run 1',None,'Run 1 · 311.242 K'),
])
def test_scan_labels_retain_distinct_banks_without_repeating_an_existing_bank(scan,bank,expected):
    assert module.scan_label(dict(scan=scan,bank=bank,condition=311.242,condition_unit='K'),'parent')==expected


@pytest.mark.parametrize('condition',[None,True,float('nan'),float('inf'),'311 K'])
def test_missing_or_invalid_conditions_do_not_become_numeric_coordinates(condition):
    assert module.scan_label(dict(condition=condition),'Hematite')=='Hematite'


def test_large_trial_portfolio_formats_labels_once_per_acquisition(monkeypatch):
    calls=[]
    label=module.scan_label
    def counted(*args):calls.append(args);return label(*args)
    monkeypatch.setattr(module,'scan_label',counted)
    rows=[dict(source='cold',parent='Hematite',metadata={'scan':'1','bank':2})]*912
    rows += [dict(source='warm',parent='Hematite',metadata={'scan':'2','bank':2})]*912
    assert module.scan_options(rows)=={'cold':'1 · bank 2','warm':'2 · bank 2'}
    assert len(calls)==2
