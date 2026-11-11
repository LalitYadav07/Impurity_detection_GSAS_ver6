"""A narrow recipe must not inherit active displacement modes from its source."""
from copy import deepcopy
from types import SimpleNamespace

from gpx_recipe.execution import hold_imported_displacement_modes


class Phase:
    def __init__(self):
        self.mode=SimpleNamespace(phase=1847,name='GM1(a)')
        self.data={'General':{'Type':'nuclear'},'Histograms':{},
                   'ISODISTORT':{'G2ModeList':[self.mode], 'modeDispl':[.027]}}
    def clear_refinements(self,*args):pass
    def set_refinements(self,*args):pass
    def clear_HAP_refinements(self,*args):pass


def project():
    phase=Phase()
    # Last fields of a native 'f' constraint: variable, vary flag, type.
    imported=[[1.,SimpleNamespace(phase=1847,name='dAx',atom=32)],phase.mode,True,'f']
    unrelated=[[1.,SimpleNamespace(phase=1847,name='AUiso',atom=32)],
               SimpleNamespace(phase=1847,name='temperature_model'),True,'f']
    same_name_other_phase=[[1.,SimpleNamespace(phase=999,name='dAx',atom=71)],
                          SimpleNamespace(phase=999,name='GM1(a)'),True,'f']
    constraints={'Phase':[imported,unrelated,same_name_other_phase],
                 'Hist':[],'HAP':[],'Global':[]}
    return SimpleNamespace(phases=lambda:[phase],histograms=lambda:[],
                           data={'Constraints':{'data':constraints}}),phase


def test_known_imported_modes_are_held_without_rewriting_equations_or_values():
    p,phase=project()
    equations=deepcopy(p.data['Constraints']['data'])
    amplitudes=deepcopy(phase.data['ISODISTORT']['modeDispl'])
    hold_imported_displacement_modes(p)
    equations['Phase'][0][-2]=False
    assert p.data['Constraints']['data']==equations
    assert phase.data['ISODISTORT']['modeDispl']==amplitudes


def test_unknown_generated_variables_remain_visible_for_preflight_rejection():
    p,_=project()
    hold_imported_displacement_modes(p)
    assert p.data['Constraints']['data']['Phase'][1][-2] is True
    assert p.data['Constraints']['data']['Phase'][2][-2] is True


def test_ordinary_projects_keep_their_source_constraints():
    p,phase=project();phase.data.pop('ISODISTORT')
    original=deepcopy(p.data)
    hold_imported_displacement_modes(p)
    assert p.data==original
