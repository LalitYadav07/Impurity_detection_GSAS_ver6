"""Native k-form matching and mode-inventory preparation, before approval."""
from __future__ import annotations
from fractions import Fraction
import json
from pathlib import Path
import re
import sys


def first_valid_form(forms,solve):
    # Native fixed forms can return a truthy {match: False}; do not let that
    # displace a valid parametric form. Retain native basis/parameter solving.
    remaining=list(forms)
    while remaining:
        index,parameters=solve(remaining)
        if type(index) is not int or not 0<=index<len(remaining) or not isinstance(parameters,dict):raise ValueError('Native k-form matching failed')
        if parameters.get('match') is False:remaining.pop(index);continue
        return remaining[index],parameters
    raise ValueError('No native k form matches the requested vector')


def match_k(cif,k,options):
    from ..snapshot import gsas
    gsas()
    # The validated Linux image predates this pure-Python upstream helper.
    # Ship its unchanged source/table with the upstream license; no additional
    # GSAS-II installation or Diffraction Studio runtime is required.
    from .vendor import k_vec_solve as solver
    forms=[]
    for value in options:
        if '(' not in value:raise ValueError('A k form has no symbolic vector')
        expression='('+value.split('(',1)[1]
        if not re.fullmatch(r'[0-9abg+*/().,\-\s]+',expression):raise ValueError('Unexpected symbolic k-form grammar')
        forms.append((value,solver.parse_expression_string(expression)))
    number=solver.grab_spg_num(cif)
    if not number:raise ValueError('The standardized parent has no space-group number')
    selected,parameters=first_valid_form(forms,lambda remaining:solver.find_kvec_form_param(number,cif,[float(Fraction(v)) for v in k],remaining))
    result=dict(kvec1=selected[0]);names={'a':'kparam11','b':'kparam21','g':'kparam31'}
    if not parameters.get('match'):
        for name,value in parameters.items():
            if str(name) in names:result[names[str(name)]]=str(Fraction(str(value)).limit_denominator(96))
    return result


def modes(child,temporary):
    from ..snapshot import gsas
    from gsas_isodistort_import import add_isodistort_phase
    G=gsas();p=G.G2Project(newgpx=str(temporary))
    phase,receipt=add_isodistort_phase(p,child,phasename='Exploration child')
    iso=phase.data['ISODISTORT']
    return dict(modes=[name.name for name in iso['G2ModeList']],normalized_modes=iso['IsoModeList'],normalization=iso['NormList'],import_receipt=receipt)


if __name__=='__main__':
    from ..execution import write_json
    request=json.loads(Path(sys.argv[2]).read_text())
    if sys.argv[1]=='match':result=match_k(request['cif'],request['k'],request['options'])
    elif sys.argv[1]=='modes':result=modes(request['child'],request['temporary'])
    elif sys.argv[1]=='export':
        from ..snapshot import gsas
        project=gsas().G2Project(gpxfile=request['source']);project.phase(request['phase']).export_CIF(request['destination']);result=dict(path=request['destination'])
    else:raise ValueError('Unknown native preparation action')
    write_json(sys.argv[3],result)
