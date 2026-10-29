"""Bind declared magnetic form factors to the actual occupied parent sites."""
import re


def element(value):
    match=re.match(r'^([A-Z][a-z]?)',str(value))
    if not match:raise ValueError('An atom or form factor has no supported chemical element')
    return match[1]


def validate_sites(atoms,sites):
    for site in sites:
        found=[a for a in atoms if a['label']==site['label']]
        if len(found)!=1:raise ValueError('Selected magnetic site is absent or ambiguous: '+site['label'])
        atom=found[0]
        if element(atom['type'])!=element(site['form_factor']):raise ValueError('Magnetic form factor changes the chemical element at '+site['label'])
        if not 0<float(atom['occupancy'])<=1:raise ValueError('Selected magnetic site needs a positive physical occupancy: '+site['label'])
