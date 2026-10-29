"""Bind declared magnetic form factors to the actual occupied parent sites."""
import re
import math


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


def sites_from_inputs(labels, form_factors, lande_g):
    """Preserve site-specific g values when several ionic species are selected."""
    if not labels or len(set(labels)) != len(labels):
        raise ValueError('Select unique magnetic parent sites')
    def mapping(value, title):
        result = {}
        for part in str(value).split(';'):
            if not part.strip():
                continue
            if part.count('=') != 1:
                raise ValueError('Use atom=value entries for ' + title)
            key, item = (x.strip() for x in part.split('='))
            if key in result or not key or not item:
                raise ValueError('Duplicate or empty site entry in ' + title)
            result[key] = item
        if set(result) != set(labels):
            raise ValueError('Supply ' + title + ' for exactly the selected sites')
        return result
    forms = mapping(form_factors, 'ionic form factors')
    gs = mapping(lande_g, 'Landé g') if '=' in str(lande_g) else dict.fromkeys(labels, lande_g)
    sites = []
    for label in labels:
        try:
            g = float(gs[label])
        except (ValueError, TypeError):
            raise ValueError('Landé g must be a finite positive number') from None
        if isinstance(gs[label], bool) or not math.isfinite(g) or not 0 < g <= 10:
            raise ValueError('Landé g must be a finite positive number at most 10')
        sites.append(dict(label=label, form_factor=forms[label], lande_g=g))
    return sites
