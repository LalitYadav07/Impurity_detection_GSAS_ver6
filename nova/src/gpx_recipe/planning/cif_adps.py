"""Traceable B-to-U CIF conversion before native phase import.

The reviewed native importer ignores anisotropic B fields. Write a derived
CIF using U = B/(8*pi^2), preserving loops and the exact original input file.
This converts nominal values; original uncertainty strings remain in the source
artifact and are not propagated into an uncertainty model by this adapter.
"""
from pathlib import Path
import hashlib
import re
import numpy as np
from .nuclear import number

VERSION = 'powgen-cif-displacement-normalization/1'


def prepare_cif(path, target):
    path, target = Path(path).resolve(), Path(target).resolve()
    if path.stat().st_size > 20 * 1024**2:
        raise ValueError('CIF displacement input exceeds its byte budget')
    data = path.read_bytes()
    if len(data) > 20 * 1024**2:
        raise ValueError('CIF displacement input exceeds its byte budget')
    if not re.search(rb'_atom_site_(?:aniso_)?b_(?:iso_or_equiv|11|22|33|12|13|23)\b', data, re.I):
        return path, None
    if target == path:
        raise ValueError('Derived displacement CIF must not overwrite its source')
    import CifFile
    cif = CifFile.ReadCif(str(path))
    conversions = []
    def values(value):
        return [value] if isinstance(value, str) else list(value)
    for name in cif.keys():
        block = cif[name]
        for key in list(block.keys()):
            if not re.fullmatch(r'_atom_site_(?:aniso_)?b_(?:iso_or_equiv|11|22|33|12|13|23)', key, re.I):
                continue
            ukey = key.lower().replace('_b_', '_u_')
            original = block[key]
            converted = np.array([number(v) / (8 * np.pi**2) for v in values(original)])
            if ukey in block:
                existing = np.array([number(v) for v in values(block[ukey])])
                if existing.shape != converted.shape or not np.allclose(existing, converted, rtol=1e-6, atol=1e-10):
                    raise ValueError('Conflicting U and B displacement parameters require explicit resolution')
                del block[key]
                action = 'consistent alternate B values removed; declared U retained'
            else:
                block.ChangeTagName(key, ukey)
                converted_text = [format(value, '.17g') for value in converted]
                block[ukey] = converted_text[0] if isinstance(original, str) else converted_text
                action = 'nominal B values converted to U'
            conversions.append({'block': name, 'source_tag': key, 'derived_tag': ukey,
                'values': len(converted), 'action': action})
        for key in ('_atom_site_adp_type', '_atom_site_thermal_displace_type'):
            if key in block:
                original = block[key]
                converted = [{'biso': 'Uiso', 'bani': 'Uani'}.get(str(v).lower(), v) for v in values(original)]
                block[key] = converted[0] if isinstance(original, str) else converted
        aniso = [f'_atom_site_aniso_u_{i}{j}' for i,j in ((1,1),(2,2),(3,3),(1,2),(1,3),(2,3))]
        if any(key in block for key in aniso) and not all(key in block for key in aniso):
            raise ValueError('Six explicit anisotropic U or B components are required')
    if not conversions:
        return path, None
    with target.open('x') as stream:
        stream.write(cif.WriteOut())
    record = {'version': VERSION, 'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_cif_sha256': hashlib.sha256(data).hexdigest(),
        'derived_cif_name': target.name, 'derived_cif_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
        'conversions': conversions, 'formula': 'U_ij = B_ij / (8*pi^2)',
        'original_cif_preserved': True, 'nominal_values_only': True,
        'adp_uncertainty_propagated': False, 'experimental_accuracy_certified': False}
    return target, record
