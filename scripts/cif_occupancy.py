"""Strict CIF loading and an explicitly activated, value-preserving review.

An archive declaration is evidence, never permission. Native analysis must
activate the exact reviewed archive/policy in its own process first.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import warnings
from decimal import Decimal
from importlib.metadata import version
from typing import Any
from ndip_contracts import sha256_file

import numpy as np

SCHEMA = 'radar-cif-occupancy-review/v1'
MAX_EXCESS = 0.001
# Preserve pymatgen's existing fractional-site equivalence tolerance. Making
# this tighter creates duplicate images of rounded special positions (e.g.
# 0.3333/0.6667) and changes the atom inventory of otherwise valid CIFs.
SITE_TOLERANCE = 1e-4
AUTH_ENV = 'RADAR_PD_CIF_OCCUPANCY_AUTHORIZATION'
_NUMBER = re.compile(r'^([+-]?(?:\d+(?:\.\d*)?|\.\d+))(?:\(\d+\))?([eE][+-]?\d+)?$')


def runtime_signature() -> dict[str, str]:
    return {'method':SCHEMA, 'pymatgen':version('pymatgen'), 'PyCifRW':version('PyCifRW')}


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(raw: bytes) -> bytes:
    return raw.decode('utf-8-sig').replace('\r\n', '\n').replace('\r', '\n').encode('utf-8')


def policy_hash(review: dict[str, Any]) -> str:
    body = {k:v for k,v in review.items() if k != 'policy_sha256'}
    return digest(json.dumps(body, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def maximum_excess(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('Reviewed occupancy excess must be a finite scalar number')
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= MAX_EXCESS:
        raise ValueError('Reviewed occupancy excess must be between 0 and 0.001')
    return number


def _number(value: Any, name: str) -> float:
    match = _NUMBER.fullmatch(str(value).strip())
    if match is None:
        raise ValueError(f'{name}: expected a finite scalar CIF number, found {value!r}')
    token = match[1] + (match[2] or '')
    exact = Decimal(token)
    number = float(exact)
    if not math.isfinite(number):
        raise ValueError(f'{name}: nonfinite value')
    if exact != 0 and number == 0:
        raise ValueError(f'{name}: nonzero CIF value underflows the numerical representation')
    return number


def _structural_block(raw: bytes):
    import CifFile
    document = CifFile.ReadCif(io.StringIO(raw.decode('utf-8-sig')))
    blocks = [document[key] for key in document.keys() if '_atom_site_fract_x' in document[key]]
    if len(blocks) != 1:
        raise ValueError('Occupancy validation requires exactly one structural CIF block')
    return blocks[0]


def model_signature(raw: bytes, *, neutral_elements: bool = False) -> str:
    """Bind all non-cell fields, including symmetry, in a lattice derivative.

    A neutron export may replace charged atom names with the same elements.
    No other atom, symmetry, displacement or constraint field is ignored.
    """
    block = _structural_block(raw)
    cell_tags = {'_cell_length_'+x for x in ('a', 'b', 'c')}
    cell_tags |= {'_cell_angle_'+x for x in ('alpha', 'beta', 'gamma')}
    cell_tags |= {'_cell_volume'}
    if neutral_elements:
        cell_tags.add('_atom_type_oxidation_number')
    result = {}
    for key in block.keys():
        tag = str(key).lower()
        if tag in cell_tags:
            continue
        value = block[key]
        def normalized(item):
            text = str(item)
            if neutral_elements and tag in {'_atom_site_type_symbol', '_atom_type_symbol'}:
                return species_key(_species(text))
            return text
        result[tag] = [normalized(item) for item in value] if isinstance(value, (list, tuple)) else normalized(value)
    return digest(json.dumps(result, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def raw_sites(raw: bytes) -> tuple[list[dict[str, Any]], str]:
    block = _structural_block(raw)
    def column(name, default=None):
        value = block.get(name, default)
        if not isinstance(value, (list, tuple)):
            raise ValueError(f'{name}: expected one value per atom site')
        return list(value)
    labels = column('_atom_site_label')
    if not labels or len(set(map(str, labels))) != len(labels):
        raise ValueError('CIF site labels must be nonempty and unique')
    columns = [column('_atom_site_fract_'+axis) for axis in 'xyz']
    symbols = column('_atom_site_type_symbol', [re.match(r'^([A-Z][a-z]?)',str(label))[1] if re.match(r'^([A-Z][a-z]?)',str(label)) else str(label) for label in labels])
    occupancies = column('_atom_site_occupancy', ['1']*len(labels))
    if any(len(values) != len(labels) for values in [*columns, symbols, occupancies]):
        raise ValueError('CIF atom-site columns have different lengths')
    # Preserve the separate oxidation-number loop as well as charged site
    # tokens. Ignoring it changes the species used for X-ray scattering.
    oxidation = {}
    if '_atom_type_oxidation_number' in block:
        types = column('_atom_type_symbol')
        numbers = column('_atom_type_oxidation_number')
        if len(types) != len(numbers):
            raise ValueError('CIF atom-type oxidation columns have different lengths')
        oxidation = {str(key): _number(value, f'{key} oxidation number') for key, value in zip(types, numbers)}
    rows = []
    for i,label in enumerate(labels):
        occ = _number(occupancies[i], f'{label} occupancy')
        if occ < 0 or occ > 1 + MAX_EXCESS:
            raise ValueError(f'{label}: raw occupancy {occ:g} is outside 0..1.001')
        xyz = [_number(values[i], f'{label} coordinate') for values in columns]
        symbol = str(symbols[i])
        resolved = symbol
        if symbol in oxidation:
            number = oxidation[symbol]
            element = species_key(_species(symbol))
            resolved = (element if element in {'D','T'} and number == 0
                        else f'{element}{abs(number):g}{"+" if number >= 0 else "-"}')
        rows.append({'label':str(label), 'symbol':symbol, 'resolved_symbol':resolved, 'occupancy':occ, 'xyz':xyz})
    # The signature deliberately excludes cell lengths/angles.
    # A reviewed lattice-only derivative must still carry identical atom data.
    signature_rows = [{key:value for key,value in row.items() if key != 'resolved_symbol'} for row in rows]
    for row in signature_rows:
        row['symbol'] = species_key(_species(row['symbol']))
    signature = digest(json.dumps(signature_rows, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
    return rows, signature


def _species(symbol: str):
    from pymatgen.core.periodic_table import DummySpecies, get_el_sp
    ion = re.fullmatch(r'([A-Z][a-z]?)([+-])(\d+(?:\.\d+)?)', symbol)
    value = ion[1]+ion[3]+ion[2] if ion else symbol
    try:
        species = get_el_sp(value)
        if isinstance(species, DummySpecies):
            raise ValueError('Dummy species are not supported')
        return species
    except Exception as exc:
        raise ValueError(f'Reviewed occupancy cannot preserve atom type {symbol!r}') from exc


def species_key(species) -> str:
    """Keep named isotopes distinct while aggregating oxidation states."""
    name = getattr(species, 'name', None)
    return name if name in {'D', 'T'} else species.symbol


def species_inventory(structure) -> dict[str, float]:
    inventory = {}
    for species, amount in structure.composition.items():
        key = species_key(species)
        inventory[key] = inventory.get(key, 0.0) + float(amount)
    return inventory


def _expanded(raw: bytes, rows: list[dict[str, Any]], limit: float):
    """Construct occupancies from raw values, not sanitized pymatgen species."""
    from pymatgen.io.cif import CifParser
    from pymatgen.core.sites import PeriodicSite
    from pymatgen.core import Composition
    parser = CifParser.from_str(raw.decode('utf-8-sig'), frac_tolerance=0, site_tolerance=SITE_TOLERANCE)
    with warnings.catch_warnings(record=True) as parser_warnings:
        warnings.simplefilter('always')
        structures = parser.parse_structures(primitive=False, check_occu=False)
    if len(structures) != 1:
        raise ValueError('Reviewed occupancy requires exactly one parsed structure')
    structure = structures[0]
    coords = np.asarray(structure.frac_coords)
    expected = [dict() for _ in structure]
    labels = [set() for _ in structure]
    for row in rows:
        sp = _species(row['resolved_symbol'])
        indices = set()
        for op in parser.symmetry_operations:
            position = np.asarray(op.operate(row['xyz']))
            delta = coords-position
            delta -= np.round(delta)
            matches = np.flatnonzero(np.max(np.abs(delta), axis=1) <= SITE_TOLERANCE + 1e-12)
            if len(matches) != 1:
                raise ValueError(f"{row['label']}: symmetry-expanded site cannot be matched uniquely")
            indices.add(int(matches[0]))
        if not indices:
            raise ValueError(f"{row['label']}: missing symmetry support")
        for index in indices:
            expected[index][sp] = expected[index].get(sp, 0.0) + row['occupancy']
            labels[index].add(row['label'])
    affected=[]; remove=[]
    for index,values in enumerate(expected):
        if not labels[index]:
            raise ValueError('Parser produced an atom site absent from the original CIF')
        total=math.fsum(values.values())
        if total > 1 + limit + 1e-12:
            raise ValueError(f"Sites {','.join(sorted(labels[index]))}: occupancy total {total:.12g} exceeds reviewed limit {1+limit:.12g}")
        if total > 1 + 1e-12:
            components = {}
            for species, value in values.items():
                key = species_key(species)
                components[key] = components.get(key, 0.0) + value
            affected.append({'labels':sorted(labels[index]),'fractional_coordinates':coords[index].tolist(),
                             'total':total,'components':components})
        nonzero={sp:value for sp,value in values.items() if value != 0}
        if not nonzero:
            remove.append(index)
        else:
            # skip_checks allows the reviewed small excess; every value was
            # independently checked above. No division by total is performed.
            structure[index]=PeriodicSite(Composition(nonzero),coords[index],structure.lattice,
                properties=structure[index].properties,label=structure[index].label,skip_checks=True)
            actual = dict(structure[index].species.items())
            if any(sp not in actual or float(actual[sp]) != value
                   for sp,value in nonzero.items()):
                raise ValueError('CIF occupancy preservation failed')
    if remove:
        structure.remove_sites(remove)
    if not len(structure):
        raise ValueError('CIF contains no occupied sites')
    if affected:
        warnings.warn('Reviewed CIF occupancy rounding retained exactly; no occupancies were normalized.', stacklevel=2)
    for notice in parser_warnings:
        if 'occupancies' not in str(notice.message) and 'rescaled' not in str(notice.message):
            warnings.warn(str(notice.message), stacklevel=2)
    return structure, affected


def _validate_review_header(review: dict[str, Any], raw: bytes) -> None:
    if not isinstance(review, dict) or review.get('schema') != SCHEMA or review.get('reviewed') is not True:
        raise ValueError('Explicit reviewed CIF occupancy policy is required')
    if review.get('normalization_applied') is not False:
        raise ValueError('Occupancy normalization is not permitted')
    limit=maximum_excess(review.get('maximum_site_excess'))
    if limit <= 0 or not str(review.get('review_note') or '').strip():
        raise ValueError('A positive reviewed allowance and review note are required')
    if review.get('policy_sha256') != policy_hash(review):
        raise ValueError('CIF occupancy policy hash mismatch')
    if digest(raw) not in {review.get('source_sha256'), review.get('canonical_sha256'), review.get('derived_sha256')}:
        raise ValueError('Reviewed CIF source hash mismatch')
    rows,signature=raw_sites(raw)
    if signature != review.get('atom_signature'):
        raise ValueError('Reviewed CIF atom/occupancy signature mismatch')
    if model_signature(raw) != review.get('model_signature'):
        raise ValueError('Reviewed CIF non-cell model/symmetry signature mismatch')
    if model_signature(raw,neutral_elements=True) != review.get('nuclear_model_signature'):
        raise ValueError('Reviewed CIF nuclear model/symmetry signature mismatch')


def _validate_evidence(review, structure, affected) -> None:
    if review.get('affected_sites') != affected or review.get('composition') != species_inventory(structure):
        raise ValueError('Reviewed CIF evidence does not match the complete source structure')


def validate_review(review: dict[str, Any], raw: bytes) -> dict[str, Any]:
    _validate_review_header(review, raw)
    rows, _ = raw_sites(raw)
    structure, affected = _expanded(raw, rows, maximum_excess(review['maximum_site_excess']))
    _validate_evidence(review, structure, affected)
    return review


def make_review(raw: bytes, limit: float, note: str) -> dict[str, Any]:
    limit=maximum_excess(limit)
    if limit <= 0 or not isinstance(note,str) or not note.strip():
        raise ValueError('An explicit positive allowance and review note are required')
    rows,signature=raw_sites(raw)
    structure,affected=_expanded(raw,rows,limit)
    review={'schema':SCHEMA,'reviewed':True,'maximum_site_excess':limit,'review_note':note.strip(),
            'builder_validation_runtime':runtime_signature(),
            'source_sha256':digest(raw),'canonical_sha256':digest(canonical_bytes(raw)),
            'atom_signature':signature,'model_signature':model_signature(raw),
            'nuclear_model_signature':model_signature(raw,neutral_elements=True),
            'normalization_applied':False,'affected_sites':affected,
            'composition':species_inventory(structure)}
    review['policy_sha256']=policy_hash(review)
    return review


def parse_structure(raw: bytes, review: dict[str, Any] | None = None):
    rows,_=raw_sites(raw)
    if review is None:
        # Rebuild from raw occupancies even in strict mode: the upstream parser
        # can otherwise silently discard negative/zero components or clamp tiny
        # positive values. The unreviewed site-total limit remains exactly one.
        return _expanded(raw, rows, 0.0)[0]
    _validate_review_header(review,raw)
    structure,affected = _expanded(raw,rows,maximum_excess(review['maximum_site_excess']))
    _validate_evidence(review,structure,affected)
    structure.properties['radar_occupancy_review'] = {
        'policy_sha256':review['policy_sha256'], 'source_sha256':review['source_sha256']}
    return structure


def load_structure(path: str | Path, review: dict[str, Any] | None = None):
    return parse_structure(Path(path).read_bytes(),review)


def sidecar(path: str | Path) -> Path:
    return Path(str(path)+'.occupancy-review.json')


def write_review(path: str | Path, review: dict[str, Any]) -> None:
    validate_review(review,Path(path).read_bytes())
    sidecar(path).write_text(json.dumps(review,sort_keys=True,indent=2,allow_nan=False)+'\n')


def authorized_review(path: str | Path) -> dict[str, Any] | None:
    """Read a review only if this analysis process explicitly activated it."""
    authorization=os.environ.get(AUTH_ENV)
    if not authorization:
        return None
    ledger=json.loads(Path(authorization).read_text())
    source=Path(path).resolve()
    root=Path(ledger['root']).resolve()
    if not source.is_relative_to(root):
        return None
    candidate=sidecar(source)
    if not candidate.exists():
        raw=source.read_bytes()
        matches=[r for r in ledger.get('approved_policies',{}).values()
                 if digest(raw) in {r.get('source_sha256'),r.get('canonical_sha256')}]
        if len(matches)>1:raise ValueError('Ambiguous activated occupancy review')
        return validate_review(matches[0],raw) if matches else None
    review=json.loads(candidate.read_text());validate_review(review,source.read_bytes())
    authority=review.get('root_policy_sha256',review['policy_sha256'])
    allowed=ledger.get('approved_policies',{})
    if authority not in allowed:
        raise ValueError('CIF occupancy policy was not activated for this configuration')
    root_review=allowed[authority]
    if (review['atom_signature'] != root_review['atom_signature']
        or review['maximum_site_excess'] != root_review['maximum_site_excess']
        or review['nuclear_model_signature'] != root_review['nuclear_model_signature']):
        raise ValueError('Derived CIF does not preserve the activated occupancy review')
    return review


def load_authorized_structure(path: str | Path):
    return load_structure(path,authorized_review(path))


def derive_review(source: str | Path, destination: str | Path, *, neutral_elements: bool = False) -> None:
    review=authorized_review(source)
    if review is None:
        return
    raw=Path(destination).read_bytes();rows,signature=raw_sites(raw)
    if signature != review['atom_signature']:
        raise ValueError('Lattice derivative changed source atom sites or occupancies')
    expected = review['nuclear_model_signature'] if neutral_elements else review['model_signature']
    if model_signature(raw,neutral_elements=neutral_elements) != expected:
        raise ValueError('Lattice derivative changed source non-cell fields or symmetry')
    structure,affected = _expanded(raw,rows,review['maximum_site_excess'])
    _validate_evidence(review,structure,affected)
    derived={**review,'derived_sha256':digest(raw),'model_signature':model_signature(raw),
             'root_policy_sha256':review.get('root_policy_sha256',review['policy_sha256']),
             'parent_policy_sha256':review['policy_sha256'],'operation':'verified lattice-only derivative'}
    derived['policy_sha256']=policy_hash(derived)
    write_review(destination,derived)


def cache_identity(source: str | Path | None) -> tuple:
    """Invalidate in-process caches when bytes, policy or activation changes."""
    authorization = os.environ.get(AUTH_ENV)
    authority = digest(Path(authorization).read_bytes()) if authorization else None
    if source is None:
        return (None, authority)
    path = Path(source).resolve(strict=True)
    declaration = sidecar(path)
    return (str(path), digest(path.read_bytes()),
            digest(declaration.read_bytes()) if declaration.exists() else None, authority)


def same_site_values(left, right) -> bool:
    def values(structure):
        return sorted((tuple(np.mod(site.frac_coords, 1.0).round(10)),
                       tuple(sorted((species_key(sp),float(n)) for sp,n in site.species.items()))) for site in structure)
    return values(left) == values(right)


def policy_set_hash(reviews: dict[str, Any]) -> str:
    return digest(json.dumps({key:value['policy_sha256'] for key,value in reviews.items()},
                             sort_keys=True,separators=(',', ':')).encode())


def canonical_activation(value: dict[str, Any] | None) -> dict[str, Any]:
    """Portable permission fields; a UI selection key is not scientific consent."""
    if value is None or value == {}:
        return {}
    fields={'acknowledged','archive_sha256','policy_set_sha256','allow_incomplete'}
    if not isinstance(value,dict) or set(value)-fields-{'source_key'}:
        raise ValueError('Invalid library review activation fields')
    if value.get('acknowledged') is not True:
        raise ValueError('Library review requires explicit acknowledgment')
    for key in ('archive_sha256','policy_set_sha256'):
        if not isinstance(value.get(key),str) or not re.fullmatch(r'[a-f0-9]{64}',value[key]):
            raise ValueError('Library review requires exact archive and policy hashes')
    incomplete=value.get('allow_incomplete',False)
    if not isinstance(incomplete,bool):
        raise ValueError('Incomplete-library permission must be boolean')
    return {key:value[key] for key in fields if key!='allow_incomplete'} | {'allow_incomplete':incomplete}


def library_failures_requiring_review(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Only exact producer-declared duplicate outcomes are ordinary deduping."""
    failures = manifest.get('failures') or []
    if not isinstance(failures,list) or any(not isinstance(row,dict) for row in failures):
        raise ValueError('Invalid library failure declarations')
    def duplicate(row):
        reason = row.get('error', row.get('reason'))
        return (reason in {'duplicate structure file', 'phase id already exists in base catalog'}
                or isinstance(reason,str) and reason.startswith('phase already exists in base database as ')
                and bool(reason.removeprefix('phase already exists in base database as ').strip()))
    return [row for row in failures if not duplicate(row)]


def activate_archive(archive: str | Path, database_root: str | Path, work_root: str | Path,
                     activation: dict[str, Any] | None) -> dict[str, Any] | None:
    """Bind a user configuration to this exact archive before scientific use."""
    os.environ.pop(AUTH_ENV,None)
    database_root=Path(database_root).resolve();work_root=Path(work_root).resolve()
    if not database_root.is_relative_to(work_root):
        raise ValueError('Reviewed database is outside this analysis workspace')
    manifest_path=database_root/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    reviews=manifest.get('occupancy_reviews') or {}
    if not isinstance(reviews,dict):
        raise ValueError('Invalid library occupancy declarations')
    failures=library_failures_requiring_review(manifest)
    if not reviews and not failures and not activation:
        return None
    activation=canonical_activation(activation)
    if not activation:
        raise ValueError('This library requires explicit review of its occupancy policy or incomplete build before analysis')
    archive_sha=sha256_file(archive);set_sha=policy_set_hash(reviews)
    if activation.get('archive_sha256') != archive_sha or activation.get('policy_set_sha256') != set_sha:
        raise ValueError('Library changed after review; review the selected archive again')
    if failures and activation.get('allow_incomplete') is not True:
        raise ValueError('The selected library has failed source structures; explicitly review the incomplete build')
    mapping=json.loads((database_root/'cif_map.json').read_text()) if reviews else {}
    approved={}
    for phase,review in reviews.items():
        source=(database_root/mapping[phase]).resolve()
        if not source.is_relative_to(database_root):
            raise ValueError('Reviewed CIF mapping escaped the selected archive')
        validate_review(review,source.read_bytes())
        write_review(source,review)
        approved[review['policy_sha256']]=review
    ledger={'schema':'radar-cif-occupancy-activation/v1','root':str(work_root),
            'archive_sha256':archive_sha,'policy_set_sha256':set_sha,'approved_policies':approved}
    destination=work_root/'cif-occupancy-activation.json'
    destination.write_text(json.dumps(ledger,sort_keys=True,indent=2,allow_nan=False)+'\n')
    os.environ[AUTH_ENV]=str(destination)
    return {'archive_sha256':archive_sha,'policy_set_sha256':set_sha,
            'reviews':reviews,'verification':'raw and symmetry-expanded source evidence recomputed',
            'validator_runtime':runtime_signature(),
            'incomplete_build_accepted':bool(failures),
            'warning':'Reviewed CIF occupancy rounding is preserved without normalization; it does not certify specimen composition.' if reviews else 'An incomplete candidate library was explicitly reviewed.'}
