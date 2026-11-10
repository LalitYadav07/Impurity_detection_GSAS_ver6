"""Reuse authoritative discovery adapters from an inspected nuclear CIF.

The service credential remains in this coordinator. Native workers receive
only captured structure data and the resulting data-only subgroup catalogs.
"""
import copy
import json
import os
from pathlib import Path
import uuid
from fractions import Fraction

from gpx_recipe.execution import write_json
from gpx_recipe.symmetry.compiler import file_sha, package_signature
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.discovery import discover_magnetic
from gpx_recipe.symmetry.isodistort_discovery import discover_structural
from .native_operations import intent_lock


def k_vector(value):
    if not isinstance(value,(list,tuple)) or len(value)!=3 or not all(isinstance(v,str) for v in value):
        raise ValueError('Use three exact rational k components in the inspected parent basis')
    result=[]
    for component in value:
        try:part=Fraction(component)
        except (ValueError,ZeroDivisionError):raise ValueError('Use finite rational k components') from None
        if part.denominator>96 or abs(part)>8:raise ValueError('The requested k vector is outside the supported rational scope')
        result.append(str(part))
    return result


def require_structural_coverage(catalog, *, policy=None):
    """Enumeration and successful construction are separate coverage claims."""
    coverage=catalog.get('coverage',{})
    selected_policy=coverage.get('opd_policy')
    rows=coverage.get('candidates')
    candidates=catalog.get('candidates')
    if (coverage.get('discovery_complete') is not True
            or selected_policy not in {'all','highest-symmetry-per-irrep'}
            or (policy is not None and selected_policy!=policy)
            or not isinstance(rows,list) or not isinstance(candidates,list)
            or type(coverage.get('discovered_count')) is not int
            or coverage['discovered_count']!=len(rows)
            or type(coverage.get('constructed_count')) is not int
            or coverage['constructed_count']!=len(candidates)):
        raise ValueError('Structural discovery coverage is incomplete or differs from the selected scope; retry discovery')
    ids=[row.get('id') for row in rows if isinstance(row,dict)]
    selected=[row.get('id') for row in candidates if isinstance(row,dict)]
    if (len(ids)!=len(rows) or len(selected)!=len(candidates)
            or any(not isinstance(value,str) or not value for value in ids+selected)
            or len(set(ids))!=len(ids) or len(set(selected))!=len(selected)):
        raise ValueError('Structural discovery has an invalid or duplicate candidate inventory')
    constructed={row['id'] for row in rows if row.get('state')=='constructed'}
    allowed={'constructed'} if selected_policy=='all' else {'constructed','deferred'}
    if (not selected or constructed!=set(selected)
            or any(row.get('state') not in allowed for row in rows)):
        raise ValueError(f'Structural candidate construction is incomplete ({len(constructed)} / {len(rows)} returned); retry discovery before preparation')
    return coverage


class PreparationDiscovery:
    def __init__(self,workspace):
        self.workspace=workspace
        self.root=workspace.root/'discovery';self.root.mkdir(parents=True,exist_ok=True)

    def discover(self,reference,kind,k,*,opd_policy='highest-symmetry-per-irrep',maximum=100):
        if kind not in {'magnetic','structural'}:raise ValueError('Choose magnetic or structural preparation')
        if opd_policy not in {'all','highest-symmetry-per-irrep'} or type(maximum) is not int or not 1<=maximum<=500:
            raise ValueError('Declare a supported structural candidate scope')
        k=k_vector(k);context=self.workspace.references(reference)
        identity=dict(source_sha256=context['source_sha256'],parent=context['parent'],kind=kind,k=k,
            implementation_sha256=package_signature(),opd_policy=opd_policy if kind=='structural' else None,
            maximum=maximum if kind=='structural' else None)
        key=digest(identity);root=self.root/key;root.mkdir(exist_ok=True)
        path=root/'receipt.json'
        with intent_lock(path):
            if path.exists():
                if path.is_symlink() or path.stat().st_size>8*1024*1024:raise ValueError('Saved preparation discovery is oversized or changed')
                receipt=json.loads(path.read_text())
                if (set(receipt)!={'schema','identity','catalog','directory','files'}
                        or receipt['schema']!='radar-preparation-discovery/v1' or receipt['identity']!=identity
                        or not isinstance(receipt['directory'],str) or not isinstance(receipt['files'],dict)):
                    raise ValueError('Saved discovery belongs to another parent, k or implementation')
                directory=(root/receipt['directory']).resolve()
                if not directory.is_relative_to(root):raise ValueError('Saved discovery escaped its workspace')
                for name,checksum in receipt['files'].items():
                    candidate=directory/name
                    if candidate.is_symlink() or not candidate.resolve().is_relative_to(directory) or file_sha(candidate)!=checksum:
                        raise ValueError('An authoritative discovery artifact changed')
                if receipt['catalog']!=json.loads((directory/'catalog.json').read_text()):
                    raise ValueError('Cached discovery differs from its authoritative catalog file')
                complete=receipt['catalog'].get('coverage',{}).get('discovery_complete') is True
                if complete and kind=='structural':
                    try:require_structural_coverage(receipt['catalog'],policy=opd_policy)
                    except ValueError:complete=False
                if complete:
                    return dict(catalog=copy.deepcopy(receipt['catalog']),directory=str(directory),cached=True)
            directory=root/('attempt-'+uuid.uuid4().hex)
            if kind=='magnetic':
                directory.mkdir()
                catalog=discover_magnetic(context['parent'],k,api_key=os.environ.get('BCS_API_KEY',''))
                write_json(directory/'catalog.json',catalog)
            else:
                source=self.workspace.queue(reference).source;runtime=self.workspace.runtime
                def native(operation,payload):
                    return runtime.native(operation,payload,directory,source=source,timeout=600)[0]
                def matcher(cif,vector,options):
                    return native('match',dict(cif=cif,k=k,options=options))
                def inventory_many(children):
                    rows=[]
                    for offset in range(0,len(children),32):
                        rows.extend(native('modes-many',dict(children=[str(p) for p in children[offset:offset+32]],workers=4))['items'])
                    return rows
                catalog=discover_structural(context['source'],context['parent'],k,directory,
                    matcher=matcher,inventory=lambda child:native('modes',dict(child=str(child),
                        temporary=str(directory/(child.stem+'.gpx')))),inventory_many=inventory_many,
                    opd_policy=opd_policy,maximum=maximum)
            if (catalog.get('schema')!='radar-symmetry-catalog/v1' or catalog.get('k')!=k
                    or catalog.get('parent_setting_sha256')!=context['parent']['setting_sha256']):
                raise ValueError('Discovery returned a different nuclear parent or k vector')
            files={str(p.relative_to(directory)):file_sha(p) for p in sorted(directory.rglob('*'))
                if p.is_file() and p.suffix in {'.cif','.json','.txt'}}
            receipt=dict(schema='radar-preparation-discovery/v1',identity=identity,catalog=catalog,
                directory=directory.name,files=files)
            write_json(path,receipt)
            return dict(catalog=copy.deepcopy(catalog),directory=str(directory),cached=False)
