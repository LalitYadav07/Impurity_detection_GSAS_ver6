#!/usr/bin/env python3
"""Native forward controls using explicit, provenance-bearing reference inputs.

This verifies implementation invariants, not agreement with measured transitions.
Run in the GSAS-II Python with nova/src on PYTHONPATH. The manifest contains
magnetic and structural cases, each with scenario, candidate and exact k. The
magnetic case adds sites; the structural case adds the constructed child_cif.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpx_recipe.planning.magnetic import execute
from gpx_recipe.planning.worker import run


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(args.manifest.read_text())
    for case in manifest.values():
        for owner,key in [(case['scenario'],'cif'),(case,'child_cif')]:
            if owner.get(key) and not Path(owner[key]).is_absolute():
                owner[key]=str((args.manifest.parent/owner[key]).resolve())
    results = []
    for kind in ('magnetic', 'structural'):
        case = manifest[kind]
        root = args.output / kind
        root.mkdir()
        scenario = {**case['scenario'], 'response_mode': 'profile_only'}
        receipt = run(scenario, root/'parent')
        source = root/'parent/model/forward.gpx'
        base = {'source_gpx': str(source.resolve()), 'source_gpx_sha256': digest(source)}
        parent = execute('context', base, root/'context')['phases'][0]
        request = {**base, 'parent_setting_sha256': parent['setting_sha256'],
                   'candidate': case['candidate'], 'k': case['k']}
        if kind == 'magnetic':
            request['sites'] = case['sites']
            prepared = execute('prepare', request, root/'initial-moments')
            coordinates = {p['name']: p['value'] for p in prepared['parameters']}
            operation = 'prepare'
        else:
            request['child_cif'] = case['child_cif']
            inventory = execute('structural-modes', {**base, 'child_cif':case['child_cif']}, root/'inventory')
            request['candidate'] = {**case['candidate'], 'modes': inventory['modes']}
            coordinates = {name: .01 for name in inventory['modes']}
            operation = 'structural-prepare'
        if not coordinates:
            raise AssertionError('The transition fixture has no independent coordinates')
        curves = {}
        for factor in (0, 1, 2):
            values = {name: factor*value for name,value in coordinates.items()}
            key = 'parameters' if kind == 'magnetic' else 'mode_values'
            prepared = execute(operation, {**request, key:values}, root/f'prepared-{factor}')
            bound = {**prepared, 'parent_cif_sha256':receipt['inputs']['cif']['sha256'],
                     'profile_sha256': receipt['profile_sha256']}
            result = run({**scenario, 'prepared_'+kind:bound}, root/f'simulated-{factor}')
            with np.load(root/f'simulated-{factor}/curves.npz', allow_pickle=False) as data:
                curves[factor] = {name:data[name].copy() for name in data.files}
            (root/f'prepared-{factor}/result.json').write_text(json.dumps(prepared,indent=2)+'\n')
            if not prepared['no_refinement'] or result['experimental_accuracy_certified']:
                raise AssertionError('Forward controls must not refine or certify experimental accuracy')
        with np.load(root/'parent/curves.npz',allow_pickle=False) as data:
            original = data['expected'].copy()
        scale = max(float(np.max(abs(original))), 1e-30)
        zero_error = float(np.max(abs(curves[0]['expected']-original))/scale)
        if zero_error > 1e-5:
            raise AssertionError(f'{kind} zero-coordinate control failed: {zero_error}')
        row = {'kind':kind, 'coordinates':coordinates, 'zero_relative_max_error':zero_error,
               'no_refinement':True, 'implementation_controls_passed':True}
        if kind == 'magnetic':
            for factor in (1,2):
                np.testing.assert_allclose(curves[factor]['nuclear'],original,rtol=1e-9,atol=1e-9)
            magnetic_scale = max(float(np.max(curves[1]['magnetic'])),1e-30)
            error = float(np.max(abs(curves[2]['magnetic']-4*curves[1]['magnetic']))/magnetic_scale)
            if magnetic_scale <= 1e-20 or error > 1e-7:
                raise AssertionError('Independent moment edits do not obey squared-moment intensity')
            row['doubled_moment_relative_max_error'] = error
        else:
            difference = float(np.linalg.norm(curves[1]['expected']-original)/np.linalg.norm(original))
            if difference < 1e-8:
                raise AssertionError('Nonzero structural modes produced no observable pattern change')
            row['nonzero_mode_relative_profile_change'] = difference
        results.append(row)
    report = {'schema':'powgen-planning-transition-controls/v1', 'results':results,
              'manifest_sha256':digest(args.manifest), 'experimental_accuracy_certified':False,
              'scope':'Native forward invariants only; no measured transition comparison',
              'script_sha256':digest(__file__)}
    package=Path(__file__).parents[1]/'nova/src/gpx_recipe'
    report['source_sha256']={name:digest(package/name) for name in
        ('planning/worker.py','planning/gsas.py','planning/magnetic.py','planning/contracts.py',
         'symmetry/native_structural.py','live/validated_backend.py')}
    (args.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
