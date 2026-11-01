"""Explicit phase corrections on a new GPX; never an automatic phase search."""
from copy import deepcopy
from hashlib import sha256
import math
from pathlib import Path

from .contracts import exact_keys

SCHEMA = 'radar-phase-correction/v1'


def validate_edit(context, request):
    exact_keys(request, {'schema', 'source_sha256', 'model_signature', 'operation', 'phase', 'name', 'cif', 'cif_sha256', 'links', 'reason'},
               {'schema', 'source_sha256', 'model_signature', 'operation', 'reason'})
    if (request['schema'] != SCHEMA or request['model_signature'] != context['model_signature']
            or request['source_sha256'] != context['source']['sha256']):
        raise ValueError('Phase correction belongs to a different GPX revision. Inspect the selected model again.')
    if not isinstance(request['reason'], str) or not 3 <= len(request['reason'].strip()) <= 4000:
        raise ValueError('Record why this phase correction is needed.')
    operation = request['operation']; phases = {p['key']: p for p in context['phases']}
    if operation not in {'add', 'remove', 'replace'}:
        raise ValueError('Choose add, remove or replace.')
    if operation != 'add' and request.get('phase') not in phases:
        raise ValueError('Choose an existing phase by its inspected identity.')
    if operation == 'remove':
        if len(phases) <= 1:
            raise ValueError('Cannot remove the last phase from a refinement model.')
        if set(request) & {'cif', 'cif_sha256', 'links', 'name'}:
            raise ValueError('Removal does not accept CIFs, new names or bank assignments.')
    else:
        text = request.get('cif')
        if not isinstance(text, str) or not 1 <= len(text.encode()) <= 1024 * 1024:
            raise ValueError('Supply a UTF-8 CIF containing one structure, at most 1 MiB.')
        if sha256(text.encode()).hexdigest() != request.get('cif_sha256'):
            raise ValueError('The supplied CIF changed after selection.')
        if operation == 'replace':
            if set(request) & {'name', 'links'}:
                raise ValueError('Replacement retains the original component name and bank assignments.')
        else:
            if 'phase' in request:
                raise ValueError('Addition has no existing target phase.')
            name = request.get('name')
            if (not isinstance(name, str) or not name.strip() or name != name.strip() or len(name) > 100
                    or any(ord(c) < 32 or c in '/\\' for c in name) or name in {p['name'] for p in phases.values()}):
                raise ValueError('Use a distinct, nonempty component name (up to 100 characters).')
            links = request.get('links'); histograms = {h['key'] for h in context['histograms']}
            if not isinstance(links, list) or not links:
                raise ValueError('Explicitly select at least one bank and its initial phase scale.')
            seen = set()
            for link in links:
                exact_keys(link, {'histogram', 'scale'}, {'histogram', 'scale'})
                if link['histogram'] not in histograms or link['histogram'] in seen:
                    raise ValueError('Choose distinct existing banks.')
                scale = link['scale']
                if type(scale) not in {float, int} or not math.isfinite(scale) or scale <= 0:
                    raise ValueError('Initial phase scales must be finite and positive.')
                seen.add(link['histogram'])
    return deepcopy(request)


def edit_model(source, output, request):
    """Transactional scientific operation. Original bytes are never written."""
    from .snapshot import snapshot, gsas, json_safe, digest
    from .execution import write_json
    from .battery.preparation import array_fingerprints
    from .symmetry.adapters.gsas_cif_import import add_verified_phase

    source = Path(source).resolve(); output = Path(output).resolve()
    if output == source or output.exists():
        raise ValueError('A phase correction requires a new output GPX.')
    context = snapshot(source); request = validate_edit(context, request)
    G2 = gsas(); project = G2.G2Project(gpxfile=str(source))
    # A native constraint can reference phases by random or numeric IDs. Until
    # transfer is implemented, reject such models instead of deleting relations.
    constraints = project.data.get('Constraints', {}).get('data', {})
    if any(constraints.get(key) for key in ('Phase', 'HAP', 'Global')):
        raise ValueError('Phase correction of a model with phase/HAP/global constraints requires explicit constraint transfer, which is not supported yet.')
    phase_map = {p['key']: p for p in context['phases']}
    target = phase_map.get(request.get('phase'))
    if target and project.data.get('Restraints', {}).get('data', {}).get(target['name']):
        raise ValueError('The target phase has restraints. Explicit restraint transfer is required before replacing/removing it.')
    observations = {h.name: array_fingerprints(h) for h in project.histograms()}
    preserved = {h.name: {key: json_safe(h.data.get(key)) for key in ('Instrument Parameters', 'Sample Parameters', 'Background', 'Limits')}
                 for h in project.histograms()}
    untouched = {p.name: deepcopy(p.data) for p in project.phases() if not target or p.name != target['name']}
    old = deepcopy(project.phase(target['name']).data) if target else None
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name('editing.gpx')
    if temporary.exists():
        raise ValueError('An incomplete edit already exists in this directory; use a new edit revision.')
    project.save(str(temporary))
    name = request.get('name') if not target else target['name']
    if target:
        project.data['Phases'].pop(name)
        for row in project.names:
            if row and row[0] == 'Phases' and name in row:
                row.remove(name)
        for hist in project.histograms():
            hist.data.get('Reflection Lists', {}).pop(name, None)
        project.data.get('Restraints', {}).get('data', {}).pop(name, None)
    if request['operation'] != 'remove':
        cif = output.parent / 'supplied_phase.cif'; cif.write_text(request['cif'])
        names = list(old['Histograms']) if old else [next(h['name'] for h in context['histograms'] if h['key'] == link['histogram']) for link in request['links']]
        phase = add_verified_phase(project, cif, phasename=name, histograms=[project.histogram(n) for n in names], fmthint='CIF', useNet=False)
        if phase.name != name:
            raise ValueError('Native import changed the requested component identity.')
        if old:
            phase.data['ranId'] = old['ranId']
            for hist_name in names:
                phase.data['Histograms'][hist_name]['Use'] = old['Histograms'][hist_name]['Use']
                phase.data['Histograms'][hist_name]['Scale'] = deepcopy(old['Histograms'][hist_name]['Scale'])
        else:
            for hist_name, link in zip(names, request['links']):
                phase.data['Histograms'][hist_name]['Scale'] = [float(link['scale']), False]
    # Cached covariance and reflections describe the previous structure. New
    # structural HAP/profile/texture defaults must be reviewed after replacement.
    project.data['Covariance']['data'] = {}
    for histogram in project.histograms():
        # The old curves/R factors are for another phase hypothesis. A corrected
        # starting model must display unevaluated, not inherit its parent's fit.
        for key in histogram.residuals:
            histogram.data['data'][0].pop(key,None)
        for array in histogram.data['data'][1][3:]:
            array[:]=0.
        histogram.data['Reflection Lists']={}
    project.index_ids()
    # Do not reset unrelated phase flags/values during an edit. The existing
    # recipe executor freezes and sets the full free set before refinement.
    project.save()
    reopened = G2.G2Project(gpxfile=str(temporary))
    for hist in reopened.histograms():
        if array_fingerprints(hist) != observations[hist.name] or any(json_safe(hist.data.get(k)) != v for k, v in preserved[hist.name].items()):
            raise ValueError('Phase correction changed observed data, masks, fit limits or instrument/background settings.')
    for phase_name, data in untouched.items():
        after = deepcopy(reopened.phase(phase_name).data)
        # IDs are reindexed by GSAS-II when a phase is removed; random component
        # identity and all actual crystallographic/HAP values must remain intact.
        data.pop('pId', None); after.pop('pId', None)
        if json_safe(after) != json_safe(data):
            raise ValueError('Phase correction changed an unrelated component: ' + phase_name)
    new_context = snapshot(temporary)
    if sha256(source.read_bytes()).hexdigest() != context['source']['sha256']:
        raise ValueError('The starting GPX changed during phase correction.')
    temporary.replace(output)
    new_context['source']['name'] = output.name
    new_context.pop('context_sha256',None)
    new_context['context_sha256'] = digest(new_context)
    receipt = dict(schema=SCHEMA, status='edited_unqualified', request=request,
                   parent_sha256=context['source']['sha256'], output_sha256=sha256(output.read_bytes()).hexdigest(),
                   component=name, preserved_observations=True, preserved_bank_assignments=bool(old),calculation_status='not_yet_evaluated',
                   note='Replacement retains component name/random identity, bank use and scales. Structural HAP defaults come from the new phase; review width/texture settings. All earlier recipes must be rebuilt.')
    write_json(output.parent / 'model_edit.json', receipt)
    return new_context, receipt
