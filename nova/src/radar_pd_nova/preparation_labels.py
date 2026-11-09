"""Bounded descriptive labels from the library's captured discovery scope.

These labels describe the declared model; they never authorize scientific
reuse or replace the native parent/site/operator checks.
"""
import json
import math

from gpx_recipe.preparation.library import confined
from gpx_recipe.symmetry.isodistort_forms import _method1_catalog_group


def scope_labels(library):
    name=library.document['specification'].get('scope')
    if not name:return {}
    path=confined(library.root,name)
    if path.stat().st_size>4*1024*1024:raise ValueError('Preparation scope display exceeds its metadata budget')
    path=library.verify_assets([name])[name]
    document=json.loads(path.read_text())
    if not isinstance(document,dict) or document.get('schema')!='radar-preparation-declared-scope/v1':return {}
    labels={}
    searches=document.get('searches',[])
    if not isinstance(searches,list):return {}
    for search in searches:
        if not isinstance(search,dict):continue
        selected=search.get('selected_candidates',[]);vector=search.get('k');coverage=search.get('coverage')
        if (search.get('kind') not in {'structural','magnetic'} or not isinstance(selected,list)
                or any(not isinstance(v,str) for v in selected) or not isinstance(vector,list)
                or len(vector)!=3 or any(not isinstance(v,str) for v in vector)
                or not isinstance(coverage,dict) or not isinstance(coverage.get('candidates'),list)):continue
        selected=set(selected)
        for row in coverage['candidates']:
            if not isinstance(row,dict) or not isinstance(row.get('id'),str) or row['id'] not in selected:continue
            key=(search.get('kind'),row['id'])
            item=dict(k=vector,label=str(row.get('label',''))[:2000])
            values=labels.setdefault(key,[])
            if item not in values:values.append(item)
    return labels


def _entry_detail(library,entry,kinds):
    if entry.get('state') != 'prepared' or entry['model']['kind'] not in kinds:
        return None
    names = [row['path'] for row in entry.get('files', [])
             if row['path'].endswith('/entry.json')]
    if not names:
        return None
    if len(names) != 1:
        raise ValueError('Prepared model has ambiguous entry metadata')
    path = confined(library.root, names[0])
    if path.stat().st_size > 8 * 1024 * 1024:
        raise ValueError('Prepared entry metadata exceeds its display budget')
    path = library.verify_assets(names)[names[0]]
    document = json.loads(path.read_text())
    model = entry['model']
    if (not isinstance(document, dict)
            or document.get('schema') != 'radar-experiment-preparation-entry/v1'
            or document.get('kind') != model['kind']
            or document.get('settings') != model.get('settings')
            or document.get('grid') != model.get('grid')):
        raise ValueError('Prepared entry metadata differs from this model')
    detail = document.get('detail')
    return detail if isinstance(detail,dict) else None


def native_dimensions(library, entry):
    """Checked completed-entry counts; not authority for numerical reuse."""
    detail=_entry_detail(library,entry,{'structural','structural-examples'})
    if detail is None:return None
    counts = tuple(detail.get(k) for k in ('mode_dimensions', 'cell_dimensions'))
    if any(type(value) is not int or value < 0 for value in counts):
        return None
    return counts


def training_description(library,entry):
    detail=_entry_detail(library,entry,{'structural-training','structural-ranking-model'})
    if detail is None or 'dataset' not in detail:return None
    from gpx_recipe.preparation.structural_training_counts import validate,partition_names
    counts=validate(detail['dataset']);parts=counts['partitions']
    names=partition_names(counts)
    split=' / '.join(f"{parts[name]['clean_pairs']:,}" for name in names)
    text=(f"{counts['unique_clean_pairs']:,} unique clean pairs · "
        f"{' / '.join(names)}: {split} · "
        f"{counts['unique_simulated_only_rows']:,} additional simulated rows · "
        f"{counts['trajectory_groups']:,} separate trajectory groups")
    if 'family_coverage' in counts:
        coverage=counts['family_coverage'];training=coverage['partitions']['train']
        group=min(coverage['classes'],key=lambda c:(training[str(c)]['clean_pairs'],
            training[str(c)]['clean_trajectory_groups'],c))
        cell=training[str(group)];pairs=cell['clean_pairs'];trajectories=cell['clean_trajectory_groups']
        text+=(f" · {len(coverage['classes'])} symmetry families; least-covered SG {group}: "
            f"{pairs:,} clean training {'pair' if pairs==1 else 'pairs'} across "
            f"{trajectories:,} {'trajectory' if trajectories==1 else 'trajectories'}")
    if entry['model']['kind']=='structural-ranking-model':
        candidates=detail.get('candidate_count');families=detail.get('mode_families')
        if type(candidates) is int and candidates>0:
            text+=f" · {candidates} original constructions retained"
        if type(families) is int and families>0:
            text+=f" · {families} irrep/k displacement families"
    return text


def training_validation(library, entry):
    """Keep clean and perturbed held-out results separate in the library UI.

    These are stored simulation diagnostics, not experimental accuracy or
    authority to remove native candidates. No numerical weights are loaded.
    """
    detail = _entry_detail(library, entry, {'structural-training','structural-ranking-model'})
    if detail is None:
        return []
    test = detail.get('test', {})
    if not isinstance(test, dict):
        raise ValueError('Training validation requires typed held-out diagnostics')
    observed = test.get('observation_metrics')
    if observed is None:
        return []
    if not isinstance(observed, dict) or not isinstance(observed.get('partitions'), dict):
        raise ValueError('Training validation needs separate clean and simulated-observation results')
    result = []
    for key, title in [('clean_native', 'Clean native simulations'),
                       ('simulated_observation', 'Simulated observation variants')]:
        row = observed['partitions'].get(key)
        if row is None:
            continue
        if not isinstance(row, dict):
            raise ValueError('Invalid held-out simulation diagnostics')
        count, recall = row.get('rows'), row.get('top5')
        if type(count) is not int or count < 0:
            raise ValueError('Held-out simulation counts need nonnegative integers')
        if count == 0 and recall is None:
            continue
        if (type(recall) not in (int, float) or not math.isfinite(recall)
                or not 0 <= recall <= 1):
            raise ValueError('Held-out simulation counts and recall need finite bounded values')
        if count:
            balanced=[]
            for field,label in [('macro_top5','Class-balanced compatible hit'),('worst_class_top5','Lowest class compatible hit')]:
                if field not in row:continue
                value=row[field]
                if (type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1):
                    raise ValueError('Per-class simulation recall needs finite bounded values')
                balanced.append(f'{label}: {value:.1%}')
            if 'label_retention' in row:
                from gpx_recipe.preparation.structural_label_retention import validate
                retention=row['label_retention']
                if not isinstance(retention,dict) or 'top5' not in retention:
                    raise ValueError('Complete compatible-label retention diagnostics required')
                retained=validate(retention['top5'],rows=count)
                if retained['requested_top_k']!=5 or not math.isclose(
                        retained['any_compatible_hit_rate'],recall,rel_tol=0.,abs_tol=1e-12):
                    raise ValueError('Compatible-label retention needs the same top-five counts and depth')
                balanced=[f"Each compatible label: {retained['macro_label_retention']:.1%} average",
                    f"Lowest label: {retained['worst_class_label_retention']:.1%}",
                    f"All labels retained: {retained['all_compatible_retention']:.1%} of patterns"]
            result.append(dict(title=title, value=f'{recall:.1%}',
                count=f'{count:,} held-out patterns',detail=' · '.join(balanced)))
    return result


def entry_label(model, labels, *, dimensions=None):
    settings=model.get('settings',{});candidate=settings.get('candidate',{})
    identifier=candidate.get('id','');kind=model['kind']
    if kind in {'structural-training','structural-ranking-model'}:
        return dict(model='Imported structural family ranking model' if kind=='structural-ranking-model' else 'Structural family ranking model',
            k='All selected k families',candidate='',
            coordinates=('Read-only trained data; native compatibility is checked for each scan' if kind=='structural-ranking-model'
                else f"Trains after {len(settings['dependencies'])} native example blocks"),basis='',origin='')
    structural=kind in {'structural','structural-examples'}
    label=candidate.get('magnetic_space_group') or candidate.get('irrep') or identifier or 'Parent and profile'
    vector=settings.get('k',[])
    choices=labels.get(('structural' if structural else kind,identifier),[])
    if vector:choices=[row for row in choices if row['k']==vector]
    if len(choices)==1:
        if not vector:vector=choices[0]['k']
        if structural:
            group=_method1_catalog_group(choices[0]['label'])
            if group:label=group['catalog_child_space_group_symbol']+' · '+label
    coordinates = ''
    if structural:
        modes = candidate.get('modes')
        if dimensions is not None:
            coordinates = f'{dimensions[0]} native displacement coordinates · {dimensions[1]} lattice coordinates'
        elif isinstance(modes, list) and modes:
            coordinates = f'{len(modes)} displacement modes in discovery; native count pending'
        else:
            coordinates = 'Native coordinate count unavailable'
        if kind=='structural-examples':
            first,last=settings['trajectory_slice']
            label+=' · training patterns'
            coordinates=f"Trajectories {first+1}–{last} of {settings['trajectories']} · "+coordinates
    return dict(model=label,k=', '.join(vector),candidate=identifier,
        coordinates=coordinates,
        basis=str(candidate.get('transformation_matrix',[])),origin=str(candidate.get('origin_shift',[])))
