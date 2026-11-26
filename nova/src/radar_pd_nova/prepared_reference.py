"""Small, checked UI hints; native admission still decides compatibility."""
from pathlib import Path


def requires_measured_reference(library):
    """Only call a reference mandatory when every ready model declares it.

    Mixed banks and older metadata remain subject to native admission. This
    reads bounded model metadata, never weights or diffraction arrays.
    """
    from gpx_recipe.preparation.structural_attention_weights import FIXED_REFERENCE_KIND
    from gpx_recipe.preparation.structural_ranking_hints import _metadata

    rows = [row for row in library.document['entries']
            if row['model']['kind'] in {'structural-training', 'structural-ranking-model'}]
    if not rows or len(rows) > 8 or any(row['state'] != 'prepared' for row in rows):
        return False
    kinds = set()
    for row in rows:
        names = [item['path'] for item in row.get('files', [])
                 if Path(item['path']).name == 'model.json']
        if len(names) != 1:
            return False
        paths = library.verify_assets(names)
        metadata, _ = _metadata(paths[names[0]])
        kinds.add(metadata.get('reference_kind'))
    return kinds == {FIXED_REFERENCE_KIND}
