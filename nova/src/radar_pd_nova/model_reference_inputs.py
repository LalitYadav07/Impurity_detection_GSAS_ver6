"""Exact current-source reference aliases; never a relaxed scientific hash."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gpx_recipe.symmetry.compiler import prepare_shared_inputs
from gpx_recipe.symmetry.contracts import digest, validate_protocol
from gpx_recipe.symmetry.templates import matches, validate_template

SCHEMA = 'radar-model-reference-input-alias/v1'


def bound_reference_protocol(template, source):
    """Return the one supported binding, or retain ordinary separate packets.

    A multi-acquisition/following template still publishes its original packet.
    No source, model, catalog, policy, or implementation field may change here.
    """
    if (template.get('schema') != 'radar-model-template/v1'
            or template.get('future') is not False
            or template.get('existing_ids') != [source.get('id')]
            or not isinstance(source.get('dataset_id'), str) or not source['dataset_id']):
        return None
    validate_template(template)
    protocol = template['protocol']
    if (protocol.get('acquisition_id') != source.get('id')
            or protocol['source']['sha256'] != source.get('source_sha256')
            or not matches(template, source, template['origin'])
            or 'template_sha256' in protocol):
        return None
    return validate_protocol(dict(copy.deepcopy(protocol), template_sha256=digest(template)))


def reference_alias(template, protocol, source, history):
    expected = bound_reference_protocol(template, source)
    if expected is None or protocol != expected or not isinstance(history, str) or not history:
        raise ValueError('Reference alias requires the exact current-source template binding')
    return dict(schema=SCHEMA, history_id=history, template_sha256=digest(template),
        reference_protocol_sha256=digest(template['protocol']),
        bound_protocol_sha256=digest(protocol),
        source={key:source[key] for key in ('id', 'dataset_id', 'source_sha256')})


def restore_reference_aliases(document, template, inputs, campaign_root):
    """Rebuild a reference only after the complete bound packet was verified.

    Legacy input maps are untouched. An alias has its own version and exact
    source/template binding, rather than treating two different hashes as equal.
    Prepared files retain their byte hashes and require current native checks.
    """
    transport = document['transport']
    aliases = transport.get('reference_aliases', {})
    if not isinstance(aliases, dict) or len(aliases) > 1:
        raise ValueError('Invalid saved reference input aliases')
    for reference_sha, alias in aliases.items():
        if not isinstance(alias, dict):raise ValueError('Invalid saved reference input alias')
        bound_sha = alias.get('bound_protocol_sha256')
        if not isinstance(bound_sha, str):raise ValueError('Invalid saved bound protocol identity')
        source = transport['sources'].get(bound_sha, {})
        if bound_sha not in inputs or reference_sha in inputs:
            raise ValueError('Saved reference alias lacks its unique bound input packet')
        bound_root = Path(inputs[bound_sha])
        protocol = json.loads((bound_root/'protocol.json').read_text())
        if digest(protocol) != bound_sha:
            raise ValueError('Saved reference alias bound protocol changed')
        expected = reference_alias(template, protocol, source, document['history_id'])
        if alias != expected or reference_sha != expected['reference_protocol_sha256']:
            raise ValueError('Saved reference alias differs from its source or approved template')
        reference = template['protocol']
        target = prepare_shared_inputs(reference, bound_root, campaign_root)
        from gpx_recipe.preparation.hints import read, restore
        try:
            prepared = read(bound_root, protocol)
            if prepared is not None:
                # Rebind metadata only. Every numeric/artifact hash is unchanged;
                # restore validates its original bytes and native trust stays false.
                prepared = copy.deepcopy(prepared)
                prepared['protocol_sha256'] = reference_sha
                prepared.pop('snapshot_sha256')
                prepared['snapshot_sha256'] = digest(prepared)
                restored = restore(prepared, reference, bound_root, target)
                transport.setdefault('prepared_inputs',{})[reference_sha] = dict(
                    status='attached' if restored['attached'] else 'unavailable',
                    attached_entries=restored['attached'],
                    discarded_entries=restored['discarded_entries'], native_qualification_required=True)
        except (OSError, ValueError, EOFError) as error:
            transport.setdefault('prepared_inputs',{})[reference_sha] = dict(
                status='unavailable', message='Reference prepared inputs could not be recovered. Current physics will be rebuilt.',
                error_type=type(error).__name__, native_qualification_required=True)
        inputs[reference_sha] = target
