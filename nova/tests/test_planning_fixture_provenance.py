"""Require an explicit provenance review when transition fixture bytes change.

The hashes verify the reviewed inventory, not a source's redistribution rights.
The fixture folder accompanies this test when it is relocated for image checks.
"""
import hashlib
import json
from pathlib import Path


FIXTURES = Path(__file__).resolve().parent / 'data' / 'planning-transitions'


def provenance():
    document = json.loads((FIXTURES / 'provenance.json').read_text())
    assert document['schema'] == 'radar-planning-fixture-provenance/v1'
    return document['files']


def test_every_transition_cif_has_reviewed_provenance_and_unchanged_contents():
    records = provenance()
    actual = {p.relative_to(FIXTURES).as_posix() for p in FIXTURES.rglob('*')
              if p.is_file() and p.suffix.lower() in {'.cif', '.mcif'}}
    assert actual == set(records), 'Review provenance before adding or removing a fixture'
    for name, record in records.items():
        assert hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest() == record['sha256'], name
        assert record['description'].strip(), name
        assert record['source_type'] in {'synthetic', 'public', 'generated'}, name
        if record['source_type'] == 'synthetic':
            assert record['construction'].strip(), name
        else:
            assert record['citations'] and all(url.startswith('https://') for url in record['citations']), name


def test_generated_transition_fixture_is_bound_to_its_reviewed_parent():
    records = provenance()
    generated = {name: row for name, row in records.items() if row['source_type'] == 'generated'}
    assert generated, 'The structural control must retain its generated-child provenance'
    for name, record in generated.items():
        assert record['generator'].strip(), name
        parent = record['parent']
        assert parent['file'] != name and parent['file'] in records, name
        assert parent['sha256'] == records[parent['file']]['sha256'], name
        assert parent['sha256'] == hashlib.sha256((FIXTURES / parent['file']).read_bytes()).hexdigest(), name
