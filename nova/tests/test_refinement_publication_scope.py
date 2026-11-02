"""Different reductions of a logical run remain independently selectable."""
from radar_pd_nova.refinement_sources import (
    RefinementScope, published_selection_key, refinement_scan_options,
)
from test_refinement_integration import create, source


def two_reductions():
    a, b = source('1'), source('2')
    a['scan_key'] = b['scan_key'] = 'same-physical-run-and-bank'
    return a, b


def test_published_selection_does_not_include_another_reduction(tmp_path):
    a, b = two_reductions()
    workspace, _ = create(tmp_path)
    workspace.observe([a, b])
    scope = RefinementScope('exp', 'scans', scans=(published_selection_key(b),))
    snapshot, token = workspace.selection(scope)
    assert [row['id'] for row in snapshot['baseline']] == ['2']
    c = source('3')
    c['scan_key'] = a['scan_key']
    workspace.observe([c])
    assert workspace.selection(scope)[1] == token


def test_new_options_use_unique_publications_even_for_same_run():
    a, b = two_reductions()
    options = refinement_scan_options([a, b])
    assert [row['value'] for row in options] == ['source:1', 'source:2']
    for option, chosen in zip(options, (a, b)):
        scope = RefinementScope('exp', 'scans', scans=(option['value'],))
        assert [s['id'] for s in (a, b) if scope.matches(s)] == [chosen['id']]


def test_saved_logical_scope_is_readable_and_explicitly_labelled():
    a, b = two_reductions()
    scope = RefinementScope('exp', 'scans', scans=(a['scan_key'],))
    before = scope.to_dict()
    assert scope.matches(a) and scope.matches(b)
    options = refinement_scan_options([a, b], scope.scans)
    assert options[-1]['value'] == a['scan_key']
    assert 'recorded logical-run selection (2 saved result(s))' in options[-1]['title']
    assert scope.to_dict() == before


def test_publication_scope_still_requires_experiment_and_bank():
    a, _ = two_reductions()
    key = published_selection_key(a)
    assert not RefinementScope('other', 'scans', scans=(key,)).matches(a)
    assert not RefinementScope('exp', 'scans', scans=(key,), banks=('9',)).matches(a)
