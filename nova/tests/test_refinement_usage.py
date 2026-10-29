"""Billing estimates use reported usage across every planner attempt."""
import math

from gpx_recipe.usage import summarize_calls, usage_labels


def luna(usage):
    return dict(provider='illumine-luna', model='gpt-6-luna', usage=usage)


def test_real_response_shape_prices_cache_writes_separately():
    # Captured usage shape from a real Luna recipe response; no scientific data.
    summary = summarize_calls([luna(dict(input_tokens=18962,
        input_tokens_details=dict(cache_write_tokens=18959, cached_tokens=0),
        output_tokens=757, output_tokens_details=dict(reasoning_tokens=0), total_tokens=19719))])
    assert summary['total_tokens'] == 19719
    assert summary['cache_write_tokens'] == 18959
    assert math.isclose(summary['estimated_usd'], (3*.10 + 18959*.125 + 757*.50)/1_000_000)
    tokens, cost = usage_labels(summary)
    assert '18,962 input' in tokens and '757 output' in tokens
    assert 'Approximate Luna' in cost and 'ORNL Azure' in cost


def test_repair_calls_include_cached_input_and_output_from_both_calls():
    summary = summarize_calls([
        luna(dict(input_tokens=1000, input_tokens_details=dict(cached_tokens=200,cache_write_tokens=300),
                  output_tokens=500,total_tokens=1500)),
        luna(dict(input_tokens=1200, input_tokens_details=dict(cached_tokens=700,cache_write_tokens=0),
                  output_tokens=400,total_tokens=1600)),
    ])
    assert summary['calls'] == summary['reported_calls'] == 2
    assert summary['input_tokens'] == 2200 and summary['cached_input_tokens'] == 900
    assert summary['cache_write_tokens'] == 300 and summary['output_tokens'] == 900
    assert math.isclose(summary['estimated_usd'], (1000*.10 + 900*.01 + 300*.125 + 900*.50)/1_000_000)


def test_long_context_multiplier_is_per_request():
    summary = summarize_calls([
        luna(dict(input_tokens=272001,output_tokens=10,total_tokens=272011)),
        luna(dict(input_tokens=100,output_tokens=10,total_tokens=110)),
    ])
    expected = (272001*.10*2 + 10*.50*1.5 + 100*.10 + 10*.50)/1_000_000
    assert math.isclose(summary['estimated_usd'], expected)


def test_missing_or_invalid_usage_never_looks_free():
    missing = summarize_calls([luna(None), luna(dict(input_tokens=100,output_tokens=20,total_tokens=120))])
    assert missing['reported_calls'] == 1 and missing['estimated_usd'] is None
    assert '1 of 2 calls reported usage' in usage_labels(missing)[0]
    assert 'Full cost unavailable' in usage_labels(missing)[1]
    invalid = summarize_calls([luna(dict(input_tokens=True,output_tokens=20))])
    assert invalid['reported_calls'] == 0 and invalid['estimated_usd'] is None
    assert 'unavailable' in usage_labels(invalid)[0]
    assert usage_labels(summarize_calls([])) == ('', '')


def test_alternative_provider_gets_tokens_without_luna_price():
    other = summarize_calls([dict(provider='sambanova',model='gpt-oss-120b',
                                  usage=dict(input_tokens=500,output_tokens=50,total_tokens=550))])
    assert other['total_tokens'] == 550 and other['estimated_usd'] is None
    assert 'Cost estimate unavailable' in usage_labels(other)[1]
