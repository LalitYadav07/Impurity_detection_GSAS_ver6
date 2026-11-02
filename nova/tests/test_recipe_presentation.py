from copy import deepcopy

from radar_pd_nova.recipe_presentation import explicit_edit_lines


def fixture():
    context=dict(phases=[dict(key='p0',name='Oxide')],histograms=[dict(key='h0',name='bank5',instrument={'Type':'PNT'})])
    stage=dict(phase_normalization=['h0'],constraints=[
        dict(kind='occupancy_sum',terms=[dict(phase='p0',atom='A',coefficient=1),dict(phase='p0',atom='B',coefficient=2)],total=1,reason='One shared site.'),
        dict(kind='uiso_equal',atoms=[dict(phase='p0',atom='A'),dict(phase='p0',atom='B')],reason='Shared displacement.')],
        fit_ranges=[dict(histogram='h0',unit='native',lower=1000,upper=2000)],
        exclusions=[dict(histogram='h0',unit='d',mode='replace',regions=[[1.,1.2],[2.,2.3]])],
        texture=[dict(phase='p0',histogram='h0',model='march-dollase',direction=[0,0,1],value=1)])
    return stage,context


def test_equations_ranges_and_texture_are_readable_without_changing_plan():
    stage,context=fixture();before=deepcopy((stage,context))
    lines=explicit_edit_lines(stage,context,physical_banks={'bank5':5})
    assert lines==[
        'Normalize phase scales to sum to 1 · Physical bank 5',
        'Constraint: 1 × occupancy(Oxide / A) + 2 × occupancy(Oxide / B) = 1. One shared site.',
        'Constraint: Uiso(Oxide / A) = Uiso(Oxide / B). Shared displacement.',
        'Fit range · Physical bank 5: 1000–2000 µs (TOF)',
        'Replace excluded regions · Physical bank 5: 1–1.2, 2–2.3 Å (d-spacing)',
        'Oxide / Physical bank 5: March–Dollase preferred orientation along [0 0 1]; starting ratio 1']
    assert (stage,context)==before


def test_unknown_native_units_are_not_inferred_and_sh_order_is_retained():
    stage,context=fixture();context['histograms'][0]['instrument']={}
    stage['texture']=[dict(phase='p0',histogram='h0',model='spherical-harmonics',order=4)]
    stage['exclusions']=[dict(histogram='h0',unit='native',mode='clear',regions=[])]
    lines=explicit_edit_lines(stage,context)
    assert 'Fit range · bank5: 1000–2000 native coordinate' in lines
    assert 'Clear excluded regions · bank5' in lines
    assert lines[-1]=='Oxide / bank5: Spherical-harmonic preferred orientation, order 4'


def test_untrusted_names_remain_plain_text_not_generated_markup():
    stage,context=fixture();context['phases'][0]['name']='<img src=x onerror=alert(1)>'
    stage['constraints'][0]['reason']='<script>example</script>'
    lines=explicit_edit_lines(stage,context)
    assert '<img src=x onerror=alert(1)>' in lines[1] and '<script>example</script>' in lines[1]
    assert all(isinstance(line,str) for line in lines)  # Consumed by escaped {{ edit }}, never v-html.
