from gpx_recipe.symmetry.native_preparation import parameter_flags


def test_legacy_integer_refinement_flags_preserve_every_instrument_value():
    source={'U':[701.3626,701.3626,0],'V':[-1157.202,-1157.202,1],'Type':['PNC','PNC',0],'Source':[' ',' ']}
    normalized=parameter_flags(source,2)
    assert normalized=={'U':[701.3626,701.3626,False],'V':[-1157.202,-1157.202,True],'Type':['PNC','PNC',False],'Source':[' ',' ']}
    assert type(source['V'][2]) is int


def test_current_values_are_not_treated_as_instrument_refinement_flags():
    source={'X':[0.,1.,False],'Zero':[0.,0.,True]}
    assert parameter_flags(source,2)==source


def test_sample_flags_are_normalized_at_their_distinct_native_index():
    assert parameter_flags({'Scale':[1.,1],'Gonio. radius':1000.,'Type':'Debye-Scherrer'},1)=={'Scale':[1.,True],'Gonio. radius':1000.,'Type':'Debye-Scherrer'}
