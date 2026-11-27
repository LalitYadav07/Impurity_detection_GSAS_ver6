"""Polynomial derivative rows must not alias unrelated native parameters."""
import numpy as np
import pytest
from gpx_recipe.live.background_derivatives import background_row_order,profile_derivative


def fixture():
    params={':0:Back;0':3.,':0:Back;1':.2,':0:Back;2':.1,'scale':2.,'q':.01}
    vary=[':0:Back;0','scale','q'];values=np.array([3.,2.,.01])
    args=({},params,{},vary,{},None,None)
    return params,vary,values,args


def test_complete_block_preserves_native_call_and_array_identity():
    p,v,x,a=fixture();v[:]=[':0:Back;0',':0:Back;1',':0:Back;2','scale'];x=np.array([p[k] for k in v])
    result=np.ones((4,2));seen=[]
    def native(values,*args):seen.append((values,args));return result
    assert profile_derivative(native,x,a,allow_projection=False) is result
    assert seen[0][0] is x and seen[0][1][3] is v


def test_partial_rows_are_completed_and_projected_without_releasing_holds():
    p,v,x,a=fixture();original=x.copy();seen=[]
    def native(values,*args):
        seen.append((values.copy(),list(args[3])))
        return np.array([[p[k]]*3 for k in args[3]])
    result=profile_derivative(native,x,a,allow_projection=True)
    assert seen[0][1]==[':0:Back;0',':0:Back;1',':0:Back;2','scale','q']
    np.testing.assert_array_equal(result,np.array([[3.]*3,[2.]*3,[.01]*3]))
    np.testing.assert_array_equal(x,original)
    assert v==[':0:Back;0','scale','q'] and p[':0:Back;1']==.2 and p[':0:Back;2']==.1


def test_partial_rows_fail_closed_without_explicit_projection():
    p,v,x,a=fixture()
    with pytest.raises(ValueError,match='source background holds are preserved'):
        profile_derivative(lambda *a:pytest.fail('native must not run'),x,a,allow_projection=False)


def test_dependent_coefficients_are_not_reinterpreted_as_independent():
    p,v,x,a=fixture()
    with pytest.raises(ValueError,match='Dependent polynomial'):
        profile_derivative(lambda *a:pytest.fail('native must not run'),x,a,allow_projection=True,dependent=[':0:Back;1'])


def test_multiple_histograms_keep_original_global_row_order():
    p,v,x,a=fixture();p.update({':1:Back;0':1.,':1:Back;1':2.})
    v[:]=[':0:Back;0','q',':1:Back;1','scale']
    assert background_row_order(p,v)==[':0:Back;0',':0:Back;1',':0:Back;2','q',':1:Back;0',':1:Back;1','scale']


def test_background_with_no_active_rows_leaves_native_call_alone():
    p,v,x,a=fixture();v[:]=['scale','q'];x=np.array([2.,.01])
    expected=np.ones((2,3))
    assert profile_derivative(lambda *a:expected,x,a,allow_projection=False) is expected


def test_incomplete_background_inventory_fails_before_native():
    p,v,x,a=fixture();del p[':0:Back;1']
    with pytest.raises(ValueError,match='incomplete coefficient'):
        profile_derivative(lambda *a:pytest.fail('native must not run'),x,a,allow_projection=True)


def test_native_failure_restores_temporary_fixed_values():
    p,v,x,a=fixture()
    def broken(values,*args):args[1][':0:Back;1']=91.;raise RuntimeError('failed native derivative')
    with pytest.raises(RuntimeError,match='failed native'):
        profile_derivative(broken,x,a,allow_projection=True)
    assert p[':0:Back;1']==.2


def test_changed_fixed_value_is_restored_and_rejected():
    p,v,x,a=fixture()
    def changed(values,*args):
        args[1][':0:Back;1']=9.
        return np.ones((len(args[3]),3))
    with pytest.raises(ValueError,match='changed a held'):
        profile_derivative(changed,x,a,allow_projection=True)
    assert p[':0:Back;1']==.2


@pytest.mark.parametrize('varied,values',[
    ([':0:Back;0',':0:Back;0'],[3.,3.]),
    ([':0:Back;0','scale'],[3.]),
])
def test_duplicate_or_missing_variable_rows_are_rejected(varied,values):
    p,v,x,a=fixture();a=(*a[:3],varied,*a[4:])
    with pytest.raises(ValueError,match='complete'):
        profile_derivative(lambda *a:pytest.fail('native must not run'),np.array(values),a,allow_projection=True)


def test_missing_native_derivative_rows_are_not_accepted():
    p,v,x,a=fixture()
    with pytest.raises(ValueError,match='row inventory'):
        profile_derivative(lambda *a:np.zeros((2,3)),x,a,allow_projection=True)


def test_held_constant_and_noncontiguous_higher_rows_use_complete_internal_block():
    p,v,x,a=fixture();v[:]=['q',':0:Back;1',':0:Back;2','scale']
    assert background_row_order(p,v)==['q',':0:Back;0',':0:Back;1',':0:Back;2','scale']
    v[:]=['q',':0:Back;0',':0:Back;2','scale']
    assert background_row_order(p,v)==['q',':0:Back;0',':0:Back;1',':0:Back;2','scale']
