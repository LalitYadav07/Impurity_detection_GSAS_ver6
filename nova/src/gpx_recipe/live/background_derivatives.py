"""Preserve polynomial holds while evaluating native analytic derivatives."""
import re

import numpy as np


def background_row_order(parameters, varied):
    """Return a complete, contiguous background layout for derivative work.

    The native profile derivative writes a whole polynomial block starting at
    Back;0. A held coefficient must occupy a temporary derivative row, not the
    row of an unrelated scale or structural parameter. This layout never
    changes the actual free list, constraints, or parameter values.
    """
    original=list(varied);blocks={}
    if len(set(original))!=len(original) or any(not isinstance(name,str) for name in original):
        raise ValueError('Native derivative variables must be a complete unique row inventory')
    for name in parameters:
        found=re.fullmatch(r'(:\d+:Back;)(\d+)',str(name))
        if found:blocks.setdefault(found[1],{})[int(found[2])]=name
    active={name:prefix for prefix,block in blocks.items() for name in block.values() if name in original}
    expanded=[];seen=set()
    for name in original:
        prefix=active.get(name)
        if prefix is None:
            expanded.append(name);continue
        if prefix in seen:continue
        seen.add(prefix);block=blocks[prefix]
        if set(block)!=set(range(len(block))):
            raise ValueError('Native polynomial background has an incomplete coefficient inventory')
        expanded.extend(block[i] for i in range(len(block)))
    return expanded


def profile_derivative(call, values, args, *, allow_projection, dependent=()):
    """Compute native derivative rows and project onto the unchanged free set."""
    parameters=args[1];varied=args[3]
    if np.asarray(values).shape!=(len(varied),):
        raise ValueError('Native derivative values do not match the complete variable rows')
    expanded=background_row_order(parameters,varied)
    if expanded==list(varied):return call(values,*args)
    if not allow_projection:
        raise ValueError('Native analytic derivatives require a complete polynomial background block; source background holds are preserved')
    extra=[name for name in expanded if name not in varied]
    if set(extra)&{str(name) for name in dependent}:
        raise ValueError('Dependent polynomial background coefficients require an explicitly qualified derivative mapping')
    fixed={name:float(parameters[name]) for name in extra}
    if not np.isfinite(list(fixed.values())).all():
        raise ValueError('Held polynomial background coefficients must be finite')
    current=dict(zip(varied,values));current.update(fixed)
    expanded_values=np.array([current[name] for name in expanded],dtype=float)
    changed=list(args);changed[3]=expanded
    modified=False
    try:
        matrix=np.asarray(call(expanded_values,*changed),dtype=float)
    finally:
        try:modified=any(not np.array_equal(parameters.get(name),value) for name,value in fixed.items())
        finally:parameters.update(fixed)
    if matrix.ndim!=2 or matrix.shape[0]!=len(expanded):
        raise ValueError('Native derivative row inventory changed during background projection')
    if modified:
        raise ValueError('Native derivative evaluation changed a held background coefficient')
    return matrix[[expanded.index(name) for name in varied]]


def analytic_background_preflight(gpx,selected_histograms):
    """Reject unsupported partial analytic blocks without releasing source holds."""
    method=str(gpx.data.get('Controls',{}).get('data',{}).get('deriv type',''))
    if 'Hessian' not in method and method!='analytic Jacobian':
        return dict(checked=False,reason='Native solver does not use analytic derivative rows')
    histograms=[gpx.histogram(name) for name in selected_histograms]
    active=[h for h in histograms if h.data['Background'][0][1]]
    if not active:return dict(checked=False,reason='Polynomial background is fixed')
    from ..snapshot import gsas
    from ..studio_state import extract_gsasii_refinement_state
    native=extract_gsasii_refinement_state(gpx.filename,gpx,gsas(),max_correlations=0)
    if native['extraction']['status']!='complete':
        raise ValueError('Cannot verify the native analytic background derivative block before refinement')
    varied=native['parameter_space']['effective_vary']
    parameters={f':{h.id}:Back;{i}':0. for h in active for i in range(int(h.data['Background'][0][2]))}
    if any(not any(name.startswith(f':{h.id}:Back;') for name in varied) for h in active):
        raise ValueError('The requested analytic background fit has no independent polynomial coefficients; source holds are preserved and no refinement was run')
    if background_row_order(parameters,varied)!=list(varied):
        raise ValueError('Native analytic refinement cannot use a partially constrained polynomial background; source holds are preserved and no refinement was run')
    return dict(checked=True,method=method,source_holds_preserved=True,
        background_variables=[name for name in varied if name in parameters])


