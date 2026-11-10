"""Compile ordinary background columns with the unchanged native function.

Only independent Back;n coefficients are represented. All other background
terms remain fixed. Native full-pattern/derivative checks are still owned by
the caller; an unavailable or failed optimization retains the original path.
"""
import hashlib
import inspect
from pathlib import Path
import re
import time

import numpy as np

from .validated_backend import PINNED_PRODUCTION_STRMATH

BACKGROUND_SOURCE='ba21fb1e4304debbf4cb43065524b728fa39695087102ffdb50371f4dca274b7'
DRIVER_SOURCE='caa9e0aac821043004b54a2382008b2c0d04abb97ac8b057d57be6044880944d'
MAX_BYTES=64*1024*1024


class BackgroundCompilerUnavailable(ValueError):
    """Use the original complete native powder evaluations."""


def _reviewed(math):
    try:
        function=math.G2pwd.getBackground
        driver=math.getPowderProfile
        return (hashlib.sha256(Path(math.__file__).read_bytes()).hexdigest()==PINNED_PRODUCTION_STRMATH
            and Path(driver.__code__.co_filename).resolve()==Path(math.__file__).resolve()
            and hashlib.sha256(inspect.getsource(driver).encode()).hexdigest()==DRIVER_SOURCE
            and Path(function.__code__.co_filename).resolve()==Path(math.G2pwd.__file__).resolve()
            and hashlib.sha256(inspect.getsource(function).encode()).hexdigest()==BACKGROUND_SOURCE)
    except (AttributeError,OSError,TypeError):return False


def compile_columns(*,math,histogram,parameters,controls,axis,names):
    """Return unweighted columns in the declared independent-variable order.

    Call the same native background routine as getPowderProfile, at zero and
    every unit coefficient. Never evaluate phases, synthesize observations,
    change fixed-background arrays, or replace a native function.
    """
    started=time.monotonic();names=list(names)
    x=np.asarray(np.ma.getdata(axis),float)
    hfx=f":{histogram['hId']}:"
    if (x.ndim!=1 or len(x)<2 or not np.isfinite(x).all() or np.any(np.diff(x)<=0)
            or not 1<=len(names)<=256 or len(set(names))!=len(names)
            or len(x)*len(names)*8>MAX_BYTES
            or any(not isinstance(key,str) or not re.fullmatch(re.escape(hfx)+r'Back;\d+',key)
                or key not in parameters for key in names)
            or controls.get(hfx+'histType') not in {'PNC','PNT'}):
        raise BackgroundCompilerUnavailable('Ordinary independent native background coefficients and a bounded current grid required')
    if not _reviewed(math):
        raise BackgroundCompilerUnavailable('Native background function or powder driver needs review')
    private=parameters.copy()
    for key in names:private[key]=0.
    fixback=histogram['Background'][1].get('fixback')
    if fixback is not None:
        fixback=np.array(fixback,copy=True)
        if fixback.shape!=x.shape or not np.isfinite(fixback).all():
            raise BackgroundCompilerUnavailable('Fixed native background needs the exact current grid')
    function=math.G2pwd.getBackground;calls=0
    def evaluate():
        nonlocal calls
        value,_=function(hfx,private,controls[hfx+'bakType'],controls[hfx+'histType'],x,fixback)
        calls+=1;value=np.asarray(np.ma.getdata(value),float)
        if value.shape!=x.shape or not np.isfinite(value).all():
            raise BackgroundCompilerUnavailable('Native background returned nonfinite or incompatible values')
        return value.copy()
    try:
        base=evaluate();columns=np.empty((len(x),len(names)))
        for i,key in enumerate(names):
            private[key]=1.;columns[:,i]=evaluate()-base;private[key]=0.
    except (KeyError,TypeError,ValueError,FloatingPointError) as error:
        raise BackgroundCompilerUnavailable('Native background columns require the original powder path: '+str(error)) from error
    return columns,dict(method='unchanged native background function',columns=len(names),
        native_background_calls=calls,full_powder_calls_avoided=len(names),
        function_sha256=BACKGROUND_SOURCE,seconds=time.monotonic()-started,
        observations_used=False,native_functions_replaced=False,
        scope='Independent ordinary background coefficients; full-pattern and derivative checks remain required')
