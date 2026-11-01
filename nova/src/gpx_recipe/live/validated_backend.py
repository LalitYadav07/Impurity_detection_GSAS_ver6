"""Opt-in, pinned GSAS-II powder correction adapter for isolated worker processes.

No installed source is edited. Unsupported versions/histograms fail closed.
This version establishes calculation identity, not experimental calibration.
Use one calculation thread per worker; the native GSAS modules are global state.
"""
from contextlib import contextmanager
import hashlib
import inspect
import os
from pathlib import Path
import threading

VERSION = 'virtual-powder-gsas-corrections/1'
PINNED_PRODUCTION_STRMATH = '6fea6023fa33a2585d091b04a479194c37f6efac4336ee29c7bef7af31e930af'
PINNED_STRMATH = '5b39f376fa8555e88bfec8496cf5c475b4fdfe7d6d78ac05caaa1363cb617938'
PINNED_LINUX_STRMATH = 'cc56d2ff8855a773cb1a48ec7576d5bdcb8850ec6a73e10d43e093874108173f'
_LOCK = threading.Lock()
_ACTIVE = None


def _replace(source, old, new, expected=1):
    if source.count(old) != expected:
        raise RuntimeError('Unrecognized GSAS expression; backend review required')
    return source.replace(old, new)


def _compile(fn, source):
    scope = dict(fn.__globals__)
    exec(compile(source, '<'+VERSION+':'+fn.__name__+'>', 'exec'), scope)
    corrected = scope[fn.__name__]
    def checked(ref, g, hfx, pfx, sg, controls, params):
        if controls[hfx+'histType'] not in ('PNC', 'PNT'):
            raise ValueError('Corrected backend supports PNC/PNT powder only')
        return corrected(ref, g, hfx, pfx, sg, controls, params)
    checked.__name__ = fn.__name__
    return checked, hashlib.sha256(source.encode()).hexdigest()


def build_functions(module):
    native_hash=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    if native_hash == PINNED_PRODUCTION_STRMATH:
        # The shared worker already contains the reviewed build-time correction.
        # Report exact identity without replacing any native function at runtime.
        hashes = {name: hashlib.sha256(inspect.getsource(getattr(module,name)).encode()).hexdigest()
                  for name in ('MagStructureFactor2','MagStructureFactorDerv','StructureFactor2')}
        return {}, hashes
    if os.environ.get('RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND') != '1':
        raise RuntimeError('Legacy correction adapter is restricted to explicit local replay; use the shared pinned worker')
    if native_hash not in (PINNED_STRMATH,PINNED_LINUX_STRMATH):
        raise RuntimeError('Unvalidated GSAS-II version; refusing correction adapter')
    functions, hashes = {}, {}
    for name, uses in [('MagStructureFactor2', 2), ('MagStructureFactorDerv', 4)]:
        fn = getattr(module, name); source = inspect.getsource(fn)
        if native_hash==PINNED_LINUX_STRMATH:
            # This exact reviewed Linux build already carries the direct-axis,
            # reciprocal-vector and per-operation thermal corrections. Never
            # apply those a second time. Nuclear powder averaging remains below.
            functions[name],hashes[name]=_compile(fn,source)
            continue
        source = _replace(source, 'uAmat = G2lat.Gmat2AB(GS)[0]',
                          'uAmat = G2lat.Gmat2AB(nl.inv(Ginv))[0]')
        if name == 'MagStructureFactor2':
            source = _replace(source, 'HM = np.inner(Bmat,H)', 'HM = np.inner(Bmat.T,H)')
        elif source.count('HM = np.inner(Bmat.T,H)') != 1:
            raise RuntimeError('Derivative reciprocal convention changed')
        source = _replace(source,
            'TMcorr = 0.539*(np.reshape(Tiso,Tuij.shape)*Tuij)[:,0,:]*Fdata*Mdata*MF/(2*Nops)',
            """thermal = np.reshape(Tiso,Tuij.shape)*Tuij
        if SGData['SGInv'] and not SGData['SGFixed']:
            thermal = np.concatenate([thermal,thermal],axis=1)
        thermal = np.tile(thermal,(1,Ncen,1))
        TMcorr = 0.539*thermal*Fdata*Mdata*MF[:,nxs,:]/(2*Nops)""")
        source = _replace(source, 'TMcorr[nxs,:,nxs,:]', 'TMcorr[nxs,:,:,:]', uses)
        functions[name], hashes[name] = _compile(fn, source)
    # For unpolarized powder, averaging +/-h cancels real/imaginary cross terms.
    # Native powder derivatives already use this averaged expression; verify
    # them independently rather than making an unnecessary derivative change.
    fn = module.StructureFactor2; source = inspect.getsource(fn)
    source = _replace(source,
        'refl.T[9] = np.sum(fas[:,:],axis=0)**2+np.sum(fbs[:,:],axis=0)**2',
        'refl.T[9] = np.sum(fas**2,axis=0)+np.sum(fbs**2,axis=0)')
    functions['StructureFactor2'], hashes['StructureFactor2'] = _compile(fn, source)
    return functions, hashes


def backend_identity():
    from GSASII import GSASIIstrMath as M, GSASIIlattice as L, GSASIIElem as E, GSASIIspc as SPC
    _, hashes = build_functions(M)
    return dict(version=VERSION, adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_sources={Path(m.__file__).name: hashlib.sha256(Path(m.__file__).read_bytes()).hexdigest()
                        for m in (M,L,E,SPC)}, corrected_functions=hashes,
        scope='PNC/PNT fixed-nuisance commensurate powder calculations; empirical response unvalidated')


def active_identity():
    return _ACTIVE.copy() if _ACTIVE else None


@contextmanager
def corrected_backend():
    global _ACTIVE
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('Use a dedicated process, not a calculation thread')
    if not _LOCK.acquire(blocking=False):
        raise RuntimeError('Backend already active; nested/concurrent contexts are forbidden')
    saved = {}
    try:
        from GSASII import GSASIIstrMath as M
        functions, _ = build_functions(M)
        identity = backend_identity()
        saved = {name:getattr(M,name) for name in functions}
        for name, fn in functions.items(): setattr(M,name,fn)
        _ACTIVE = identity
        yield identity
    finally:
        if saved:
            for name, fn in saved.items(): setattr(M,name,fn)
        _ACTIVE = None
        _LOCK.release()
