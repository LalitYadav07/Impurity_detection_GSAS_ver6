"""Build-time, hash-pinned correction of the reviewed GSAS-II moment kernel.

Original immutable images remain available. This module does not patch a
running process. A corrected image has a distinct source/function/stack hash
and must be inspected and approved as a new scientific protocol.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

PATCH_VERSION = 'radar-gsas-powder-geometry/3'
ORIGINAL_FILE_SHA256 = '2c9875e52546547b6ac565d1d0f25aabf5f06b1e1aba43fa8dd4cf359353f559'
ORIGINAL_FUNCTION_SHA256 = {
    'MagStructureFactor2': '82076dc0571cad6d6e2882c5f7ced9003f08af94449aa293b0fb4c5c2ae7b9f6',
    'MagStructureFactorDerv': 'df959b6cb576c83efbea4117f2ad4b73768253310a6188254dd08033e07b72f7',
}
DESKTOP_FILE_SHA256 = '5b39f376fa8555e88bfec8496cf5c475b4fdfe7d6d78ac05caaa1363cb617938'
DESKTOP_FUNCTION_SHA256 = {
    'MagStructureFactor2': '42e5cad268a08efa8bee0a9dc0f99dcf8a2d0765b22a70e2146ac5ec50a688d0',
    'MagStructureFactorDerv': 'bb3b63ec70cbcfa5099b6c1cc0d9e9e5ffe5fef56c4404ba3e9339db111c4f6e',
}
NUCLEAR_FUNCTION_SHA256 = 'bee9200e0751ec07a84d819a02a6ceff84f9a759a2f99d2725f17c997aca3385'
NUCLEAR_DERIVATIVE_SHA256 = 'a3985fab5a791bbe60a52b1822545491c48eb1a659e3ea32b3ba23c43a5391c6'


def corrected_nuclear_function(original):
    """Average Friedel intensities for unpolarized neutron powder only.

    The imaginary scattering lengths can make F(h) and F(-h) different.
    Their intensities are averaged; their amplitudes must not be added.
    Single-crystal and X-ray paths retain the original behavior.
    """
    if sha256(original) != NUCLEAR_FUNCTION_SHA256:
        raise ValueError('The pinned nuclear powder function changed')
    return replace_once(original,
        'refl.T[9] = np.sum(fas[:,:],axis=0)**2+np.sum(fbs[:,:],axis=0)**2',
        "refl.T[9] = (np.sum(fas**2,axis=0)+np.sum(fbs**2,axis=0) if hType in ('PNC','PNT') "
        "else np.sum(fas[:,:],axis=0)**2+np.sum(fbs[:,:],axis=0)**2)")


def corrected_nuclear_derivative(original):
    """Differentiate the two occurrences of each off-diagonal neutron Uij.

    UijtoU6 stores one off-diagonal element. In h^T U h, U12, U13 and
    U23 each occur twice. Keep other radiation/geometry paths unchanged.
    """
    if sha256(original) != NUCLEAR_DERIVATIVE_SHA256:
        raise ValueError('The pinned nuclear powder derivative changed')
    updated = replace_once(original,
        '    for i in range(len(Tdata)):         #loop over atoms',
        "    nuclear_offdiag_factor = 2.0 if hType in ('PNC','PNT') else 1.0\n"
        '    for i in range(len(Tdata)):         #loop over atoms')
    for index, axis in enumerate(('12', '13', '23'), start=3):
        updated = replace_once(updated,
            "dFdvDict[pfx+'AU" + axis + ":'+str(i)] = dFdua.T[" + str(index) + "][i]",
            "dFdvDict[pfx+'AU" + axis + ":'+str(i)] = nuclear_offdiag_factor*dFdua.T[" + str(index) + "][i]")
    updated = updated.replace('    #should not be *2.0!',
                              '    #neutron off-diagonal tensor derivative')
    return updated


def sha256(value):
    return hashlib.sha256(value.encode() if isinstance(value,str) else value).hexdigest()


def function_source(source,name):
    tree=ast.parse(source)
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name==name]
    if len(nodes)!=1:raise ValueError('A unique pinned GSAS-II function is required: '+name)
    return ''.join(inspect.getblock(source.splitlines(keepends=True)[nodes[0].lineno-1:]))


def replace_once(source,old,new):
    if source.count(old)!=1:raise ValueError('Unrecognized GSAS-II expression; source review required')
    return source.replace(old,new)


def corrected_source(source):
    """Return a reviewed source variant, without executing or modifying it."""
    source_sha=sha256(source)
    desktop=source_sha==DESKTOP_FILE_SHA256
    if source_sha not in {ORIGINAL_FILE_SHA256,DESKTOP_FILE_SHA256}:
        raise ValueError('Unreviewed GSAS-II source; refusing magnetic-kernel correction')
    original_functions=DESKTOP_FUNCTION_SHA256 if desktop else ORIGINAL_FUNCTION_SHA256
    patched=source
    for name in original_functions:
        original=function_source(source,name)
        if sha256(original)!=original_functions[name]:
            raise ValueError('The pinned GSAS-II function changed: '+name)
        updated=original
        arguments=ast.parse(original).body[0].args
        if arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
            raise ValueError('The reviewed magnetic call signature changed')
        call=', '.join(arg.arg for arg in arguments.args)
        fallback_name='_radar_original_'+name
        unsupported=(f'        return {fallback_name}({call})' if desktop else
            "        raise ValueError('Reviewed magnetic geometry supports commensurate PNC/PNT powder only')")
        updated=replace_once(updated,'    g = nl.inv(G)',
            "    if calcControls[hfx+'histType'] not in ('PNC','PNT'):\n"+
            unsupported+'\n'+
            '    g = nl.inv(G)')
        if desktop:
            updated=replace_once(updated,'uAmat = G2lat.Gmat2AB(GS)[0]',
                'uAmat = G2lat.Gmat2AB(nl.inv(Ginv))[0] #direct unit-axis moment components')
            if name=='MagStructureFactor2':
                updated=replace_once(updated,'HM = np.inner(Bmat,H)',
                    'HM = np.inner(Bmat.T,H)')
        elif name=='MagStructureFactor2':
            updated=replace_once(updated,'uAmat = G2lat.Gmat2AB(GS)[0] #RVD',
                'uAmat = G2lat.Gmat2AB(nl.inv(Ginv))[0] #direct unit-axis moment components')
        else:
            updated=replace_once(updated,'Kdata = np.inner(Gdata.T,uAmat.T).T',
                'Kdata = np.inner(Gdata.T,uAmat).T')
        if name=='MagStructureFactorDerv':
            # U12/U13/U23 occur twice in h^T U h. UijtoU6 stores one
            # off-diagonal entry, so these derivatives need that factor two.
            for index,axis in enumerate(('12','13','23'),start=3):
                updated=replace_once(updated,
                    "dFdvDict[pfx+'AU"+axis+":'+str(i)] = dFdua.T["+str(index)+"][i]",
                    "dFdvDict[pfx+'AU"+axis+":'+str(i)] = 2.*dFdua.T["+str(index)+"][i]")
        # An anisotropic displacement tensor rotates with each symmetry mate.
        # The native expression reused only the first operation's factor.
        updated=replace_once(updated,
            'TMcorr = 0.539*(np.reshape(Tiso,Tuij.shape)*Tuij)[:,0,:]*Fdata*Mdata*MF/(2*Nops)',
            """thermal = np.reshape(Tiso,Tuij.shape)*Tuij
        if SGData['SGInv'] and not SGData['SGFixed']:
            thermal = np.concatenate([thermal,thermal],axis=1)
        thermal = np.tile(thermal,(1,Ncen,1))
        TMcorr = 0.539*thermal*Fdata*Mdata*MF[:,nxs,:]/(2*Nops)""")
        expected_uses=2 if name=='MagStructureFactor2' else 4
        if updated.count('TMcorr[nxs,:,nxs,:]')!=expected_uses:
            raise ValueError('The thermal-factor consumer layout changed')
        updated=updated.replace('TMcorr[nxs,:,nxs,:]','TMcorr[nxs,:,:,:]')
        if desktop:
            fallback=replace_once(original,'def '+name+'(','def '+fallback_name+'(')
            patched=replace_once(patched,original,fallback+'\n\n'+updated)
        else:
            patched=replace_once(patched,original,updated)
    nuclear = function_source(source, 'StructureFactor2')
    patched = replace_once(patched, nuclear, corrected_nuclear_function(nuclear))
    nuclear_derivative = function_source(source, 'StructureFactorDerv2')
    patched = replace_once(patched, nuclear_derivative,
                           corrected_nuclear_derivative(nuclear_derivative))
    ast.parse(patched)
    all_original = {**original_functions, 'StructureFactor2': NUCLEAR_FUNCTION_SHA256,
                    'StructureFactorDerv2': NUCLEAR_DERIVATIVE_SHA256}
    record=dict(version=PATCH_VERSION,original_file_sha256=source_sha,
        corrected_file_sha256=sha256(patched),
        original_functions=all_original,
        corrected_functions={name:sha256(function_source(patched,name)) for name in all_original},
        scope='Commensurate PNC/PNT magnetic geometry, symmetry-rotated ADPs, nuclear Friedel intensity averaging and nuclear anisotropic derivatives',
        rationale='Use consistent Cartesian transforms, average nuclear powder intensities over h and -h, and differentiate both off-diagonal Uij contributions')
    if desktop:
        record.update(source_variant='desktop-dcd09eb',non_powder_magnetic_fallback_preserved=True)
    return patched,record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true',required=True,
        help='Apply only while building a new immutable scientific image')
    parser.parse_args()
    spec=importlib.util.find_spec('GSASII.GSASIIstrMath')
    if not spec or not spec.origin:raise RuntimeError('The pinned GSAS-II module is unavailable')
    path=Path(spec.origin)
    source=path.read_text()
    updated,record=corrected_source(source)
    path.write_text(updated)
    if sha256(path.read_bytes())!=record['corrected_file_sha256']:
        raise RuntimeError('The written GSAS-II source differs from the reviewed variant')
    print(json.dumps(record,sort_keys=True))


if __name__=='__main__':main()
