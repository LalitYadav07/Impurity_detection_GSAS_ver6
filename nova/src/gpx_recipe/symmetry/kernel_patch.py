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

PATCH_VERSION = 'radar-gsas-moment-geometry/1'
ORIGINAL_FILE_SHA256 = '2c9875e52546547b6ac565d1d0f25aabf5f06b1e1aba43fa8dd4cf359353f559'
ORIGINAL_FUNCTION_SHA256 = {
    'MagStructureFactor2': '82076dc0571cad6d6e2882c5f7ced9003f08af94449aa293b0fb4c5c2ae7b9f6',
    'MagStructureFactorDerv': 'df959b6cb576c83efbea4117f2ad4b73768253310a6188254dd08033e07b72f7',
}


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
    if sha256(source)!=ORIGINAL_FILE_SHA256:
        raise ValueError('Unreviewed GSAS-II source; refusing magnetic-kernel correction')
    patched=source
    for name in ORIGINAL_FUNCTION_SHA256:
        original=function_source(source,name)
        if sha256(original)!=ORIGINAL_FUNCTION_SHA256[name]:
            raise ValueError('The pinned GSAS-II function changed: '+name)
        updated=original
        updated=replace_once(updated,'    g = nl.inv(G)',
            "    if calcControls[hfx+'histType'] not in ('PNC','PNT'):\n"
            "        raise ValueError('Reviewed magnetic geometry supports commensurate PNC/PNT powder only')\n"
            '    g = nl.inv(G)')
        if name=='MagStructureFactor2':
            updated=replace_once(updated,'uAmat = G2lat.Gmat2AB(GS)[0] #RVD',
                'uAmat = G2lat.Gmat2AB(nl.inv(Ginv))[0] #direct unit-axis moment components')
        else:
            updated=replace_once(updated,'Kdata = np.inner(Gdata.T,uAmat.T).T',
                'Kdata = np.inner(Gdata.T,uAmat).T')
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
        patched=replace_once(patched,original,updated)
    ast.parse(patched)
    record=dict(version=PATCH_VERSION,original_file_sha256=ORIGINAL_FILE_SHA256,
        corrected_file_sha256=sha256(patched),
        original_functions=ORIGINAL_FUNCTION_SHA256.copy(),
        corrected_functions={name:sha256(function_source(patched,name)) for name in ORIGINAL_FUNCTION_SHA256},
        scope='Commensurate PNC/PNT unit-axis magnetic moment coordinates and symmetry-rotated atomic displacement tensors',
        rationale='Preserve the direct-axis metric and the same Cartesian transform in forward and nonmagnetic derivatives')
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
