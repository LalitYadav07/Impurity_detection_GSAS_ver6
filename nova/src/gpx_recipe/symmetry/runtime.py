"""Location-independent identity of the validated native numerical stack."""
import hashlib
import importlib
from pathlib import Path
import sys
from .contracts import digest

# GSAS-II 5843 Linux image with the hash-pinned, independently checked magnetic
# geometry/displacement correction. Legacy immutable workers keep their own
# original identity; they must not be repinned under an existing protocol.
VALIDATED_STACKS={'bc0e552f3b163884c89d9a9a7114d5e8477d030640edcde5a0e2fdc773660b92'}


def native_stack():
    import numpy,scipy
    from ..snapshot import gsas
    G=gsas();prefix=G.__package__+'.' if G.__package__ else '';files={}
    for name in ['GSASIIscriptable','GSASIIspc','GSASIIlattice','GSASIIElem','GSASIImath','GSASIIstrMain','GSASIIstrMath','GSASIIstrIO','GSASIImapvars','pypowder','pyspg']:
        module=importlib.import_module(prefix+name);files[name]=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    magnetic=importlib.import_module('pymatgen.symmetry.maggroups')
    files['pymatgen.symmetry.maggroups']=hashlib.sha256(Path(magnetic.__file__).read_bytes()).hexdigest()
    files['BNS_database']=hashlib.sha256(Path(magnetic.MAGSYMM_DATA).read_bytes()).hexdigest()
    stack=dict(platform=sys.platform,files=files,numpy=numpy.__version__,scipy=scipy.__version__)
    return dict(native_stack_sha256=digest(stack),native_stack=stack)
