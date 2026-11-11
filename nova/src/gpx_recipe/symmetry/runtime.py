"""Location-independent identity of the validated native numerical stack."""
import hashlib
import importlib
from pathlib import Path
import sys
from .contracts import digest

# GSAS-II 5843 Linux image with the hash-pinned, independently checked magnetic
# geometry/displacement and nuclear-powder derivative correction. Legacy immutable workers keep their own
# original identity; they must not be repinned under an existing protocol.
VALIDATED_STACKS={'1faf6d3e04b66da688334aa3e56da387a60963c4ef229269d9220a644c0ae70d'}


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
