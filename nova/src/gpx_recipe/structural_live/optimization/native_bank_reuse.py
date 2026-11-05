"""Research execution adapter: exact native fixed-cell response reuse in a bank.

The saved bank still contains full native curves at the declared discrete starts.
This changes only their construction schedule. It never interpolates intensity,
shares responses between candidates or replaces the final native likelihood.
"""
import hashlib,inspect,json,time
from pathlib import Path
import numpy as np
from .native_cell_intensity import ExactCellIntensity
from .offline_native_start_bank import build

class NativeBankSource:
    def __init__(self,source):
        self.source=source;self.cache=ExactCellIntensity(source,max_cells=1)
        self.cached_calls=0;self.direct_calls=0;self.checked=False
    def __getattr__(self,name):return getattr(self.source,name)
    def native(self,point):
        point=np.asarray(point,float)
        if self.source.mode_dimensions and np.any(point[:self.source.mode_dimensions]!=0) and np.all(point[self.source.mode_dimensions:]==0):
            if not self.checked:
                self.cache.verify(point);self.checked=True
            self.cached_calls+=1;return self.cache.evaluate(point)
        self.direct_calls+=1;return self.source.native(point)

def build_reusing_profiles(source,path,policy='multiscale_v1'):
    proxy=NativeBankSource(source);receipt=build(proxy,path,policy=policy)
    receipt['construction_execution']=dict(method='GSAS-II native fixed-cell profile reuse',cached_calls=proxy.cached_calls,direct_calls=proxy.direct_calls,
        diagnostics=proxy.cache.diagnostics(),implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_target_context_unchanged=True,additional_independent_checks_required=True)
    Path(path).with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt
