"""Fixed synthetic GPU acceptance, separate from any experimental analysis.

Never count a skipped GPU test, a visible GPU, or a CPU fallback as passing.
The installed trusted fixtures exercise the actual shared three-curve trainer.
"""
import datetime
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
import zipfile


def main():
    out=Path(sys.argv[1]);archive=Path(sys.argv[2])
    if out.exists() or out.is_symlink():raise ValueError('CUDA acceptance needs a fresh owned output directory')
    out.mkdir(parents=True);started=time.monotonic()
    record=dict(schema='radar-ndip-cuda-acceptance/v1',state='not_qualified',
        entered_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),gpu_acceptance_qualified=False,
        scientific_result=False,candidate_pruning_authorized=False,
        scope='Synthetic trainer, mode head, deterministic pooling, local recovery and fresh-worker checkpoint transport')
    exit_code=1
    try:
        import torch
        from gpx_recipe.preparation.structural_training_backend import cuda_backend
        from gpx_recipe.preparation.structural_cuda_checkpoint import runtime
        from gpx_recipe.symmetry.compiler import package_signature
        from gpx_recipe.symmetry.resources import allocation
        cuda_backend()
        record.update(runtime=runtime(),cpu_allocation=allocation(1),implementation_sha256=package_signature(),
            maximum_application_gpu_memory_bytes=4096*1024**2,exclusive_gpu_reservation_claimed=False)
        environment=dict(os.environ,PYTHONPATH=os.environ.get('PYTHONPATH','')+':/opt/radar-cuda-validation',
            OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with (out/'console.log').open('w') as console:
            tested=subprocess.run([sys.executable,'-m','pytest','-q','-o','cache_dir=/tmp/radar-cuda-pytest',
                '-o','junit_family=xunit1','--junitxml='+str(out/'acceptance.xml'),
                '/opt/radar-cuda-validation/test_structural_training_cuda.py','-k','real_cuda'],
                stdout=console,stderr=subprocess.STDOUT,env=environment)
        cases=ET.parse(out/'acceptance.xml').findall('.//testcase')
        record['tests']=[dict(name=c.attrib['name'],seconds=float(c.attrib['time']),
            skipped=c.find('skipped') is not None,failed=c.find('failure') is not None or c.find('error') is not None,
            measurements={p.attrib['name']:int(p.attrib['value']) for p in c.findall('./properties/property')
                if p.attrib['name'] in ('cuda_peak_allocated_bytes','cuda_peak_reserved_bytes')}) for c in cases]
        if tested.returncode or len(cases)<7 or any(c.find(k) is not None for c in cases for k in ('skipped','failure','error')):
            raise ValueError('All seven real CUDA acceptance cases must pass without skips')
        record.update(state='qualified_synthetic_gpu_acceptance',gpu_acceptance_qualified=True)
        exit_code=0
    except Exception as error:
        record.update(state='gpu_acceptance_needs_attention',reason=str(error)[:1200],error=type(error).__name__)
    record['seconds']=time.monotonic()-started
    usage=resource.getrusage(resource.RUSAGE_CHILDREN)
    record['child_cpu_seconds']=usage.ru_utime+usage.ru_stime
    (out/'acceptance.json').write_text(json.dumps(record,indent=2)+'\n')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as packet:
        for name in ('acceptance.json','acceptance.xml','console.log'):
            path=out/name
            if path.exists():packet.write(path,name)
    print(json.dumps(record),flush=True)
    return exit_code


if __name__=='__main__':raise SystemExit(main())
