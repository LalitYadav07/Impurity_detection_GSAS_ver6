"""Read-only deployment preflight. Success is not scientific qualification."""
import argparse,importlib.util,json,math,sys
from pathlib import Path


def validate_config(config):
    for key in ('facility_root','archive','jobs'):
        if not isinstance(config.get(key),str) or not Path(config[key]).is_absolute():raise ValueError(f'{key} must be an explicit absolute server path')
    if not isinstance(config.get('local_replay',False),bool):raise ValueError('local_replay must be a server-configured boolean')
    for key,lo,hi,default in [('poll_seconds',1,3600,30),('reference_timeout_seconds',1,3600,600),('job_timeout_seconds',1,86400,3600)]:
        value=config.get(key,default)
        if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or not lo<=value<=hi:raise ValueError(f'Invalid {key}')
    if not isinstance(config.get('allow_conditional_stream',False),bool):raise ValueError('allow_conditional_stream must be a server boolean')
    if isinstance(config.get('max_active_jobs',1),bool) or not isinstance(config.get('max_active_jobs',1),int) or not 1<=config.get('max_active_jobs',1)<=8:raise ValueError('Invalid native job capacity')
    if config.get('slot_root') and not Path(config['slot_root']).is_absolute():raise ValueError('slot_root must be absolute')
    for key in ('worker_command','checkpoint_prepare_command','profile_polish_command'):
        if key in config and (not isinstance(config[key],list) or not config[key] or not all(isinstance(x,str) and x and '\x00' not in x for x in config[key])):raise ValueError(f'{key} must be an executable argument list')
    return config


def inspect(manifest,native=False):
    checks=[]
    def check(label,fn):
        try:detail=fn();checks.append(dict(check=label,passed=True,detail=detail))
        except Exception as exc:checks.append(dict(check=label,passed=False,detail=f'{type(exc).__name__}: {exc}'))
    config=json.loads(Path(manifest).read_text());check('manifest',lambda:bool(validate_config(config)))
    from .session import WebmonSession
    def bindings():
        session=WebmonSession(manifest);count=0
        for scope,options in session.config.get('checkpoint_options',{}).items():
            ipts,run=scope.split('/')
            for option in options:
                session.select_checkpoint(ipts,run,option['id']);b,req=session.request(ipts,run,k=option['binding']['k'],ion=option['binding']['ion'],lower=.1,upper=min(9,option['binding']['maximum_moment']),accepted=True);count+=1
        if not count:raise ValueError('No reviewed checkpoint preparations are configured')
        return count
    check('checkpoint bindings and native-input checksums',bindings)
    if native:
        for module in ('GSASII.GSASIIscriptable','gsas_magnetic_preflight','gpx_recipe.symmetry.native_magnetic'):
            def available(name=module):
                spec=importlib.util.find_spec(name)
                if not spec:raise ImportError(name)
                return str(spec.origin)
            check(module,available)
        def backend():
            from .validated_backend import corrected_backend
            with corrected_backend() as identity:return identity
        check('pinned native correction backend',backend)
    return dict(schema='webmon-deployment-preflight/v1',checks=checks,passed=all(c['passed'] for c in checks),scientifically_qualified=False,forecast_enabled=False,production_release_ready=False,scope='Local process adapter; live snapshot preparation, covariance calibration and Linux acceptance still required')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest');parser.add_argument('--native',action='store_true');parser.add_argument('--output');a=parser.parse_args()
    try:result=inspect(a.manifest,a.native)
    except Exception as exc:result=dict(passed=False,error=str(exc),production_release_ready=False)
    output=json.dumps(result,indent=2)
    if a.output:Path(a.output).write_text(output)
    print(output);return 0 if result['passed'] else 1

if __name__=='__main__':sys.exit(main())
