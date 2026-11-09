"""Bounded diagnostic sample-broadening trials, with the instrument held fixed.
No automatic physical interpretation or scientific acceptance of the winner.
"""
import json,sys
from pathlib import Path
from .contracts import file_hash
from .io import atomic_json


def polish(receipt_path):
    from .sample_profile_trials import sample_width_trials
    receipt_path=Path(receipt_path).resolve();r=json.loads(receipt_path.read_text());out=receipt_path.parent
    choice=r['base_option'];source=Path(choice['source_path'])
    if file_hash(source)!=choice['source_sha256']:raise ValueError('Reviewed GPX changed')
    report=sample_width_trials(source,out)
    if report['selected_for_review'] is None:raise ValueError('No physically bounded numerically successful diagnostic trial')
    best=next(v for v in report['trials'] if v['name']==report['selected_for_review'])
    if file_hash(source)!=choice['source_sha256']:raise ValueError('Original source changed')
    prep=out/'preparation-input.json';atomic_json(prep,dict(base_option=choice,job_id='bounded-profile-diagnostic',source_path=best['path'],source_sha256=file_hash(best['path'])))
    from .prepare_reference import prepare
    result=prepare(prep);result['profile_diagnostic']=str(out/'profile-trials.json')
    result['title']='Profile diagnostic · '+result['title'];atomic_json(out/'option.json',result)
    return result

if __name__=='__main__':polish(sys.argv[1])
