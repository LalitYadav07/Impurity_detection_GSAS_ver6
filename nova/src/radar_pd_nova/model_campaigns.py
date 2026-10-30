"""Explicit model campaign enrollment, independent of the visible UI selection."""
import copy
from pathlib import Path
import shutil
from contextlib import nullcontext

from gpx_recipe.symmetry.compiler import assets,confined,file_sha
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.templates import matches,rebind,validate_template
from gpx_recipe.execution import write_json
from .symmetry_sources import start_choices,resolve_start


def origin(source,workspace):
    if not source.get('recipe_job_id'):return 'radar'
    job=workspace.state['jobs'][source['recipe_job_id']]
    return 'recipe:'+job['recipe_id']+':'+source['recipe_stage']


def choices(workspace,experiment):
    with getattr(workspace,'lock',nullcontext()):
        return _choices(workspace,experiment)


def _choices(workspace,experiment):
    result=[]
    for entry in start_choices(workspace,experiment):
        if 'source' in entry:source=copy.deepcopy(entry['source']);kind='radar'
        else:
            job=workspace.state['jobs'][entry['job_id']]
            source=copy.deepcopy(job['source']);source['id']=entry['id']
            kind='recipe:'+job['recipe_id']+':'+entry['stage']
        result.append((source,kind))
    return result


def queue_sources(campaign,workspace):
    template=validate_template(campaign.setting('template'))
    entries=campaign.setting('enrollments',{})
    following=campaign.setting('following',False) and not campaign.setting('paused',False)
    if campaign.setting('paused',False):return entries
    for source,kind in choices(workspace,template['scope']['experiment']):
        key=source['id']
        if key in entries or not matches(template,source,kind):continue
        captured=key in template['existing_ids']
        future=following and key not in template['seen_ids']
        if not (captured or future):continue
        if len(entries)>=template['maximum_scans']:
            campaign.set_setting('following',False);campaign.set_setting('budget_message','Acquisition budget reached; review a new campaign to enroll more scans.')
            break
        entries[key]=dict(id=key,status='queued',scan=source['scan_label'],sample=source['sample_label'],bank=source.get('bank'))
    campaign.set_setting('enrollments',entries)
    return entries


def enroll_source(campaign,workspace,key,inspect,*,transport=None):
    """Recheck an exact source and native setting before a per-scan protocol."""
    template=validate_template(campaign.setting('template'));entries=campaign.setting('enrollments',{})
    entry=entries[key];entry['status']='preparing';campaign.set_setting('enrollments',entries)
    try:
        source=resolve_start(workspace,template['scope']['experiment'],key)
        source=workspace.runtime.resolve(source)
        if campaign.setting('paused',False):raise ValueError('Campaign paused before enrollment')
        reference=template['protocol'];root=campaign.root/'enrollment-inputs'/digest([key,source['source_sha256']])[:24]
        root.mkdir(parents=True,exist_ok=True)
        reference_root=Path(campaign.setting('reference_input_root'))
        # Capture each acquisition and child catalogue once, independent of start count.
        for asset in assets(reference):
            original=Path(source['source_path']) if asset is reference['source'] else confined(reference_root,asset['path'])
            target=confined(root,asset['path'],exists=False);target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():shutil.copyfile(original,target)
            if file_sha(target)!=(source['source_sha256'] if asset is reference['source'] else asset['sha256']):
                raise ValueError('A captured model input changed')
        context=(template['context'] if key==reference.get('acquisition_id') and source['source_sha256']==reference['source']['sha256']
                 else inspect(source,root))
        protocol=rebind(template,context,source,origin(source,workspace))
        metadata=dict(scan=source['scan_label'],sample=source['sample_label'],bank=source.get('bank'),series=source['experiment'])
        # Use the same qualified condition metadata as ordinary Refinement trends.
        from .scan_conditions import qualified_metadata
        condition=qualified_metadata(source.get('metadata') or {}).get('temperature') or {}
        if isinstance(condition,dict) and condition.get('value') is not None and condition.get('unit'):
            metadata.update(condition=condition['value'],condition_unit=condition['unit'])
        if campaign.setting('paused',False):raise ValueError('Campaign paused while its input was inspected')
        campaign.enroll(protocol,root,approved_hash=digest(protocol),metadata=metadata)
        if transport:transport.bind_source(digest(protocol),source)
        entry.update(status='enrolled',protocol=digest(protocol),source_sha256=source['source_sha256'])
    except Exception as exc:
        entry.update(status='queued' if campaign.setting('paused',False) else 'needs_review',reason=str(exc))
    entries[key]=entry;campaign.set_setting('enrollments',entries)
    return entry


def restore_history(root,runtime):
    """Restore owned campaign ledgers plus immutable input-HDA references."""
    import gzip,json,re,uuid
    from gpx_recipe.symmetry.bundle import unpack
    from gpx_recipe.symmetry.campaign import Campaign
    from gpx_recipe.symmetry.compiler import prepare_shared_inputs
    from .model_runtime import CAMPAIGN_LABEL,identifier as checked_id,document_label
    seen=set();empty=0
    for offset in range(0,5000,500):
        rows=runtime.service.search_history_datasets(query=CAMPAIGN_LABEL,limit=500,offset=offset,include_generated=True)
        empty=empty+1 if not rows else 0
        if empty>=2:break
        for row in rows:
            name=document_label(row.get('name'))
            if not name.startswith(CAMPAIGN_LABEL):continue
            identifier=name[len(CAMPAIGN_LABEL):]
            if identifier in seen:continue
            seen.add(identifier)
            if not re.fullmatch('campaign-[a-f0-9]{12}',identifier):continue
            campaign=Campaign(Path(root)/identifier,shared_inputs=True)
            if campaign.trials():continue
            archive=campaign.root/('restore-'+uuid.uuid4().hex+'.json.gz')
            runtime.service._download_dataset(row['id'],archive)
            try:
                with gzip.open(archive,'rb') as f:payload=f.read(128*1024*1024+1)
                if len(payload)>128*1024*1024:raise ValueError('Model checkpoint exceeds its recovery bound')
                document=json.loads(payload)
            finally:archive.unlink(missing_ok=True)
            if document.get('schema')!='radar-model-campaign/v1' or document.get('history_id')!=runtime.service.history_id or document.get('campaign')!=identifier:
                raise ValueError('The saved model campaign belongs to another History/identity')
            transport=document['transport'];inputs={}
            for sha,dataset in transport['inputs'].items():
                if not re.fullmatch('[a-f0-9]{64}',sha):raise ValueError('Invalid saved input identity')
                target=campaign.root/'recovering'/uuid.uuid4().hex;target.mkdir(parents=True)
                packet=target/'inputs.zip';runtime.service._download_dataset(checked_id(dataset),packet)
                unpack(packet,target/'unpacked');protocol=json.loads((target/'unpacked/protocol.json').read_text())
                if digest(protocol)!=sha:raise ValueError('The saved model input catalogue changed')
                inputs[sha]=prepare_shared_inputs(protocol,target/'unpacked',campaign.root)
            template=validate_template(document['ledger']['settings']['template'])
            if digest(template['protocol']) not in inputs:raise ValueError('The reviewed reference inputs are absent from the checkpoint')
            campaign.restore(document['ledger'],inputs)
            campaign.set_setting('reference_input_root',str(inputs[digest(template['protocol'])]))
            # Local paths in an external checkpoint never select files on this machine.
            for source in transport['sources'].values():source.pop('source_path',None);source.pop('context',None)
            write_json(campaign.root/'galaxy.json',transport)
            # A restored viewer is read-only until the user chooses Continue.
            campaign.set_setting('restored',True)
    return seen
