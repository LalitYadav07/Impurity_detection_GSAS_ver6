"""Explicit model campaign enrollment, independent of the visible UI selection."""
import copy
import json
from collections import Counter
from pathlib import Path
import shutil
from contextlib import nullcontext

from gpx_recipe.symmetry.compiler import assets,confined,file_sha
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.symmetry.templates import matches,rebind,validate_template,trial_count
from gpx_recipe.execution import write_json
from .symmetry_sources import start_choices,resolve_start,recipe_source

START_BARRIER_SCHEMA='radar-model-start-barrier/v1'
START_BARRIER_MESSAGE='Initial model preparation did not finish. Review the same model setup and use Start again; Continue cannot skip its source checkpoint.'


def start_barrier_ready(campaign):
    """Legacy campaigns predate this marker; new staged campaigns need proof."""
    marker=campaign.setting('start_barrier')
    if marker is None:return True
    return (isinstance(marker,dict) and marker.get('schema')==START_BARRIER_SCHEMA
            and marker.get('state')=='complete'
            and marker.get('template_sha256')==digest(campaign.setting('template')))


def store_start_barrier(campaign,marker,*,complete=False,following=False,current_only=False):
    """Atomically arm or disarm one owned initial Start, including on restore."""
    with campaign.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT body FROM settings WHERE key='start_barrier'").fetchone()
        current=json.loads(row['body']) if row else None
        if current_only and (not isinstance(current,dict) or current.get('attempt')!=marker['attempt']):return False
        _set_setting(db,'start_barrier',dict(marker,state='complete' if complete else 'incomplete'))
        _set_setting(db,'paused',not complete)
        _set_setting(db,'following',bool(complete and following))
    return True


def scope_progress(template,entries):
    """Keep the reviewed total visible while individual scans enter the ledger."""
    if not entries:return ''
    counts=Counter(entry['status'] for entry in entries)
    label=(f"{len(entries)} captured scans · {counts['enrolled']} prepared · "
           f"{counts['queued']+counts['preparing']} awaiting preparation · {counts['needs_review']} need review.")
    if template.get('protocol'):
        label+=f" Reviewed budget: {len(entries)*trial_count(template):,} starts before per-scan compatibility checks."
        policy=template['protocol'].get('policy',{})
        if 'cycles' in policy:
            label+=f" Up to {policy['cycles']} refinement cycles per start."
            if policy['cycles']==1:label+=' Preliminary one-cycle screening.'
    return label


def portable_source(source):
    """Capture Galaxy references and acquisition provenance, never local paths."""
    source=copy.deepcopy(source)
    for key in ['source_path','context','inspection','inspection_error']:
        source.pop(key,None)
    return source


def captured_source(entry,template):
    source=entry.get('source')
    if (entry.get('origin')!='radar' or not isinstance(source,dict) or
            source.get('id')!=entry.get('id') or not matches(template,source,'radar')):
        raise ValueError('The captured acquisition is absent or outside the reviewed campaign scope')
    if (source['id'] not in template['existing_ids'] and
            not (template.get('future') and source['id'] not in template['seen_ids'])):
        raise ValueError('The captured acquisition was not selected at approval or eligible for future enrollment')
    if not source.get('collection_id') or not source.get('collection_name'):
        raise ValueError('The captured acquisition has no immutable Galaxy publication reference')
    return portable_source(source)


def restore_captured_sources(campaign,workspace):
    if getattr(getattr(workspace,'runtime',None),'kind',None)!='galaxy':return 0
    template=validate_template(campaign.setting('template'))
    if len(campaign.setting('enrollments',{}))>template['maximum_scans']:
        raise ValueError('Captured acquisitions exceed the reviewed campaign budget')
    sources=[]
    for entry in campaign.setting('enrollments',{}).values():
        if entry.get('origin')=='radar' and entry.get('source'):
            source=captured_source(entry,template)
            current=workspace.state.get('sources',{}).get(source['id'])
            if current and any(current.get(field)!=source.get(field) for field in
                    ['collection_id','collection_name','internal_source_sha256','published_sha256']):
                raise ValueError('The captured publication differs from the current source; review its revision')
            if current and current.get('live_watch'):
                if not matches(template,current,'radar'):
                    raise ValueError('The reviewed acquisition mapping changed; review the campaign scope')
                continue
            sources.append(source)
    if sources:workspace.observe(sources)
    return len(sources)


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
            source=recipe_source(workspace,job);source['id']=entry['id']
            kind='recipe:'+job['recipe_id']+':'+entry['stage']
        result.append((source,kind))
    return result


def queue_sources(campaign,workspace):
    if not start_barrier_ready(campaign):return campaign.setting('enrollments',{})
    template=validate_template(campaign.setting('template'))
    available=choices(workspace,template['scope']['experiment'])
    # Input inspection can finish while a newly arriving acquisition is being
    # queued. Merge under the SQLite writer guard, never replace an old snapshot.
    with campaign.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        settings={r['key']:json.loads(r['body']) for r in db.execute(
            "SELECT key,body FROM settings WHERE key IN ('enrollments','following','paused')")}
        entries=settings.get('enrollments',{})
        if settings.get('paused',False):return entries
        following=settings.get('following',False)
        for source,kind in available:
            key=source['id']
            if key in entries or not matches(template,source,kind):continue
            captured=key in template['existing_ids']
            future=following and key not in template['seen_ids']
            if not (captured or future):continue
            if len(entries)>=template['maximum_scans']:
                _set_setting(db,'following',False)
                _set_setting(db,'budget_message','Acquisition budget reached; review a new campaign to enroll more scans.')
                break
            entries[key]=dict(id=key,status='queued',scan=source['scan_label'],sample=source['sample_label'],bank=source.get('bank'),
                source=portable_source(source),origin=kind)
        _set_setting(db,'enrollments',entries)
    return entries


def _set_setting(db,key,value):
    db.execute('INSERT OR REPLACE INTO settings VALUES(?,?)',(key,json.dumps(value,allow_nan=False)))


def _set_enrollment(campaign,key,value):
    with campaign.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row=db.execute("SELECT body FROM settings WHERE key='enrollments'").fetchone()
        entries=json.loads(row['body']) if row else {}
        entries[key]=value;_set_setting(db,'enrollments',entries)


def prioritize_acquisition(campaign, key):
    """Change admission order only within the already captured finite cohort."""
    with campaign.connect() as db:
        db.execute('BEGIN IMMEDIATE')
        row = db.execute("SELECT body FROM settings WHERE key='enrollments'").fetchone()
        entries = json.loads(row['body']) if row else {}
        entry = entries.get(key) if isinstance(key, str) else None
        if not entry or entry.get('id') != key or entry.get('status') != 'queued':
            raise ValueError('Choose a captured acquisition that is still awaiting preparation')
        _set_setting(db, 'priority_acquisition', key)
    return entry


def acquisition_rows(template, entries):
    """Keep the reviewed cohort order across sorted-JSON checkpoint recovery."""
    captured = {key: index for index, key in enumerate(template.get('existing_ids', []))}
    def order(entry):
        source = entry.get('source') or {}
        run = source.get('run_number')
        number = int(str(run)) if str(run).isdecimal() else float('inf')
        bank = entry.get('bank')
        bank_number = int(str(bank)) if str(bank).isdecimal() else float('inf')
        return (captured.get(entry['id'], len(captured)), number, bank_number,
                str(entry.get('scan') or ''), entry['id'])
    return sorted(entries.values(), key=order)


def next_acquisition(campaign, entries):
    """Finish acknowledged preparation before honoring a queued priority scan."""
    preparing = next((entry for entry in entries.values() if entry['status'] == 'preparing'), None)
    if preparing:
        return preparing
    priority = campaign.setting('priority_acquisition')
    preferred = entries.get(priority) if isinstance(priority, str) else None
    if preferred and preferred['status'] == 'queued':
        return preferred
    return next((entry for entry in acquisition_rows(campaign.setting('template') or {}, entries)
                 if entry['status'] == 'queued'), None)


def enroll_source(campaign,workspace,key,inspect,*,transport=None):
    """Recheck an exact source and native setting before a per-scan protocol."""
    if not start_barrier_ready(campaign):raise ValueError(START_BARRIER_MESSAGE)
    template=validate_template(campaign.setting('template'));entries=campaign.setting('enrollments',{})
    entry=entries[key];entry['status']='preparing';_set_enrollment(campaign,key,entry)
    try:
        try:source=resolve_start(workspace,template['scope']['experiment'],key)
        except ValueError:
            # A fresh viewer need not have the Live monitor open. Recover the
            # exact captured publication, then verify its bytes and native
            # compatibility through the same runtime as ordinary enrollment.
            source=captured_source(entry,template)
            current=workspace.state.get('sources',{}).get(key)
            if current and current.get('live_watch') and not matches(template,current,'radar'):
                raise ValueError('The reviewed acquisition mapping changed; review the campaign scope')
            if current and any(current.get(field)!=source.get(field) for field in
                    ['collection_id','collection_name','internal_source_sha256','published_sha256']):
                raise ValueError('The captured publication differs from the current source; review its revision')
            workspace.observe([source])
        source=workspace.runtime.resolve(source)
        if entry.get('origin')=='radar':entry['source']=portable_source(source)
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
        if transport:
            # A simultaneous checkpoint must never contain a new protocol
            # without the Galaxy inputs required to restore its trials.
            transport.prepare_source(protocol,root,source)
        if campaign.setting('paused',False):raise ValueError('Campaign paused while its input was inspected')
        campaign.enroll(protocol,root,approved_hash=digest(protocol),metadata=metadata)
        entry.update(status='enrolled',protocol=digest(protocol),source_sha256=source['source_sha256'])
    except Exception as exc:
        entry.update(status='queued' if campaign.setting('paused',False) else 'needs_review',reason=str(exc))
    _set_enrollment(campaign,key,entry)
    return entry


def _recovery_source(document,sha,protocol,template):
    """Recover an old reference binding only through its exact enrolled copy.

    Early thin checkpoints captured the reviewed reference packet before its
    per-acquisition protocol, without binding the former to a Galaxy GPX. The
    latter can supply that binding only when the saved ledger proves it is the
    same approved reference, including its exact bytes and campaign scope.
    """
    transport=document['transport']
    source=transport['sources'].get(sha)
    if source is None and sha==digest(template['protocol']):
        key=protocol.get('acquisition_id')
        entry=document['ledger']['settings'].get('enrollments',{}).get(key) or {}
        rebound_sha=entry.get('protocol')
        rebound=next((json.loads(row['body']) for row in document['ledger']['protocols']
                      if row['hash']==rebound_sha),None)
        expected=copy.deepcopy(protocol)
        expected['template_sha256']=digest(template)
        candidate=transport['sources'].get(rebound_sha) or {}
        if (entry.get('status')=='enrolled' and entry.get('origin')==template['origin']
                and entry.get('source_sha256')==protocol['source']['sha256']
                and rebound is not None and digest(rebound)==rebound_sha
                and rebound==expected and candidate.get('id')==key
                and candidate.get('source_sha256')==protocol['source']['sha256']
                and matches(template,candidate,entry['origin'])):
            source=candidate
    source=portable_source(source or {})
    if source.get('source_sha256')!=protocol['source']['sha256']:
        raise ValueError('The saved source reference differs from its approved protocol')
    return source


def restore_history(root,runtime,*,experiment=None,progress=None):
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
            if campaign.trials() and not campaign.setting('restore_in_progress',False):continue
            archive=campaign.root/('restore-'+uuid.uuid4().hex+'.json.gz')
            runtime.service._download_dataset(row['id'],archive)
            try:
                with gzip.open(archive,'rb') as f:payload=f.read(128*1024*1024+1)
                if len(payload)>128*1024*1024:raise ValueError('Model checkpoint exceeds its recovery bound')
                document=json.loads(payload)
            finally:archive.unlink(missing_ok=True)
            if document.get('schema')!='radar-model-campaign/v1' or document.get('history_id')!=runtime.service.history_id or document.get('campaign')!=identifier:
                raise ValueError('The saved model campaign belongs to another History/identity')
            template=validate_template(document['ledger']['settings']['template'])
            # A new NOMAD search must not wait for an unrelated POWGEN ledger,
            # its GPX inputs and every completed refinement to be recovered.
            if experiment is not None and template['scope']['experiment'] != experiment:
                continue
            transport=document['transport'];inputs={}
            total=len(transport['inputs'])
            for index,(sha,dataset) in enumerate(transport['inputs'].items(),1):
                if progress:progress(f'Recovering {identifier}: input {index} of {total}. No new refinements are being submitted.')
                if not re.fullmatch('[a-f0-9]{64}',sha):raise ValueError('Invalid saved input identity')
                target=campaign.root/'recovering'/uuid.uuid4().hex;target.mkdir(parents=True)
                packet=target/'inputs.zip';runtime.service._download_dataset(checked_id(dataset),packet)
                unpack(packet,target/'unpacked');protocol=json.loads((target/'unpacked/protocol.json').read_text())
                if digest(protocol)!=sha:raise ValueError('The saved model input catalogue changed')
                source_path=target/'unpacked'/protocol['source']['path']
                if not source_path.is_file():
                    from gpx_recipe.symmetry.bundle import hydrate_source
                    source=_recovery_source(document,sha,protocol,template)
                    resolved=runtime.resolve(source)
                    hydrate_source(target/'unpacked',resolved['source_path'],
                        {asset['path']:asset['sha256'] for asset in assets(protocol)},protocol['source']['sha256'])
                    transport['sources'][sha]=portable_source(resolved)
                inputs[sha]=prepare_shared_inputs(protocol,target/'unpacked',campaign.root)
            if digest(template['protocol']) not in inputs:raise ValueError('The reviewed reference inputs are absent from the checkpoint')
            if progress:progress(f'Rebuilding {len(document["ledger"]["trials"]):,} saved starts for {identifier}. No new refinements are being submitted.')
            from .model_ledger_recovery import restore_ledger
            restore_ledger(campaign,document['ledger'],inputs,
                progress=(lambda message: progress(f'{identifier}: {message}. No new refinements are being submitted.')) if progress else None)
            campaign.set_setting('reference_input_root',str(inputs[digest(template['protocol'])]))
            # Local paths in an external checkpoint never select files on this machine.
            for source in transport['sources'].values():source.pop('source_path',None);source.pop('context',None)
            write_json(campaign.root/'galaxy.json',transport)
            # A restored viewer is read-only until the user chooses Continue.
            campaign.set_setting('restored',True)
            campaign.set_setting('checkpoint_updated_utc',row.get('update_time') or '')
            campaign.set_setting('restore_in_progress',False)
    return seen
