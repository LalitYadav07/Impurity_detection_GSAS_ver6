"""Exact multi-HDA input capture, hostile archives, and old packet isolation."""
import copy
import json
from pathlib import Path
import shutil
import stat
import zipfile

import pytest

from gpx_recipe.preparation import structural_input_parts as parts
from gpx_recipe.preparation import structural_jobs as jobs
from gpx_recipe.symmetry.compiler import file_sha,package_signature
from test_structural_paired_preparation import inputs


def captured(root,monkeypatch):
    example,first=inputs(root)
    second=root/'second';shutil.copytree(first,second)
    path=root/'plan.json';jobs.create_plan(example['source'],[first,second],path)
    names=['parent.cif','profile.instprm','plan.json']+jobs.plan_inputs(path)
    request=dict(schema='radar-model-native/v1',operation=parts.OPERATION,
        implementation_sha256=package_signature(),expected_source_sha256=file_sha(root/'parent.cif'),
        dispatch_key='d'*24,timeout=120,payload=dict(source='parent.cif',instrument='profile.instprm',
            plan='plan.json',destination='trained',grid=example['grid'],
            settings=dict(representation=example['settings']['representation'])),
        input_hashes={name:file_sha(root/name) for name in names})
    # Real closed packets, with only their size thresholds reduced for tests.
    monkeypatch.setattr(parts,'MAX_PART_BYTES',12000)
    monkeypatch.setattr(parts,'MAX_METADATA',7000)
    return request,names


def packet(root,request,out):
    published={}
    out.mkdir()
    def publish(signature,path):
        key='hda'+str(len(published));target=out/(key+'.zip');shutil.copyfile(path,target)
        published[key]=target;return key
    reference=parts.prepare(root,request,out/'temporary',publish)
    primary=published[reference['primary']]
    return primary,[published[key] for key in reference['parts']],reference


def rewrite(path,mutate):
    with zipfile.ZipFile(path) as archive:
        rows=[(item,archive.read(item)) for item in archive.infolist()]
    with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as archive:
        for item,data in rows:
            if item.filename==parts.INDEX:
                index=json.loads(data);mutate(index);data=json.dumps(index).encode()
            archive.writestr(item,data)


def test_parts_reconstruct_the_complete_request_and_never_publish_transport_metadata(tmp_path,monkeypatch):
    root=tmp_path/'capture';request,names=captured(root,monkeypatch)
    primary,more,reference=packet(root,request,tmp_path/'packets')
    assert len(more)==2 and parts.references(reference)==(reference['primary'],reference['parts'])
    target=parts.stage(primary,more,request,tmp_path/'worker-inputs')
    assert {str(p.relative_to(target)):file_sha(p) for p in target.rglob('*') if p.is_file()}==request['input_hashes']
    assert not (target/parts.INDEX).exists() and not list((tmp_path/'packets/temporary').iterdir())
    assert len(jobs.checked_blocks(target/'plan.json',jobs.read_plan(target/'plan.json')))==2
    with pytest.raises(ValueError,match='fresh owned'):parts.stage(primary,more,request,target)


@pytest.mark.parametrize('change',['source','profile','plan','operation','representation','extra_file','grid'])
def test_foreign_or_untyped_scope_cannot_claim_multipart_budget(tmp_path,monkeypatch,change):
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    primary,more,_=packet(root,request,tmp_path/'packets')
    changed=copy.deepcopy(request)
    if change=='source':changed['expected_source_sha256']='f'*64
    elif change=='profile':
        with zipfile.ZipFile(primary) as archive:rows=[(entry,archive.read(entry)) for entry in archive.infolist()]
        with zipfile.ZipFile(primary,'w') as archive:
            for entry,data in rows:archive.writestr(entry,b'foreign calibration\n' if entry.filename=='profile.instprm' else data)
    elif change=='plan':changed['payload']['plan']='missing.json'
    elif change=='operation':changed['operation']='inspect'
    elif change=='representation':changed['payload']['settings'].pop('representation')
    elif change=='grid':changed['payload']['grid']['bank']=2
    else:changed['input_hashes']['unreviewed.json']='e'*64
    with pytest.raises((ValueError,OSError)):
        parts.stage(primary,more,changed,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists()


@pytest.mark.parametrize('change',['missing','order','bytes','file_hash','total','duplicate','case_collision','traversal','symlink'])
def test_missing_corrupt_or_ambiguous_archives_fail_without_a_published_partial_input(tmp_path,monkeypatch,change):
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    primary,more,_=packet(root,request,tmp_path/'packets')
    if change=='missing':more=more[:-1]
    elif change=='order':more=list(reversed(more))
    elif change=='bytes':
        with more[0].open('ab') as stream:stream.write(b'changed')
    elif change=='file_hash':rewrite(primary,lambda index:index['files']['examples/examples.npz'].update(sha256='0'*64))
    elif change=='total':monkeypatch.setattr(jobs,'MAX_PAIRED_INPUT_BYTES',1)
    elif change=='duplicate':rewrite(primary,lambda index:index['parts'][1].update(files=index['parts'][0]['files']))
    else:
        # Even a self-consistent archive checksum cannot authorize new paths,
        # case aliases, or a symlink within an otherwise declared member list.
        old=more[0]
        with zipfile.ZipFile(old) as archive:rows=[(e,archive.read(e)) for e in archive.infolist()]
        with zipfile.ZipFile(old,'w') as archive:
            for i,(entry,data) in enumerate(rows):
                if i==0:
                    if change=='case_collision':entry.filename=entry.filename.upper()
                    elif change=='traversal':entry.filename='../outside.json'
                    else:entry.external_attr=(stat.S_IFLNK|0o777)<<16
                archive.writestr(entry,data)
        rewrite(primary,lambda index:index['parts'][0].update(sha256=parts.sha256(old),bytes=old.stat().st_size))
    with pytest.raises((ValueError,OSError)):
        parts.stage(primary,more,request,tmp_path/'rejected')
    assert not (tmp_path/'rejected').exists() and not (tmp_path/'outside.json').exists()


def test_input_budget_is_typed_and_preserves_the_single_archive_limit(tmp_path,monkeypatch):
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    monkeypatch.setattr(jobs,'MAX_INPUT_BYTES',1)
    # Paired plans are bounded independently; all original blocks survive.
    value=jobs.create_plan(root/'parent.cif',[root/'examples',root/'second'],root/'larger-plan.json')
    assert len(jobs.checked_blocks(root/'larger-plan.json',value))==2
    from test_structural_training_jobs import inputs as old_inputs
    with pytest.raises(ValueError,match='typed byte budget'):old_inputs(tmp_path/'legacy')
    from gpx_recipe.symmetry import bundle
    monkeypatch.setattr(bundle,'MAX_BYTES',1)
    from gpx_recipe.symmetry.bundle import pack
    pack(root,tmp_path/'legacy.zip',['parent.cif'])
    with pytest.raises(ValueError,match='expanded-size'):parts.stage(tmp_path/'legacy.zip',[],request,tmp_path/'old')


@pytest.mark.parametrize('value',[dict(schema=parts.REFERENCES,primary='x',parts=['x']),
    dict(schema=parts.REFERENCES,primary='x',parts=[]),dict(schema=parts.REFERENCES,primary='x',parts=['../x']),
    dict(schema='unknown',primary='x',parts=['y']),dict(schema=parts.REFERENCES,primary='x',parts=['y'],token='secret')])
def test_durable_references_cannot_hide_paths_duplicates_or_secrets(value):
    with pytest.raises(ValueError):parts.references(value)


def test_source_changed_before_capture_is_not_uploaded(tmp_path,monkeypatch):
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    (root/'second/examples.npz').write_bytes(b'different')
    with pytest.raises(ValueError,match='changed'):
        parts.prepare(root,request,tmp_path/'packets',lambda *args:pytest.fail('No unverified input should upload'))


def test_multipart_continuation_retains_the_exact_numerical_checkpoint_and_all_examples(tmp_path,monkeypatch):
    from gpx_recipe.preparation import structural_training_transfer as transfer
    from test_structural_training_transfer import owned
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    checkpoint=owned(root/'checkpoint')
    transfer.snapshot(checkpoint,root/'continuation',identity='a'*64,epochs=60)
    jobs.create_plan(root/'parent.cif',[root/'examples',root/'second'],root/'continued.json',resume=root/'continuation')
    request['payload']['plan']='continued.json'
    names=['parent.cif','profile.instprm','continued.json']+jobs.plan_inputs(root/'continued.json')
    request['input_hashes']={name:file_sha(root/name) for name in names}
    primary,more,_=packet(root,request,tmp_path/'packets')
    target=parts.stage(primary,more,request,tmp_path/'fresh-worker')
    assert len(jobs.checked_blocks(target/'continued.json',jobs.read_plan(target/'continued.json')))==2
    for name in (transfer.STATE,transfer.RECEIPT):
        assert (target/'resume'/name).read_bytes()==(root/'resume'/name).read_bytes()
    receipt,_=transfer.read(target/'resume',identity='a'*64,epochs=60)
    assert receipt['warmup_completed']==20 and receipt['trained_model'] is False


def test_control_index_rejects_aggregate_archive_padding_before_any_part_download(tmp_path,monkeypatch):
    root=tmp_path/'capture';request,_=captured(root,monkeypatch)
    primary,more,_=packet(root,request,tmp_path/'packets')
    inventory=parts.inspect_parts(primary,request,len(more),tmp_path)
    assert len(inventory['parts'])==2
    # The declared expanded cohort fits. Inflated ZIP/trailing bytes still
    # cannot turn 32 nominal parts into a larger unchecked download budget.
    monkeypatch.setattr(jobs,'MAX_PAIRED_INPUT_BYTES',20000)
    monkeypatch.setattr(parts,'MAX_BYTES',20000)
    rewrite(primary,lambda index:[p.update(bytes=20000) for p in index['parts']])
    # 40,000 compressed bytes plus a 20,000-byte normal control allowance fit.
    parts.inspect_parts(primary,request,len(more),tmp_path)
    monkeypatch.setattr(jobs,'MAX_PAIRED_INPUT_BYTES',19000)
    with pytest.raises(ValueError,match='aggregate transport'):
        parts.inspect_parts(primary,request,len(more),tmp_path)
