"""Native readers can save implicitly; independent audits must use copies."""
import importlib.util
from pathlib import Path
import pytest


def load(name):
    path=Path(__file__).resolve().parents[2]/'benchmarks/gpx_recipes'/(name+'.py')
    spec=importlib.util.spec_from_file_location('audit_isolation_'+name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_native_verify_keeps_delivered_and_source_bytes_even_if_reader_saves(monkeypatch,tmp_path):
    worker=load('native_worker');source=tmp_path/'input.gpx';output=tmp_path/'polished.gpx';checkpoint=tmp_path/'polished_s1.gpx'
    for p in [source,output,checkpoint]:p.write_bytes(p.name.encode())
    before={p:p.read_bytes() for p in [source,output,checkpoint]}
    def reader(src,out,*args):
        assert src!=source and out!=output
        src.write_bytes(b'implicit source save');out.write_bytes(b'implicit output save');out.with_name('polished_s1.gpx').write_bytes(b'implicit checkpoint save')
        return {'checks':['copy audit']}
    monkeypatch.setattr(worker,'_verify_copies',reader)
    assert worker.verify(source,output,{'stages':[{'id':'s1'}]}, {}, {})=={'checks':['copy audit']}
    assert all(p.read_bytes()==data for p,data in before.items())


def test_post_audit_keeps_original_bytes_when_reader_raises(monkeypatch,tmp_path):
    import json
    audit=load('post_audit');(tmp_path/'inputs').mkdir();source=tmp_path/'inputs/p.gpx';source.write_bytes(b'original');work=tmp_path/'case';work.mkdir();cp=work/'refined_s1.gpx';cp.write_bytes(b'checkpoint');(work/'plan.json').write_text(json.dumps({'stages':[{'id':'s1'}]}))
    def reader(*args):
        local=args[-1];(local/'source.gpx').write_bytes(b'implicit save');(local/'refined_s1.gpx').write_bytes(b'implicit save');raise ValueError('native read failure')
    monkeypatch.setattr(audit,'_inspect_copies',reader)
    with pytest.raises(ValueError,match='native read failure'):audit.inspect(tmp_path,'model',{'case':'A','artifact_directory':str(work)},{'A':{'project':'p'}})
    assert source.read_bytes()==b'original' and cp.read_bytes()==b'checkpoint'
