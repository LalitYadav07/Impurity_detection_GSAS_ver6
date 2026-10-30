"""Previously failing language and four-beamline integration contracts."""
from dataclasses import replace
from types import SimpleNamespace

import pytest

from radar_pd_nova.live_assistant_language import ReplyContext, SampleOption, interpret_reply
from radar_pd_nova.live_beamlines import BEAMLINES
from radar_pd_nova.live_setup_preview import inspect_inventory


@pytest.mark.parametrize('message,preset', [
    ('Ignore the Al can peaks','Al_fcc'),('exclude aluminium container peaks','Al_fcc'),
    ('mask aluminum reflections','Al_fcc'),('exclude Cu reference peaks','Cu_fcc'),
    ('mask vanadium can peaks','V_bcc')])
def test_explicit_can_peak_masks_preserve_other_presets_and_support_undo(message,preset):
    from radar_pd_nova.live_assistant import SetupConversation
    before=['Cu_fcc']
    context=ReplyContext(step='chemistry',beamline='HB2A',reference_presets=tuple(before))
    reply=interpret_reply(message,context)
    assert reply.action=='patch'
    assert reply.value=={'live_reference_masks_enabled':True,
                         'live_reference_mask_presets':list(dict.fromkeys([*before,preset]))}
    state=SimpleNamespace(live_reference_masks_enabled=False,live_reference_mask_presets=before.copy())
    conversation=SetupConversation();conversation.apply(state,reply.value,reply.message)
    assert state.live_reference_masks_enabled
    conversation.undo_last(state)
    assert not state.live_reference_masks_enabled and state.live_reference_mask_presets==before


@pytest.mark.parametrize('text',["do not exclude Al can peaks","mask all peaks","ignore Fe can peaks","exclude Al can peaks and start monitoring"])
def test_unreviewable_or_combined_mask_commands_do_not_partially_apply(text):
    reply=interpret_reply(text,ReplyContext(step='chemistry'))
    assert reply.action!='patch'


def test_can_chemistry_alone_does_not_mask_and_clear_masks_is_explicit():
    reply=interpret_reply('can Al',ReplyContext(step='chemistry'))
    assert reply.value=={'live_environment_elements':'Al'}
    reply=interpret_reply('clear reference masks',ReplyContext(step='settings',reference_presets=('Al_fcc',)))
    assert reply.value=={'live_reference_masks_enabled':False,'live_reference_mask_presets':[]}


@pytest.mark.parametrize('alias,code', [('POWGEN','PG3'),('NOMAD','NOM'),('HB2A','HB2A'),('HB-2A','HB2A'),('SNAP','SNAP')])
def test_registered_beamline_reply_keeps_exact_proposal(alias, code):
    reply = interpret_reply(alias + ' IPTS-21886')
    assert reply.action == 'patch'
    assert reply.value == {'live_beamline':code,'powgen_ipts':'IPTS-21886'}


@pytest.mark.parametrize('identity', ['oxide-sample','sample-1','bank-reference','low T sample','histogram-reference'])
@pytest.mark.parametrize('prefix',['sample ID ',''])
def test_combined_reply_preserves_complete_exact_sample_identity(identity,prefix):
    option = SampleOption(identity,identity+' — known specimen')
    context = ReplyContext(step='selection',samples=(option,),shown_samples=(option,),banks=(2,))
    reply = interpret_reply(prefix+identity+'; bank 2',context)
    assert reply.action == 'patch'
    assert reply.value['powgen_processing_sample_ids'] == [identity]
    assert reply.value['powgen_processing_banks'] == [2]
    rejected = interpret_reply(prefix+identity+'; bank 200',context)
    assert rejected.action != 'patch'  # no partial edit when one part is invalid


@pytest.mark.parametrize('instrument,facility,unit', [('HB2A','HFIR','degrees 2θ'),('SNAP','SNS','µs')])
def test_preview_uses_registered_reader_facility_and_native_units(tmp_path,monkeypatch,instrument,facility,unit):
    path=tmp_path/'native.dat';path.write_text('native reader owns format')
    run=SimpleNamespace(source_path=str(path),run_id='one',run_number=1)
    called=[]
    def runs(definition,policy):
        called.append((definition.facility,policy))
        return [run]
    def setup(source,definition,policy):
        signature=(path.stat().st_size,path.stat().st_mtime_ns)
        return SimpleNamespace(source_signature=signature,as_dict=lambda:{'owner':instrument}),[{'title':'one','value':1}]
    monkeypatch.setitem(BEAMLINES,instrument,replace(BEAMLINES[instrument],preview_runs=runs,preview_setup=setup))
    from radar_pd_nova.powgen_watch import WatchDefinition
    monkeypatch.setattr(WatchDefinition,'source_directory',property(lambda self:str(tmp_path)))
    inventory=inspect_inventory(instrument,'IPTS-21886',BEAMLINES[instrument].default_folder,
        SimpleNamespace(identity=lambda run:{'sample_id':'specimen','sample_name':'known'}),
        normalization='monitor',product_family='reviewed-family')
    assert called==[(facility,{'normalization':'monitor','product_family':'reviewed-family'})]
    assert inventory['unit']==unit
    assert inventory['setup_key']==BEAMLINES[instrument].controller.setup_metadata_key
