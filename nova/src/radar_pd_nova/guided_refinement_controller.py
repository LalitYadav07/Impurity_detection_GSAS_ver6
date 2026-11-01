"""Small guided form; scientific choices and compilation stay in Python rules."""
import asyncio
from copy import deepcopy
from hashlib import sha256
import os
from pathlib import Path
import uuid

from gpx_recipe import guided
from gpx_recipe.execution import write_json
from gpx_recipe.model_edits import SCHEMA as EDIT_SCHEMA


class GuidedRefinementMixin:
    def _init_guided_refinement(self):
        self._guided_state=None; self._guided_source_id=None; self._guided_proposal=None
        self._guided_binding=None
        self._guided_epoch=uuid.uuid4().hex
        defaults=dict(rf_setup_mode='guided',rf_guided_busy=False,rf_guide_question=None,rf_guide_token='',
            rf_guide_answer=None,rf_guide_text='',rf_guide_proposal='',rf_guide_can_accept=False,
            rf_guide_complete=False,rf_guide_summary=[],rf_guide_summary_notice='',rf_transfer_notice='',rf_edit_operation='replace',rf_edit_phase=None,
            rf_edit_name='',rf_edit_cif_path=None,rf_edit_cif_text='',rf_edit_banks=[],rf_edit_scale=1.,rf_edit_reason='',
            rf_edit_phase_options=[],rf_edit_bank_options=[],rf_edit_source_sha='')
        for key,value in defaults.items():setattr(self.server.state,key,value)

    def _guided_source_changed(self):
        if not hasattr(self,'_guided_state'):return
        s=self.server.state
        source=(self._refinement_workspace.state['sources'].get(s.rf_source) if self._refinement_workspace else None) or {}
        context=source.get('context') or {}
        binding=(s.rf_source,source.get('source_sha256') or context.get('source',{}).get('sha256'))
        if self._guided_binding!=binding:
            self._guided_binding=binding
            self._guided_epoch=uuid.uuid4().hex;self._guided_state=None;self._guided_proposal=None
            self._guided_source_id=s.rf_source
            s.rf_guide_question=None;s.rf_guide_complete=False;s.rf_guide_summary=[];s.rf_guide_summary_notice=''
            s.rf_guide_proposal='';s.rf_guide_can_accept=False
            s.rf_guide_answer=None;s.rf_guide_text='';s.rf_edit_phase=None
            s.rf_edit_name='';s.rf_edit_cif_path=None;s.rf_edit_cif_text='';s.rf_edit_banks=[];s.rf_edit_scale=1.;s.rf_edit_reason=''
        s.rf_edit_source_sha=(context.get('source') or {}).get('sha256','')
        s.rf_edit_phase_options=[dict(title=p['name'],value=p['key']) for p in context.get('phases',[])]
        banks=source.get('metadata',{}).get('physical_banks',{})
        s.rf_edit_bank_options=[dict(title=f"Bank {banks[h['name']]}" if h['name'] in banks else h['name'],value=h['key']) for h in context.get('histograms',[])]
        if context and self._guided_state is None and self._refinement_workspace:
            draft=self._refinement_workspace.state.get('drafts',{}).get(getattr(s,'rf_experiment',''),{})
            saved=(draft.get('guided') if draft.get('source_id')==s.rf_source and draft.get('plan') else None)
            saved=saved or self._refinement_workspace.state.get('guided_setups',{}).get(s.rf_source)
            if saved:
                try:guided.next_question(context,saved)
                except ValueError:
                    # Reusing a portable recipe preserves its original decisions
                    # as read-only provenance, not a new editable setup.
                    if draft.get('guided') is saved and draft.get('context',{}).get('source',{}).get('sha256')==context['source']['sha256']:
                        s.rf_guide_summary=guided.decision_summary(context,saved)
                        s.rf_guide_summary_notice='These decisions were recorded for the saved recipe’s original starting project. Review their applicability to the selected GPX before approval.'
                else:
                    self._guided_state=deepcopy(saved);self._guided_render(persist=False)

    def _guided_context(self):
        s=self.server.state;w=self._refinement_workspace
        if not w or not self._guided_state or self._guided_source_id!=s.rf_source:
            raise ValueError('Start guided setup for the selected GPX first.')
        context=w.state['sources'][s.rf_source]['context']
        guided.check_binding(context,self._guided_state)
        return context

    def _guided_render(self, *, persist=True):
        s=self.server.state;context=self._guided_context()
        question=guided.next_question(context,self._guided_state)
        s.rf_guide_question=question;s.rf_guide_token=guided.question_token(context,self._guided_state)
        s.rf_guide_complete=question is None
        s.rf_guide_answer=[] if question and question['kind']=='multiple' else None
        s.rf_guide_text='';s.rf_guide_proposal='';s.rf_guide_can_accept=False;self._guided_proposal=None
        s.rf_guide_summary=guided.decision_summary(context,self._guided_state)
        s.rf_guide_summary_notice=''
        if persist:
            with self._refinement_workspace.lock:
                self._refinement_workspace.state.setdefault('guided_setups',{})[s.rf_source]=deepcopy(self._guided_state)
                self._refinement_workspace.save()
        s.flush()

    def start_guided_refinement(self,**_):
        s=self.server.state;w=self._refinement_workspace
        if s.rf_busy or s.rf_guided_busy or not w or not s.rf_source:return
        source_id=s.rf_source;epoch=self._guided_epoch
        self._refinement_hold_enrollment();w.discard_draft(s.rf_experiment)
        s.rf_guided_busy=True;s.rf_error='';s.rf_message='';s.rf_request='';s.flush()
        async def work():
            try:
                source=await asyncio.to_thread(w.inspect,source_id)
                if epoch!=self._guided_epoch or source_id!=s.rf_source:return
                self._guided_source_id=source_id;self._guided_state=guided.start(source['context'])
                self._guided_binding=(source_id,source['source_sha256'])
                self._guided_render()
            except Exception as exc:
                if epoch==self._guided_epoch and source_id==s.rf_source:s.rf_error=str(exc)
            finally:s.rf_guided_busy=False;self._refinement_sync();s.flush()
        self._refinement_schedule(work(),'guided-refinement-start')

    def answer_guided_refinement(self,payload=None,**_):
        s=self.server.state
        if s.rf_busy or s.rf_guided_busy:return
        try:
            payload=payload or {};context=self._guided_context()
            if payload.get('token')!=guided.question_token(context,self._guided_state):
                raise ValueError('This answer belongs to an earlier question. Review the current form.')
            question=guided.next_question(context,self._guided_state);value=payload.get('value')
            if question is None:raise ValueError('All questions are already answered. Review and build the recipe.')
            if question['kind']=='number':
                if isinstance(value,bool) or value is None:raise ValueError('Enter a finite positive ion count per unit cell.')
                value=float(value)
            self._guided_state=guided.answer(context,self._guided_state,question['id'],value)
            self._refinement_workspace.discard_draft(s.rf_experiment)
            s.rf_error='';self._guided_render();self._refinement_sync()
        except Exception as exc:s.rf_error=str(exc)
        s.flush()

    def interpret_guided_answer(self,payload=None,**_):
        s=self.server.state
        if s.rf_busy or s.rf_guided_busy:return
        try:
            payload=payload or {};context=self._guided_context();state=deepcopy(self._guided_state)
            if payload.get('token')!=guided.question_token(context,state):raise ValueError('The question changed; enter the answer again.')
            utterance=payload.get('text');epoch=self._guided_epoch
        except Exception as exc:s.rf_error=str(exc);return
        s.rf_guided_busy=True;s.rf_error='';s.rf_guide_can_accept=False;s.flush()
        def call():
            from gpx_recipe.providers import configure_provider,IllumineLuna
            configure_provider('illumine-luna',os.getenv('RADAR_PD_REFINEMENT_KEY_FILE') or None)
            return guided.interpret(context,state,utterance,IllumineLuna())
        async def work():
            try:
                proposal=await asyncio.to_thread(call)
                write_json(self._refinement_workspace.root/'guided_calls'/(uuid.uuid4().hex+'.json'),proposal)
                if epoch!=self._guided_epoch or proposal['token']!=guided.question_token(self._guided_context(),self._guided_state):return
                self._guided_proposal=proposal
                s.rf_guide_can_accept=proposal['acceptable']
                q=guided.next_question(context,state);labels={o['value']:o['title'] for o in q['options']}
                value=proposal['value']
                display=', '.join(labels.get(x,x) for x in value) if isinstance(value,list) else labels.get(value,value) if isinstance(value,str) else value
                s.rf_guide_proposal=('Suggested answer: '+str(display)+'. Review before using it.' if proposal['acceptable']
                    else 'No answer applied. '+('; '.join(proposal['raw']['unhandled']) or 'The interpretation was ambiguous; use the displayed choices or clarify your answer.'))
            except Exception as exc:
                if epoch==self._guided_epoch:s.rf_error=str(exc)
            finally:s.rf_guided_busy=False;s.flush()
        self._refinement_schedule(work(),'guided-answer-interpret')

    def accept_guided_answer(self,payload=None,**_):
        s=self.server.state
        if s.rf_busy or s.rf_guided_busy:return
        try:
            proposal=self._guided_proposal;context=self._guided_context()
            if proposal is None:raise ValueError('There is no current suggestion to accept.')
            if not payload or payload.get('text')!=proposal['utterance'] or payload.get('token')!=proposal['token']:
                raise ValueError('The answer text or question changed. Interpret the current answer again.')
            updated=guided.accept_proposal(context,self._guided_state,proposal)
            if proposal.get('usage'):updated.setdefault('model_calls',[]).append(proposal['usage'])
            self._guided_state=updated;s.rf_error=''
            self._refinement_workspace.discard_draft(s.rf_experiment)
            self._guided_render();self._refinement_sync()
        except Exception as exc:s.rf_error=str(exc)
        s.flush()

    def compile_guided_refinement(self,**_):
        s=self.server.state;w=self._refinement_workspace
        if s.rf_busy or s.rf_guided_busy:return
        try:
            context=self._guided_context();state=deepcopy(self._guided_state)
            plan=guided.build_plan(context,state);source_id=s.rf_source;experiment=s.rf_experiment;epoch=self._guided_epoch
            s.rf_request=plan['request'];s.rf_guided_busy=True;s.rf_error='';s.rf_message='';s.flush()
        except Exception as exc:s.rf_error=str(exc);return
        async def work():
            try:
                draft=await asyncio.to_thread(w.prepare,experiment,source_id,plan['request'],guided=state)
                if draft and epoch==self._guided_epoch and s.rf_source==source_id and s.rf_experiment==experiment:
                    s.rf_message='Guided recipe checked by the native engine. Review its free parameters and assumptions before approving.'
            except Exception as exc:
                if epoch==self._guided_epoch and s.rf_source==source_id and s.rf_experiment==experiment:s.rf_error=str(exc)
            finally:s.rf_guided_busy=False;self._refinement_sync();self._refinement_checkpoint();s.flush()
        self._refinement_schedule(work(),'guided-recipe-preview')

    def correct_refinement_phase(self,payload=None,**_):
        s=self.server.state;w=self._refinement_workspace
        if s.rf_busy or s.rf_guided_busy or not w:return
        try:
            payload=payload or {};source_id=payload.get('source');source=w.state['sources'].get(source_id,{})
            if source_id!=s.rf_source or not source.get('context'):raise ValueError('Inspect the selected GPX before correcting phases.')
            context=source['context']
            if payload.get('source_sha256')!=context['source']['sha256']:
                raise ValueError('The phase list changed. Inspect the current model before correcting it.')
            request=dict(schema=EDIT_SCHEMA,source_sha256=context['source']['sha256'],model_signature=context['model_signature'],
                         operation=payload.get('operation'),reason=payload.get('reason'))
            if request['operation']!='add':request['phase']=payload.get('phase')
            if request['operation']!='remove':
                text=payload.get('cif_text') or ''
                if payload.get('cif_path'):
                    from .uploads import read_browser_upload
                    text=read_browser_upload(payload['cif_path'], max_bytes=1024*1024, suffix='.cif').decode('utf-8-sig')
                request.update(cif=text,cif_sha256=sha256(text.encode()).hexdigest())
            if request['operation']=='add':
                request.update(name=payload.get('name'),links=[dict(histogram=h,scale=float(payload.get('scale'))) for h in payload.get('banks',[])])
            from gpx_recipe.model_edits import validate_edit
            validate_edit(context,request)
        except Exception as exc:s.rf_error=str(exc);s.flush();return
        self._refinement_hold_enrollment();s.rf_guided_busy=True;s.rf_error='';s.flush()
        async def work():
            try:
                edited=await asyncio.to_thread(w.edit_source,source_id,request)
                if s.rf_source==source_id:
                    s.rf_scope='scans';s.rf_scans=[edited['scan_key']];s.rf_banks=[];s.rf_future=False;s.rf_source=edited['id'];s.rf_request=''
                    self._guided_source_changed()
                s.rf_message='New phase-model revision created. Review its phase/bank settings and restart guided setup; previous recipes were invalidated.'
            except Exception as exc:s.rf_error=str(exc)
            finally:s.rf_guided_busy=False;self._refinement_sync();self._refinement_checkpoint();s.flush()
        self._refinement_schedule(work(),'refinement-phase-correction')
