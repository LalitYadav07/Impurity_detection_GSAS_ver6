"""Optional k survey in Model exploration; source-bound and setup-only."""
import asyncio
import json
import os
from pathlib import Path
import queue
import threading
import time
import uuid

from gpx_recipe.symmetry.k_search import KSearch,NativeRunner,SearchStopped
from gpx_recipe.symmetry.discovery import discover_magnetic
from gpx_recipe.symmetry.chemistry import sites_from_inputs
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.execution import write_json
from .symmetry_plots import empty
from .symmetry_sources import resolve_start
from .k_search_plot import evidence


class KSearchMixin:
    def _init_k_search(self):
        self._k_task=None;self._k_cancel=None;self._k_result=None;self._k_result_identity=None
        s=self.server.state
        for name,value in dict(sx_k_budget=60,sx_k_limit=6,sx_k_workers=4,sx_k_busy=False,sx_k_message='',sx_k_rows=[],
            sx_k_choices=[],sx_k_selected='',sx_k_axis='q',sx_k_reset=0,sx_k_detail='',sx_k_notes=[],sx_k_plot=empty('Run a k survey to compare magnetic templates'),sx_k_has_result=False).items():setattr(s,name,value)
        s.change('sx_kind','sx_parent','sx_histogram','sx_sites','sx_form_factors','sx_lande','sx_directions','sx_mag_d_min','sx_max_moment','sx_sign_starts','sx_start_source','rf_experiment','sx_k_budget','sx_k_limit','sx_k_workers')(self._k_inputs_changed)
        s.change('sx_k_selected','sx_k_axis','sx_k_reset')(self._k_plot_changed)

    def _k_identity(self):
        s=self.server.state;d=self._symmetry_draft or {}
        return digest(dict(source=s.sx_start_source,experiment=s.rf_experiment,sha=d.get('context',{}).get('sha256'),
            kind=s.sx_kind,parent=s.sx_parent,histogram=s.sx_histogram,sites=list(s.sx_sites),ions=s.sx_form_factors,g=s.sx_lande,
            directions=list(s.sx_directions),cutoff=s.sx_mag_d_min,maximum=s.sx_max_moment,signs=s.sx_sign_starts,budget=s.sx_k_budget,limit=s.sx_k_limit,workers=getattr(s,'sx_k_workers',4)))

    def _k_inputs_changed(self,**_):
        if getattr(self,'_k_cancel',None):self._k_cancel.set()
        self._k_result=None;self._k_result_identity=None
        s=self.server.state;s.sx_k_has_result=False;s.sx_k_rows=[];s.sx_k_choices=[];s.sx_k_notes=[]
        s.sx_k_message='Search inputs changed; run the k survey again.' if getattr(s,'sx_k_busy',False) else ''

    def _k_plot_changed(self,**_):
        if not getattr(self,'_k_result',None):return
        s=self.server.state
        try:
            f=next(r for r in self._k_result['families'] if ','.join(r['k'])==s.sx_k_selected)
            s.sx_k_detail=(f"{f['simulated']} simulated / {f.get('available_starts',0)} eligible starts · {f['discovered']} catalog embeddings. "+f.get('reason',''))
            s.sx_k_plot=evidence(self._k_result,s.sx_k_selected,axis=s.sx_k_axis,reset=s.sx_k_reset)
        except Exception as exc:s.sx_k_plot=empty(str(exc))

    def reset_k_search_zoom(self):self.server.state.sx_k_reset+=1

    def cancel_k_search(self):
        if self._k_cancel:self._k_cancel.set();self.server.state.sx_k_message='Stopping the current native operation…'

    def find_candidate_k(self):
        if self._k_task and not self._k_task.done():return
        async def run():
            s=self.server.state;identity=None;cancel=threading.Event();self._k_cancel=cancel;wall_started=time.monotonic()
            s.sx_k_busy=True;s.sx_busy=True;s.sx_error='';s.sx_k_has_result=False;s.sx_k_message='Preparing the k survey';s.flush()
            try:
                d=self._symmetry_draft
                if not d or d['source_id']!=s.sx_start_source or s.sx_kind!='magnetic':raise ValueError('Inspect the selected nuclear GPX and choose magnetic sites first')
                identity=self._k_identity()
                sites=sites_from_inputs(s.sx_sites,s.sx_form_factors,s.sx_lande)
                source=await asyncio.to_thread(resolve_start,self._refinement_store(),s.rf_experiment,s.sx_start_source)
                source=await asyncio.to_thread(self._refinement_store().runtime.resolve,source)
                if source['source_sha256']!=d['context']['sha256']:raise ValueError('The published GPX changed; inspect the current revision')
                remote=self._model_remote()
                if remote:native=remote.k_runner(source,cancel)
                else:
                    transport=self._symmetry_transport()
                    native=NativeRunner(transport.command,environment=transport.environment,cancel_event=cancel)
                def discover(parent,k,**kwargs):return discover_magnetic(parent,k,api_key=os.environ.get('BCS_API_KEY',''),**kwargs)
                search=KSearch(self._symmetry_root()/'k-cache',native,discover,**({'clock':native.clock} if remote else {}))
                mailbox=queue.Queue();root=self._symmetry_root()/'k-surveys'/uuid.uuid4().hex
                work=asyncio.create_task(asyncio.to_thread(search.run,source,root,parent=s.sx_parent,histogram=s.sx_histogram,
                    sites=sites,directions=list(s.sx_directions),maximum_moment=float(s.sx_max_moment),magnetic_d_min=float(s.sx_mag_d_min),
                    relative_signs=bool(s.sx_sign_starts),templates_per_k=int(s.sx_k_limit),budget_seconds=int(s.sx_k_budget),
                    workers=min(4,max(1,int(s.sx_k_workers))),progress=mailbox.put,cancel=cancel))
                while not work.done():
                    if identity!=self._k_identity():cancel.set()
                    latest=None
                    while not mailbox.empty():latest=mailbox.get_nowait()
                    if latest and identity==self._k_identity():
                        s.sx_k_message=latest['message']
                        if remote and latest.get('stage')=='simulate':s.sx_k_message+=' · requested pool; Galaxy allocation can reduce this'
                        s.flush()
                    await asyncio.sleep(.15)
                result=await work
                if cancel.is_set() or identity!=self._k_identity():return
                self._k_result=result;self._k_result_identity=identity
                s.sx_k_rows=[dict(k='('+', '.join(f['k'])+')',score=(f'{f["score"]:.3g}' if f.get('score') is not None else '—'),
                    sampled=(f'{f["simulated"]} / {f["available_starts"]}' if 'available_starts' in f else 'Not checked'),catalog=str(f['discovered']) if f['discovery'] in {'cached','complete'} else 'Unavailable',
                    status=f['status'].replace('_',' '),value=','.join(f['k']),reason=f.get('reason','')) for f in result['families']]
                s.sx_k_choices=[dict(title='('+', '.join(f['k'])+') · '+f['status'].replace('_',' '),value=','.join(f['k'])) for f in result['families']]
                s.sx_k_notes=coverage_notes(result)
                s.sx_k_selected=s.sx_k_choices[0]['value'] if s.sx_k_choices else ''
                summary={'shortlist':'Candidate shortlist','ambiguous':'Several k families remain competitive','no_support':'No supported magnetic template in this sample','incomplete':'Partial survey; increase the budget to check remaining families'}[result['status']]
                wall_seconds=time.monotonic()-wall_started
                timing=f'{wall_seconds:.1f} s total'
                if remote:timing+=f' · {result["elapsed_seconds"]:.1f} s budgeted work; remainder includes staging and queueing'
                s.sx_k_message=f'{summary} · {timing} · {sum(f["simulated"] for f in result["families"])} native profiles. Scores are not probabilities.'
                s.sx_k_has_result=True;self._k_plot_changed()
            except Exception as exc:
                if cancel.is_set():s.sx_k_message='K survey stopped; no refinement plan was changed.'
                else:s.sx_error=str(exc);s.sx_k_message='K survey could not complete.'
            finally:
                s.sx_k_busy=False;s.sx_busy=False;self._k_cancel=None;s.flush()
        self._k_task=self._refinement_schedule(run(),'radar-k-survey')

    def use_candidate_k(self):
        s=self.server.state;result=self._k_result;d=self._symmetry_draft
        try:
            if not result or not d or self._k_result_identity!=self._k_identity():raise ValueError('The source or settings changed; run the k survey again')
            family=next(f for f in result['families'] if ','.join(f['k'])==s.sx_k_selected)
            catalog=result['catalogs'].get(s.sx_k_selected)
            if not catalog:raise ValueError('This k has no complete subgroup catalog. Enter it manually and discover candidates when the service is available.')
            if 'compatible_candidate_ids' not in family:raise ValueError('The native compatibility inventory is unfinished. Increase the survey budget, or discover candidates manually.')
            if catalog['parent_setting_sha256']!=next(p for p in d['context']['phases'] if p['name']==s.sx_parent)['setting_sha256']:raise ValueError('The selected catalog belongs to another parent setting')
            s.sx_k=', '.join(family['k']);self._symmetry_unapprove()
            d.pop('protocol',None);write_json(d['root']/'catalog.json',catalog)
            d.update(catalog=catalog,discovery_identity=(s.sx_kind,s.sx_parent,s.sx_k,s.sx_start_source))
            s.sx_candidate_options=[dict(title=(c['magnetic_space_group'] or c['irrep'])+' · '+c['id'][-6:],value=c['id']) for c in catalog['candidates']]
            compatible=set(family['compatible_candidate_ids'])
            canted=getattr(s,'sx_moment_model','collinear')=='symmetry_allowed'
            s.sx_candidates=[c['id'] for c in catalog['candidates'] if canted or c['id'] in compatible];s.sx_mode_options=[];s.sx_modes=[]
            s.sx_discovery_scope=f'{len(catalog["candidates"])} magnetic embeddings in the cached catalog; {len(compatible)} support the sampled collinear sites, directions and k translations. The k survey sampled only a subset.'
            if canted:s.sx_discovery_scope+=' All catalog embeddings remain selected for the canted search; its site symmetry and propagation translations are checked per trial.'
            s.sx_discovered=True;s.sx_plan_ready=False;s.sx_approved=False
            s.sx_k_message='Selected k and its complete subgroup catalog are ready. Review candidates and build the finite refinement plan.'
        except Exception as exc:s.sx_error=str(exc)


def coverage_notes(result):
    notes=list(result.get('errors',[]))
    for family in result['families']:
        label='('+', '.join(family['k'])+')'
        if family.get('reason'):notes.append(label+': '+family['reason'])
        failures=family.get('simulation_failures',[])
        if failures:notes.append(f'{label}: {len(failures)} native starts failed; first reason: '+failures[0]['reason'])
        rejections=family.get('simulation_rejections',[])
        if rejections:notes.append(f'{label}: {len(rejections)} starts were not scored because their moments did not carry the requested k.')
    return notes
