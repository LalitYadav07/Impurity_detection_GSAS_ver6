"""Model exploration inside Refinement, sharing published RADAR-PD sources."""
from __future__ import annotations
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import uuid
from fractions import Fraction

from gpx_recipe.jobs import scientific_environment
from gpx_recipe.process import run_scientific
from gpx_recipe.symmetry.campaign import Campaign,verify_receipt
from gpx_recipe.symmetry.compiler import confined
from gpx_recipe.symmetry.compiler import file_sha,package_signature,verify_inputs
from gpx_recipe.symmetry.contracts import digest,validate_protocol
from gpx_recipe.symmetry.discovery import discover_magnetic
from gpx_recipe.symmetry.local_transport import LocalTransport
from gpx_recipe.symmetry.worker import require_runtime
from gpx_recipe.symmetry.chemistry import validate_sites,sites_from_inputs
from gpx_recipe.symmetry.planner import make_intent
from gpx_recipe.symmetry.isodistort_discovery import discover_structural
from gpx_recipe.symmetry.plans import build_plan
from gpx_recipe.symmetry.contracts import iter_trials
from .symmetry_workspace import campaign_rows,group_models,view_for,choose_group,disposition_reason,coverage_rows,scope_rows,parameter_rows
from . import symmetry_plots as plots
from .symmetry_sources import start_choices,resolve_start


class SymmetryMixin:
    def _init_symmetry(self):
        self._symmetry_rows=[];self._symmetry_all_rows=[];self._symmetry_groups=[];self._symmetry_draft=None;self._symmetry_approved='';self._symmetry_task=None;self._symmetry_loading=False;self._symmetry_refresh_pending=False;self._symmetry_explicit_selection=False;self._symmetry_display_identity=None;self._symmetry_loading_identity=None
        s=self.server.state
        defaults=dict(sx_campaign='',sx_campaigns=[],sx_source='',sx_source_options=[],sx_busy=False,sx_error='',sx_message='',sx_progress='',sx_groups=[],sx_group='',sx_members=[],sx_selected='',sx_search='',sx_status='all',
            sx_overview=plots.empty('Open a campaign to inspect its coverage'),sx_fit=plots.empty('Select an audited checkpoint'),sx_axis='d',sx_reset=0,sx_pins=[],sx_pin_options=[],sx_parameters=[],sx_details=[],sx_intensity_scale='linear',
            sx_parent='',sx_parents=[],sx_histogram='',sx_histograms=[],sx_k='0, 0, 0',sx_sites=[],sx_site_options=[],sx_form_factors='',sx_lande=2.,sx_directions=['x','y','z'],sx_adequate=None,sx_max_moment=6.,sx_cycles=20,sx_slots=2,
            sx_kind='magnetic',sx_mag_d_min=0.,sx_max_mode=.5,sx_start_mode=.1,sx_candidate_options=[],sx_candidates=[],sx_mode_options=[],sx_modes=[],sx_discovered=False,sx_discovery_scope='',
            sx_plan_ready=False,sx_approved=False,sx_plan_summary='',sx_plan_candidates=[],sx_setup=False,sx_has_context=False,sx_has_gpx=False,sx_has_receipt=False,sx_resumable=False,
            sx_request='',sx_planning_summary='',sx_planning_notes=[],sx_sign_starts=True,sx_coverage=[],sx_group_limit=12,sx_result_scope='experiment',sx_start_source='',sx_start_sources=[],sx_results_current=False)
        for key,value in defaults.items():setattr(s,key,value)
        s.change('rf_mode')(self._symmetry_mode_changed)
        s.change('rf_experiment','sx_result_scope')(self._symmetry_mode_changed)
        s.change('sx_campaign')(self._symmetry_campaign_changed)
        s.change('sx_group')(self._symmetry_group_changed)
        s.change('sx_source')(self._symmetry_source_changed)
        s.change('sx_selected','sx_axis','sx_pins','sx_reset','sx_intensity_scale')(self._symmetry_plot_changed)
        s.change('sx_kind','sx_parent','sx_histogram','sx_k','sx_sites','sx_form_factors','sx_lande','sx_directions','sx_adequate','sx_max_moment','sx_mag_d_min','sx_max_mode','sx_start_mode','sx_cycles','sx_request','sx_sign_starts','rf_source','sx_start_source')(self._symmetry_unapprove)
        s.change('sx_parent')(self._symmetry_parent_changed)
        s.change('sx_start_source')(self._symmetry_start_changed)

    def _symmetry_root(self):
        root=Path(os.environ.get('RADAR_PD_SYMMETRY_ROOT') or (Path(getattr(self.service,'output_root',tempfile.gettempdir()))/'symmetry'))
        root.mkdir(parents=True,exist_ok=True);return root.resolve()

    def _symmetry_transport(self):
        python=os.environ.get('RADAR_PD_SYMMETRY_PYTHON','')
        if not python:raise ValueError('The symmetry worker destination is not configured. Ordinary refinement remains available.')
        environment={}
        for name in ['GSASII_ROOT','RADAR_GPX_DOCKER_IMAGE']:
            if os.environ.get(name):environment[name]=os.environ[name]
        return LocalTransport([python],environment=environment)

    def _symmetry_mode_changed(self,**_):
        if self.server.state.rf_mode=='models':self.refresh_symmetry()

    def _symmetry_unapprove(self,**_):
        self._symmetry_approved='';self.server.state.sx_approved=False;self.server.state.sx_plan_ready=False

    def _symmetry_parent_changed(self,**_):
        d=self._symmetry_draft;s=self.server.state
        if d:s.sx_site_options=[dict(title=a['label']+' ('+a['type']+')',value=a['label']) for p in d['context']['phases'] if p['name']==s.sx_parent for a in p['atoms']]
        if not getattr(self,'_symmetry_applying_intent',False):s.sx_sites=[]

    def _symmetry_start_changed(self,**_):
        self._symmetry_draft=None
        s=self.server.state;s.sx_has_context=False;s.sx_discovered=False;s.sx_planning_summary='';s.sx_planning_notes=[]

    def _symmetry_native(self,action,payload,root):
        transport=self._symmetry_transport();uid=uuid.uuid4().hex;request=root/(uid+'-request.json');output=root/(uid+'-response.json')
        request.write_text(json.dumps(payload)+'\n');env={**scientific_environment(),**transport.environment,'RADAR_GPX_DOCKER_WORKDIR':str(root)}
        with (root/(uid+'-native.log')).open('w') as log:
            result=run_scientific(transport.command+['-m','gpx_recipe.symmetry.isodistort_native',action,str(request),str(output)],timeout=120,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode or not output.is_file():raise ValueError('Native preparation failed; retained log '+uid+'-native.log')
        return json.loads(output.read_text())

    def _symmetry_campaign_changed(self,**_):
        if not self._symmetry_loading:
            self._symmetry_explicit_selection=False;self.refresh_symmetry()

    def refresh_symmetry(self):
        if self._symmetry_loading:
            s=self.server.state
            if (s.sx_campaign,s.rf_experiment,s.sx_result_scope)!=self._symmetry_loading_identity:
                self._symmetry_refresh_pending=True;s.sx_results_current=False
            return
        async def load():
            self._symmetry_loading=True;s=self.server.state;s.sx_busy=True;s.sx_error=''
            self._symmetry_loading_identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
            s.sx_results_current=getattr(self,'_symmetry_display_identity',None)==self._symmetry_loading_identity
            s.flush()
            try:
                choices=start_choices(self._refinement_store(),s.rf_experiment)
                s.sx_start_sources=[dict(title=entry['title'],value=entry['id']) for entry in choices]
                if s.sx_start_source not in {entry['id'] for entry in choices}:
                    s.sx_start_source=s.rf_source if s.rf_source in {entry['id'] for entry in choices} else (choices[0]['id'] if choices else '')
                roots=sorted(self._symmetry_root().glob('*/campaign.sqlite'))
                s.sx_campaigns=[dict(title=p.parent.name,value=p.parent.name) for p in roots]
                if s.sx_campaign not in {p.parent.name for p in roots}:s.sx_campaign=roots[-1].parent.name if roots else ''
                if not s.sx_campaign:
                    s.sx_message='Select a completed RADAR-PD GPX above, then prepare model exploration.';return
                root=next(p.parent for p in roots if p.parent.name==s.sx_campaign)
                identity=(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
                self._symmetry_loading_identity=identity
                rows,summaries=await asyncio.to_thread(campaign_rows,root,experiment=identity[1] if identity[2]=='experiment' else None)
                if identity!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                    self._symmetry_refresh_pending=True;return
                rows=scope_rows(rows,s.rf_experiment,s.sx_result_scope)
                self._symmetry_all_rows=rows
                s.sx_coverage=coverage_rows(rows)
                s.sx_resumable=any(summary['pending_count'] for summary in summaries)
                counts=Counter(r['status'] for r in rows)
                current=sum(r['execution_state']=='terminal' for r in rows)
                s.sx_progress=f'{current} / {len(rows)} terminal receipts in this view · '+', '.join(f'{n} {k.replace("_"," ")}' for k,n in sorted(counts.items()))
                if s.sx_result_scope=='experiment':s.sx_progress+=f' · whole campaign: {sum(v["received_count"] for v in summaries)} / {sum(v["declared_count"] for v in summaries)}'
                s.sx_message=('Qualified structures remain hypotheses. Matching calculated curves do not establish a unique space group.' if rows else 'No campaign results belong to the selected experiment. Choose another experiment or view the whole campaign.')
                options={r['source']:(' · '.join(str(v) for v in [r['metadata'].get('scan') or r['parent'],
                    (f"{r['metadata']['condition']:g} {r['metadata'].get('condition_unit','')}" if isinstance(r['metadata'].get('condition'),(int,float)) else None),
                    ('bank '+str(r['metadata']['bank']) if r['metadata'].get('bank') else None)] if v)) for r in rows}
                s.sx_source_options=[dict(title=label,value=key) for key,label in options.items()]
                if s.sx_source not in options:s.sx_source=next(iter(options),'')
                await asyncio.to_thread(self._symmetry_source_data,s.sx_source)
                if identity!=(s.sx_campaign,s.rf_experiment,s.sx_result_scope):
                    self._symmetry_refresh_pending=True;return
                self._symmetry_publish_source()
                self._symmetry_display_identity=identity;s.sx_results_current=True
            except Exception as exc:s.sx_error=str(exc)
            finally:
                s.sx_results_current=getattr(self,'_symmetry_display_identity',None)==(s.sx_campaign,s.rf_experiment,s.sx_result_scope)
                self._symmetry_loading=False;s.sx_busy=False;s.flush()
                if self._symmetry_refresh_pending:
                    self._symmetry_refresh_pending=False;self.refresh_symmetry()
        self._refinement_schedule(load(),'radar-symmetry-refresh')

    def _symmetry_source_data(self,source):
        rows=[r for r in self._symmetry_all_rows if r['source']==source]
        self._symmetry_rows=rows;self._symmetry_groups=group_models(rows)

    def _symmetry_publish_source(self):
        s=self.server.state;groups=self._symmetry_groups;rows=self._symmetry_rows
        s.sx_groups=[dict(id=g['id'],title=g['representative']['name'] if g['representative']['has_view'] else g['representative']['status'].replace('_',' ').capitalize(),count=len(g['members']),rwp=(f"{g['representative']['rwp']:.3f}%" if g['representative']['rwp'] is not None else '—'),status=g['representative']['status'].replace('_',' ')) for g in groups]
        s.sx_overview=plots.overview(rows,s.sx_selected)
        s.sx_pin_options=[dict(title=r['name']+' · '+r['candidate'][-6:]+' · '+r['seed'],value=r['id']) for r in rows if r['has_view']]
        pinnable={r['id'] for r in rows if r['has_view']}
        s.sx_pins=[uid for uid in s.sx_pins if uid in pinnable][-2:]
        s.sx_group=choose_group(groups,s.sx_selected,explicit=self._symmetry_explicit_selection)
        self._symmetry_group_changed()

    def _symmetry_source_changed(self,**_):
        if self._symmetry_loading:return
        self._symmetry_explicit_selection=False
        self.server.state.sx_group_limit=12
        self._symmetry_source_data(self.server.state.sx_source);self._symmetry_publish_source()

    def select_symmetry_group(self,uid):
        self._symmetry_explicit_selection=True;self.server.state.sx_group=uid

    def _symmetry_user_selected(self,**_):self._symmetry_explicit_selection=True

    def _symmetry_group_changed(self,**_):
        s=self.server.state;group=next((g for g in self._symmetry_groups if g['id']==s.sx_group),None)
        members=group['members'] if group else []
        s.sx_members=[dict(title=r['name']+' · '+r['candidate'][-6:]+' · '+r['seed']+' · '+r['status'].replace('_',' '),value=r['id']) for r in members]
        if s.sx_selected not in {r['id'] for r in members}:s.sx_selected=members[0]['id'] if members else ''
        self._symmetry_plot_changed()

    def _symmetry_plot_changed(self,**_):
        s=self.server.state;row=next((r for r in self._symmetry_rows if r['id']==s.sx_selected),None)
        if len(s.sx_pins)>2:s.sx_pins=s.sx_pins[-2:]
        pins=[r for r in self._symmetry_rows if r['id'] in s.sx_pins]
        try:
            s.sx_overview=plots.overview(self._symmetry_rows,s.sx_selected)
            s.sx_fit=plots.fit(row,pins,axis=s.sx_axis,reset=s.sx_reset,intensity_scale=s.sx_intensity_scale)
            s.sx_details=[];s.sx_parameters=[];s.sx_has_gpx=bool(row and row.get('has_gpx') and row['has_view'])
            s.sx_has_receipt=bool(row and row.get('receipt_sha256'))
            if row:
                reason=disposition_reason(row)
                s.sx_details=[dict(label='Status',value=row['status'].replace('_',' ')),dict(label='Reason',value=reason),
                    dict(label='Candidate',value=row['name']+(' · BNS '+row['bns'] if row['bns'] else '')),dict(label='Embedding',value=json.dumps(row['embedding'])),dict(label='Start',value=row['seed']),
                    dict(label='Worker',value='GSAS-II '+row.get('native_version','pending')+' · implementation '+row['worker_revision'][:12])]
                if row['kind']=='magnetic':s.sx_details.append(dict(label='Complete BNS operator audit',value='Passed' if row.get('bns_audited') else 'Not present in this archived or unfinished trial'))
                s.sx_details.append(dict(label='Parent-basis k',value='('+', '.join(row['k'])+')'))
                s.sx_details.append(dict(label='Starting GPX revision',value=row['source_sha256'][:16]))
                if row.get('sites'):s.sx_details.append(dict(label='Magnetic sites',value='; '.join(site['label']+'='+site['form_factor']+' (g='+str(site['lande_g'])+')' for site in row['sites'])))
                if row.get('baseline_rwp') is not None and row.get('rwp') is not None:
                    baseline=row['baseline_rwp']
                    s.sx_details.append(dict(label='Nuclear baseline Rwp',value=f'{baseline:.6f}% · selected change {row["rwp"]-baseline:+.6f} percentage points'))
                if row['has_view']:
                    view=view_for(row)
                    s.sx_parameters=parameter_rows(row,view)
        except Exception as exc:s.sx_error=str(exc)

    def symmetry_reset_zoom(self):self.server.state.sx_reset+=1

    def download_symmetry_artifact(self,kind):
        s=self.server.state
        try:
            row=next(r for r in self._symmetry_rows if r['id']==s.sx_selected)
            receipt=verify_receipt(row['directory'],dict(id=row['id'],protocol_sha256=row['protocol']))
            if digest(receipt)!=row.get('receipt_sha256'):raise ValueError('The receipt changed; refresh evidence before downloading')
            if kind=='gpx':
                if not row.get('has_gpx') or not row['has_view']:raise ValueError('No independently audited native checkpoint is available')
                path=confined(row['directory'],row['gpx_relative']);self.download_file(row['id']+'-'+path.name,'application/octet-stream',path.read_bytes())
            elif kind=='receipt':self.download_file(row['id']+'-receipt.json','application/json',(json.dumps(receipt,indent=2)+'\n').encode())
            else:raise ValueError('Unknown native artifact')
        except Exception as exc:s.sx_error=str(exc);s.flush()

    def symmetry_overview_click(self,event):
        points=event.get('points',[]) if isinstance(event,dict) else []
        uid=points[0].get('customdata') if points else None
        for group in self._symmetry_groups:
            if any(r['id']==uid for r in group['members']):self._symmetry_explicit_selection=True;self.server.state.sx_group=group['id'];self.server.state.sx_selected=uid;return

    def inspect_symmetry_source(self):
        async def inspect():
            s=self.server.state;s.sx_busy=True;s.sx_error='';s.sx_has_context=False;s.flush()
            try:
                transport=self._symmetry_transport();w=self._refinement_store()
                identity=(s.rf_experiment,s.sx_start_source)
                source=await asyncio.to_thread(resolve_start,w,s.rf_experiment,s.sx_start_source)
                captured=await asyncio.to_thread(w.runtime.resolve,source)
                root=self._symmetry_root()/'drafts'/uuid.uuid4().hex;root.mkdir(parents=True)
                shutil.copyfile(captured['source_path'],root/'source.gpx')
                env={**scientific_environment(),**transport.environment,'RADAR_GPX_DOCKER_WORKDIR':str(root)}
                def call():
                    with (root/'inspect.log').open('w') as log:
                        return run_scientific(transport.command+['-m','gpx_recipe.symmetry.inspect',str(root/'source.gpx'),str(root/'context.json')],timeout=120,env=env,stdout=log,stderr=subprocess.STDOUT)
                completed=await asyncio.to_thread(call)
                if completed.returncode or not (root/'context.json').exists():raise ValueError('Native inspection failed; the captured log is available for review')
                context=json.loads((root/'context.json').read_text());require_runtime(context['runtime'],context['runtime']['magnetic_function_sha256'])
                if identity!=(s.rf_experiment,s.sx_start_source):raise ValueError('The starting checkpoint changed during inspection; inspect the current selection')
                if context['sha256']!=captured['source_sha256']:raise ValueError('The captured GPX revision changed during native inspection')
                self._symmetry_draft=dict(root=root,context=context,source=source,source_id=identity[1],experiment=identity[0])
                s.sx_discovered=False;s.sx_planning_summary='';s.sx_planning_notes=[]
                s.sx_parents=[dict(title=p['name']+' · '+p['space_group'],value=p['name']) for p in context['phases'] if p['type']=='nuclear']
                s.sx_histograms=[dict(title=h['name'],value=h['name']) for h in context['histograms']]
                s.sx_parent=s.sx_parents[0]['value'] if s.sx_parents else '';s.sx_histogram=s.sx_histograms[0]['value'] if s.sx_histograms else ''
                s.sx_site_options=[dict(title=p['name']+' / '+a['label']+' ('+a['type']+')',value=a['label']) for p in context['phases'] if p['name']==s.sx_parent for a in p['atoms']]
                s.sx_has_context=True;s.sx_setup=True;s.sx_message='Parent setting, operations and instrument coordinates captured from this exact GPX.'
            except Exception as exc:s.sx_error=str(exc)
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(inspect(),'radar-symmetry-inspect')

    def plan_symmetry_request(self):
        """A setup-only call; discovery and native dispatch still need review."""
        async def plan():
            s=self.server.state;s.sx_busy=True;s.sx_error='';s.flush()
            try:
                d=self._symmetry_draft
                if not d or d['source_id']!=s.sx_start_source:raise ValueError('Inspect this exact source GPX first')
                identity=(s.sx_start_source,s.sx_request)
                result=await asyncio.to_thread(make_intent,s.sx_request,d['context'])
                if identity!=(s.sx_start_source,s.sx_request):raise ValueError('The source or request changed during planning; plan again')
                (d['root']/'planning.json').write_text(json.dumps(result,indent=2)+'\n')
                d['planning']=result
                intent=result['intent'];s.sx_planning_notes=intent['assumptions']+intent['questions']+intent['unsupported']
                s.sx_planning_summary=f"{len(result['calls'])} setup call(s) · "+('needs clarification' if result['status']=='needs_input' else 'ready to review and discover candidates')
                self._symmetry_unapprove();s.sx_discovered=False
                if intent['settings'] is not None:
                    settings=intent['settings'];self._symmetry_applying_intent=True
                    try:
                        s.sx_kind=settings['kind'];s.sx_parent=settings['parent'];s.sx_histogram=settings['histogram']
                        s.sx_k=', '.join(settings['k']);s.sx_sites=[site['label'] for site in settings['sites']]
                        s.sx_form_factors='; '.join(site['label']+'='+site['form_factor'] for site in settings['sites'])
                        gs={site['lande_g'] for site in settings['sites']}
                        s.sx_lande=str(next(iter(gs))) if len(gs)==1 else '; '.join(site['label']+'='+str(site['lande_g']) for site in settings['sites'])
                        s.sx_directions=settings['directions'];s.sx_adequate=settings['adequate_rwp'];s.sx_max_moment=settings['maximum_moment']
                        s.sx_mag_d_min=settings['magnetic_d_min'];s.sx_cycles=settings['cycles'];s.sx_sign_starts=settings['relative_sign_starts'];s.flush()
                    finally:self._symmetry_applying_intent=False
            except Exception as exc:s.sx_error=str(exc)
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(plan(),'radar-symmetry-plan')

    def discover_symmetry_candidates(self):
        async def discover():
            s=self.server.state;s.sx_busy=True;s.sx_error='';s.flush()
            try:
                d=self._symmetry_draft
                if not d or d['source_id']!=s.sx_start_source:raise ValueError('Inspect the selected source GPX first')
                parent=next(p for p in d['context']['phases'] if p['name']==s.sx_parent);k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')]
                identity=(s.sx_kind,s.sx_parent,s.sx_k,s.sx_start_source);root=d['root']
                if s.sx_kind=='magnetic':catalog=await asyncio.to_thread(discover_magnetic,parent,k,api_key=os.environ.get('BCS_API_KEY',''))
                else:
                    parent_cif=root/'parent.cif'
                    await asyncio.to_thread(self._symmetry_native,'export',dict(source=str(root/'source.gpx'),phase=s.sx_parent,destination=str(parent_cif)),root)
                    discovery=root/('isodistort-'+uuid.uuid4().hex)
                    def prepare():
                        return discover_structural(parent_cif,parent,k,discovery,matcher=lambda cif,vector,options:self._symmetry_native('match',dict(cif=cif,k=vector,options=options),root),
                            inventory=lambda child:self._symmetry_native('modes',dict(child=str(child),temporary=str(root/(child.stem+'.gpx'))),root))
                    catalog=await asyncio.to_thread(prepare)
                    for candidate in catalog['candidates']:candidate['child']['path']=str((discovery/candidate['child']['path']).relative_to(root))
                if identity!=(s.sx_kind,s.sx_parent,s.sx_k,s.sx_start_source):raise ValueError('Inputs changed during discovery; discover again for the current parent and k')
                if d.get('planning'):
                    catalog.setdefault('provenance',{})['setup_planning_sha256']=digest(d['planning'])
                (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n');d.update(catalog=catalog,discovery_identity=identity)
                s.sx_candidate_options=[dict(title=(c['magnetic_space_group'] or c['irrep'])+' · '+c['id'][-6:],value=c['id']) for c in catalog['candidates']]
                s.sx_candidates=[c['id'] for c in catalog['candidates']]
                s.sx_mode_options=[dict(title=(c['irrep'] or c['id'])+' / '+mode,value=c['id']+'|'+mode) for c in catalog['candidates'] for mode in c['modes']]
                s.sx_modes=[m['value'] for m in s.sx_mode_options]
                coverage=catalog.get('coverage',{});s.sx_discovery_scope=(f"{coverage.get('discovered_count',len(catalog['candidates']))} OPDs discovered · {len(catalog['candidates'])} constructed in the declared tier · deferred/failed branches retained in the discovery record" if s.sx_kind=='structural' else f"{len(catalog['candidates'])} magnetic embeddings discovered in this exact parent setting")
                s.sx_discovered=True;s.sx_plan_ready=False;s.sx_approved=False
            except Exception as exc:s.sx_error=str(exc)
            finally:s.sx_busy=False;s.flush()
        self._refinement_schedule(discover(),'radar-symmetry-discovery')

    def build_symmetry_plan(self):
        s=self.server.state;s.sx_error=''
        try:
            d=self._symmetry_draft
            if not d or d.get('discovery_identity')!=(s.sx_kind,s.sx_parent,s.sx_k,s.sx_start_source):raise ValueError('Discover candidates for the current parent and k first')
            if not s.sx_adequate or float(s.sx_adequate)<=0:raise ValueError('Enter an experiment-specific adequate Rwp (%) before building the plan')
            sites=[]
            if s.sx_kind=='magnetic':
                sites=sites_from_inputs(s.sx_sites,s.sx_form_factors,s.sx_lande)
            settings=dict(acquisition_id=d['source_id'],kind=s.sx_kind,parent=s.sx_parent,histogram=s.sx_histogram,k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')],candidates=list(s.sx_candidates),modes=list(s.sx_modes),sites=sites,directions=list(s.sx_directions),adequate_rwp=float(s.sx_adequate),maximum_moment=float(s.sx_max_moment),magnetic_d_min=float(s.sx_mag_d_min),maximum_mode=float(s.sx_max_mode),start_mode=float(s.sx_start_mode),cycles=int(s.sx_cycles),relative_sign_starts=s.sx_sign_starts)
            p=build_plan(d['context'],d['catalog'],d['root'],settings);verify_inputs(p,d['root']);d.update(protocol=p,settings=settings)
            s.sx_plan_candidates=[dict(name=c['magnetic_space_group'] or c['irrep'],bns=c['bns_number'] or '—',basis=str(c['transformation_matrix']),origin=str(c['origin_shift']),modes=', '.join(c['modes']) or 'Selected collinear moments') for c in p['candidates']]
            free='Selected modes, cell, background and global scale' if s.sx_kind=='structural' else 'Selected collinear moments; nuclear model and nuisance parameters fixed'
            if s.sx_kind=='magnetic':free+=('; all fitted d spacings' if settings['magnetic_d_min']==0 else f"; magnetic d ≥ {settings['magnetic_d_min']} Å")
            s.sx_plan_summary=f"{len(p['candidates'])} embeddings · {sum(1 for _ in iter_trials(p))} finite trials. {free}. Instrument, occupancy and displacement parameters fixed. Adequacy: Rwp ≤ {s.sx_adequate}%."
            s.sx_plan_ready=True;s.sx_approved=False
        except Exception as exc:s.sx_error=str(exc)

    def approve_symmetry(self):
        d=self._symmetry_draft;s=self.server.state
        if not d or not s.sx_plan_ready:return
        self._symmetry_approved=digest(d['protocol']);s.sx_approved=True

    def start_symmetry_campaign(self):
        if self._symmetry_task and not self._symmetry_task.done():return
        async def run():
            s=self.server.state;s.sx_error=''
            try:
                d=self._symmetry_draft
                if not d or not s.sx_plan_ready or not s.sx_approved or self._symmetry_approved!=digest(d['protocol']):raise ValueError('Approve the current finite plan first')
                if d['source_id']!=s.sx_start_source:raise ValueError('The source changed after approval')
                current=await asyncio.to_thread(resolve_start,self._refinement_store(),s.rf_experiment,s.sx_start_source)
                current=await asyncio.to_thread(self._refinement_store().runtime.resolve,current)
                if d['source_id']!=s.sx_start_source or d.get('experiment',s.rf_experiment)!=s.rf_experiment:raise ValueError('The source selection changed while its revision was checked')
                if current.get('source_sha256') and current['source_sha256']!=d['protocol']['source']['sha256']:raise ValueError('The published GPX revision changed after approval; inspect it again')
                c=Campaign(self._symmetry_root()/('campaign-'+self._symmetry_approved[:12]));transport=self._symmetry_transport()
                source=d['source'];metadata=dict(scan=source['scan_label'],sample=source['sample_label'],bank=source.get('bank'),series=source['experiment'])
                await asyncio.to_thread(c.enroll,d['protocol'],d['root'],approved_hash=self._symmetry_approved,metadata=metadata)
                s.sx_campaign=c.root.name;s.sx_setup=False
                await self._drive_symmetry_campaign(c,transport)
            except Exception as exc:s.sx_error=str(exc)
            finally:s.flush()
        self._symmetry_task=self._refinement_schedule(run(),'radar-symmetry-campaign')

    async def _drive_symmetry_campaign(self,campaign,transport):
        while True:
            trials=await asyncio.to_thread(campaign.tick,transport,slots=int(self.server.state.sx_slots))
            self.refresh_symmetry()
            if all(r['state'] in {'terminal','cancelled','needs_recovery'} for r in trials):break
            await asyncio.sleep(3)

    def continue_symmetry_campaign(self):
        if self._symmetry_task and not self._symmetry_task.done():return
        async def resume():
            s=self.server.state;s.sx_error=''
            try:
                root=(self._symmetry_root()/s.sx_campaign).resolve()
                if root.parent!=self._symmetry_root() or not (root/'campaign.sqlite').is_file():raise ValueError('Select an existing campaign')
                campaign=Campaign(root)
                await asyncio.to_thread(campaign.verify_sealed_continuation)
                await self._drive_symmetry_campaign(campaign,self._symmetry_transport())
            except Exception as exc:s.sx_error=str(exc)
            finally:s.flush()
        self._symmetry_task=self._refinement_schedule(resume(),'radar-symmetry-continue')

    def cancel_symmetry_selected(self):
        async def cancel():
            try:
                s=self.server.state;c=Campaign(self._symmetry_root()/s.sx_campaign)
                await asyncio.to_thread(c.cancel,s.sx_selected,self._symmetry_transport());self.refresh_symmetry()
            except Exception as exc:self.server.state.sx_error=str(exc)
        self._refinement_schedule(cancel(),'radar-symmetry-cancel')
