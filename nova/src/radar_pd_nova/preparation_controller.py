"""Experiment-preparation interaction, with independent asynchronous ownership."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
import uuid
from urllib.parse import urlsplit,urlunsplit

from gpx_recipe.preparation.reference import validate_grid
from gpx_recipe.preparation.library import MAX_ARCHIVE_BYTES as MAX_BYTES
from gpx_recipe.symmetry.compiler import file_sha
from gpx_recipe.symmetry.contracts import digest
from gpx_recipe.structural_live.defaults import DEFAULT_CELL_RADIUS
from .battery_preparation import checked_upload, capture_file
from .model_runtime import identifier
from .preparation_training_runtime import PreparationRuntime
from .preparation_discovery import PreparationDiscovery,k_vector,require_structural_coverage
from .preparation_workspace import PreparationWorkspace
from .preparation_limits import PREPARATION_JOBS
from .uploads import inspect_cif_upload


def finite_number(value,*,integer=False):
    if isinstance(value,bool):raise ValueError('Use a finite number')
    try:number=float(value)
    except (TypeError,ValueError):raise ValueError('Use a finite number') from None
    if not math.isfinite(number) or (integer and not number.is_integer()):raise ValueError('Use a finite whole number' if integer else 'Use a finite number')
    return int(number) if integer else number


def download_url(base,dataset):
    """Use the public Galaxy origin/prefix, never credentials in a link."""
    parsed=urlsplit(str(base))
    if (parsed.scheme not in {'http','https'} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment or any(c.isspace() for c in str(base))):
        raise ValueError('Configure a public HTTP(S) Galaxy address without credentials, query or fragment')
    return urlunsplit((parsed.scheme,parsed.netloc,
        parsed.path.rstrip('/')+'/api/datasets/'+identifier(dataset)+'/display','to_ext=zip',''))


class PreparationMixin:
    def _init_preparation(self):
        self._preparation_workspace=None;self._prep_driver=None;self._prep_tasks=set();self._prep_stopping=False
        self._prep_banks={};self._prep_bank_names={};self._prep_queue_errors={}
        self._prep_context=None;self._prep_catalogs=[];self._prep_discovered_identity=None
        self._prep_site_inputs={};self._prep_loaded_reference='';self._prep_return=None
        self._prep_input_revision=0
        self._prep_profile_handoff=None
        s=self.server.state
        values=dict(prep_busy=False,prep_operation='',prep_error='',prep_message='',prep_tab='new',
            prep_name='Experiment preparation',prep_parent_path='',prep_profile_path='',prep_bank_label='',
            prep_profile_bank='1',prep_grid_kind='PNT',prep_lower='3000',prep_upper='180000',prep_bins='2500',
            prep_bank_rows=[],prep_reference='',prep_reference_options=[],prep_context_ready=False,prep_parent_title='',
            prep_kind='magnetic',prep_k_options=[],prep_selected_k=[],prep_custom_k='',prep_site_options=[],prep_sites=[],
            prep_site_rows=[],prep_maximum_moment='6',prep_mode_radius='.05',prep_cell_radius=format(DEFAULT_CELL_RADIUS,'g'),
            prep_opd_policy='all',prep_construct_limit='100',prep_discovery_rows=[],
            prep_discovery_ready=False,prep_candidate_count=0,prep_selected_banks=[],prep_inspected_banks=[],
            prep_parallel_jobs=PREPARATION_JOBS,prep_library='',prep_library_options=[],prep_entries=[],prep_coverage='',prep_active_jobs=0,
            prep_library_kind='',prep_can_pause=False,prep_can_resume=False,
            prep_page=1,prep_pages=1,prep_entry_filter='all',prep_stage_filter='all',prep_stage_rows=[],
            prep_entry_search='',prep_table_count=0,
            prep_training_record_open=False,prep_training_record_title='',prep_training_record=None,
            prep_saved='',prep_saved_options=[],prep_import_path='',prep_import_source='upload',
            prep_history_archive='',prep_history_archives=[],prep_download_href='')
        for key,value in values.items():setattr(s,key,value)
        s.change('prep_parent_path')(self._preparation_inputs_changed)
        s.change('prep_reference')(self._preparation_reference_changed)
        s.change('prep_kind','prep_selected_k','prep_opd_policy','prep_construct_limit')(self._preparation_scope_changed)
        s.change('prep_sites')(self._preparation_site_selection_changed)
        s.change('prep_selected_banks')(self._preparation_count_changed)
        s.change('prep_library')(self._preparation_library_changed)
        s.change('prep_page','prep_entry_filter','prep_stage_filter','prep_entry_search')(self._preparation_table_changed)
        s.change('prep_profile_path')(self._preparation_profile_changed)

    def _preparation_store(self):
        if self._preparation_workspace is None:
            base=Path(os.environ.get('RADAR_PD_PREPARATION_ROOT') or
                Path(getattr(self.service,'output_root',None) or tempfile.gettempdir())/'experiment-preparation')
            root=base/digest([str(self.service.history_id or 'unconfigured')])[:16]
            runtime=PreparationRuntime(root/'runtime',self.service,max_active_jobs=PREPARATION_JOBS)
            self._preparation_workspace=PreparationWorkspace(root,runtime)
        return self._preparation_workspace

    def _prep_schedule(self,coroutine,name):
        try:loop=asyncio.get_running_loop()
        except RuntimeError:coroutine.close();return None
        task=loop.create_task(coroutine,name=name);self._prep_tasks.add(task)
        task.add_done_callback(self._prep_tasks.discard);return task

    def _prep_action(self,coroutine,title):
        if self.server.state.prep_busy:
            coroutine.close();return None
        try:asyncio.get_running_loop()
        except RuntimeError:coroutine.close();return None
        s=self.server.state;s.prep_busy=True;s.prep_operation=title;s.prep_error='';s.prep_message=title;s.flush()
        return self._prep_schedule(self._prep_perform(coroutine,title),'radar-preparation-action')

    async def _prep_perform(self,coroutine,title):
        s=self.server.state;s.prep_busy=True;s.prep_operation=title;s.prep_error='';s.prep_message=title;s.flush()
        try:await coroutine
        except ValueError as exc:s.prep_error=str(exc)
        except Exception as exc:s.prep_error=title+' could not finish ('+type(exc).__name__+'). Your saved jobs and artifacts remain available.'
        finally:s.prep_busy=False;s.prep_operation='';s.flush()

    def open_preparation_workspace(self,**_):
        s=self.server.state
        if s.workspace_view!='preparation':self._prep_return=(s.workspace_view,s.setup_collapsed)
        s.workspace_view='preparation';s.setup_collapsed=True
        if getattr(s,'live_assistant_open',False):s.live_assistant_open=False
        self._prep_refresh();s.flush()

    def close_preparation_workspace(self,**_):
        s=self.server.state;s.workspace_view,s.setup_collapsed=self._prep_return or ('monitor',False)
        s.prep_training_record_open=False
        if hasattr(s,'sx_preparation_notice'):s.sx_preparation_notice=self._prepared_selection_caption()
        if hasattr(s,'webmon_preparation_notice'):s.webmon_preparation_notice=self._prepared_selection_caption()
        s.flush()

    def _prep_refresh(self):
        store=self._preparation_store();s=self.server.state;rows=store.registry.summaries()
        s.prep_library_options=[dict(title=f"{r['name']} · {r['prepared'] if r['prepared'] is not None else '?'} / {r['total'] or '?'} prepared",value=r['id']) for r in rows]
        references=[]
        for row in rows:
            if row['available'] and row['kinds']==['reference']:
                references.append(dict(title=row['name'],value=row['id']))
        s.prep_reference_options=references
        if not s.prep_library:
            selected=store.registry.selected_key()
            if selected in {row['id'] for row in rows}:s.prep_library=selected
        if s.prep_library:self._preparation_table_changed()

    def _preparation_inputs_changed(self,**_):
        self._prep_input_revision+=1
        self._prep_context=None;self._prep_loaded_reference=''
        self.server.state.prep_context_ready=False;self.server.state.prep_reference=''
        self._preparation_scope_changed()

    def _preparation_reference_changed(self,**_):
        self._prep_context=None;self._prep_loaded_reference='';self._prep_site_inputs={}
        s=self.server.state;s.prep_context_ready=False;s.prep_parent_title='';s.prep_site_options=[];s.prep_sites=[]
        s.prep_k_options=[];s.prep_selected_k=[];self._preparation_scope_changed()
        if s.prep_reference:self._prep_schedule(self._prep_load_reference(s.prep_reference),'radar-preparation-parent')

    async def _prep_load_reference(self,key):
        s=self.server.state
        try:context=await asyncio.to_thread(self._preparation_store().references,key)
        except ValueError as exc:
            if s.prep_reference==key:
                try:pending=not self._preparation_store().registry.get(key).coverage()['complete']
                except ValueError:pending=False
                if pending:s.prep_message='Parent/profile inspection is queued or running. Its status is shown in Libraries.'
                else:s.prep_error=str(exc)
                s.flush()
            return
        if s.prep_reference!=key:return
        self._prep_context=context;self._prep_loaded_reference=key;parent=context['parent']
        s.prep_parent_title=parent['name']+' · '+parent['space_group']
        s.prep_site_options=[dict(title=a['label']+' ('+a['type']+')',value=a['label']) for a in parent['atoms']]
        self._prep_site_inputs={a['label']:dict(label=a['label'],element=a['type'],
            form_factor=a['type'] if '+' in a['type'] or '-' in a['type'] else '',lande_g='2') for a in parent['atoms']}
        points=context['banks'][0]['points']['candidates'];options=[]
        for row in points:
            vector=k_vector(row['k']);token=','.join(vector)
            if token not in {r['value'] for r in options}:
                options.append(dict(value=token,title='('+', '.join(vector)+')'+(' · '+', '.join(row.get('labels',[])) if row.get('labels') else '')))
        s.prep_k_options=options;s.prep_selected_k=[r['value'] for r in options]
        s.prep_inspected_banks=[dict(title=b['bank'],value=b['bank']) for b in context['banks']]
        s.prep_selected_banks=[b['bank'] for b in context['banks']];s.prep_context_ready=True
        self._preparation_scope_changed()
        if hasattr(self, 'apply_planning_preparation_intent'):
            self.apply_planning_preparation_intent(context)
        if not s.prep_busy:s.prep_message='Input inspection complete. Review the parent, banks and model scope below.'
        s.flush()

    def _scope_identity(self):
        s=self.server.state
        return digest([s.prep_reference,s.prep_kind,s.prep_selected_k,s.prep_opd_policy,s.prep_construct_limit])

    def _preparation_scope_changed(self,**_):
        self._prep_catalogs=[];self._prep_discovered_identity=None
        s=self.server.state;s.prep_discovery_rows=[];s.prep_discovery_ready=False;s.prep_candidate_count=0

    def _preparation_count_changed(self,**_):
        self.server.state.prep_candidate_count=sum(len(c['catalog']['candidates']) for c in self._prep_catalogs)*len(self.server.state.prep_selected_banks)

    def _preparation_profile_changed(self,**_):
        s=self.server.state;widget=getattr(self,'_prep_profile_upload',None)
        path=getattr(widget,'current_path',None)
        if path is None or str(path)!=s.prep_profile_path or path.is_symlink() or not path.is_file():
            self._prep_profile_handoff=None;return
        try:
            with path.open('rb') as stream:raw=stream.read(1024*1024+1)
        except OSError:
            self._prep_profile_handoff=None;return
        if len(raw)>1024*1024:self._prep_profile_handoff=None;return
        text=raw.decode('utf-8-sig',errors='replace')
        found=re.search(r'(?:^|[;\n])\s*Type\s*:\s*(PN[TC])\b',text,re.I)
        if found is None:found=re.search(r'^INS\s+HTYPE\s+(PN[TC])\b',text,re.I|re.M)
        kind=found[1].upper() if found else None
        reviewed=getattr(self,'_prep_profile_handoff',None)
        if (reviewed and reviewed['path']==str(path) and reviewed['kind']==kind
                and hashlib.sha256(raw).hexdigest()==reviewed['sha256']):
            # Trame executes upload change listeners on flush, after the
            # handoff has populated its reviewed range. Keep that range only
            # while this exact uploaded profile and native type still match.
            s.prep_grid_kind=kind
            return
        self._prep_profile_handoff=None
        if found:
            s.prep_grid_kind=kind
            s.prep_lower='10' if s.prep_grid_kind=='PNC' else '3000'
            s.prep_upper='160' if s.prep_grid_kind=='PNC' else '180000'

    def add_preparation_bank(self,**_):
        s=self.server.state
        try:
            if len(self._prep_banks)>=64:raise ValueError('One inspection supports up to 64 bank contexts')
            path=checked_upload(self._prep_profile_upload,s.prep_profile_path)
            if path.suffix.lower() not in {'.instprm','.prm','.ins','.inst'}:raise ValueError('Choose a supported GSAS-II instrument profile')
            grid=validate_grid(kind=s.prep_grid_kind,lower=finite_number(s.prep_lower),upper=finite_number(s.prep_upper),
                bins=finite_number(s.prep_bins,integer=True),bank=finite_number(s.prep_profile_bank,integer=True))
            label=s.prep_bank_label.strip() or path.stem+' · bank '+str(grid['bank'])
            if len(label)>100 or label in self._prep_banks:raise ValueError('Give each bank a unique name, up to 100 characters')
            destination=self._preparation_store().root/'uploads'/(uuid.uuid4().hex+path.suffix.lower())
            destination.parent.mkdir(parents=True,exist_ok=True);capture_file(path,destination,maximum=1024*1024)
            self._prep_banks[label]=dict(instrument=destination,grid=grid)
            self._prep_bank_names[label]=path.name
            s.prep_bank_rows=[dict(label=k,profile=self._prep_bank_names.get(k,v['instrument'].name),bank=v['grid']['bank'],kind=v['grid']['kind'],
                range=f"{v['grid']['lower']:g}–{v['grid']['upper']:g}") for k,v in self._prep_banks.items()]
            s.prep_bank_label='';s.prep_error='';self._preparation_inputs_changed()
        except ValueError as exc:s.prep_error=str(exc)
        s.flush()

    def remove_preparation_bank(self,label):
        if label in self._prep_banks:
            self._prep_banks.pop(label)
            self._prep_bank_names.pop(label,None)
            self.server.state.prep_bank_rows=[r for r in self.server.state.prep_bank_rows if r['label']!=label]
            self._preparation_inputs_changed()

    def inspect_preparation_inputs(self,**_):
        return self._prep_action(self._inspect_preparation_inputs(),'Inspecting the parent and bank profiles')

    async def _inspect_preparation_inputs(self):
        s=self.server.state;revision=self._prep_input_revision
        source=checked_upload(self._prep_parent_upload,s.prep_parent_path,maximum=10*1024*1024)
        inspect_cif_upload(source.read_bytes(),source.name)
        if not self._prep_banks:raise ValueError('Add at least one bank profile')
        store=self._preparation_store()
        key=await asyncio.to_thread(store.inspect,name=s.prep_name.strip()[:160]+' · input inspection',source=source,banks=copy.deepcopy(self._prep_banks))
        if revision!=self._prep_input_revision:
            self._prep_refresh();s.prep_message='Inputs changed. The captured plan was saved without starting jobs; inspect the current inputs again.';return
        await self._prep_start_inspection(key)

    async def _prep_start_inspection(self,key):
        s=self.server.state;store=self._preparation_store()
        s.prep_library=key;s.prep_reference=key;s.prep_tab='new';self._prep_refresh()
        await asyncio.to_thread(store.start,key,maximum_jobs=finite_number(s.prep_parallel_jobs,integer=True))
        s.prep_message='Parent/profile inspection submitted. No experimental scans are being refined.'
        self._prep_start_driver()

    def inspect_preparation_library(self,**_):
        async def inspect():
            key=await asyncio.to_thread(self._preparation_store().inspect_from_library,self.server.state.prep_library)
            await self._prep_start_inspection(key)
        return self._prep_action(inspect(),'Inspecting the library’s captured parent and profiles')

    def add_preparation_k(self,**_):
        s=self.server.state
        try:
            vector=k_vector([v.strip() for v in s.prep_custom_k.split(',')]);token=','.join(vector)
            if token not in {r['value'] for r in s.prep_k_options}:
                s.prep_k_options=[*s.prep_k_options,dict(title='('+', '.join(vector)+') · supplied',value=token)]
            s.prep_selected_k=list(dict.fromkeys([*s.prep_selected_k,token]));s.prep_custom_k=''
            self._preparation_scope_changed();s.prep_error=''
        except ValueError as exc:s.prep_error=str(exc)

    def _preparation_site_selection_changed(self,**_):
        s=self.server.state;s.prep_site_rows=[copy.deepcopy(self._prep_site_inputs[label]) for label in s.prep_sites if label in self._prep_site_inputs]

    def set_preparation_ion(self,label,value):
        if label not in self._prep_site_inputs:return
        self._prep_site_inputs[label]['form_factor']=str(value).strip()[:32];self._preparation_site_selection_changed()

    def set_preparation_g(self,label,value):
        if label not in self._prep_site_inputs:return
        self._prep_site_inputs[label]['lande_g']=str(value).strip()[:32];self._preparation_site_selection_changed()

    def discover_preparation_models(self,**_):
        return self._prep_action(self._discover_preparation_models(),'Discovering the declared model families')

    async def _discover_preparation_models(self):
        s=self.server.state;reference=s.prep_reference;identity=self._scope_identity();kind=s.prep_kind
        if not s.prep_context_ready or not reference:raise ValueError('Complete a parent/profile inspection first')
        allowed={r['value'] for r in s.prep_k_options};selected=list(s.prep_selected_k)
        if not selected or len(selected)>128 or len(selected)!=len(set(selected)) or any(k not in allowed for k in selected):
            raise ValueError('Select a nonempty finite k scope from the displayed parent-basis options')
        client=PreparationDiscovery(self._preparation_store());catalogs=[];rows=[]
        for index,token in enumerate(selected):
            if identity!=self._scope_identity():return
            s.prep_message=f'Discovering family {index+1} / {len(selected)}: ({token}). No refinement has started.';s.flush()
            try:
                item=await asyncio.to_thread(client.discover,reference,kind,token.split(','),opd_policy=s.prep_opd_policy,
                    maximum=finite_number(s.prep_construct_limit,integer=True))
                catalog=item['catalog'];ready=catalog.get('coverage',{}).get('discovery_complete') is True
                if kind=='structural':require_structural_coverage(catalog,policy=s.prep_opd_policy)
                rows.append(dict(k='('+token+')',count=len(catalog['candidates']),state='Available' if ready else 'Incomplete discovery'))
                if ready:catalogs.append(item)
            except ValueError as exc:rows.append(dict(k='('+token+')',count=0,state=str(exc)))
            if identity==self._scope_identity():s.prep_discovery_rows=copy.deepcopy(rows);s.flush()
        if identity!=self._scope_identity():return
        self._prep_catalogs=catalogs;self._prep_discovered_identity=identity
        s.prep_discovery_ready=len(catalogs)==len(selected) and all(c['catalog']['candidates'] for c in catalogs)
        self._preparation_count_changed()
        s.prep_message=('Review the scope and magnetic-site assumptions, then prepare the selected models.' if s.prep_discovery_ready
            else 'Some declared families are unavailable or empty. Review their messages; retry discovery or explicitly revise the selected scope.')

    def start_preparation_models(self,**_):
        return self._prep_action(self._start_preparation_models(),'Capturing and starting the selected preparation')

    async def _start_preparation_models(self):
        s=self.server.state
        if not s.prep_discovery_ready or self._prep_discovered_identity!=self._scope_identity():
            raise ValueError('Discover the currently selected model scope first')
        identity=self._prep_model_identity();sites=[]
        for label in s.prep_sites:
            row=self._prep_site_inputs.get(label)
            if row is None or not row['form_factor']:raise ValueError('Supply an ionic magnetic form factor for every selected site')
            sites.append(dict(label=label,form_factor=row['form_factor'],lande_g=finite_number(row['lande_g'])))
        store=self._preparation_store()
        key=await asyncio.to_thread(store.create_models,s.prep_reference,name=s.prep_name.strip(),kind=s.prep_kind,
            catalogs=copy.deepcopy(self._prep_catalogs),bank_labels=list(s.prep_selected_banks),sites=sites,
            maximum_moment=finite_number(s.prep_maximum_moment),mode_radius=finite_number(s.prep_mode_radius),
            cell_radius=finite_number(s.prep_cell_radius))
        if identity!=self._prep_model_identity():
            self._prep_refresh();s.prep_message='Setup changed. The captured model plan was saved without starting jobs; review the current settings.';return
        s.prep_library=key;s.prep_tab='libraries';self._prep_refresh()
        await asyncio.to_thread(store.start,key,maximum_jobs=finite_number(s.prep_parallel_jobs,integer=True))
        s.prep_message='Preparation submitted. These are reusable model calculations, not experimental fit results.'
        self._prep_start_driver()

    def _prep_model_identity(self):
        s=self.server.state
        return digest([self._scope_identity(),s.prep_name,s.prep_selected_banks,s.prep_sites,
            [self._prep_site_inputs.get(label) for label in s.prep_sites],
            s.prep_maximum_moment,s.prep_mode_radius,s.prep_cell_radius])

    def _prep_start_driver(self):
        self._prep_stopping=False
        if self._prep_driver is None or self._prep_driver.done():
            self._prep_driver=self._prep_schedule(self._prep_drive(),'radar-preparation-queue')

    async def _prep_drive(self):
        store=self._preparation_store();s=self.server.state
        while not self._prep_stopping:
            busy=False
            for key,queue in list(store._queues.items()):
                try:result=await asyncio.to_thread(queue.tick)
                except Exception as exc:
                    message='Preparation queue needs attention ('+type(exc).__name__+'). Existing Galaxy jobs were not restarted.'
                    self._prep_queue_errors[key]=message;s.prep_error=message
                    # A transient observation failure is not a terminal job.
                    # Keep polling the same handles without resubmitting them.
                    busy=busy or not isinstance(exc,ValueError)
                    s.flush();continue
                previous=self._prep_queue_errors.pop(key,None)
                if previous and s.prep_error==previous:
                    s.prep_error=next(reversed(self._prep_queue_errors.values()),'')
                busy=busy or bool(result['active'] or (not result['queue']['paused'] and not result['coverage']['complete']))
                if key==s.prep_reference and self._prep_loaded_reference!=key and result['coverage']['all_prepared']:
                    await self._prep_load_reference(key)
                if key==s.prep_library:
                    s.prep_active_jobs=result['active'];self._preparation_table_changed()
                    if result.get('blocked_training') and not result['active'] and not s.prep_busy:
                        s.prep_message='Ranking-model training is waiting for failed pattern blocks. Retry the entries needing attention, then resume preparation. Completed blocks remain available.'
                    elif result['coverage']['complete'] and not s.prep_busy:
                        if not result['coverage']['all_prepared']:
                            s.prep_message='Preparation stopped with entries needing attention. Successful sibling artifacts remain available.'
                        elif s.prep_library_kind=='Input inspection':
                            s.prep_message='Input inspection complete. Review the parent, banks and model scope in New preparation.'
                        else:
                            s.prep_message='Preparation complete. These calculations still need actual-scan compatibility checks before fitting.'
            self._prep_refresh();s.flush()
            if not busy:break
            await asyncio.sleep(2)

    def _preparation_library_changed(self,**_):
        s=self.server.state;s.prep_page=1;s.prep_download_href='';s.prep_active_jobs=0
        s.prep_training_record_open=False;s.prep_training_record=None
        if s.prep_library:self._preparation_store().registry.select(s.prep_library)
        self._preparation_table_changed()

    def _capture_prepared_model_inputs(self,protocol,root):
        """Called off the UI thread before a new immutable source dispatch."""
        enabled=((protocol['kind']=='magnetic' and protocol['policy'].get('magnetic_optimizer')=='profiled_quadratic')
            or (protocol['kind']=='structural' and protocol['policy'].get('structural_optimizer')=='prepared_nonlinear'))
        if not enabled:return None
        return self._capture_prepared_inputs(protocol,root)

    def _capture_prepared_inputs(self,context,root):
        """Common selection for independent Model and Live request contracts."""
        from gpx_recipe.preparation.hints import read,capture
        saved=read(root,context)
        if saved is not None:return saved
        store=self._preparation_store()
        if not store.registry.selected_key():return None
        return capture(store.registry.get(),context,root)

    def _prepared_selection_caption(self):
        # A read-only Results/Live refresh must not create a runtime, registry
        # or filesystem workspace. Actual selection/capture opens that store
        # explicitly, including its durable saved selection when appropriate.
        if self._preparation_workspace is None:
            return 'Open Experiment preparation to choose or review a saved library. Model searches check compatibility before reuse.'
        try:
            registry=self._preparation_workspace.registry;selected=registry.selected_key()
            if not selected:
                return 'No preparation library selected. Model searches build the required physics when they run.'
            row=next(row for row in registry.summaries() if row['id']==selected)
            if not row['available']:raise ValueError('Selected preparation needs review')
            if 'structural-ranking-model' in row['kinds']:
                return (f"Trained structural model: {row['name']}. "
                    'Compatible scans use its scores to prioritize the complete native candidate search. '
                    'This library contains no prepared native starting curves.')
            if 'structural' in row['kinds']:
                return (f"Prepared library: {row['name']} · {row['prepared']} / {row['total']} entries prepared. "
                    'Structural Model calculations check the current geometry and complete starting bank before reuse; changed contexts rebuild automatically.')
            if 'magnetic' not in row['kinds']:
                return (f"Prepared library: {row['name']}. This selection contains no magnetic scattering assets. "
                    'Magnetic model searches build the required physics when they run.')
            return (f"Prepared library: {row['name']} · {row['prepared']} / {row['total']} entries prepared. "
                'New magnetic Model and Live calculations check each scan’s compatibility before reuse.')
        except (OSError,ValueError):
            return 'The selected preparation needs review. Model searches can rebuild the required physics.'

    def _preparation_table_changed(self,**_):
        s=self.server.state
        s.prep_can_pause=False;s.prep_can_resume=False
        if not s.prep_library:
            s.prep_entries=[];s.prep_table_count=0;s.prep_pages=1
            s.prep_coverage='';s.prep_library_kind='';s.prep_stage_rows=[]
            return
        try:
            store=self._preparation_store()
            library=store.registry.get(s.prep_library);coverage=library.coverage();counts=coverage['counts']
            queue=store._queues.get(s.prep_library)
            queue_state=queue.snapshot() if queue else None
            paused=queue_state['paused'] if queue_state else True
            from .preparation_labels import training_progress
            continuing=training_progress(library,queue_state)
            unfinished=bool(counts['pending'] or counts['preparing'])
            s.prep_can_pause=bool(unfinished and queue and not paused)
            from gpx_recipe.preparation.structural_dependencies import ready
            runnable=bool(counts['preparing'] or any(row['state']=='pending' and ready(library,row['model'])
                for row in library.document['entries']))
            s.prep_can_resume=bool(runnable and paused)
            kinds={r['model']['kind'] for r in library.document['entries']}
            s.prep_library_kind='Input inspection' if kinds=={'reference'} else 'Model preparation'
            if kinds=={'structural-ranking-model'}:s.prep_library_kind='Imported trained structural model'
            if library.document.get('storage'):s.prep_library_kind='Compiled model library'
            s.prep_coverage=f"{counts['prepared']} prepared · {counts['preparing']} in preparation · {counts['pending']} pending · {counts['failed']+counts['cancelled']+counts['unsupported']} need attention · {coverage['total']} total"
            if library.document.get('storage'):
                s.prep_coverage+=' · Compiled analysis copy; training arrays remain in the original preparation History.'
            titles={'reference':'Parent/profile inspection','magnetic':'Magnetic responses',
                'structural':'Native starting curves','structural-examples':'Training pattern blocks',
                'structural-training':'Ranking models','structural-ranking-model':'Imported ranking models'}
            s.prep_stage_rows=[dict(kind=kind,title=titles[kind],total=sum(r['model']['kind']==kind for r in library.document['entries']),
                prepared=sum(r['model']['kind']==kind and r['state']=='prepared' for r in library.document['entries']))
                for kind in titles if kind in kinds]
            from .preparation_labels import scope_labels,entry_label,native_dimensions,training_description,training_validation
            labels=scope_labels(library)
            text=str(s.prep_entry_search).strip().lower()[:200];rows=[]
            for row in library.document['entries']:
                model=row['model'];candidate=model.get('settings',{}).get('candidate',{})
                display=entry_label(model,labels)
                reason=row['reason']
                if library.document.get('storage') and model['kind']=='structural-examples':
                    reason=(reason+' ' if reason else '')+'Training arrays archived; checksums retained.'
                if model['kind']=='structural-training' and row['state']=='preparing' and not reason:
                    reason=continuing.get(row['id'],'Training preparation is in progress. Recover its saved queue to inspect Galaxy status.')
                if model['kind']=='structural-training' and row['state']=='pending' and not reason:
                    keys=model['settings']['dependencies']
                    completed=sum(library.row(key)['state']=='prepared' for key in keys)
                    reason=(('Training patterns ready. Resume preparation to start training.' if paused else
                             'Training patterns ready; waiting for a training slot.') if completed==len(keys) else
                            f'Waiting for native training patterns: {completed} / {len(keys)} blocks prepared.')
                    failed=sum(library.row(key)['state'] in {'failed','cancelled','unsupported'} for key in keys)
                    if failed:reason+=f" {failed} {'block needs' if failed==1 else 'blocks need'} attention before training can start."
                if s.prep_entry_filter!='all' and row['state']!=s.prep_entry_filter:continue
                if s.prep_stage_filter!='all' and model['kind']!=s.prep_stage_filter:continue
                if text and text not in (' '.join(display.values())+' '+model['bank']+' '+reason).lower():continue
                rows.append(dict(id=row['id'],bank=model['bank'],**display,state=row['state'],reason=reason,
                    has_training_record=row['state']=='prepared' and model['kind'] in {'structural-training','structural-ranking-model'}))
            s.prep_table_count=len(rows);s.prep_pages=max(1,math.ceil(len(rows)/25))
            page=min(max(1,finite_number(s.prep_page,integer=True)),s.prep_pages)
            if page!=s.prep_page:s.prep_page=page
            visible=rows[(page-1)*25:page*25]
            entries={row['id']:row for row in library.document['entries']}
            # Verify only the displayed page's small metadata, never every
            # model's numerical payload on each progress refresh.
            for row in visible:
                entry=entries[row['id']]
                row.update(entry_label(entry['model'],labels,dimensions=native_dimensions(library,entry)))
                description=training_description(library,entry)
                if description:row['coordinates']=description
                row['validation']=training_validation(library,entry)
            s.prep_entries=visible
        except (OSError,ValueError) as exc:
            s.prep_entries=[];s.prep_table_count=0;s.prep_pages=1;s.prep_stage_rows=[]
            s.prep_error=str(exc)

    def show_preparation_training_record(self,entry_id,**_):
        key=self.server.state.prep_library
        async def show():
            from .preparation_training_record import describe
            if not key:raise ValueError('Select a preparation library first')
            library=self._preparation_store().registry.get(key);entry=library.row(entry_id)
            record=await asyncio.to_thread(describe,library,entry)
            if self.server.state.prep_library!=key or self.server.state.workspace_view!='preparation':
                self.server.state.prep_message='';return
            if record is None:raise ValueError('This entry has no completed model training record')
            self.server.state.prep_training_record_title=library.document['name']+' · '+entry['model']['bank']
            self.server.state.prep_training_record=record
            self.server.state.prep_training_record_open=True
            self.server.state.prep_message='Showing recorded model training; no calculation was submitted.'
        return self._prep_action(show(),'Reading the saved training record')

    def pause_preparation(self,**_):
        async def pause():
            await asyncio.to_thread(self._preparation_store().pause,self.server.state.prep_library)
            self.server.state.prep_message='Preparation paused. Acknowledged Galaxy jobs may finish; no new batches will be submitted.'
        return self._prep_action(pause(),'Pausing new preparation work')

    def resume_preparation(self,**_):
        async def resume():
            await asyncio.to_thread(self._preparation_store().start,self.server.state.prep_library,
                maximum_jobs=finite_number(self.server.state.prep_parallel_jobs,integer=True))
            self._prep_start_driver()
        return self._prep_action(resume(),'Resuming pending preparation')

    def retry_preparation_entry(self,entry):
        async def retry():
            store=self._preparation_store();key=self.server.state.prep_library
            await asyncio.to_thread(store.registry.get(key).retry,entry)
            await asyncio.to_thread(store.start,key,maximum_jobs=finite_number(self.server.state.prep_parallel_jobs,integer=True))
            self._prep_start_driver()
        return self._prep_action(retry(),'Retrying the selected failed entry')

    def refresh_preparation_saved(self,**_):
        async def refresh():
            store=self._preparation_store()
            rows=await asyncio.to_thread(store.saved)
            self.server.state.prep_saved_options=[dict(title=r['name'],value=r['dataset_id']) for r in rows]
            archives=await asyncio.to_thread(self.service.search_history_datasets,query='',limit=200,include_generated=True)
            self.server.state.prep_history_archives=[dict(title=r.get('name') or r['id'],value=r['id']) for r in archives
                if r.get('state')=='ok' and (r.get('extension') or r.get('file_ext') or '').lower()=='zip']
        return self._prep_action(refresh(),'Finding saved preparation in this History')

    def recover_preparation(self,**_):
        async def recover():
            key=await asyncio.to_thread(self._preparation_store().recover,identifier(self.server.state.prep_saved))
            self.server.state.prep_library=key;self.server.state.prep_tab='libraries';self._prep_refresh();self._prep_start_driver()
        return self._prep_action(recover(),'Recovering the original preparation jobs and artifacts')

    def import_preparation(self,**_):
        async def imported():
            s=self.server.state;store=self._preparation_store();owned=None
            try:
                if s.prep_import_source=='upload':path=checked_upload(self._prep_library_upload,s.prep_import_path,maximum=64*1024*1024)
                elif s.prep_import_source=='history':
                    if s.prep_history_archive not in {r['value'] for r in s.prep_history_archives}:raise ValueError('Select a ZIP from the current History list')
                    path=store.root/'imports'/(uuid.uuid4().hex+'.zip');path.parent.mkdir(parents=True,exist_ok=True);owned=path
                    await asyncio.to_thread(self.service._download_dataset,identifier(s.prep_history_archive),path,maximum_bytes=MAX_BYTES)
                else:raise ValueError('Choose a portable file or a ZIP in this History')
                result=await asyncio.to_thread(store.import_archive,path)
            finally:
                if owned is not None:owned.unlink(missing_ok=True)
            s.prep_library=result['library'];s.prep_message=result['message'];s.prep_tab='libraries';self._prep_refresh()
        return self._prep_action(imported(),'Importing a checked portable preparation library')

    def save_preparation_download(self,**_):
        async def saved():
            s=self.server.state;store=self._preparation_store();key=s.prep_library
            download_url(self.service.galaxy_url,'validation')
            path=store.root/'exports'/(uuid.uuid4().hex+'.zip');path.parent.mkdir(parents=True,exist_ok=True)
            try:
                await asyncio.to_thread(store.registry.export,key,path)
                dataset=identifier(await asyncio.to_thread(self.service.upload_document,path,
                    label='RADAR-PD prepared library '+store.registry.get(key).document['id'],wait_ready=True))
            finally:path.unlink(missing_ok=True)
            if s.prep_library==key:s.prep_download_href=download_url(self.service.galaxy_url,dataset)
            s.prep_message=('The portable library is saved in this Galaxy History. Download it using the link below. '
                'Completed training is exported as a compiled analysis copy; its original arrays stay in the preparation History.')
        return self._prep_action(saved(),'Saving the portable library to History')

    async def _preparation_shutdown(self):
        self._prep_stopping=True;store=getattr(self,'_preparation_workspace',None)
        if store:
            for queue in list(store._queues.values()):await asyncio.to_thread(queue.pause)
        task=getattr(self,'_prep_driver',None)
        if task is not None and not task.done():await task
