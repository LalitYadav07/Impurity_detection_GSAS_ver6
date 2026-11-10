"""Independent pre-experiment POWGEN workspace; never reads an IPTS."""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
import threading
import uuid
from fractions import Fraction
from decimal import Decimal, InvalidOperation

import numpy as np

from gpx_recipe.planning.contracts import validate_scenario
from gpx_recipe.planning.instrument import profiles
from gpx_recipe.planning.exposure_inputs import exposure_label

from .planning_storage import PlanningStorageMixin

ISOTOPE_MIXTURE = 'Atom fractions'

# Working ranges used by the current-cycle end-to-end calculations. These are
# inside the measured response and published numerical profile domains.
CURRENT_RANGES = {
    '2026B_HighRes_60HzB1_CWL0p8.instprm': (.3, 3.1),
    '2026B_HighRes_60HzB2_CWL1p5.instprm': (.55, 7.24),
    '2026B_HighRes_60HzB3_CWL2p665.instprm': (1.1, 7.24),
}


def cycle_label(value):
    match=re.fullmatch(r'(\d{4})[-_]([12])(?:_([12]))?_11A_CAL',value)
    return (match[1]+''.join('A' if part=='1' else 'B' for part in match.groups()[1:] if part)) if match else value


def _isotope_percent(fraction):
    return str(Decimal(str(fraction))*100)


def _isotope_atom_fraction(percent):
    try:
        if len(str(percent))>128:raise ValueError('Isotope percentage is too long')
        return float(Decimal(str(percent))/100)
    except (InvalidOperation, OverflowError) as exc:
        raise ValueError('A finite isotope percentage is required') from exc

FIELDS = ('plan_configuration','plan_custom_profile','plan_profile_bank','plan_cif','plan_mcif',
          'plan_comparison_cif','plan_mass','plan_radius','plan_packing','plan_beam_width',
          'plan_beam_height','plan_seconds','plan_charge','plan_exposure_basis','plan_current_ua','plan_power','plan_energy','plan_yield_ratio',
          'plan_d_min','plan_d_max','plan_step','plan_seed','plan_moment_multiplier',
          'plan_isotopes','plan_response_mode','plan_container','plan_intensity_calibration','plan_resolution_model') + tuple(
              f'plan_phase_{index}_{field}' for index in range(1,4)
              for field in ('cif','percent','size','microstrain'))


class PlanningMixin(PlanningStorageMixin):
    def _init_planning(self):
        s = self.server.state
        self._planning_restoring=False
        self._planning_owned_inputs={}
        self._planning_previous_workspace=None
        defaults = dict(plan_step_ui=0,plan_beamline='PG3',plan_cycle='2026-2_11A_CAL',plan_transition='none',
            plan_name='New experiment',plan_saved_id='',plan_saved_options=[],plan_save_message='',
            plan_cell_values=['']*6,plan_site_rows=[],plan_form_factor_options={},
            plan_moment_sites=[],plan_displacement_sites=[],plan_structural_note='',plan_child_cell_note='',
            plan_configuration='2026B_HighRes_60HzB2_CWL1p5.instprm',
            plan_custom_profile='',plan_profile_bank=1,plan_cif='',plan_mcif='',plan_comparison_cif='',
            plan_mass=1.,plan_radius=.3,plan_packing=.5,plan_beam_width=1.,plan_beam_height=4.,
            plan_seconds='60, 300, 1800',plan_power=1.4,plan_energy=1.,plan_yield_ratio=None,
            plan_charge='25, 125, 750',plan_exposure_basis='current_time',plan_current_ua=1500.,plan_pattern_view='reduced',
            plan_plot_mode='pattern',plan_error_model='predictive',plan_error_note='',
            plan_container='PAC06',
            plan_intensity_calibration='vanadium',plan_resolution_model='cycle_table',
            plan_d_min=.55,plan_d_max=7.24,plan_step=.0004,plan_seed=42,plan_moment_multiplier=1.,
            plan_isotopes='{}',plan_response_mode='current_cycle',plan_busy=False,plan_can_cancel=False,plan_error='',plan_standalone=False,
            plan_message='Upload a nuclear CIF and select a POWGEN configuration.',
            plan_current=False,plan_result_ready=False,plan_look=0,plan_look_options=[],
            plan_figure={'data':[],'layout':{}},plan_qualification='',plan_warnings=[],plan_summary=[],
            plan_isotope_options=[],plan_advanced=False,plan_k='0, 0, 0',plan_sites=[],plan_site_options=[],
            plan_form_factors='{}',plan_site_policy='all',plan_candidate='',plan_candidates=[],plan_parameters='{}',
            plan_parameter_rows=[],plan_structural_parameter_rows=[],
            plan_magnetic_message='',plan_coordinate_note='',plan_structural_candidate='',plan_structural_candidates=[],
            plan_structural_parameters='{}',plan_structural_cell='',plan_structural_message='',
            plan_can_promote_parent=False,plan_promoted_parent=False,plan_parent_note='',
            plan_job_id='',plan_job_state='',plan_job_operation='',plan_recoverable_jobs=[],plan_recover_job_id='',
            plan_bundle_upload='')
        for index in range(1,4):
            for field,value in (('cif',''),('percent',0.),('size',''),('microstrain','')):
                defaults[f'plan_phase_{index}_{field}']=value
        for key, value in defaults.items(): setattr(s,key,value)
        self._planning_profiles=profiles()
        s.plan_cycles=[{'title':cycle_label(cycle),'value':cycle,
                       'props':{'disabled':not any(row['forward_supported'] for row in self._planning_profiles if row['cycle']==cycle)}} for cycle in
                       sorted({row['cycle'] for row in self._planning_profiles},reverse=True)]
        self._planning_profile_options()
        self._planning_saved_list()
        self._planning_profile_context=(s.plan_configuration,s.plan_custom_profile)
        self._planning_generation=0;self._planning_output=None;self._planning_cancel=None
        self._planning_receipt_sha256=None;self._planning_parent_receipt_sha256=None
        self._planning_input_errors={};self._planning_isotope_source=self._planning_structure_sources()
        self._planning_parent_output=None;self._planning_catalog=None;self._planning_prepared=None;self._planning_magnetic_request=None
        self._planning_structural_catalog=None;self._planning_prepared_kind=None
        self._planning_parent_state=None
        self._planning_runtime_cache=None;self._planning_native_artifacts={}
        s.change(*FIELDS)(self._planning_edited)
        s.change('plan_name')(self._planning_name_edited)
        s.change('plan_look','plan_pattern_view','plan_plot_mode','plan_error_model')(self._planning_plot)
        s.change('plan_resolution_model')(self._planning_resolution_edited)
        s.change('plan_cif','plan_custom_profile','plan_configuration','plan_profile_bank','plan_isotopes',
                 *(f'plan_phase_{index}_cif' for index in range(1,4)))(self._planning_context_edited)
        s.change('plan_k')(self._planning_k_edited)
        s.change('plan_candidate','plan_sites','plan_form_factors','plan_site_policy')(self._planning_magnetic_edited)
        s.change('plan_parameters')(self._planning_edited)
        s.change('plan_structural_candidate')(self._planning_magnetic_edited)
        s.change('plan_structural_parameters','plan_structural_cell')(self._planning_edited)
        s.change('plan_cycle')(self._planning_cycle_edited)
        s.change('plan_transition')(self._planning_transition_edited)
        s.change('plan_sites','plan_form_factors')(self._planning_site_rows)
        s.change('plan_d_min','plan_d_max','plan_step','plan_resolution_model')(self._planning_context_edited)

    def select_planning_tab(self, value, **_):
        if value=='planning':self.open_planning_workspace()
        elif value=='analysis' and self.server.state.workspace_view=='planning':self.close_planning_workspace()

    def _planning_profile_options(self):
        s=self.server.state
        rows=sorted(self._planning_profiles,key=lambda row:row['id'] not in CURRENT_RANGES)
        s.plan_configurations=[{'title':('Current count response · ' if row['id'] in CURRENT_RANGES else '')+row['title'],
            'value':row['id'],'props':{'disabled':not row['forward_supported']}} for row in rows]
        s.plan_visible_configurations=[{'title':f"{row['mode']} · {row['wavelength']} Å · {row['frequency']:g} Hz"
                +(' · shape only' if row['id'] not in CURRENT_RANGES else ''),
            'value':row['id'],'props':{'disabled':not row['forward_supported']}}
            for row in rows if row['cycle']==s.plan_cycle]
        counts=s.plan_configuration in CURRENT_RANGES and not s.plan_custom_profile
        s.plan_response_options=([{'title':'Counts and noise · measured 2026B response','value':'current_cycle'}] if counts else [])+[
            {'title':'Pattern shape preview','value':'profile_only'}]
        s.plan_resolution_options=([{'title':'Published 2026B resolution table','value':'cycle_table'}] if counts else [])+[
            {'title':'Selected instrument profile','value':'profile'}]
        s.plan_configuration_note=('The matching instrument profile, resolution table, vanadium response and container references are selected automatically. Absolute intensity and noise remain conditional estimates.'
            if counts else 'An instrument profile is available for this setting. A matching count-response calibration is unavailable, so this setting supports a shape preview.')

    def _planning_cycle_edited(self, **_):
        if self._planning_restoring:return
        s=self.server.state
        options=[row for row in self._planning_profiles if row['cycle']==s.plan_cycle and row['forward_supported']]
        if options and s.plan_configuration not in {row['id'] for row in options}:
            s.plan_configuration=options[0]['id']
        self._planning_profile_options()

    def _planning_transition_edited(self, **_):
        if self._planning_restoring:return
        s=self.server.state
        if s.plan_transition!='mcif':s.plan_mcif=''
        if s.plan_transition!='comparison':s.plan_comparison_cif=''
        if s.plan_transition!='magnetic' and s.plan_promoted_parent:self.reset_planning_nuclear_parent()
        self._planning_magnetic_edited()

    def update_planning_k_component(self, index, value):
        if int(index) not in range(3):return
        parts=(str(self.server.state.plan_k).split(',')+['0']*3)[:3]
        parts[int(index)]=str(value or '0').strip()[:32]
        self.server.state.plan_k=', '.join(parts)

    def _planning_site_rows(self, **_):
        if self._planning_restoring:return
        s=self.server.state
        try:
            factors=json.loads(s.plan_form_factors)
            if set(factors)-set(s.plan_sites):
                factors={k:v for k,v in factors.items() if k in s.plan_sites};s.plan_form_factors=json.dumps(factors)
            atoms={row['value']:row.get('atom_type','') for row in s.plan_site_options}
            s.plan_site_rows=[{'label':label,'form_factor':self._planning_form_factor(label,atoms[label],factors),
                'options':s.plan_form_factor_options.get(label,[])}
                for label in s.plan_sites if label in atoms]
        except (TypeError,ValueError):s.plan_site_rows=[]

    def _planning_form_factor(self, label, atom_type, choices):
        if label in choices:return choices[label]
        options=self.server.state.plan_form_factor_options.get(label,[])
        if atom_type in options:return atom_type
        return options[0] if len(options)==1 else ''

    def update_planning_site(self, label, field, value):
        s=self.server.state
        if label not in s.plan_sites or field!='form_factor':return
        key='plan_form_factors'
        mapping=json.loads(getattr(s,key) or '{}')
        mapping[label]=str(value or '').strip()[:40]
        setattr(s,key,json.dumps(mapping))
        self._planning_site_rows()

    def update_planning_cell(self, index, value):
        if int(index) not in range(6):return
        s=self.server.state
        values=list(s.plan_cell_values);values[int(index)]=str(value or '').strip()[:40]
        s.plan_cell_values=values;self._planning_edited()
        try:
            if any(values) and not all(values):raise ValueError('Complete all six child lattice parameters, or leave all six blank.')
            numbers=list(map(float,values)) if any(values) else []
            if numbers and not np.isfinite(numbers).all():raise ValueError('Child lattice parameters must be finite.')
            s.plan_structural_cell=json.dumps(numbers) if numbers else ''
            self._planning_input_errors.pop('cell',None);s.plan_error=''
        except ValueError as exc:
            self._planning_input_errors['cell']=str(exc);s.plan_error=str(exc)

    def inherit_planning_cell(self, **_):
        self.server.state.plan_cell_values=['']*6
        self.server.state.plan_structural_cell=''
        self._planning_input_errors.pop('cell',None)
        self._planning_edited();self.server.state.plan_error=''

    def edit_planning_child_cell(self, **_):
        if self._planning_prepared_kind!='structural':return
        values=self._planning_prepared.get('controls',{}).get('cell',[])
        if len(values)!=6:return
        self.server.state.plan_cell_values=[str(v) for v in values]
        self.server.state.plan_structural_cell=json.dumps(values)
        self._planning_input_errors.pop('cell',None)
        self._planning_edited();self.server.state.plan_error=''

    def _planning_validate_transition(self):
        s=self.server.state
        if s.plan_transition in {'magnetic','structural'} and self._planning_prepared_kind!=s.plan_transition:
            raise ValueError('Prepare the selected transition before simulating, or select Nuclear structure only.')
        if s.plan_transition=='mcif' and not s.plan_mcif:raise ValueError('Upload the magnetic CIF for this scenario.')
        if s.plan_transition=='comparison' and not s.plan_comparison_cif:raise ValueError('Upload the comparison nuclear CIF.')

    def _planning_execute(self, request, directory, event):
        operation='planning-parent' if Path(directory).name.startswith('parent-') else 'planning-simulate'
        return self._planning_dispatch(operation,request,directory,event)

    def _planning_native_input(self, value):
        original=Path(value)
        if original.is_symlink():raise ValueError('Native planner inputs cannot be symbolic links')
        path=original.resolve(strict=True)
        record=self._planning_native_artifacts.get(str(path))
        if (not record or not path.is_relative_to(self._planning_root())
                or hashlib.sha256(path.read_bytes()).hexdigest()!=record.get('sha256')):
            raise ValueError('Prepare this native model in the planner, or recover its verified Galaxy result. Uploaded GPX files are not executable planner inputs.')
        return path

    def _planning_runtime(self):
        if self._planning_runtime_cache is not None:return self._planning_runtime_cache
        from .planning_runtime import GalaxyPlanningRuntime,LocalPlanningRuntime
        service=getattr(self,'service',None)
        local=bool(os.environ.get('RADAR_PD_PLANNING_PYTHON') or os.environ.get('RADAR_PD_LOCAL_EXPERIMENTAL_BACKEND')=='1')
        if not local and service is not None and all(getattr(service,key,None) for key in ('galaxy_url','galaxy_key','history_id')):
            remote=self._model_remote() if hasattr(self,'_model_remote') else None
            if remote is None:
                from .model_runtime import GalaxyModelRuntime
                remote=GalaxyModelRuntime(self._planning_root()/'runtime',service)
            self._planning_runtime_cache=GalaxyPlanningRuntime(remote,self._planning_root()/'runtime',
                check_text=self._planning_input,check_native=self._planning_native_input,
                progress=self._planning_job_progress,checkpoint=self._planning_checkpoint_operation)
        elif local:
            self._planning_runtime_cache=LocalPlanningRuntime(check_text=self._planning_input,check_native=self._planning_native_input)
        else:
            raise RuntimeError('The managed planner worker is unavailable. Connect the Galaxy scientific service to run this plan.')
        return self._planning_runtime_cache

    def _planning_job_progress(self, *, operation, state, job_id='', message='', cancel=None):
        loop=getattr(self,'_planning_progress_loop',None)
        try:current=asyncio.get_running_loop()
        except RuntimeError:current=None
        if loop is not None and current is not loop:
            try:
                loop.call_soon_threadsafe(lambda:self._planning_job_progress(
                    operation=operation,state=state,job_id=job_id,message=message,cancel=cancel))
            except RuntimeError:
                pass  # A closed viewer must not interrupt the durable worker.
            return
        s=self.server.state
        if self._planning_cancel is None or self._planning_cancel is not cancel or self._planning_cancel.is_set():return
        s.plan_job_operation=operation;s.plan_job_state=state;s.plan_job_id=job_id
        if job_id:
            s.plan_message=f'{operation.removeprefix("planning-").replace("-"," ")} · Galaxy {state} · {job_id}'
            if message:s.plan_message+=' · '+str(message)[:400]
        else:
            s.plan_message=str(message)[:400] if message else str(state)
        s.flush()

    def _planning_dispatch(self, operation, request, directory, event):
        runtime=self._planning_runtime()
        result=runtime.execute(operation,request,directory,event)
        self._planning_native_artifacts.update(runtime.artifacts)
        return result

    async def _ensure_planning_parent(self, generation, event):
        if self._planning_parent_output is not None:
            self._planning_parent_receipt()
            return
        s=self.server.state
        s.plan_message='Preparing the nuclear structure calculation.';s.flush()
        request=self._planning_request()
        request.update(mcif=None,comparison_cif=None,response_mode='profile_only',intensity_calibration='vanadium')
        directory=self._planning_root()/('parent-'+uuid.uuid4().hex)
        receipt=await asyncio.to_thread(self._planning_execute,request,directory,event)
        if generation!=self._planning_generation or event.is_set():return
        self._planning_parent_output=directory/'simulation'
        self._planning_parent_receipt_sha256=hashlib.sha256((self._planning_parent_output/'result.json').read_bytes()).hexdigest()
        self._planning_isotope_rows(receipt)
        s.plan_message='Structure prepared. Isotope choices and transition discovery are ready.'

    def prepare_planning_parent(self, **_):
        async def prepare():
            s=self.server.state
            if s.plan_busy:return
            s.plan_busy=True;s.plan_error='';generation=self._planning_generation
            event=self._planning_begin_calculation()
            try:await self._ensure_planning_parent(generation,event)
            except Exception as exc:
                if generation==self._planning_generation and not event.is_set():s.plan_error=str(exc)
            finally:
                self._planning_finish_job(event)
        self._planning_schedule(prepare(),'radar-planning-parent')

    def _planning_resolution_edited(self, **_):
        if self._planning_restoring:return
        if self.server.state.plan_resolution_model=='cycle_table':
            self.server.state.plan_intensity_calibration='vanadium'

    def _planning_context_edited(self, **_):
        if self._planning_restoring:return
        s=self.server.state
        profile_context=(s.plan_configuration,s.plan_custom_profile)
        if profile_context!=self._planning_profile_context:
            self._planning_profile_context=profile_context
            if s.plan_configuration in CURRENT_RANGES and not s.plan_custom_profile:
                s.plan_d_min,s.plan_d_max=CURRENT_RANGES[s.plan_configuration]
                s.plan_response_mode='current_cycle';s.plan_resolution_model='cycle_table'
            else:
                s.plan_response_mode='profile_only';s.plan_resolution_model='profile'
        if s.plan_configuration!='2026B_HighRes_60HzB2_CWL1p5.instprm' or s.plan_custom_profile:
            s.plan_intensity_calibration='vanadium'
        if self._planning_structure_sources()!=self._planning_isotope_source:
            self._planning_isotope_source=self._planning_structure_sources()
            s.plan_isotope_options=[];s.plan_isotopes='{}'
        self._planning_profile_options()
        self._planning_parent_output=None
        self._planning_parent_receipt_sha256=None
        self._planning_parent_state=None
        s.plan_promoted_parent=False;s.plan_parent_note=''
        self._planning_k_edited()

    def load_planning_example(self, **_):
        if self.server.state.plan_busy:return
        from gpx_recipe.planning import __file__ as planning_file
        s=self.server.state
        s.plan_cif=str(Path(planning_file).parent/'data/examples/silicon.cif')
        s.plan_mcif='';s.plan_comparison_cif='';s.plan_isotopes='{}'
        s.plan_cycle='2026-2_11A_CAL';s.plan_transition='none'
        s.plan_configuration='2026B_HighRes_60HzB2_CWL1p5.instprm';s.plan_custom_profile='';s.plan_profile_bank=1
        s.plan_mass=.95;s.plan_radius=.295;s.plan_packing=.3730165664701342
        s.plan_beam_width=1.;s.plan_beam_height=4.
        s.plan_response_mode='current_cycle';s.plan_resolution_model='cycle_table'
        s.plan_intensity_calibration='vanadium';s.plan_container='PAC06'
        s.plan_d_min=.55;s.plan_d_max=7.24;s.plan_step=.0004
        s.plan_moment_multiplier=1.
        s.plan_exposure_basis='current_time';s.plan_current_ua=1500.;s.plan_seconds='60, 300, 1800'
        for index in range(1,4):
            for field,value in [('cif',''),('percent',0.),('size',''),('microstrain','')]:
                setattr(s,f'plan_phase_{index}_{field}',value)
        self._planning_context_edited();self._planning_edited()
        s.plan_message='Silicon example loaded with assumed 0.950 g mass and 40 mm fill. Select Simulate to calculate it.'
        s.flush()

    def _planning_structure_sources(self):
        s=self.server.state
        return tuple(str(value) for value in [s.plan_cif]+
            [getattr(s,f'plan_phase_{index}_cif') for index in range(1,4)])

    def _planning_receipt(self, root, expected):
        raw=(root/'result.json').read_bytes()
        if not expected or hashlib.sha256(raw).hexdigest()!=expected:
            raise ValueError('The simulation receipt changed; calculate the scenario again')
        return json.loads(raw)

    def _planning_parent_receipt(self):
        root=self._planning_parent_output
        if root is None:raise ValueError('Calculate a nuclear-only preview first')
        receipt=self._planning_receipt(root,self._planning_parent_receipt_sha256)
        relative=receipt.get('parent_model_artifact','model/forward.gpx')
        path=(root/relative).resolve()
        if (not path.is_relative_to(root.resolve()) or
            hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['artifacts'].get(relative)):
            raise ValueError('The nuclear parent model changed; calculate a new preview')
        return receipt

    def _planning_parent_request(self, receipt):
        root=self._planning_parent_output
        source=root/receipt.get('parent_model_artifact','model/forward.gpx')
        request={'source_gpx':str(source),'source_gpx_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
        if self._planning_parent_state is not None:
            request['nuclear_reference']={'schema':'powgen-planning-nuclear-reference/v1',
                'source_gpx_sha256':request['source_gpx_sha256'],
                'nuclear_cell_multiplier':self._planning_parent_state['nuclear_cell_multiplier'],
                'structural_state':{k:v for k,v in self._planning_parent_state.items() if k!='gpx'}}
        return request

    def promote_planning_structural_parent(self, **_):
        s=self.server.state
        if s.plan_busy:return
        try:
            if not s.plan_current or not s.plan_can_promote_parent:
                raise ValueError('Simulate the current structural-only model before using it as a parent')
            root=self._planning_output
            receipt=self._planning_receipt(root,self._planning_receipt_sha256)
            structural=receipt.get('structural_state')
            if not structural or receipt.get('magnetic_state'):
                raise ValueError('Choose a structural-only simulation as the nuclear parent')
            relative=receipt.get('parent_model_artifact','model/forward.gpx')
            path=(root/relative).resolve()
            if (not path.is_relative_to(root.resolve()) or
                hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['artifacts'].get(relative)):
                raise ValueError('The structural parent changed; simulate it again')
            self._planning_parent_output=root
            self._planning_parent_receipt_sha256=self._planning_receipt_sha256
            self._planning_parent_state={**structural,'gpx':str(path),
                'gpx_sha256':receipt['artifacts'][relative],
                'selected_result_sha256':self._planning_receipt_sha256,
                'prepared_structural_gpx_sha256':structural['gpx_sha256']}
            self._planning_k_edited();s.plan_k='0, 0, 0'
            s.plan_transition='magnetic'
            s.plan_promoted_parent=True
            s.plan_parent_note='Using the simulated structural child as nuclear parent. Enter magnetic k in this child cell’s reciprocal basis. The original nuclear-cell intensity reference is preserved.'
            s.plan_message='Structural parent selected. Find magnetic subgroups and prepare its allowed moments.'
            s.plan_error='';s.flush()
        except Exception as exc:s.plan_error=str(exc)

    def reset_planning_nuclear_parent(self, **_):
        if self.server.state.plan_busy:return
        self._planning_context_edited()
        self.server.state.plan_message='Uploaded nuclear structure selected. Calculate a nuclear preview to discover subgroups.'

    def _planning_k_edited(self, **_):
        if self._planning_restoring:return
        self._planning_catalog=None;self.server.state.plan_candidates=[];self.server.state.plan_candidate=''
        self._planning_structural_catalog=None;self.server.state.plan_structural_candidates=[];self.server.state.plan_structural_candidate=''
        self._planning_magnetic_edited()

    def _planning_magnetic_edited(self, **_):
        if self._planning_restoring:return
        self._planning_input_errors={}
        self._planning_prepared=None;self._planning_magnetic_request=None
        self._planning_prepared_kind=None
        self.server.state.plan_coordinate_note=''
        self.server.state.plan_parameter_rows=[];self.server.state.plan_structural_parameter_rows=[]
        self.server.state.plan_moment_sites=[];self.server.state.plan_displacement_sites=[]
        self.server.state.plan_structural_note='';self.server.state.plan_child_cell_note=''
        self._planning_edited()

    def _planning_parameter_rows(self, kind, values):
        prefix='plan_parameters' if kind=='magnetic' else 'plan_structural_parameters'
        rows='plan_parameter_rows' if kind=='magnetic' else 'plan_structural_parameter_rows'
        setattr(self.server.state,prefix,json.dumps(values,indent=2))
        controls=(self._planning_prepared or {}).get('controls')
        if controls:
            from gpx_recipe.planning.transition_controls import control_view
            parameters,sites=control_view(controls,values,kind)
            setattr(self.server.state,rows,parameters)
            setattr(self.server.state,'plan_moment_sites' if kind=='magnetic' else 'plan_displacement_sites',sites)
        else:
            setattr(self.server.state,rows,[{'name':name,'value':value,'symbol':f'q{i+1}',
                'label':name+(' (μB)' if kind=='magnetic' else ' (Å)'), 'description':''}
                for i,(name,value) in enumerate(values.items())])

    def _update_planning_parameter(self, kind, name, value):
        self._planning_edited()
        try:
            field='plan_parameters' if kind=='magnetic' else 'plan_structural_parameters'
            values=json.loads(getattr(self.server.state,field));number=float(value)
            limit=30 if kind=='magnetic' else 2
            if name not in values or not np.isfinite(number) or abs(number)>limit:
                raise ValueError(f'Supply a finite allowed coordinate within ±{limit}')
            values[name]=number;self._planning_parameter_rows(kind,values);self.server.state.plan_error=''
            self._planning_input_errors.pop((kind,name),None)
        except (ValueError,TypeError) as exc:
            self._planning_input_errors[(kind,name)]=str(exc);self.server.state.plan_error=str(exc)

    def update_planning_magnetic_parameter(self, name, value):
        self._update_planning_parameter('magnetic',name,value)

    def update_planning_structural_parameter(self, name, value):
        self._update_planning_parameter('structural',name,value)

    def update_planning_isotope(self, element, value):
        self._planning_edited()
        try:
            entry=next(row for row in self.server.state.plan_isotope_options if row['element']==element)
            if value not in entry['options'] + [ISOTOPE_MIXTURE]:raise ValueError('Select a listed isotope')
            choices=json.loads(self.server.state.plan_isotopes)
            if value == ISOTOPE_MIXTURE:
                choice = choices.get(element, entry.get('selection',entry['selected']))
                if not isinstance(choice,dict):choice={choice:1.}
                entry={**entry,'abundances':[{'label':label,'percent':_isotope_percent(choice.get(label,0.))} for label in entry['options']]}
                choices[element]=choice
            else:
                choices[element]=value
            self.server.state.plan_isotopes=json.dumps(choices)
            self.server.state.plan_isotope_options=[{**entry,'selected':value} if row['element']==element else row for row in self.server.state.plan_isotope_options]
            self.server.state.plan_error=''
        except (ValueError,TypeError,StopIteration):self.server.state.plan_error='Select an isotope from the current structure preview'

    def _planning_isotopes(self):
        from gpx_recipe.planning.isotopes import selections
        choices=json.loads(self.server.state.plan_isotopes or '{}')
        for entry in self.server.state.plan_isotope_options:
            if entry['selected'] != ISOTOPE_MIXTURE:continue
            rows=entry.get('abundances',[])
            if (len(rows)!=len(entry['options']) or
                {row['label'] for row in rows} != set(entry['options'])):
                raise ValueError('Use the isotope choices from the current structure preview')
            fractions={row['label']:_isotope_atom_fraction(row['percent']) for row in rows}
            choices[entry['element']]={label:number for label,number in fractions.items() if number != 0}
        return selections(choices)

    def update_planning_isotope_fraction(self, element, label, value):
        self._planning_edited()
        try:
            entries=self.server.state.plan_isotope_options
            entry=next(row for row in entries if row['element']==element)
            if entry['selected']!=ISOTOPE_MIXTURE or label not in entry['options']:
                raise ValueError('Choose atom fractions for a listed isotope')
            updated={**entry,'abundances':[{**row,'percent':str(value)} if row['label']==label else row for row in entry['abundances']]}
            self.server.state.plan_isotope_options=[updated if row['element']==element else row for row in entries]
            choices=self._planning_isotopes()
            self.server.state.plan_isotopes=json.dumps(choices)
            self.server.state.plan_error=''
        except (ValueError,TypeError,StopIteration) as exc:
            self.server.state.plan_error='Use finite isotope atom percentages totaling 100%. '+str(exc)

    def _planning_isotope_rows(self, receipt):
        self.server.state.plan_isotope_options=[{
            'element':element,'options':options,
            'selection':receipt['isotope_selection'][element],
            'selected':ISOTOPE_MIXTURE if isinstance(receipt['isotope_selection'][element],dict) else receipt['isotope_selection'][element],
            'abundances':[{'label':label,'percent':_isotope_percent(receipt['isotope_selection'][element].get(label,0.))} for label in options]
                         if isinstance(receipt['isotope_selection'][element],dict) else []}
            for element,options in receipt['isotope_options'].items()]

    def _planning_native(self, operation, request, directory):
        name='planning-magnetic-prepare' if operation=='prepare' else 'planning-'+operation
        return self._planning_dispatch(name,request,directory,self._planning_cancel or threading.Event())

    def discover_planning_magnetic_groups(self, **_):
        async def discover():
            s=self.server.state
            if s.plan_busy:return
            s.plan_busy=True;s.plan_error='';generation=self._planning_generation
            event=self._planning_begin_calculation()
            try:
                await self._ensure_planning_parent(generation,event)
                if generation!=self._planning_generation or event.is_set():return
                parent_output=self._planning_parent_output
                parent_receipt=self._planning_parent_receipt()
                k=[str(Fraction(part.strip())) for part in s.plan_k.split(',')]
                if len(k)!=3 or any(Fraction(part).denominator>96 for part in k):
                    raise ValueError('Supply one exact commensurate k vector with denominators at most 96')
                request=self._planning_parent_request(parent_receipt)
                root=parent_output.parent/('magnetic-discovery-'+uuid.uuid4().hex)
                context=await asyncio.to_thread(self._planning_native,'context',request,root)
                parent=next(p for p in context['phases'] if p['type']=='nuclear')
                from gpx_recipe.symmetry.discovery import discover_magnetic
                s.plan_message='Finding magnetic subgroups with Bilbao…';s.flush()
                catalog=await asyncio.to_thread(discover_magnetic,parent,k,api_key=os.environ.get('BCS_API_KEY',''),timeout_seconds=60)
                if generation!=self._planning_generation or event.is_set():return
                self._planning_catalog=(catalog,parent,request,root)
                s.plan_candidates=[{'title':c['magnetic_space_group']+' · '+c['id'][-8:],'value':c['id']} for c in catalog['candidates']]
                s.plan_form_factor_options=context.get('magnetic_form_factors',{})
                s.plan_site_options=[{'title':a['label']+' ('+a['type']+')','value':a['label'],'atom_type':a['type']}
                    for a in parent['atoms'] if s.plan_form_factor_options.get(a['label'])]
                self._planning_site_rows()
                s.plan_magnetic_message=f"{len(catalog['candidates'])} subgroup embeddings returned for the supplied parent and k. Select sites and a subgroup to prepare its allowed moments."
                (root/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
                s.plan_message=('Magnetic subgroups ready. Select a subgroup and sites to prepare allowed moments.'
                    if catalog['candidates'] else 'Magnetic subgroup search complete. No subgroups were returned for this parent and k.')
            except Exception as exc:
                if generation==self._planning_generation and not event.is_set():s.plan_error=str(exc)
            finally:
                self._planning_finish_job(event)
        self._planning_schedule(discover(),'radar-planning-magnetic-discovery')

    def prepare_planning_magnetic_moments(self, **_):
        async def prepare():
            s=self.server.state
            if s.plan_busy:return
            s.plan_busy=True;s.plan_error='';generation=self._planning_generation
            event=self._planning_begin_calculation()
            try:
                if self._planning_catalog is None:raise ValueError('Discover subgroups for the current parent and k first')
                catalog,parent,source,root=self._planning_catalog
                candidate=next(c for c in catalog['candidates'] if c['id']==s.plan_candidate)
                if not s.plan_sites:raise ValueError('Select the magnetic parent sites')
                factors=json.loads(s.plan_form_factors or '{}')
                atoms={a['label']:a for a in parent['atoms']}
                if not isinstance(factors,dict) or set(factors)-set(s.plan_sites):
                    raise ValueError('Form factors must refer to the selected magnetic sites')
                sites=[{'label':label,'form_factor':self._planning_form_factor(label,atoms[label]['type'],factors)} for label in s.plan_sites]
                if any(not site['form_factor'] for site in sites):
                    raise ValueError('Choose the magnetic ion for each selected site; its oxidation state is not specified by this CIF.')
                request={**source,'parent_setting_sha256':parent['setting_sha256'],'candidate':candidate,
                         'sites':sites,'site_policy':s.plan_site_policy,'k':catalog['k']}
                prepared=await asyncio.to_thread(self._planning_native,'prepare',request,root/('prepare-'+uuid.uuid4().hex))
                if generation!=self._planning_generation or event.is_set():return
                self._planning_prepared=prepared;self._planning_magnetic_request=request;self._planning_prepared_kind='magnetic'
                self._planning_parameter_rows('magnetic',{p['name']:p['value'] for p in prepared['parameters']})
                s.plan_coordinate_note=prepared['coordinate_convention']
                s.plan_magnetic_message=f"{len(prepared['controls']['sites'])} magnetic sites · {len(prepared['parameters'])} independent moment parameters. Enter moments below; linked components update automatically."
                s.plan_message='Magnetic controls ready. Enter moment values and simulate.'
            except Exception as exc:
                if generation==self._planning_generation and not event.is_set():s.plan_error=str(exc)
            finally:
                self._planning_finish_job(event)
        self._planning_schedule(prepare(),'radar-planning-magnetic-prepare')

    def discover_planning_structural_groups(self, **_):
        async def discover():
            s=self.server.state
            if s.plan_busy:return
            s.plan_busy=True;s.plan_error='';generation=self._planning_generation
            event=self._planning_begin_calculation()
            try:
                await self._ensure_planning_parent(generation,event)
                if generation!=self._planning_generation or event.is_set():return
                parent_output=self._planning_parent_output
                parent_receipt=self._planning_parent_receipt()
                k=[str(Fraction(part.strip())) for part in s.plan_k.split(',')]
                if len(k)!=3 or any(Fraction(part).denominator>96 for part in k):raise ValueError('Supply one exact commensurate k vector')
                request=self._planning_parent_request(parent_receipt)
                root=parent_output.parent/('structural-discovery-'+uuid.uuid4().hex)
                context=await asyncio.to_thread(self._planning_native,'context',request,root/'context')
                parent=next(p for p in context['phases'] if p['type']=='nuclear')
                exported=await asyncio.to_thread(self._planning_native,'export-parent',request,root/'export')
                from gpx_recipe.symmetry.isodistort_discovery import discover_structural
                from gpx_recipe.symmetry.isodistort_forms import IsodistortFormClient
                s.plan_message='Discovering and constructing structural candidates with ISODISTORT…';s.flush()
                def build():
                    def checked_post(url,**kwargs):
                        if event.is_set():raise asyncio.CancelledError()
                        from gpx_recipe.symmetry.isodistort_discovery import fixed_post
                        return fixed_post(url,**kwargs)
                    client=IsodistortFormClient(post=checked_post,timeout_s=60,max_attempts=2,sleep=event.wait)
                    def matcher(cif,vector,options):
                        if event.is_set():raise asyncio.CancelledError()
                        return self._planning_native('structural-match',{**request,'cif_text':cif,'k':vector,'options':options},root/('match-'+uuid.uuid4().hex))
                    def inventory(child):
                        if event.is_set():raise asyncio.CancelledError()
                        child=Path(child).resolve(strict=True)
                        if not child.is_relative_to(root/'catalog') or child.suffix.lower()!='.cif':
                            raise ValueError('The discovered child is outside this planning catalog')
                        self._planning_owned_inputs[str(child)]=hashlib.sha256(child.read_bytes()).hexdigest()
                        return self._planning_native('structural-modes',{**request,'child_cif':str(child)},root/('modes-'+uuid.uuid4().hex))
                    return discover_structural(exported['cif'],parent,k,root/'catalog',matcher=matcher,inventory=inventory,
                        opd_policy='highest-symmetry-per-irrep',maximum=32,client=client)
                catalog=await asyncio.to_thread(build)
                if generation!=self._planning_generation or event.is_set():return
                self._planning_structural_catalog=(catalog,parent,request,root/'catalog')
                coverage_labels={row['id']:row.get('label','') for row in catalog['coverage'].get('candidates',[]) if 'id' in row}
                def label(candidate):
                    match=re.search(r'\)\s+(\d+)\s+([^,]+),\s*basis=',coverage_labels.get(candidate['id'],''))
                    group=(' · '+match[2]+' (No. '+match[1]+')') if match else ''
                    return (candidate['irrep'] or candidate['id'])+group+' · '+candidate['id'][-6:]
                s.plan_structural_candidates=[{'title':label(c),'value':c['id']} for c in catalog['candidates']]
                c=catalog['coverage']
                s.plan_structural_message=f"{c['discovered_count']} order-parameter directions discovered; {c['constructed_count']} constructed in the highest-symmetry-per-irrep tier. Deferred and failed branches remain in the catalog."
                s.plan_message=('Structural candidates ready. Select a child structure to prepare its displacement modes.'
                    if catalog['candidates'] else 'Structural discovery complete. No child structures were constructed for this parent and k.')
            except Exception as exc:
                if generation==self._planning_generation and not event.is_set():s.plan_error=str(exc)
            finally:
                self._planning_finish_job(event)
        self._planning_schedule(discover(),'radar-planning-structural-discovery')

    def prepare_planning_structural_modes(self, **_):
        async def prepare():
            s=self.server.state
            if s.plan_busy:return
            self._planning_edited()
            s.plan_busy=True;s.plan_error='';generation=self._planning_generation
            event=self._planning_begin_calculation()
            try:
                if self._planning_structural_catalog is None:raise ValueError('Discover structural children first')
                catalog,parent,source,root=self._planning_structural_catalog
                candidate=next(c for c in catalog['candidates'] if c['id']==s.plan_structural_candidate)
                request={**source,'parent_setting_sha256':parent['setting_sha256'],
                    'candidate':candidate,'child_cif':str(root/candidate['child']['path']),'k':catalog['k']}
                values={name:0. for name in candidate['modes']}
                s.plan_message='Preparing native displacement modes and checking the zero-distortion pattern…';s.flush()
                prepared=await asyncio.to_thread(self._planning_native,'structural-prepare',
                    {**request,'mode_values':values},root.parent/('prepare-'+uuid.uuid4().hex))
                if generation!=self._planning_generation or event.is_set():return
                self._planning_prepared=prepared;self._planning_prepared_kind='structural'
                self._planning_magnetic_request=request
                self._planning_parameter_rows('structural',values)
                controls=prepared['controls'];s.plan_structural_note=controls['convention']
                s.plan_child_cell_note=controls['space_group']+' · '+', '.join(
                    f'{label} = {value:.6g}{unit}' for label,value,unit in zip(
                        ['a','b','c','α','β','γ'],controls['cell'],[' Å']*3+['°']*3))
                s.plan_structural_message=f"{len(values)} native displacement modes · {len(controls['sites'])} child sites. Zero-distortion pattern verified against the parent."
                if controls['gauge_rank']:
                    s.plan_structural_message+=' Some combinations shift the origin and do not change diffraction intensities.'
                s.plan_message='Structural controls ready. Enter amplitudes and simulate.'
            except Exception as exc:
                if generation==self._planning_generation and not event.is_set():s.plan_error=str(exc)
            finally:self._planning_finish_job(event)
        self._planning_schedule(prepare(),'radar-planning-structural-prepare')

    def open_planning_workspace(self, **_):
        s=self.server.state
        if getattr(s,'workspace_view','monitor')!='planning':
            self._planning_previous_workspace=(getattr(s,'workspace_view','monitor'),getattr(s,'setup_collapsed',False))
        s.workspace_view='planning';s.setup_collapsed=True;s.flush()

    def close_planning_workspace(self, **_):
        s=self.server.state
        s.workspace_view,s.setup_collapsed=self._planning_previous_workspace or ('monitor',False)
        selected=s.workspace_view;collapsed=s.setup_collapsed
        if hasattr(self,'select_workspace_view'):self.select_workspace_view(selected)
        s.setup_collapsed=collapsed
        s.flush()

    def _planning_edited(self, **_):
        if self._planning_restoring:return
        self._planning_generation+=1
        s=self.server.state;s.plan_current=False;s.plan_result_ready=False
        s.plan_save_message=''
        s.plan_can_promote_parent=False
        s.plan_message='Inputs changed. Simulate the current scenario to see its pattern.'
        s.plan_figure={'data':[],'layout':{}};s.plan_qualification='';s.plan_summary=[];s.plan_warnings=[]
        if self._planning_cancel is not None:
            self._planning_cancel.set()
            s.plan_can_cancel=False

    def _planning_name_edited(self, **_):
        if not self._planning_restoring:self.server.state.plan_save_message=''

    def _planning_request(self):
        if self._planning_input_errors:
            raise ValueError('Correct the invalid transition coordinates before simulating')
        s=self.server.state
        if any(s.plan_cell_values) and not all(s.plan_cell_values):
            raise ValueError('Complete all six child lattice parameters, or leave all six blank.')
        request=dict(schema='powgen-planning-scenario/v1',configuration=s.plan_configuration,
            cif=s.plan_cif,mcif=s.plan_mcif or None,comparison_cif=s.plan_comparison_cif or None,
            custom_profile=s.plan_custom_profile or None,profile_bank=int(s.plan_profile_bank),
            sample=dict(mass_g=float(s.plan_mass),radius_cm=float(s.plan_radius),packing_fraction=float(s.plan_packing),
                        beam_width_cm=float(s.plan_beam_width),beam_height_cm=float(s.plan_beam_height)),
            beam={},
            d_min=float(s.plan_d_min),d_max=float(s.plan_d_max),relative_step=float(s.plan_step),
            seed=int(s.plan_seed),moment_multiplier=float(s.plan_moment_multiplier),
            isotopes=self._planning_isotopes(),response_mode=s.plan_response_mode,container=s.plan_container,
            intensity_calibration=s.plan_intensity_calibration if s.plan_response_mode=='current_cycle' else 'vanadium',
            resolution_model=s.plan_resolution_model)
        if s.plan_exposure_basis=='charge':
            request['charge_uah']=[float(t.strip()) for t in str(s.plan_charge).split(',') if t.strip()]
        else:
            request['seconds']=[float(t.strip()) for t in str(s.plan_seconds).split(',') if t.strip()]
            if s.plan_exposure_basis=='current_time':
                request['beam']['average_current_ua']=float(s.plan_current_ua)
            elif s.plan_exposure_basis=='power_time':
                request['beam'].update(power_mw=float(s.plan_power),
                    proton_energy_gev=1.3 if s.plan_response_mode=='current_cycle' else float(s.plan_energy))
            else:raise ValueError('Select charge, time/current or time/power exposure')
        request['additional_phases']=[]
        for index in range(1,4):
            cif=getattr(s,f'plan_phase_{index}_cif')
            percent=getattr(s,f'plan_phase_{index}_percent')
            size=getattr(s,f'plan_phase_{index}_size')
            strain=getattr(s,f'plan_phase_{index}_microstrain')
            if not cif:
                if (str(percent).strip() and float(percent)!=0) or str(size).strip() or str(strain).strip():
                    raise ValueError(f'Upload the CIF for additional phase {index}, or clear its settings')
                continue
            row={'cif':cif,'mass_fraction':float(percent)/100}
            if str(size).strip():row['crystallite_size_um']=float(size)
            if str(strain).strip():row['microstrain']=float(strain)
            request['additional_phases'].append(row)
        if s.plan_response_mode=='historical_estimate' and s.plan_yield_ratio is not None and str(s.plan_yield_ratio).strip():
            request['beam']['neutron_yield_ratio']=float(s.plan_yield_ratio)
        return validate_scenario(request)

    def run_planning_simulation(self, **_):
        if self.server.state.plan_busy:return
        self._planning_schedule(self._run_planning_simulation(),'radar-powgen-planning')

    async def _run_planning_simulation(self):
        if self.server.state.plan_busy:return
        s=self.server.state;s.plan_busy=True;s.plan_error='';s.plan_current=False;s.plan_result_ready=False
        s.plan_message='Preparing the selected structure and instrument calculation.';s.flush()
        generation=self._planning_generation
        event=self._planning_begin_calculation()
        try:
            request=self._planning_request()
            self._planning_validate_transition()
            root=self._planning_root()
            directory=root/uuid.uuid4().hex;directory.mkdir()
            if self._planning_prepared is not None:
                if self._planning_prepared_kind=='structural':
                    model_request={**self._planning_magnetic_request,'mode_values':json.loads(s.plan_structural_parameters),
                                   'cell':json.loads(s.plan_structural_cell) if str(s.plan_structural_cell).strip() else None}
                    operation='structural-prepare'
                else:
                    model_request={**self._planning_magnetic_request,'parameters':json.loads(s.plan_parameters)}
                    operation='prepare'
                prepared=await asyncio.to_thread(self._planning_native,operation,model_request,directory/'prepared-model')
                if self._planning_prepared_kind=='structural':
                    # The resolved native cell alone cannot distinguish an
                    # inherited lattice from an explicit override. Retain the
                    # submitted intent in the sealed simulation receipt.
                    prepared={**prepared,'requested_cell':model_request['cell']}
                parent_receipt=self._planning_parent_receipt()
                request['prepared_'+self._planning_prepared_kind]={**prepared,'parent_cif_sha256':parent_receipt['inputs']['cif']['sha256'],
                    'profile_sha256':parent_receipt['profile_sha256']}
            elif self._planning_parent_state is not None:
                parent_receipt=self._planning_parent_receipt()
                request['prepared_structural']={**self._planning_parent_state,
                    'parent_cif_sha256':parent_receipt['inputs']['cif']['sha256'],
                    'profile_sha256':parent_receipt['profile_sha256']}
            output=directory/'simulation'
            receipt=await asyncio.to_thread(self._planning_execute,request,directory,event)
            if generation!=self._planning_generation or event.is_set():return
            self._planning_output=output;s.plan_current=True;s.plan_result_ready=True
            self._planning_receipt_sha256=hashlib.sha256((output/'result.json').read_bytes()).hexdigest()
            if not request.get('mcif') and not request.get('prepared_magnetic') and not request.get('prepared_structural'):
                self._planning_parent_output=output
                self._planning_parent_receipt_sha256=self._planning_receipt_sha256
                self._planning_parent_state=None
            s.plan_can_promote_parent=bool(receipt.get('structural_state') and not receipt.get('magnetic_state'))
            s.plan_qualification={'profile_only':'Peak-shape preview · counting-time response unavailable',
                'historical_estimate':'Historical response estimate · experimental accuracy unqualified',
                'current_cycle':'2026B measured response · conditional prediction, validation pending'}[receipt['mode']]
            s.plan_warnings=receipt['warnings']
            s.plan_summary=[{'label':'Configuration','value':receipt['configuration']},
                            {'label':'Intensity','value':receipt['intensity_unit']},
                            {'label':'Independent experimental acceptance','value':'Pending'},
                            {'label':'Structure','value':' + '.join(p['spacegroup'] for p in receipt['phases'])}]
            response=receipt.get('response',{})
            sample=response.get('sample_assumptions',{})
            geometry=response.get('geometry',{})
            references=response.get('measured_reference_runs',{})
            if sample and geometry:
                s.plan_summary.extend([
                    {'label':'Assumed specimen','value':f"{sample['mass_g']:g} g · radius {sample['radius_cm']:g} cm · packing {sample['packing_fraction']:g}"},
                    {'label':'Calculated fill height','value':f"{geometry['fill_height_cm']:.4g} cm from the declared mass, density and packing"}])
            if references:
                s.plan_summary.append({'label':'Cycle reference runs','value':'; '.join(f'{name}: PG3_{run}' for name,run in references.items())})
            if receipt.get('mixture'):
                s.plan_summary.append({'label':'Declared crystalline mass fractions',
                    'value':'; '.join(f"{p['name']}: {100*p['mass_fraction']:g}%" for p in receipt['phases'] if p['type']=='nuclear')})
            self._planning_isotope_rows(receipt)
            s.plan_look_options=[{'title':exposure_label(look),'value':i}
                                 for i,look in enumerate(receipt['looks'])] or [{'title':'Expected profile','value':0}]
            s.plan_look=0;s.plan_message='Simulation ready. Select an exposure to compare its counting noise.'
            self._planning_plot()
        except Exception as exc:
            if generation==self._planning_generation:
                s.plan_error=str(exc);s.plan_message='Simulation stopped; review the reported input or calibration issue.'
        finally:
            self._planning_finish_job(event)

    def _planning_finish_job(self, event):
        if self._planning_cancel is not event:return
        self._planning_cancel=None
        s=self.server.state;s.plan_busy=False;s.plan_can_cancel=False
        if event.is_set():s.plan_message='Calculation cancelled. Your inputs are preserved.'
        s.flush()

    def cancel_planning_simulation(self, **_):
        if self._planning_cancel is None:return
        self._planning_generation+=1
        self._planning_cancel.set()
        self.server.state.plan_can_cancel=False
        self.server.state.plan_message='Stopping the calculation. An active external-service request may take up to 60 seconds to return.';self.server.state.flush()

    def _planning_plot(self, **_):
        s=self.server.state
        if not s.plan_current or self._planning_output is None:return
        root=self._planning_output
        def load(relative):
            path=(root/relative).resolve()
            if not path.is_relative_to(root.resolve()) or hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['artifacts'].get(relative):
                raise ValueError('A simulation artifact changed; run the current scenario again')
            with np.load(path,allow_pickle=False) as data:return {k:data[k].copy() for k in data.files}
        try:
            receipt=self._planning_receipt(root,self._planning_receipt_sha256)
            arrays=load('curves.npz');d=arrays['d']
            raw=s.plan_pattern_view=='counts' and 'raw_rate' in arrays and bool(receipt['looks'])
            charge=receipt['looks'][int(s.plan_look)]['cumulative_charge_uah'] if raw else 1.
            if raw:
                arrays={**arrays,'expected':arrays['raw_rate']*charge,
                    'nuclear':arrays['raw_nuclear_rate']*charge,'magnetic':arrays['raw_magnetic_rate']*charge}
            traces=[dict(x=d.tolist(),y=arrays['expected'].tolist(),name='Expected total',mode='lines',line={'color':'#15543c'})]
            uncertainty_view=bool(receipt['looks']) and s.plan_plot_mode=='uncertainty'
            s.plan_error_note='Shape preview only: this configuration has no calibrated count prediction or statistical error bars.'
            if receipt['looks']:
                look=receipt['looks'][int(s.plan_look)];data=load('reduced/'+look['sufficient_statistics'])
                y=data['counts'] if raw else data['y']
                if raw:
                    counting=np.sqrt(data.get('expected_counts',arrays['expected']))
                    predictive=np.sqrt(data['raw_predictive_variance'])
                else:
                    counting=np.sqrt(data['counting_variance'] if 'counting_variance' in data else
                        np.maximum(0.,data['diagonal_variance']-data['shared_reference_variance']))
                    predictive=data['sigma']
                error=counting if s.plan_error_model=='counting' else predictive
                traces.insert(0,dict(x=d.tolist(),y=y.tolist(),name='Synthetic · '+exposure_label(look),
                    mode='markers',marker={'size':3,'color':'#52779d'},
                    customdata=error.tolist(),hovertemplate='d = %{x:.5g} Å<br>Intensity = %{y:.5g}<br>±1σ = %{customdata:.4g}<extra>%{fullData.name}</extra>',
                    error_y={'type':'data','array':error.tolist(),'visible':True,'thickness':.7,'width':2}))
                s.plan_error_note=('Bars show ±1σ '+('sample counting noise only.' if s.plan_error_model=='counting' else
                    'counting noise plus finite calibration/reference uncertainty (the exported error convention).')+
                    ' These are per-bin standard deviations, not a full uncertainty on the physical model. Reference errors can be correlated between bins and exposures. Small bars may need zooming; use Error-bar size to inspect σ directly.')
                if uncertainty_view:
                    traces=[dict(x=d.tolist(),y=values.tolist(),name=label,mode='lines',line={'color':color,'width':2})
                        for values,label,color in [(counting,'Sample counting σ','#52779d'),
                            (predictive,'Counting + references σ','#bd5b37')]]
                else:
                    traces.extend([
                        dict(x=d.tolist(),y=(arrays['expected']-error).tolist(),name='Expected −1σ',
                            mode='lines',line={'width':0},showlegend=False,hoverinfo='skip',legendgroup='sigma'),
                        dict(x=d.tolist(),y=(arrays['expected']+error).tolist(),name='Expected ±1σ',
                            mode='lines',line={'width':0},fill='tonexty',fillcolor='rgba(82,119,157,0.13)',
                            hoverinfo='skip',legendgroup='sigma')])
            if not uncertainty_view:
                for key,color in [('nuclear','#457b9d'),('magnetic','#bd5b37'),('comparison_nuclear','#8c63a8')]:
                    if key in arrays and np.any(arrays[key]):traces.append(dict(x=d.tolist(),y=arrays[key].tolist(),name=key.replace('_',' ').title(),mode='lines',line={'color':color,'width':1},visible='legendonly' if key!='comparison_nuclear' else True))
            if not uncertainty_view and 'comparison-curves.npz' in receipt['artifacts']:
                comparison=load('comparison-curves.npz')
                traces.append(dict(x=d.tolist(),y=(comparison['raw_rate']*charge if raw else comparison['expected']).tolist(),name='Comparison structure',mode='lines',line={'color':'#8c63a8'}))
            unit='Accepted neutron counts / bin' if raw else receipt['intensity_unit']
            s.plan_figure={'data':traces,'layout':{'height':500,'margin':{'l':75,'r':15,'t':45,'b':110},
                'title':{'text':s.plan_qualification.replace(' · ','<br>'),'font':{'size':12}},
                'xaxis':{'title':{'text':'d spacing (Å)'}},'yaxis':{'title':{'text':('Standard deviation σ · ' if uncertainty_view else '')+unit}},
                'legend':{'orientation':'h','y':-.2,'yanchor':'top'},'uirevision':str(root)+s.plan_pattern_view+s.plan_plot_mode,'template':'plotly_white'}}
        except Exception as exc:
            s.plan_error=str(exc);s.plan_current=False;s.plan_result_ready=False;s.plan_figure={'data':[],'layout':{}}

    def export_planning_simulation(self, **_):
        if not self.server.state.plan_current or self._planning_output is None:return
        self.export_planning_plan()
