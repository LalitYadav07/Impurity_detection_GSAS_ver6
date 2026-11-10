"""Reviewed planner-input drafts; synthetic observations never enter analysis.

The complete data-only plan accompanies the draft. Native parent/profile
inspection remains authoritative, and unsupported structure assumptions are
kept visible instead of being replaced with the original warm/natural CIF.
"""
from __future__ import annotations

import copy
from datetime import datetime,timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import uuid
import zipfile

import numpy as np

from gpx_recipe.symmetry.contracts import digest
from . import planning_bundle as bundle
from .battery_preparation import capture_file
from .diffraction_coordinates import TofCalibration
from .uploads import inspect_cif_upload,store_browser_upload

SCHEMA='radar-planner-input-handoff/v1'
PREPARATION_FIELDS=('prep_name','prep_parent_path','prep_profile_path','prep_profile_bank',
    'prep_grid_kind','prep_lower','prep_upper','prep_bins','prep_bank_label','prep_reference','prep_kind',
    'prep_selected_k','prep_sites','prep_site_rows','prep_custom_k','prep_opd_policy','prep_construct_limit')
SINGLE_FIELDS=('radiation','instrument_mode','main_cif_source','main_cif_path','instrument_source',
    'instrument_path','fit_start','fit_end','bank','input_source','data_path','history_data_id','ipts','run_number')
LIVE_FIELDS=('live_beamline','live_radiation','live_instrument_mode','live_configuration_source',
    'powgen_main_cif_source','powgen_main_cif_path','powgen_instrument_source','powgen_instrument_path',
    'live_fit_start','live_fit_end','powgen_bank_scope','powgen_processing_banks','powgen_wavelength',
    'powgen_ipts','powgen_processing_sample_ids','powgen_configuration_dataset_id')
CONTEXT_FIELDS=('plan_transition','plan_k','plan_sites','plan_site_rows','plan_form_factors','plan_site_policy',
    'plan_candidate','plan_parameters','plan_structural_candidate','plan_structural_parameters',
    'plan_structural_cell','plan_promoted_parent')


def _sha(raw):return hashlib.sha256(raw).hexdigest()


def _bank(value):
    if isinstance(value,bool):raise ValueError('Enter the physical source bank as a positive integer')
    try:number=int(value)
    except (ValueError,TypeError):raise ValueError('Enter the physical source bank as a positive integer') from None
    if str(number)!=str(value).strip() or not 1<=number<=10000:
        raise ValueError('Enter the physical source bank as a positive integer')
    return number


def _require_analysis_profile(path,*,source_bank,profile_bank):
    """Reject bank selections the analysis input contract cannot preserve.

    Preparation sends its explicit instrument bank to the native worker.
    Analysis instead lets the native reader select from the data bank, and
    cannot carry a separate instrument-bank selection. Keep every profile byte
    intact rather than extracting only numeric fields and losing native tables.
    """
    path=Path(path);text=path.read_text(encoding='utf-8-sig')
    message=('Analysis and Live handoffs require a standalone selected-bank profile. '
        'Export the selected instrument bank as a single-bank GSAS-II profile, '
        'then select it in the planner before handing off these inputs.')
    if path.suffix.lower()=='.instprm':
        headers=re.findall(r'(?mi)^#\s*Bank\s+(\d+)\b',text)
        fields=re.findall(r'(?m)(?:^|;)\s*Bank\s*:',text)
        if len(headers)>1 or len(fields)>1:raise ValueError(message)
        if headers and (int(headers[0])!=profile_bank or source_bank!=profile_bank):
            raise ValueError(message+' Its explicit bank header cannot be remapped by the analysis input contract.')
    else:
        counts=[line[12:].strip() for line in text.splitlines() if line[:12]=='INS   BANK  ']
        # Without an explicit selection, the native legacy loader accepts only
        # one bank and reads bank 1, independently of the physical data bank.
        if profile_bank!=1 or any(value!='1' for value in counts):raise ValueError(message)
        banks={line[3:6].strip() for line in text.splitlines()
            if line.startswith('INS') and line[3:6].strip().isdigit()}
        if banks-{'1'}:raise ValueError(message)


def _snapshot(app,target):
    s=app.server.state
    fields=PREPARATION_FIELDS if target=='preparation' else LIVE_FIELDS if target=='live' else SINGLE_FIELDS
    values={key:copy.deepcopy(getattr(s,key,None)) for key in fields}
    values['workflow_mode']=getattr(s,'workflow_mode','single')
    if target=='preparation':
        values['banks']={name:dict(instrument=str(row['instrument']),grid=copy.deepcopy(row['grid']))
            for name,row in getattr(app,'_prep_banks',{}).items()}
        values['revision']=getattr(app,'_prep_input_revision',0)
    return values


def _unsupported(app,request,chemistry):
    from gpx_recipe.planning.isotopes import MONOISOTOPIC_NATURAL
    reasons=[];s=app.server.state
    for element,choice in request['isotopes'].items():
        # A 100% isotope selection is native-supported in the planner, but
        # unchanged CIF ingestion has no field applying General.Isotope.
        symbol=re.match(r'[A-Z][a-z]?',element)
        symbol=symbol[0] if symbol else element
        if symbol not in chemistry:
            reasons.append('Isotope '+element+': the corresponding atom type is absent from the uploaded CIF; inspect the native structure before handoff.')
            continue
        inherent={'D':'2','T':'3'}.get(symbol)
        natural=(isinstance(choice,str) and (choice=='Nat. Abund.' or
            MONOISOTOPIC_NATURAL.get(symbol)==choice.strip() or inherent==choice.strip()))
        if not natural:
            reasons.append('Isotope '+element+': '+('atom-fraction mixtures' if isinstance(choice,dict) else
                'the selected single-isotope override')+' requires a native structure export; the original CIF does not encode this planner override.')
    if request.get('additional_phases'):
        reasons.append('The selected multiphase composition needs a reviewed downstream phase model; copying only the primary CIF would lose phases or mass fractions.')
    if getattr(app,'_planning_parent_state',None) is not None or getattr(s,'plan_promoted_parent',False):
        reasons.append('The promoted distorted parent needs a qualified native CIF export before this handoff; copying the original warm parent would change the selected parent.')
    if any(request.get(key) is not None for key in ('crystallite_size_um','microstrain','march_dollase','texture_axis')):
        reasons.append('Planner sample broadening or texture settings need a reviewed downstream model and cannot be encoded by copying the original CIF.')
    return reasons


class PlanningHandoffMixin:
    def _init_planning_handoff(self):
        self._planning_pending_handoff=None
        self._planning_applied_handoffs={}
        self._planning_preparation_intent=None
        s=self.server.state
        for key,value in dict(plan_handoff_open=False,plan_handoff_target='',plan_handoff_rows=[],
                plan_handoff_warnings=[],plan_handoff_error='',plan_handoff_can_apply=False,
                plan_handoff_message='',plan_handoff_source_bank='',plan_handoff_source_bank_required=False,
                plan_handoff_parent_choice_required=False,plan_handoff_use_original_parent=False,
                plan_handoff_intent_available=False).items():setattr(s,key,value)
        s.change('plan_handoff_source_bank','plan_handoff_use_original_parent')(self._planning_handoff_bank_changed)

    def _planning_handoff_identity(self):
        s=self.server.state
        return digest(dict(request=self._planning_request(),
            context={key:copy.deepcopy(getattr(s,key,None)) for key in CONTEXT_FIELDS},
            generation=self._planning_generation))

    def _planning_handoff_profile(self,request):
        from gpx_recipe.planning.instrument import selected_profile
        custom=request.get('custom_profile')
        if custom:custom=str(self._planning_input(custom))
        profile=selected_profile(request['configuration'],custom)
        path=Path(profile['profile'])
        if not 0<path.stat().st_size<=1024*1024:raise ValueError('The downstream profile limit is 1 MiB')
        raw=path.read_bytes()
        if _sha(raw)!=profile['sha256']:raise ValueError('The selected instrument profile changed')
        return profile,path,raw

    def _planning_handoff_mapping(self,pending):
        s=self.server.state;profile=pending['profile'];request=pending['request']
        source_bank=_bank(profile['source_bank'] if profile.get('source_bank') is not None else s.plan_handoff_source_bank)
        profile_bank=_bank(request['profile_bank'] if request.get('custom_profile') else profile['profile_bank'])
        calibration=TofCalibration.from_profile(pending['profile_path'],source_bank=source_bank,
            profile_bank=profile_bank,expected_sha256=profile['sha256'])
        d=np.array([request['d_min'],request['d_max']],dtype=float)
        tof=calibration.tof(d)
        if not np.isfinite(tof).all() or not 0<tof[0]<tof[1] or not np.allclose(calibration.spacing(tof),d,rtol=1e-9,atol=1e-10):
            raise ValueError('The selected profile has no valid increasing TOF conversion over the planned d range')
        bins=math.ceil(math.log(request['d_max']/request['d_min'])/request['relative_step'])
        if pending['target']=='preparation':
            from gpx_recipe.preparation.reference import validate_grid
            validate_grid(kind='PNT',lower=float(tof[0]),upper=float(tof[1]),bins=bins,bank=profile_bank)
        return dict(source_bank=source_bank,profile_bank=profile_bank,profile_sha256=profile['sha256'],
            d_range_angstrom=d.tolist(),tof_range_microseconds=tof.tolist(),bins=bins,
            conversion='checksum-verified native profile coefficients; not a copied simulated curve')

    def _planning_handoff_blockers(self,pending):
        reasons=list(pending['unsupported']);s=self.server.state;target=pending['target']
        if pending['requires_parent_choice'] and not s.plan_handoff_use_original_parent:
            reasons.append('Explicitly choose the original nuclear parent as the starting setup, with the transition retained as pending intent. The simulated child, moments and response will not be imported.')
        if getattr(s,'plan_busy',False):reasons.append('Wait for the current planner calculation before handing off its inputs.')
        if target=='preparation' and getattr(s,'prep_busy',False):
            reasons.append('The preparation workspace is busy; wait before replacing its draft inputs.')
        if target=='single' and getattr(s,'busy',False):reasons.append('Wait for the active analysis submission before replacing its setup draft.')
        if target=='single' and getattr(s,'radiation','neutron') not in {'neutron',None,''}:
            reasons.append('This setup contains an X-ray draft. Open a neutron setup explicitly before applying POWGEN inputs; its measured data will not be reinterpreted.')
        if target=='single' and getattr(s,'bank',None):
            match=re.fullmatch(r'(?:bank[ _-]?)?(\d+)',str(s.bank).strip(),re.I)
            planned=pending['profile'].get('source_bank') or s.plan_handoff_source_bank
            if match and planned and int(match[1])!=_bank(planned):
                reasons.append('The selected analysis source bank differs from the planner bank. Select matching real data before applying this profile.')
        if target=='live':
            if getattr(s,'powgen_monitoring',False) or getattr(s,'live_launch_busy',False):
                reasons.append('Stop or finish the active Live launch before changing its draft inputs.')
            if getattr(s,'live_beamline','PG3')!='PG3':reasons.append('Open a POWGEN Live draft before applying a POWGEN plan.')
        try:
            mapping=self._planning_handoff_mapping(pending)
            if target in {'single','live'}:
                _require_analysis_profile(pending['profile_path'],source_bank=mapping['source_bank'],
                    profile_bank=mapping['profile_bank'])
        except ValueError as exc:reasons.append(str(exc))
        return reasons

    def _planning_handoff_bank_changed(self,**_):
        pending=getattr(self,'_planning_pending_handoff',None)
        if pending is None:return
        reasons=self._planning_handoff_blockers(pending)
        self.server.state.plan_handoff_can_apply=not reasons
        self.server.state.plan_handoff_error=' '.join(reasons)
        rows=pending.get('review_rows',[])
        try:
            mapping=self._planning_handoff_mapping(pending)
            rows=[*rows,dict(label='Physical source bank',value=str(mapping['source_bank'])),
                dict(label='Converted native TOF range',value='–'.join(f'{x:.8g}' for x in mapping['tof_range_microseconds'])+' μs')]
        except ValueError:pass
        self.server.state.plan_handoff_rows=rows

    def use_planning_inputs_for_preparation(self,**_):self._planning_offer_handoff('preparation')

    def use_planning_inputs_for_analysis(self,**_):
        self._planning_offer_handoff('live' if getattr(self.server.state,'workflow_mode','single')=='powgen' else 'single')

    def _planning_offer_handoff(self,target):
        s=self.server.state
        try:
            if s.plan_busy:raise ValueError('Wait for the planner calculation to finish before reviewing a handoff')
            request=self._planning_request();source=self._planning_input(request['cif'])
            if source.suffix.lower()!='.cif':raise ValueError('The target requires a nuclear .cif input')
            raw=source.read_bytes();source_metadata=inspect_cif_upload(raw,source.name)
            if not 0<len(raw)<=10*1024*1024:raise ValueError('The downstream nuclear CIF limit is 10 MiB')
            profile,profile_path,profile_raw=self._planning_handoff_profile(request)
            document,files=self._planning_build_bundle()
            if _sha(files[document['settings']['plan_cif']])!=_sha(raw):raise ValueError('The nuclear CIF changed while capturing the plan')
            root=self._planning_root()/'handoffs'/uuid.uuid4().hex;root.mkdir(parents=True)
            source_copy=root/source.name;capture_file(source,source_copy,maximum=10*1024*1024,expected_sha256=_sha(raw))
            profile_copy=root/('profile'+profile_path.suffix.lower())
            capture_file(profile_path,profile_copy,maximum=1024*1024,expected_sha256=profile['sha256'])
            pending=dict(target=target,request=request,profile=profile,profile_path=profile_copy,
                source_path=source_copy,source_sha256=_sha(raw),source_name=source.name,profile_name=profile_path.name,
                identity=self._planning_handoff_identity(),draft=_snapshot(self,target),root=root,
                document=document,files=files,unsupported=_unsupported(self,request,source_metadata['elements']),
                requires_parent_choice=getattr(s,'plan_transition','none')!='none',
                transition_context={key:copy.deepcopy(getattr(s,key,None)) for key in CONTEXT_FIELDS})
            self._planning_pending_handoff=pending
            s.plan_handoff_target={'single':'Analysis setup','live':'Live setup','preparation':'Experiment preparation'}[target]
            s.plan_handoff_source_bank_required=profile.get('source_bank') is None
            s.plan_handoff_parent_choice_required=pending['requires_parent_choice']
            s.plan_handoff_use_original_parent=False
            s.plan_handoff_source_bank=profile.get('source_bank') or ''
            pending['review_rows']=[dict(label='Nuclear CIF',value=source.name),dict(label='Instrument profile',value=profile_path.name),
                dict(label='Profile block',value=str(request['profile_bank'] if request.get('custom_profile') else profile['profile_bank'])),
                dict(label='Planned d range',value=f"{request['d_min']:g}–{request['d_max']:g} Å"),
                dict(label='Draft replacement',value='Only the reviewed structure, profile and range settings. Existing measured data, active results and saved libraries stay in place.')]
            s.plan_handoff_rows=pending['review_rows']
            s.plan_handoff_warnings=[
                'Review the destination draft before running native parent/profile inspection or analysis. No job starts from this handoff.',
                'The planner resolution convolution and sample/exposure assumptions are retained in the intent bundle. Downstream native calculations use their own checked profile model.',
                'No simulated counts or curves become measured observations. Verify the real instrument, source bank and experiment selection.']
            if target=='preparation':s.plan_handoff_warnings.append('Preparation creates its own native grid across the converted range; the original relative step and full plan remain in the intent bundle.')
            self._planning_write_handoff_intent(pending,mapping=None)
            s.plan_handoff_intent_available=True;s.plan_handoff_message='Review the proposed input draft. The complete plan is available as a data-only intent bundle.'
            self._planning_handoff_bank_changed();s.plan_handoff_open=True;s.plan_error=''
        except (ValueError,OSError,KeyError) as exc:
            self._planning_pending_handoff=None;s.plan_handoff_can_apply=False
            s.plan_handoff_error=str(exc);s.plan_handoff_intent_available=False
        s.flush()

    def _planning_write_handoff_intent(self,pending,*,mapping):
        document=pending['document'];plan=bundle.archive(document,pending['files'])
        if len(plan)+pending['source_path'].stat().st_size+pending['profile_path'].stat().st_size>bundle.MAX_BYTES:
            raise ValueError('The complete handoff exceeds the bounded portable plan limit')
        intent=dict(schema=SCHEMA,target=pending['target'],created_utc=datetime.now(timezone.utc).isoformat(),
            plan='plan.zip',plan_sha256=_sha(plan),source_plan_identity=pending['identity'],
            source_implementation=document['implementation'],mapping=mapping,
            nuclear_cif=dict(path='inputs/parent.cif',sha256=pending['source_sha256']),
            profile=dict(path='inputs/'+pending['profile_path'].name,sha256=pending['profile']['sha256'],
                configuration=pending['profile']['id'],original_name=pending['profile_name']),
            unsupported_assumptions=pending['unsupported'],requires_native_reinspection=True,
            structure_scope='original_nuclear_parent' if not pending['requires_parent_choice'] or self.server.state.plan_handoff_use_original_parent else 'transition_not_applied',
            transition_context=pending['transition_context'],transition_response_imported=False,
            scientific_result=False,measured_observations_included=False)
        output=io.BytesIO()
        with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('intent.json',bundle.json_bytes(intent));archive.writestr('plan.zip',plan)
            archive.writestr('inputs/parent.cif',pending['source_path'].read_bytes())
            archive.writestr('inputs/'+pending['profile_path'].name,pending['profile_path'].read_bytes())
        path=pending['root']/'intent.zip';path.write_bytes(output.getvalue());pending['intent_path']=path
        pending['intent_sha256']=_sha(output.getvalue())

    def export_planning_handoff(self,**_):
        pending=getattr(self,'_planning_pending_handoff',None)
        if pending is None:return
        raw=pending['intent_path'].read_bytes()
        if _sha(raw)!=pending['intent_sha256']:raise ValueError('The captured input intent changed')
        self.download_file('planner-input-intent.zip','application/zip',raw)

    def cancel_planning_handoff(self,**_):
        self._planning_pending_handoff=None
        self.server.state.plan_handoff_open=False
        self.server.state.plan_handoff_can_apply=False
        self.server.state.plan_handoff_intent_available=False
        self.server.state.plan_handoff_message='Input handoff cancelled; the existing draft is unchanged.'
        self.server.state.flush()

    def _planning_register_uploads(self,pending):
        contents={kind:pending['source_path' if kind=='source' else 'profile_path'].read_bytes() for kind in ('source','profile')}
        if (_sha(contents['source'])!=pending['source_sha256'] or _sha(contents['profile'])!=pending['profile']['sha256']):
            raise ValueError('The reviewed handoff input changed')
        return {kind:store_browser_upload(raw,pending['source_name' if kind=='source' else 'profile_name'])
            for kind,raw in contents.items()}

    def _planning_install_upload(self,saved,state_key,widget=None):
        if widget is not None:
            widget.current_path=saved
            for attr,value in (('display_model',saved.name),('error_model',''),('client_model',None)):
                if getattr(widget,attr,None):setattr(self.server.state,getattr(widget,attr),value)
        setattr(self.server.state,state_key,str(saved))
        return saved

    def confirm_planning_handoff(self,**_):
        pending=getattr(self,'_planning_pending_handoff',None);s=self.server.state
        if pending is None:return
        try:
            if self._planning_handoff_identity()!=pending['identity']:raise ValueError('The planner changed after review. Open a new handoff review.')
            if _snapshot(self,pending['target'])!=pending['draft']:raise ValueError('The destination draft changed after review. Open a new handoff review.')
            source=self._planning_input(pending['request']['cif'])
            profile,_,_=self._planning_handoff_profile(pending['request'])
            if _sha(source.read_bytes())!=pending['source_sha256'] or profile['sha256']!=pending['profile']['sha256']:
                raise ValueError('The reviewed planner input changed. Open a new handoff review.')
            reasons=self._planning_handoff_blockers(pending)
            if reasons:raise ValueError(' '.join(reasons))
            mapping=self._planning_handoff_mapping(pending);target=pending['target']
            if target=='preparation':
                parent_widget=getattr(self,'_prep_parent_upload',None);profile_widget=getattr(self,'_prep_profile_upload',None)
                if parent_widget is None or profile_widget is None:raise ValueError('Open the preparation input panel before applying this draft.')
            self._planning_write_handoff_intent(pending,mapping=mapping)
            if target=='preparation':self._planning_apply_preparation(pending,mapping,parent_widget,profile_widget)
            else:self._planning_apply_analysis(pending,mapping)
            self._planning_applied_handoffs[target]=dict(intent_path=str(pending['intent_path']),sha256=pending['intent_sha256'],mapping=mapping)
            s.plan_handoff_open=False;s.plan_handoff_error='';s.plan_handoff_can_apply=False
            s.plan_handoff_message='Input draft applied. Review the native structure/profile inspection and select real data before analysis.'
        except (ValueError,OSError,KeyError) as exc:s.plan_handoff_error=str(exc);s.plan_handoff_can_apply=False
        s.flush()

    def _planning_apply_preparation(self,pending,mapping,parent_widget,profile_widget):
        s=self.server.state
        store=self._preparation_store();destination=store.root/'uploads'/(uuid.uuid4().hex+pending['profile_path'].suffix)
        destination.parent.mkdir(parents=True,exist_ok=True)
        capture_file(pending['profile_path'],destination,maximum=1024*1024,expected_sha256=mapping['profile_sha256'])
        uploads=self._planning_register_uploads(pending)
        self._prep_profile_handoff=dict(path=str(uploads['profile']),sha256=mapping['profile_sha256'],kind='PNT')
        self._planning_install_upload(uploads['source'],'prep_parent_path',parent_widget)
        self._planning_install_upload(uploads['profile'],'prep_profile_path',profile_widget)
        s.prep_name=(str(s.plan_name).strip()+' · preparation')[:160]
        s.prep_profile_bank=str(mapping['profile_bank']);s.prep_grid_kind='PNT'
        s.prep_lower=format(mapping['tof_range_microseconds'][0],'.17g');s.prep_upper=format(mapping['tof_range_microseconds'][1],'.17g')
        s.prep_bins=str(mapping['bins']);label='POWGEN source bank '+str(mapping['source_bank'])
        grid=dict(kind='PNT',lower=mapping['tof_range_microseconds'][0],upper=mapping['tof_range_microseconds'][1],
            bins=mapping['bins'],bank=mapping['profile_bank'])
        self._prep_banks={label:dict(instrument=destination,grid=grid)}
        self._prep_bank_names={label:pending['profile_name']}
        s.prep_bank_rows=[dict(label=label,profile=pending['profile_name'],bank=grid['bank'],kind='PNT',range=f"{grid['lower']:g}–{grid['upper']:g}")]
        s.prep_bank_label='';s.prep_error='';self._preparation_inputs_changed()
        context=pending['transition_context']
        s.prep_kind='magnetic' if context['plan_transition'] in {'magnetic','mcif'} else 'structural'
        s.prep_custom_k=str(context['plan_k']);s.prep_tab='new'
        self._planning_preparation_intent=dict(source_sha256=pending['source_sha256'],
            profile_sha256=mapping['profile_sha256'],profile_bank=mapping['profile_bank'],
            context=copy.deepcopy(context),intent_path=str(pending['intent_path']),intent_sha256=pending['intent_sha256'])
        s.prep_message='Planner input draft ready. Inspect the nuclear parent and exact profile before generating or reusing calculations.'
        self.open_preparation_workspace()

    def apply_planning_preparation_intent(self,context):
        """After native inspection, restore only matched editable k/site inputs.

        Call from _prep_load_reference after its native parent/site controls are
        initialized. No candidate, simulated moment or response is installed.
        """
        pending=getattr(self,'_planning_preparation_intent',None)
        if pending is None:return False
        from .preparation_discovery import k_vector
        from gpx_recipe.symmetry.chemistry import validate_sites
        try:
            if context.get('source_sha256')!=pending['source_sha256']:
                raise ValueError('Planner intent belongs to a different inspected nuclear parent')
            matching=[row for row in context['banks'] if
                row['grid']['bank']==pending['profile_bank'] and
                _sha(Path(row['instrument']).read_bytes())==pending['profile_sha256']]
            if not matching:raise ValueError('Planner intent belongs to a different inspected instrument profile')
            intent=pending['context']
            if intent['plan_transition'] in {'mcif','comparison'}:
                self.server.state.prep_message='Original parent and profile inspected. The supplied transition file remains in the planner intent; review its native transition coordinates separately before choosing candidates.'
                return True
            vector=k_vector([part.strip() for part in intent['plan_k'].split(',')]);token=','.join(vector)
            labels=intent['plan_sites'] or []
            forms={row['label']:row['form_factor'] for row in intent.get('plan_site_rows',[]) if row.get('form_factor')}
            forms.update(json.loads(intent['plan_form_factors'] or '{}'))
            sites=[]
            for label in labels:
                if label not in self._prep_site_inputs or not forms.get(label):
                    raise ValueError('Review the planner magnetic sites and ionic form factors against the inspected parent')
                sites.append(dict(label=label,form_factor=forms[label]))
            validate_sites(context['parent']['atoms'],sites)
            s=self.server.state
            if token not in {row['value'] for row in s.prep_k_options}:
                s.prep_k_options=[*s.prep_k_options,dict(value=token,title='('+', '.join(vector)+') · planner intent')]
            s.prep_selected_k=[token];s.prep_custom_k=''
            for site in sites:self._prep_site_inputs[site['label']]['form_factor']=site['form_factor']
            if sites:s.prep_sites=labels;self._preparation_site_selection_changed()
            self._preparation_scope_changed()
            s.prep_message='Planner k and matched site inputs are ready for review. Discover native candidates explicitly; simulated moments and responses have not been imported.'
            return True
        except (ValueError,KeyError,TypeError) as exc:
            self.server.state.prep_error=str(exc);return False

    def _planning_apply_analysis(self,pending,mapping):
        s=self.server.state;live=pending['target']=='live'
        uploads=self._planning_register_uploads(pending)
        self._planning_install_upload(uploads['source'],'powgen_main_cif_path' if live else 'main_cif_path',
            getattr(self,'_powgen_main_cif_upload' if live else '_main_cif_upload',None))
        self._planning_install_upload(uploads['profile'],'powgen_instrument_path' if live else 'instrument_path',
            getattr(self,'_powgen_instrument_upload' if live else '_instrument_upload',None))
        if live:
            s.powgen_main_cif_source='computer';s.powgen_instrument_source='computer'
            s.live_configuration_source='inline';s.live_radiation='neutron';s.live_instrument_mode='tof'
            s.live_fit_start,s.live_fit_end=mapping['tof_range_microseconds']
            s.powgen_bank_scope='selected';s.powgen_processing_banks=[mapping['source_bank']]
            if pending['profile'].get('wavelength'):s.powgen_wavelength=str(pending['profile']['wavelength'])
        else:
            s.main_cif_source='upload';s.instrument_source='upload';s.radiation='neutron';s.instrument_mode='tof'
            s.fit_start,s.fit_end=mapping['tof_range_microseconds']
            if not getattr(s,'bank',None):s.bank=str(mapping['source_bank'])
        if hasattr(self,'select_primary_workspace'):self.select_primary_workspace('analysis')
        else:self.close_planning_workspace()
        s.setup_collapsed=False
        s.notice='Planner input draft ready. Check the real data, physical bank and native structure/profile interpretation before submitting.'
        # The detailed review section can be below the fold. Keep the next
        # action visible on the analysis landing page as well.
        s.workspace_notice=s.notice
        s.workspace_notice_kind='info'
