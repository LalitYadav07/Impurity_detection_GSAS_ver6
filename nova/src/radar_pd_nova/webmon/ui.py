"""WebMON workspace in the normal RADAR-PD experiment tabs.

The local replay backend is explicitly opt-in. It does not impersonate Galaxy
or submit processing jobs. Production source browsing works after the usual
monitor start; scientific jobs require a sealed reviewed context manifest.
"""
import asyncio,hashlib,json,os,shutil,time,uuid,logging
from pathlib import Path
from trame.widgets import html,plotly,client
from trame.widgets import vuetify3 as v
from .session import WebmonSession,digest
from . import plots
from .contracts import verify_result,verify_posterior,validate_request
from .processes import stop_process,wait_process,spawn_process
from .viewdata import sampler_summary, full_posterior, moment_rows, STATUS
from datetime import datetime

def request_identity(request):
    # Transient cache/download paths are not scientific identity. The request
    # already binds each input by checksum; use stable packet member names.
    from gpx_recipe.live.packet import INPUTS,canonical
    portable={key:(INPUTS[key] if key in INPUTS and value else value) for key,value in request.items()}
    return hashlib.sha256(canonical(portable)).hexdigest()


class WebmonMixin:
    def _init_webmon(self):
        s=self.server.state;manifest=os.environ.get('RADAR_PD_WEBMON_CONTEXT','')
        s.update(dict(webmon_checkpoints=[],webmon_checkpoint='',webmon_checkpoint_axis='d',webmon_checkpoint_summary='',webmon_width_rows=[],webmon_polished_jobs=[],webmon_polished_job=''))
        production=bool(getattr(getattr(self,'service',None),'galaxy_url',None) and getattr(self.service,'history_id',None))
        self._webmon_configured=bool(manifest) or production;self._webmon_config_error=''
        s.webmon_calculation_status='Not started'
        s.webmon_mapping_bank='';s.webmon_mapping_factor='';s.webmon_mapping_normalization='';s.webmon_mapping_producer='';s.webmon_mapping_evidence='';s.webmon_mapping_accepted=False;s.webmon_prepare_accepted=False;s.webmon_refine_transfer=False
        s.webmon_published_sources=[];s.webmon_published_source='';s.webmon_source_notice='Select a published GPX; producer bank/export/normalization mapping and native model preparation are checked separately.'
        self._webmon_stream=None;self._webmon_queue=None
        s.webmon_stream_active=False;s.webmon_stream_message='Automatic updates are off.'
        try:
            if not manifest and production:
                # Private per-history work roots; ordinary setup needs no
                # prebuilt science manifest just to browse the live source.
                history_hash=hashlib.sha256(str(self.service.history_id).encode()).hexdigest()[:24]
                base=Path(getattr(self.service,'output_root','/tmp'))/'live-snapshots'/history_hash
                manifest=dict(facility_root=os.environ.get('RADAR_PD_SNS_ROOT','/SNS'),archive=str(base/'snapshots'),jobs=str(base/'jobs'),checkpoint_registry=str(base/'checkpoints.json'),local_replay=False)
            self._webmon=WebmonSession(manifest) if manifest else None
        except (OSError,ValueError,KeyError) as exc:
            self._webmon=None;self._webmon_config_error=f'WebMON configuration unavailable: {exc}'
            logging.getLogger(__name__).error(self._webmon_config_error)
        self._webmon_colors={};self._webmon_pattern=None;self._webmon_payloads={};self._webmon_out=None;self._webmon_plot_binding=None;self._webmon_saved_paths={};self._webmon_request={};self._webmon_last_preview=None;self._webmon_model_paths=set();self._webmon_process=None;self._webmon_cancelled=False;self._webmon_task=None;self._webmon_poll_task=None;self._webmon_binding=None;self._webmon_start_scope=None
        s.update(dict(webmon_follow_latest=True,webmon_stale=False,webmon_analysis_revision='',webmon_local=bool(self._webmon and self._webmon.config.get('local_replay')),webmon_runs=[],webmon_run='',webmon_target='',webmon_message='Start the experiment monitor, then select a stored or current WebMON scan.',webmon_reference='',webmon_accepted=False,webmon_k='0, 0, 0',webmon_ion='',webmon_lower=.1,webmon_upper=9.,webmon_busy=False,webmon_rows=[],webmon_model='',webmon_models=[],webmon_details='',webmon_has_result=False,webmon_context_ready=False,webmon_revision='',webmon_latency='',webmon_latest='',webmon_health='No inference yet',webmon_reference_rows=[],webmon_poll_seconds=30))
        s.update(dict(webmon_table_selected=[],webmon_compared=[],webmon_axis='tof',webmon_axes=[{'title':'TOF (µs)','value':'tof'}],webmon_errors=False,webmon_band=True,webmon_ticks=True,webmon_sampler_summary="",webmon_fit_metrics=[],webmon_component=False,webmon_region='full',webmon_filter='All',webmon_search='',webmon_display_rows=[],webmon_moment_rows=[],webmon_pair_a=0,webmon_pair_b=1,webmon_site_options=[],webmon_saved=[],webmon_saved_job='',webmon_checked='Not checked',webmon_connection='Not connected',webmon_count_summary='No candidates analyzed',webmon_validity='No fit assessed',webmon_boundary='Not assessed',webmon_sampling='Not assessed',webmon_fit_caption='',webmon_lower_used=None,webmon_upper_used=None))
        if self._webmon_config_error:s.webmon_message=self._webmon_config_error
    def _webmon_is_local(self):
        return getattr(getattr(self,'_webmon',None),'config',{}).get('local_replay') is True

    def webmon_attach_monitor(self):
        # Called after the normal Live monitor starts; capture is independent of
        # the completion-only reducer. Starting the monitor does not approve fits.
        if not self._webmon or str(self.server.state.live_beamline) != 'PG3':
            return
        self._webmon_start_scope = str(self.server.state.powgen_ipts)
        self.webmon_refresh(quiet=True)
        if self._webmon_poll_task is None or self._webmon_poll_task.done():
            self._webmon_poll_task = asyncio.create_task(self._webmon_poll())

    def _webmon_local_command(self, name):
        if not self._webmon_is_local():
            raise ValueError('Native live jobs require a configured Galaxy transport on NDIP; the local replay command is unavailable')
        command = self._webmon.config.get(name)
        if not command:
            raise ValueError('No local replay command configured for ' + name)
        return command

    def _webmon_ui(self):
        with html.Section(v_show="experiment_result_view === 'webmon' && live_beamline === 'PG3'",classes='radar-result-card wm-workspace'):
            client.Style("""
              .radar-workspace-inner:has(.wm-workspace:not([style*="display: none"])) {max-width:1920px;}
              .wm-workspace {grid-column:1/-1;min-width:0;padding:18px!important;}
              .wm-workspace p {font-size:13px;color:#526374;margin:5px 0 10px;}
              .wm-workspace h2 {font-size:23px!important;margin:0;}
              .wm-workspace h3 {font-size:16px;margin:0 0 8px;color:#22394d;}
              .wm-top,.wm-controls,.wm-stats,.wm-assumptions {display:grid;gap:12px;align-items:start;}
              .wm-top {grid-template-columns:1fr auto;align-items:center;margin-bottom:12px;}
              .wm-controls {grid-template-columns:minmax(240px,2fr) minmax(180px,1fr) auto;}
              .wm-stats {grid-template-columns:repeat(4,minmax(0,1fr));margin:12px 0;}
              .wm-stat {padding:12px;border:1px solid #dce3ea;border-radius:8px;background:#f8fafc;}
              .wm-stat span {display:block;font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:#647589;}
              .wm-stat strong {font-size:14px;display:block;margin-top:5px;}
              .wm-warning {border-left:4px solid #b36c1a;background:#fff8eb;}
              .wm-main {display:grid;grid-template-columns:minmax(0,1fr) 410px;gap:18px;align-items:start;}
              .wm-card {border:1px solid #dce3ea;border-radius:8px;padding:15px;min-width:0;background:white;}
              .wm-toolbar {display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:8px 0;}
              .wm-toolbar .v-input {min-width:125px;flex:1;}
              .wm-assumptions {grid-template-columns:repeat(4,minmax(130px,1fr));}
              .wm-plot {position:relative;min-width:0;height:700px;}
              .wm-smallplot {position:relative;min-width:0;height:320px;overflow:auto;}
              .wm-workspace .v-expansion-panel-title {background:#f5f8fb!important;color:#293e50!important;min-height:42px!important;}
              .wm-workspace .v-data-table {font-size:12px;}
              .wm-workspace .v-btn {text-transform:none;letter-spacing:0;}
              .wm-workspace .v-field {font-size:13px;}
              .wm-workspace details {margin:12px 0;}
              .wm-workspace summary {cursor:pointer;font-weight:600;color:#385269;padding:9px 0;}
              .wm-workspace pre {font-size:11px;overflow:auto;max-height:260px;background:#f6f8fb;padding:12px;}
              @media(max-width:1200px){.wm-main{grid-template-columns:1fr}.wm-stats{grid-template-columns:repeat(2,1fr)}}
              @media(max-width:700px){.wm-controls,.wm-assumptions{grid-template-columns:1fr}.wm-top{grid-template-columns:1fr}.wm-stats{grid-template-columns:1fr}}
            """)
            with html.Div(classes='wm-top'):
                with html.Div():
                    html.H2('In-progress scans')
                    html.P('POWGEN WebMON source · inspect provisional patterns and conditional magnetic fits.')
                v.VChip('Archived-data replay',v_if='webmon_local',size='small',variant='outlined')
            with html.Div(classes='wm-controls'):
                v.VSelect(label='Viewed scan',v_model=('webmon_run',),items=('webmon_runs',),item_title='label',item_value='value',update_modelValue=(self.webmon_select,'[$event]'),density='compact',hide_details=True)
                v.VCheckbox(label='Follow latest scan',v_model=('webmon_follow_latest',),update_modelValue=(self.webmon_follow_changed,'[$event]'),density='compact',hide_details=True)
                v.VBtn('Refresh source',click=self.webmon_refresh,disabled=('!powgen_monitoring',),variant='outlined',size='small')
            html.P('Source: {{ webmon_connection }} · Last checked {{ webmon_checked }} · Latest {{ webmon_latest }} · Viewed {{ webmon_run }} · Analyzed {{ webmon_target || "none" }}')
            v.VAlert(v_if='webmon_target && webmon_target !== webmon_run',text=("'You are browsing a different scan. Analysis remains pinned to run ' + webmon_target",),type='info',variant='tonal',density='compact')
            v.VAlert(v_if='webmon_stale',text='Input revision changed. Displayed results are stale; prepare and review the new inputs before reuse.',type='warning',variant='tonal',density='compact')
            with html.Div(classes='wm-stats'):
                for label,value,extra in [('Calculation','{{ webmon_busy ? "Running" : webmon_calculation_status }}',''),('Scientific use','{{ webmon_validity }}','wm-warning'),('Candidate settings','{{ webmon_count_summary }}',''),('Counting decision','Unavailable · calibration required','wm-warning')]:
                    with html.Div(classes='wm-stat '+extra):html.Span(label);html.Strong(value)
            html.P('{{ webmon_message }}')
            with v.VExpansionPanels(variant='accordion'):
                with v.VExpansionPanel(title='Review nuclear reference & magnetic assumptions'):
                    with v.VExpansionPanelText():
                        with html.Div(v_if='!webmon_local && !webmon_context_ready'):
                            v.VBtn('Load published GPX sources',click=self.webmon_load_published_sources,disabled=('webmon_busy',),variant='outlined')
                            v.VSelect(label='Published source GPX',v_model=('webmon_published_source',),items=('webmon_published_sources',),update_modelValue=(self.webmon_published_source_changed,'[$event]'),density='compact')
                            html.P('{{ webmon_source_notice }}')
                            v.VBtn('Inspect selected GPX in Refinement',click=self.webmon_open_published_source,disabled=('!webmon_published_source || webmon_busy',),variant='text')
                            v.VBtn('Set magnetic models in Refinement',click=self.webmon_open_model_setup,disabled=('!webmon_published_source || webmon_busy',),variant='outlined')
                            html.P('Use the existing Model exploration setup or Luna assistant to inspect the selected parent, select sites/k and discover candidates. Return here after discovery; this does not start the completed-scan campaign.')
                            v.VCheckbox(label='Refit existing scales/background during snapshot preparation',v_model=('webmon_refine_transfer',),density='compact')
                            v.VCheckbox(label='I reviewed the current Model exploration parent, sites, k and candidate set for this live run',v_model=('webmon_prepare_accepted',),density='compact')
                            v.VBtn('Prepare reviewed live models in Galaxy',click=self.webmon_prepare_from_models,disabled=('!webmon_prepare_accepted || !webmon_published_source || webmon_busy',),variant='outlined')
                            with html.Details():
                                html.Summary('Reviewed producer mapping')
                                html.P('Use this only after checking the live producer calibration. The GPX instrument DIFC is not evidence for the live export conversion. Saving records your mapping; native preparation and model review are still required.')
                                v.VTextField(label='Physical bank',v_model=('webmon_mapping_bank',),density='compact')
                                v.VTextField(label='Export TOF/d factor (µs/Å)',v_model=('webmon_mapping_factor',),density='compact')
                                v.VTextField(label='Normalization convention',v_model=('webmon_mapping_normalization',),density='compact')
                                v.VTextField(label='Producer configuration SHA-256',v_model=('webmon_mapping_producer',),density='compact')
                                v.VTextField(label='Where this mapping was verified',v_model=('webmon_mapping_evidence',),density='compact')
                                v.VCheckbox(label='I checked this bank, conversion and normalization against the producer configuration',v_model=('webmon_mapping_accepted',),density='compact')
                                v.VBtn('Save reviewed mapping',click=self.webmon_save_mapping,disabled=('!webmon_published_source || !webmon_mapping_accepted || webmon_busy',),variant='outlined')
                        v.VSelect(label='Starting nuclear checkpoint',v_model=('webmon_checkpoint',),items=('webmon_checkpoints',),update_modelValue=(self.webmon_checkpoint_changed,'[$event]'),disabled=('webmon_busy',),density='compact')
                        html.P('{{ webmon_reference }}')
                        v.VDataTable(headers=("[{title:'Input',key:'label'},{title:'Reviewed value',key:'value'}]",),items=('webmon_reference_rows',),items_per_page=-1,hide_default_footer=True,density='compact')
                        with html.Div(v_if='webmon_checkpoint'):
                            html.H3('Check the original nuclear fit')
                            html.P('{{ webmon_checkpoint_summary }}')
                            v.VSelect(label='Reference plot coordinate',v_model=('webmon_checkpoint_axis',),items=([dict(title='d (Å)',value='d'),dict(title='Q (Å⁻¹)',value='q'),dict(title='TOF (µs)',value='tof')],),update_modelValue=(self.webmon_checkpoint_axis_changed,'[$event]'),density='compact')
                            with html.Div(style='position:relative;height:510px'):self._webmon_checkpoint_plot=plotly.Figure(display_mode_bar=True)
                            v.VDataTable(headers=([dict(title='Peak d (Å)',key='d'),dict(title='Autoreduce width (Å)',key='observed'),dict(title='GPX width (Å)',key='calculated'),dict(title='WebMON width (Å)',key='webmon')],),items=('webmon_width_rows',),items_per_page=-1,hide_default_footer=True,density='compact')
                            html.P('Widths are local half-height estimates, not refined uncertainties. The green curve is the reference run’s WebMON scan, rescaled only for shape comparison. This does not validate normalization or its error bars.')
                            v.VBtn('Polish selected GPX in Refinement',click=self.webmon_open_refinement,disabled=('webmon_busy',),variant='outlined')
                            html.P('Opens the existing Luna recipe workflow for this exact GPX. Review its phases, fit limits, background and explicitly supported variables before applying changes.')
                            v.VBtn('Test bounded sample broadening',click=self.webmon_profile_polish,v_if='webmon_local',disabled=('webmon_busy',),variant='outlined')
                            html.P('Diagnostic single-phase trials with the instrument fixed. A lower Rwp does not establish physical size or strain; inspect residuals and a suitable standard before accepting.',v_if='webmon_local')
                            v.VBtn('Find completed polishes',click=self.webmon_find_polishes,v_if='webmon_local',disabled=('webmon_busy',),variant='text')
                            v.VSelect(label='Completed polish of this checkpoint',v_model=('webmon_polished_job',),items=('webmon_polished_jobs',),v_if='webmon_local && webmon_polished_jobs.length',disabled=('webmon_busy',),density='compact')
                            v.VBtn('Prepare polished GPX for magnetic review',click=self.webmon_prepare_polish,v_if='webmon_local && webmon_polished_jobs.length',disabled=('webmon_busy || !webmon_polished_job',),variant='outlined')
                        v.VCheckbox(label='Use this reviewed nuclear revision for conditional analysis',v_model=('webmon_accepted',),disabled=('!webmon_context_ready || webmon_busy',),density='compact')
                        with html.Div(classes='wm-assumptions'):
                            v.VTextField(label='Propagation vector k',v_model=('webmon_k',),disabled=('webmon_busy',),density='compact')
                            v.VTextField(label='Magnetic ion',v_model=('webmon_ion',),disabled=('webmon_busy',),density='compact')
                            v.VTextField(label='Minimum moment (µB)',type='text',inputmode='decimal',v_model=('webmon_lower',),disabled=('webmon_busy',),density='compact')
                            v.VTextField(label='Maximum moment (µB)',type='text',inputmode='decimal',v_model=('webmon_upper',),disabled=('webmon_busy',),density='compact')
                        html.P('Site inclusion, moment bounds and candidate omissions follow the explicitly reviewed model assumptions; unresolved settings are not exclusions from the data.')
                        v.VBtn('Start magnetic analysis',click=self.webmon_start,disabled=('!webmon_accepted || !webmon_context_ready || webmon_busy',),color='#15543c')
                        v.VBtn('{{ webmon_local ? "Cancel calculation" : "Stop waiting for job" }}',click=self.webmon_cancel,disabled=('!webmon_busy',),variant='text')
                    with html.Div(classes='wm-toolbar'):
                        v.VBtn('Analyze updates for this run',click=self.webmon_enable_stream,disabled=('webmon_busy || webmon_stream_active || !webmon_accepted',),variant='outlined')
                        v.VBtn('Pause automatic analysis',click=self.webmon_pause_stream,disabled=('!webmon_stream_active',),variant='text')
                    html.P('{{ webmon_stream_message }}')
                with v.VExpansionPanel(title='Restore a saved analysis'):
                    with v.VExpansionPanelText():
                        html.P('Reload a completed analysis of this exact scan and reviewed reference without rerunning inference.')
                        v.VSelect(label='Saved analysis',v_model=('webmon_saved_job',),items=('webmon_saved',),item_title='title',item_value='value',density='compact')
                        v.VBtn('Load saved results',click=self.webmon_restore,disabled=('!webmon_saved_job || webmon_busy',),variant='outlined')
            with html.Div(v_show='!webmon_has_result || webmon_run !== webmon_target'):
                html.H3('Scan preview',classes='mt-4')
                html.P('Stored reduced density. Exposure is unverified. Review inputs above or load a saved analysis.')
                with html.Div(classes='wm-smallplot'):self._webmon_observed=plotly.Figure(display_mode_bar=True)
            with html.Div(v_show='webmon_has_result',classes='mt-4'):
                with html.Div(classes='wm-main'):
                    with html.Div(classes='wm-card'):
                        html.H3('Compare diffraction fits')
                        html.P('{{ webmon_fit_caption }}')
                        v.VSelect(label='Compare up to five settings · first is difference reference',v_model=('webmon_compared',),items=('webmon_models',),item_title='title',item_value='value',multiple=True,chips=True,closable_chips=True,update_modelValue=(self.webmon_compare,'[$event]'),density='compact',hide_details=True)
                        with html.Div(classes='wm-toolbar'):
                            v.VSelect(label='Coordinate',v_model=('webmon_axis',),items=('webmon_axes',),update_modelValue=(self.webmon_control, "['webmon_axis', $event]"),density='compact',hide_details=True)
                            v.VSelect(label='Region',v_model=('webmon_region',),items=([dict(title='Full range',value='full'),dict(title='Largest fit difference',value='difference'),dict(title='Large d / long TOF',value='large_d')],),update_modelValue=(self.webmon_control, "['webmon_region', $event]"),density='compact',hide_details=True)
                            v.VCheckbox(label='Error bars',v_model=('webmon_errors',),update_modelValue=(self.webmon_control, "['webmon_errors', $event]"),hide_details=True,density='compact')
                            v.VCheckbox(label='95% envelopes',v_model=('webmon_band',),update_modelValue=(self.webmon_control, "['webmon_band', $event]"),hide_details=True,density='compact')
                            v.VCheckbox(label='Fixed baseline',v_model=('webmon_component',),update_modelValue=(self.webmon_control, "['webmon_component', $event]"),hide_details=True,density='compact')
                        v.VCheckbox(label='Reflection ticks',v_model=('webmon_ticks',),update_modelValue=(self.webmon_control, "['webmon_ticks', $event]"),hide_details=True,density='compact')
                        with html.Div(v_for='metric in webmon_fit_metrics',key='metric.id',style='font-size:13px;padding:3px 0'):
                            html.Span('{{ metric.text }}',style=('metric.style',))
                        with html.Div(classes='wm-plot'):self._webmon_fit=plotly.Figure(display_mode_bar=True)
                        html.P('Shared zoom across fit, reflection ticks, residual and fit-difference panels. Δ fit is relative to the first selected setting, in supplied σ units; it is not a discrimination probability. Each color has its own pointwise 95% credible envelope of the mean curve, using all saved posterior draws. These omit observation noise and fixed-baseline uncertainty; a narrow envelope can still be wrong.')
                        html.P('Intensity stays in TOF-normalized units ×10⁶ for d and Q views. N: nuclear reflections; M1, M2, …: magnetic reflections in comparison order. Ticks show nonzero calculated reflections at each saved best sample; hover for phase and hkl. Separate nuclear/background curves are not available.')
                    with html.Div(classes='wm-card'):
                        html.H3('Moment uncertainty')
                        v.VSelect(label='Inspect setting',v_model=('webmon_model',),items=('webmon_models',),item_title='title',item_value='value',update_modelValue=(self.webmon_show_model,'[$event]'),density='compact')
                        v.VAlert(text=('webmon_boundary',),type='warning',variant='tonal',density='compact')
                        html.P('{{ webmon_sampling }}')
                        v.VBtn('Use selected checkpoint in Refinement',click=self.webmon_open_posterior_refinement,disabled=('!webmon_model || webmon_busy',),variant='outlined')
                        html.P('Uses a separately saved project at the selected posterior maximum. It is conditional on the fixed nuclear response and carries no inherited covariance.')
                        html.P('Intervals are conditional on the fixed nuclear model. Small intervals do not establish accuracy.')
                        with html.Div(classes='wm-smallplot'):self._webmon_moments=plotly.Figure(display_mode_bar=True)
                        with html.Div(v_for='moment in webmon_moment_rows',key='moment.site',style='border-top:1px solid #dce3ea;padding:10px 0;font-size:13px'):
                            html.Strong('{{ moment.site }} · {{ moment.median }} µB')
                            html.P('95% interval: {{ moment.interval }} µB')
                            html.P('Relative half-width: {{ moment.precision }} · {{ moment.warning }}')

                with html.Div(classes='wm-card mt-4'):
                    html.H3('Candidate settings')
                    html.P('Assumption exclusions and policy omissions are separate from data rejection. No experimental exclusions are authorized in this replay.')
                    with html.Div(classes='wm-toolbar'):
                        v.VTextField(label='Search group, BNS or setting',v_model=('webmon_search',),density='compact',hide_details=True)
                        v.VSelect(label='Status filter',v_model=('webmon_filter',),items=(['All','Replica screen passed','Sampling unresolved','Excluded by assumption','Omitted by policy'],),update_modelValue=(self.webmon_control, "['webmon_filter', $event]"),density='compact',hide_details=True)
                    v.VDataTable(headers=("[{title:'Magnetic group',key:'name'},{title:'BNS',key:'bns'},{title:'Setting',key:'setting'},{title:'State',key:'status'},{title:'Parameters',key:'dimensions'},{title:'Rwp (%)',key:'rwp'},{title:'χ² / dof',key:'reduced_chi2'},{title:'Detail',key:'reason'}]",),items=('webmon_display_rows',),search=('webmon_search',),items_per_page=10,density='compact',show_select=True,item_value='posterior',item_selectable='selectable',v_model=('webmon_table_selected',),update_modelValue=(self.webmon_table_select,'[$event]'))
                    html.P('Select sampled rows to compare their fits and inspect moments. Comparison is limited to five settings.')
                with html.Details():
                    html.Summary('Statistical diagnostics · selected setting')
                    with html.Div(classes='wm-main'):
                        with html.Div(classes='wm-card'):
                            html.H3('Correlation between site moments')
                            with html.Div(classes='wm-toolbar',v_if='webmon_site_options.length > 1'):
                                v.VSelect(label='Horizontal site',v_model=('webmon_pair_a',),items=('webmon_site_options',),update_modelValue=(self.webmon_control, "['webmon_pair_a', $event]"),density='compact')
                                v.VSelect(label='Vertical site',v_model=('webmon_pair_b',),items=('webmon_site_options',),update_modelValue=(self.webmon_control, "['webmon_pair_b', $event]"),density='compact')
                            with html.Div(classes='wm-smallplot'):self._webmon_correlation=plotly.Figure(display_mode_bar=True)
                        with html.Div(classes='wm-card'):
                            html.H3('How the sampler explored this setting')
                            html.P('β is data influence, not scan time. Weight ESS is deliberately held near 80% of particles by adaptive steps; its flat line is expected and is not the number of independent posterior samples.')
                            html.P('{{ webmon_sampler_summary }}')
                            html.P('These are mechanics of the first recorded replica, not evidence of a good fit or global convergence. Check residuals, assumption pressure and replica agreement separately.')
                            with html.Div(style='position:relative;height:420px'):self._webmon_sampler=plotly.Figure(display_mode_bar=True)
                with html.Details():
                    html.Summary('Provenance & numerical details')
                    html.P('Viewed revision {{ webmon_revision }} · Analyzed revision {{ webmon_analysis_revision }} · {{ webmon_latency }}')
                    html.Pre('{{ webmon_details }}',style='white-space:pre-wrap')
    def webmon_check_local(self):
        s=self.server.state
        try:
            if str(s.live_beamline)!='PG3':raise ValueError('This acceptance mirror is POWGEN only')
            rows=self._webmon.runs(s.powgen_ipts)
            if not rows:raise ValueError('No stored WebMON scans in this IPTS mirror')
            s.powgen_preflight_ready=True;s.powgen_preflight_status='ready';s.powgen_preflight_message=f'Local replay: {len(rows)} stored WebMON scans. No Galaxy submission.'
            s.powgen_source_directory=str(self._webmon.scope(s.powgen_ipts)/'shared/autoreduce')
        except Exception as e:s.powgen_preflight_ready=False;s.powgen_preflight_message=str(e)
        s.flush();return bool(s.powgen_preflight_ready)
    async def webmon_start_local_monitor(self):
        s=self.server.state
        if not self.webmon_check_local():return
        self._webmon_start_scope=str(s.powgen_ipts);s.powgen_monitoring=True;s.workflow_mode='powgen';s.workspace_view='experiment';s.experiment_result_view='webmon'
        s.powgen_message='Read-only local archive monitoring; production processing is not active.'
        self.webmon_refresh();s.flush()
        if self._webmon_poll_task is None or self._webmon_poll_task.done():self._webmon_poll_task=asyncio.create_task(self._webmon_poll())
    async def _webmon_poll(self):
        while self.server.state.powgen_monitoring:
            await asyncio.sleep(float(self._webmon.config.get('poll_seconds',30)))
            if self.server.state.powgen_monitoring:self.webmon_refresh(quiet=True)
            if self.server.state.powgen_monitoring and getattr(self,'_webmon_stream',None):self.webmon_stream_tick()
    def webmon_refresh(self,quiet=False,**_):
        s=self.server.state
        try:
            if str(s.live_beamline)!='PG3':raise ValueError('WebMON is currently available for POWGEN only')
            if not self._webmon:raise ValueError('No source/checkpoint adapter configured for this server')
            if self._webmon_start_scope and str(s.powgen_ipts)!=self._webmon_start_scope:raise ValueError('IPTS changed; stop and restart the experiment monitor')
            s.webmon_checked=datetime.now().astimezone().strftime('%H:%M:%S %Z');s.webmon_connection=f"Archive reachable · checking every {float(self._webmon.config.get('poll_seconds',30)):g} s";s.webmon_runs=self._webmon.runs(s.powgen_ipts);s.webmon_latest=s.webmon_runs[0]['run'] if s.webmon_runs else 'none'
            if s.webmon_runs:
                if s.webmon_follow_latest:self.webmon_select(s.webmon_runs[0]['run'],follow=True)
                elif not quiet:self.webmon_select(s.webmon_run or s.webmon_runs[0]['run'])
            if quiet and s.webmon_target and not getattr(self,'_webmon_stream',None):
                try:self._webmon.restore_binding(s.powgen_ipts,s.webmon_target,getattr(self,'_webmon_request',{}) or {})
                except (ValueError,OSError,KeyError):
                    s.webmon_stale=True;s.webmon_accepted=False;s.webmon_context_ready=False
                    s.webmon_message='Source revision changed. Displayed inference belongs to the previous snapshot; a new prepared context is required.'
        except Exception as e:
            s.webmon_message=str(e);s.webmon_connection='Source unavailable — retry Refresh source'
            s.webmon_accepted=False;s.webmon_context_ready=False;s.webmon_stale=bool(s.webmon_has_result)
        s.flush()
    def webmon_follow_changed(self,value,**_):
        self.server.state.webmon_follow_latest=bool(value)
        if value:self.webmon_refresh(quiet=True)

    def webmon_select(self,value,follow=False,**_):
        s=self.server.state
        if not follow:s.webmon_follow_latest=False
        s.webmon_run=str(value)
        try:
            row,receipt,pattern=self._webmon.capture(s.powgen_ipts,value)
            preview=(str(s.powgen_ipts),str(value),receipt['content_sha256'])
            if follow and preview==self._webmon_last_preview:
                if self._webmon_binding:self._webmon.binding(s.powgen_ipts,value)
                return
            self._webmon_last_preview=preview;s.webmon_accepted=False;s.webmon_context_ready=False
            s.webmon_polished_jobs=[];s.webmon_polished_job=''
            if s.webmon_target==str(value) and s.webmon_analysis_revision!=receipt['content_sha256'][:12]:s.webmon_stale=True
            self._webmon_pattern=pattern;self._webmon_observed.update(plots.observed(pattern));s.webmon_revision=receipt['content_sha256'][:12]
            s.webmon_message=f"Stored scan {value}: {row['sample']}, {row['temperature']} K. Final run duration {row['final_duration']} s; last live snapshot exposure is unverified."
            options=self._webmon.checkpoints(s.powgen_ipts,value)
            s.webmon_checkpoints=[dict(title=c['title'],value=c['id']) for c in options]
            s.webmon_checkpoint=''
            try:
                choice=self._webmon.selected_checkpoint(s.powgen_ipts,value)
                if choice:
                    s.webmon_checkpoint=choice['id'];self._webmon_review_checkpoint(choice)
                b=self._webmon.binding(s.powgen_ipts,value);self._webmon_binding=b;s.webmon_context_ready=True
                s.webmon_reference=b['description'];s.webmon_reference_rows=b['review']
                s.webmon_k=', '.join(str(v) for v in b['k']);s.webmon_ion=b['ion']
            except (ValueError,OSError,KeyError) as e:self._webmon_binding=None;s.webmon_reference=str(e);s.webmon_reference_rows=[]
            self.webmon_saved_list()
        except Exception as e:
            s.webmon_message=str(e);s.webmon_connection='Source unavailable — retry Refresh source'
            s.webmon_accepted=False;s.webmon_context_ready=False;s.webmon_stale=bool(s.webmon_has_result)
        s.flush()
    def _webmon_review_checkpoint(self,choice):
        import numpy as np
        s=self.server.state
        if digest(choice['view_path'])!=choice['view_sha256']:raise ValueError('Checkpoint review arrays changed; regenerate the verification')
        vfy=choice['verification'];b=choice.get('binding') or {}
        with np.load(choice['view_path'],allow_pickle=False) as data:
            self._webmon_checkpoint_plot.update(plots.checkpoint_review(data,axis=s.webmon_checkpoint_axis,difc=b.get('display_export_difc'),revision=choice['source_sha256']))
        error=vfy.get('native_saved_curve_relative_error')
        comparison=f'Saved curve relative difference from recomputed native model: {error:.2g}.' if isinstance(error,(int,float)) else 'Saved-curve consistency has not been measured.'
        s.webmon_checkpoint_summary=f"Original run {choice['run']} · {choice['role']} · Rwp {vfy['rwp_recomputed']:.3f}%. Target WebMON run {s.webmon_run}. {comparison} Inspect the reference residuals."
        def width(peak,key):
            value=peak.get(key)
            return f"{value['fwhm_A']:.5f}" if value else 'Unavailable'
        s.webmon_width_rows=[dict(d=p['d_A'],observed=width(p,'final_observed'),calculated=width(p,'radar_calculated'),webmon=width(p,'webmon_observed')) for p in vfy.get('peaks',[])]

    def webmon_checkpoint_axis_changed(self,value,**_):
        s=self.server.state;s.webmon_checkpoint_axis=value
        try:
            choice=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_run)
            if choice:self._webmon_review_checkpoint(choice)
        except Exception as exc:s.webmon_message=str(exc)
        s.flush()

    def webmon_checkpoint_changed(self,value,**_):
        s=self.server.state
        if s.webmon_busy:
            current=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_run)
            s.webmon_checkpoint=current['id'] if current else '';s.flush();return
        self.webmon_pause_stream()
        # Acceptance and analysis belong to a checkpoint revision, not just a run.
        s.webmon_accepted=False;s.webmon_context_ready=False;s.webmon_has_result=False
        s.webmon_target='';s.webmon_model='';s.webmon_models=[];s.webmon_rows=[];s.webmon_compared=[];s.webmon_table_selected=[]
        s.webmon_count_summary='No candidates analyzed';s.webmon_validity='No fit assessed'
        s.webmon_polished_jobs=[];s.webmon_polished_job=''
        self._webmon_payloads={};self._webmon_binding=None;self._webmon_model_paths=set();self._webmon_plot_binding=None
        try:
            choice=self._webmon.select_checkpoint(s.powgen_ipts,s.webmon_run,value)
            s.webmon_checkpoint=choice['id'];self._webmon_review_checkpoint(choice)
            b=self._webmon.binding(s.powgen_ipts,s.webmon_run);self._webmon_binding=b
            s.webmon_reference=b['description'];s.webmon_reference_rows=b['review']
            s.webmon_k=', '.join(str(v) for v in b['k']);s.webmon_ion=b['ion'];s.webmon_context_ready=True
            s.webmon_message='Starting checkpoint changed. Review its original fit and explicitly accept this revision before conditional magnetic analysis.'
        except Exception as exc:s.webmon_reference=str(exc);s.webmon_reference_rows=[];s.webmon_message=str(exc)
        self.webmon_saved_list();s.flush()

    def webmon_open_refinement(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        try:
            from ..refinement_controller import refinement_experiment_options
            choice=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_run)
            if not choice or not choice.get('refinement_source'):raise ValueError('No published refinement source is attached to this checkpoint')
            source=choice['refinement_source']
            if source['source_sha256']!=choice['source_sha256']:raise ValueError('Refinement source differs from the selected checkpoint')
            self._webmon_handoff_source(source,'Inspect this nuclear starting model. Refine its existing background, phase scale and unit cell while keeping atomic coordinates and instrument/profile parameters fixed. Preserve the current phases and fit range.')
        except Exception as exc:s.webmon_message=f'Refinement handoff failed: {exc}'
        s.flush()

    async def webmon_load_published_sources(self,**_):
        s=self.server.state
        try:
            task=getattr(self,'_refinement_refresh_task',None)
            if task and not task.done():await task
            else:await self._refresh_refinement_sources()
            from ..refinement_sources import experiment_key
            experiment=experiment_key(str(self.service.history_id),'PG3','IPTS-'+str(s.powgen_ipts).removeprefix('IPTS-'))
            w=self._refinement_store()
            self._webmon_published_catalog={key:dict(value) for key,value in w.state['sources'].items() if value['experiment']==experiment}
            s.webmon_published_sources=[dict(title=f"{v['scan_label']} · {v['sample_label']}",value=key) for key,v in self._webmon_published_catalog.items()]
            if s.webmon_published_source not in self._webmon_published_catalog:s.webmon_published_source=''
            s.webmon_source_notice=f'{len(s.webmon_published_sources)} published GPX sources in this IPTS. Inspect a reference, then resolve the live producer mapping before analysis.'
        except Exception as exc:s.webmon_source_notice='Published source lookup failed: '+str(exc)
        s.flush()

    def webmon_published_source_changed(self,value,**_):
        s=self.server.state;s.webmon_published_source=value
        source=getattr(self,'_webmon_published_catalog',{}).get(value)
        if source:
            s.webmon_mapping_bank=str(source.get('bank') or '');s.webmon_mapping_accepted=False
            s.webmon_source_notice=f"Selected {source['scan_label']}, {source['sample_label']}, bank {source.get('bank') or 'unresolved'}. The GPX's instrument calibration does not establish the live export conversion. Producer mapping and native preparation are still required."
        s.flush()

    def webmon_open_model_setup(self,**_):
        s=self.server.state
        try:
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            if not source:raise ValueError('Select a published GPX first')
            self._webmon_handoff_source(source,'Prepare commensurate magnetic candidates for a provisional live snapshot. Inspect this exact nuclear parent and choose the magnetic sites, form factors and k before discovery. No completed-scan campaign is requested.')
            s.rf_mode='models';s.sx_kind='magnetic';s.sx_moment_model='symmetry_allowed'
            s.sx_start_source=source['id'];s.sx_source=source['id'];s.sx_follow_new=False;s.sx_existing='selected'
            self.inspect_symmetry_source()
        except Exception as exc:s.webmon_source_notice=str(exc)
        s.flush()

    async def webmon_prepare_from_models(self,**_):
        import copy
        from fractions import Fraction
        from gpx_recipe.symmetry.chemistry import sites_from_inputs
        from gpx_recipe.live.io import atomic_json
        from .galaxy_runtime import GalaxyLiveRuntime
        s=self.server.state
        if s.webmon_busy:return
        try:
            if not s.webmon_prepare_accepted:raise ValueError('Review the magnetic setup before preparation')
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            d=getattr(self,'_symmetry_draft',None);mapping=getattr(self,'_webmon_reviewed_mapping',None)
            if not source or not d or d.get('source_id')!=source['id'] or d.get('discovery_identity')!=self._discovery_identity() or s.sx_kind!='magnetic':
                raise ValueError('Inspect and discover magnetic candidates for this exact source in Model exploration first')
            launch=(str(s.powgen_ipts),str(s.webmon_run),source['id'])
            row,receipt,_=self._webmon.capture(s.powgen_ipts,s.webmon_run)
            if not mapping or mapping['source_id']!=source['id'] or mapping['snapshot_sha256']!=receipt['content_sha256']:
                raise ValueError('Review the producer mapping for the current source snapshot first')
            material=await asyncio.to_thread(self._refinement_store().runtime.resolve,source)
            sites=sites_from_inputs(s.sx_sites,s.sx_form_factors,s.sx_lande)
            catalog=copy.deepcopy(d['catalog']);selected=set(s.sx_candidates)
            catalog['candidates']=[c for c in catalog['candidates'] if c['id'] in selected]
            if not catalog['candidates']:raise ValueError('Select at least one discovered magnetic candidate')
            root=Path(self._webmon.config['jobs'])/'preparations'/uuid.uuid4().hex;root.mkdir(parents=True)
            atomic_json(root/'catalog.json',catalog)
            req=dict(action='prepare',source_gpx=material['source_path'],source_sha256=material['source_sha256'],
                catalog=str(root/'catalog.json'),catalog_sha256=digest(root/'catalog.json'),observation_source=row['path'],observation_source_sha256=receipt['content_sha256'],
                mapping=mapping,parent_phase=s.sx_parent,k=[str(Fraction(v.strip())) for v in s.sx_k.split(',')],sites=sites,
                source_revision=f"PG3_{s.webmon_run}:"+receipt['content_sha256'],reference_revision=f"PG3_{source['run_number']}:"+material['source_sha256'],
                upper=float(s.sx_max_moment),lower=0.,refine_transfer=bool(s.webmon_refine_transfer),workers=min(4,max(1,int(s.sx_slots))),particles=256)
            s.webmon_busy=True;s.webmon_source_notice='Preparing the reviewed snapshot in a Galaxy worker…';s.flush()
            key=request_identity(req)[:24]
            runtime=GalaxyLiveRuntime(Path(self._webmon.config['jobs'])/'galaxy',self.service)
            import threading
            self._webmon_remote_cancel=threading.Event()
            self._webmon_task=asyncio.current_task()
            result=await asyncio.to_thread(runtime.call,req,root/'dispatch',dispatch_key=key,cancel_event=self._webmon_remote_cancel)
            if (str(s.powgen_ipts),str(s.webmon_run),s.webmon_published_source)!=launch:
                raise ValueError('Selection changed while preparing. The Galaxy result is retained; review it for the original scan before adopting it.')
            prepared=result['preparation'];result_root=Path(result['directory'])
            inference=json.loads(Path(prepared['request']).read_text())
            for field in ('source_gpx','catalog','preflight'):inference[field]=str(result_root/inference[field])
            atomic_json(root/'inference-request.json',inference)
            forms={v['label']:v['form_factor'] for v in sites};ions=set(forms.values())
            binding=dict(reference_gpx=material['source_path'],reference_sha256=material['source_sha256'],source_gpx=prepared['source_gpx'],source_sha256=prepared['source_sha256'],
                source_dat=row['path'],source_dat_sha256=receipt['content_sha256'],request=str(root/'inference-request.json'),k=req['k'],ion=next(iter(ions)) if len(ions)==1 else 'per-site',site_forms=forms,
                maximum_moment=req['upper'],display_export_difc=prepared['display_export_difc'],
                description='Galaxy-prepared fixed nuclear response; empirical covariance/normalization qualification remains pending.',
                review=[dict(label='Source',value=source['scan_label']),dict(label='Live snapshot',value=receipt['content_sha256'][:12]),dict(label='Candidates',value=str(len(catalog['candidates']))),dict(label='Mapping',value=mapping['evidence'])])
            choice=dict(id='prepared-'+key,title='Reviewed '+source['scan_label'],run=source['run_number'],role='Nuclear reference',source_path=material['source_path'],source_sha256=material['source_sha256'],
                refinement_source=source,binding=binding,view_path=prepared['reference_view'],view_sha256=digest(prepared['reference_view']),
                verification=dict(rwp_recomputed=prepared['original_reference_rwp'],native_saved_curve_relative_error=prepared.get('native_saved_curve_relative_error'),peaks=[]))
            scope=f"IPTS-{str(s.powgen_ipts).removeprefix('IPTS-')}/{s.webmon_run}"
            self._webmon.config.setdefault('checkpoint_options',{}).setdefault(scope,[]).append(choice)
            if self._webmon.registry:self._webmon.registry.add(scope,choice)
            self._webmon.selected_checkpoints[scope]=choice['id']
            self.webmon_select(s.webmon_run)
            s.webmon_source_notice='Prepared in Galaxy. Inspect the reference and assumptions, then accept this revision to start conditional live analysis.'
        except Exception as exc:s.webmon_source_notice='Live preparation stopped: '+str(exc)
        finally:
            if self._webmon_task is asyncio.current_task():self._webmon_task=None
            s.webmon_busy=False;s.flush()

    def webmon_save_mapping(self,**_):
        from .mapping import reviewed_mapping
        from gpx_recipe.live.io import atomic_json
        s=self.server.state
        try:
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            if not source:raise ValueError('Select a published source GPX first')
            _,receipt,_=self._webmon.capture(s.powgen_ipts,s.webmon_run)
            mapping=reviewed_mapping(source=source,receipt=receipt,bank=s.webmon_mapping_bank,
                export_difc=s.webmon_mapping_factor,normalization=s.webmon_mapping_normalization,
                producer_sha256=s.webmon_mapping_producer,evidence=s.webmon_mapping_evidence,
                accepted=s.webmon_mapping_accepted)
            name=hashlib.sha256(json.dumps(mapping,sort_keys=True).encode()).hexdigest()
            path=Path(self._webmon.config['jobs'])/'mappings'/(name+'.json');atomic_json(path,mapping)
            self._webmon_reviewed_mapping=mapping
            s.webmon_source_notice='Reviewed producer mapping saved. Native source preparation and a reviewed magnetic catalogue are still required; no fit or counting decision was launched.'
        except Exception as exc:s.webmon_source_notice=str(exc)
        s.flush()

    def webmon_open_published_source(self,**_):
        s=self.server.state
        try:
            source=getattr(self,'_webmon_published_catalog',{}).get(s.webmon_published_source)
            if not source:raise ValueError('Select a published source first')
            self._webmon_handoff_source(source,'Inspect this published GPX as a possible nuclear reference for an in-progress POWGEN scan. Preserve the recorded bank, calibration, fit limits and phases. Do not assume its TOF calibration is the live d-spacing export convention.')
        except Exception as exc:s.webmon_source_notice=str(exc)
        s.flush()

    def _webmon_handoff_source(self,source,prompt):
        from ..refinement_controller import refinement_experiment_options
        s=self.server.state;w=self._refinement_store();w.observe([source])
        self._refinement_internal=True
        try:
            s.rf_experiments=[dict(title=v,value=k) for k,v in refinement_experiment_options(w).items()]
            s.rf_experiment=source['experiment'];s.rf_source=source['id'];s.rf_scope='scans';s.rf_scans=[source['scan_key']]
            s.rf_samples=[];s.rf_banks=[];s.rf_existing=True;s.rf_future=False;s.rf_mode='recipe';s.rf_selected_job=''
            s.rf_request=prompt;s.experiment_result_view='refinement';s.flush()
        finally:self._refinement_internal=False
        self._refinement_sync();self.inspect_refinement_project()

    def webmon_open_posterior_refinement(self,**_):
        import copy
        s=self.server.state
        if s.webmon_busy:return
        try:
            post=self._webmon_post(s.webmon_model)
            path=post.get('fitted_gpx');export=post.get('gpx_export') or {}
            if not path:raise ValueError('This result has no verified posterior project; the starting project is not a fitted output')
            choice=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_target)
            if not choice or not choice.get('refinement_source'):raise ValueError('The original experiment source is unavailable')
            if choice['source_sha256']!=self._webmon_request['reference_revision'].split(':')[-1]:raise ValueError('Select the reference used by this posterior first')
            source=copy.deepcopy(choice['refinement_source']);checksum=export['fitted_gpx_sha256']
            for key in ('dataset_id','collection_id','collection_name','galaxy_job_id'):source.pop(key,None)
            row=next(r for r in s.webmon_rows if r.get('posterior')==s.webmon_model)
            if not self._webmon_is_local() and not row.get('fitted_dataset_id'):
                raise ValueError('The verified posterior GPX has not been mapped to a Galaxy collection element yet')
            if row.get('fitted_dataset_id'):source['dataset_id']=row['fitted_dataset_id']
            source.update(id='live-posterior-'+checksum[:24],record_uid='live-posterior-'+checksum[:24],source_path=path,source_sha256=checksum,
                internal_source_sha256=checksum,published_sha256=checksum,collection_name=Path(path).name,collection_id='',base_rwp=post['rwp'],
                origin='conditional_live_posterior',parent_source_sha256=choice['source_sha256'],snapshot_revision=self._webmon_request['source_revision'])
            self._webmon_handoff_source(source,'Inspect this conditional magnetic checkpoint and its fixed nuclear reference. Preserve its phases, moment constraints, fit limits and instrument parameters. Propose refinements for review; do not infer native convergence or uncertainty from the saved starting state.')
        except Exception as exc:s.webmon_message=f'Posterior handoff unavailable: {exc}'
        s.flush()

    def webmon_find_polishes(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        try:
            choice=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_run);w=self._refinement_store()
            jobs=[j for j in w.state['jobs'].values() if j['status']=='completed' and j['source'].get('source_sha256')==choice['source_sha256'] and (j.get('result') or {}).get('final_gpx')]
            s.webmon_polished_jobs=[dict(title=f"{j['source']['scan_label']} · job {j['id'][:12]}",value=j['id']) for j in jobs]
            s.webmon_polished_job=jobs[-1]['id'] if jobs else ''
            s.webmon_message=f'{len(jobs)} completed polish(es) of this exact starting checkpoint. Native re-preparation and review are required before magnetic use.'
        except Exception as exc:s.webmon_message=str(exc)
        s.flush()

    def webmon_prepare_polish(self,**_):
        s=self.server.state
        if s.webmon_busy or not s.webmon_polished_job:return
        launch=(str(s.powgen_ipts),str(s.webmon_run),s.webmon_checkpoint,s.webmon_polished_job)
        self._webmon_cancelled=False
        s.webmon_busy=True;s.webmon_accepted=False;s.flush()
        self._webmon_task=asyncio.create_task(self._webmon_prepare_polish(launch))

    async def _webmon_prepare_polish(self,launch):
        s=self.server.state;ipts,run,checkpoint,uid=launch;lease=None
        try:
            from ..refinement_runtime import verify_result_files
            choice=self._webmon.select_checkpoint(ipts,run,checkpoint);w=self._refinement_store();job=w.state['jobs'].get(uid)
            if not job or job['status']!='completed' or job['source'].get('source_sha256')!=choice['source_sha256']:raise ValueError('Select a completed polish of this exact checkpoint')
            result=job['result'];verify_result_files(result['directory'],result)
            if result.get('source_sha256')!=choice['source_sha256']:raise ValueError('Polish receipt has different source ancestry')
            command=self._webmon_local_command('checkpoint_prepare_command')
            if not command:raise ValueError('No native checkpoint preparation adapter is configured')
            preparation_id=hashlib.sha256(json.dumps([ipts,run,checkpoint,result['output_sha256'],time.time_ns()]).encode()).hexdigest()
            out=Path(self._webmon.config['jobs'])/'references'/preparation_id;out.mkdir(parents=True,exist_ok=False)
            receipt=out/'preparation-input.json';receipt.write_text(json.dumps(dict(base_option=choice,job_id=uid,source_path=str(Path(result['directory'])/result['final_gpx']),source_sha256=result['output_sha256']),indent=2))
            s.webmon_message='Checking the polished GPX and rebuilding native magnetic preparation…';s.flush()
            lease=await self._webmon_native_slot()
            if self._webmon_cancelled:raise ValueError('Reference preparation cancelled')
            with (out/'native.log').open('w') as log:
                process=await spawn_process(*command,str(receipt),stdout=log,stderr=log)
                self._webmon_process=process
                if self._webmon_cancelled:await stop_process(process);raise ValueError('Reference preparation cancelled')
                code=await wait_process(process,float(self._webmon.config.get('reference_timeout_seconds',600)))
            if code:raise ValueError(f'Native re-preparation failed; inspect {out / "native.log"}. The previous checkpoint is retained.')
            if self._webmon_cancelled:raise ValueError('Reference preparation cancelled; output not adopted')
            new=json.loads((out/'option.json').read_text());self._webmon.register_checkpoint(ipts,run,new)
            s.webmon_calculation_status='Reference prepared'
            # Preserve selected experiment if the user browsed elsewhere while preparation ran.
            if str(s.powgen_ipts)==ipts and str(s.webmon_run)==run:
                s.webmon_checkpoints=[dict(title=c['title'],value=c['id']) for c in self._webmon.checkpoints(ipts,run)]
                s.webmon_busy=False;self.webmon_checkpoint_changed(new['id'])
                s.webmon_message='Polished checkpoint prepared as a new revision. Inspect its original fit, then review the provisional transfer before conditional analysis.'
        except Exception as exc:
            s.webmon_calculation_status='Cancelled' if self._webmon_cancelled else 'Failed'
            s.webmon_message=str(exc)
        finally:
            if self._webmon_process and self._webmon_process.returncode is None:await stop_process(self._webmon_process)
            if lease:lease.release()
            self._webmon_process=None;s.webmon_busy=False;s.flush()

    def webmon_profile_polish(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        self.webmon_pause_stream()
        try:
            choice=self._webmon.selected_checkpoint(s.powgen_ipts,s.webmon_run)
            command=self._webmon_local_command('profile_polish_command')
            if not command or not choice:raise ValueError('No bounded profile diagnostic is configured for this server/checkpoint')
            launch=(str(s.powgen_ipts),str(s.webmon_run),choice,command)
            s.webmon_busy=True;s.webmon_accepted=False;self._webmon_cancelled=False
            async def execute():
                ipts,run,base,command=launch;lease=None
                out=Path(self._webmon.config['jobs'])/'references'/uuid.uuid4().hex;out.mkdir(parents=True)
                from .stream import atomic_json
                atomic_json(out/'profile-input.json',dict(base_option=base))
                try:
                    s.webmon_message='Testing physically bounded sample broadening with the instrument fixed…';s.flush()
                    lease=await self._webmon_native_slot()
                    if self._webmon_cancelled:raise ValueError('Profile diagnostic cancelled')
                    with (out/'native.log').open('w') as log:
                        self._webmon_process=await spawn_process(*command,str(out/'profile-input.json'),stdout=log,stderr=log)
                        if self._webmon_cancelled:await stop_process(self._webmon_process)
                        code=await wait_process(self._webmon_process,self._webmon.config.get('reference_timeout_seconds',600))
                    if code or self._webmon_cancelled:raise ValueError(f'Profile diagnostic not adopted; inspect {out / "native.log"}')
                    new=json.loads((out/'option.json').read_text());self._webmon.register_checkpoint(ipts,run,new)
                    if str(s.powgen_ipts)==ipts and str(s.webmon_run)==run:
                        s.webmon_checkpoints=[dict(title=c['title'],value=c['id']) for c in self._webmon.checkpoints(ipts,run)]
                        s.webmon_busy=False;self.webmon_checkpoint_changed(new['id'])
                    s.webmon_calculation_status='Reference prepared';s.webmon_message='Bounded profile diagnostic prepared. Review its residuals; physical attribution and counting predictions remain unqualified.'
                except Exception as exc:
                    s.webmon_calculation_status='Cancelled' if self._webmon_cancelled else 'Failed';s.webmon_message=str(exc)
                finally:
                    if self._webmon_process and self._webmon_process.returncode is None:await stop_process(self._webmon_process)
                    if lease:lease.release()
                    self._webmon_process=None;s.webmon_busy=False;s.flush()
            self._webmon_task=asyncio.create_task(execute())
        except Exception as exc:s.webmon_message=str(exc)
        s.flush()

    def webmon_pause_stream(self,**_):
        if getattr(self,'_webmon_queue',None):self._webmon_queue.close()
        self._webmon_stream=None
        self.server.state.webmon_stream_active=False
        self.server.state.webmon_stream_message='Automatic updates paused. An active calculation may finish; Cancel stops it.'

    def webmon_enable_stream(self,**_):
        from .stream import ObservationStream,LatestQueue
        from fractions import Fraction
        s=self.server.state
        if s.webmon_busy:return
        try:
            if self._webmon.config.get('allow_conditional_stream') is not True:
                raise ValueError('This server has not enabled conditional stream analysis')
            k=[str(Fraction(x.strip())) for x in str(s.webmon_k).split(',')]
            b,req=self._webmon.request(s.powgen_ipts,s.webmon_run,k=k,ion=s.webmon_ion,lower=float(s.webmon_lower),upper=float(s.webmon_upper),accepted=s.webmon_accepted)
            directory=Path(self._webmon.config['jobs'])/'streams'/uuid.uuid4().hex
            self._webmon_stream=ObservationStream(self._webmon,s.powgen_ipts,s.webmon_run,b,req,directory)
            self._webmon_queue=LatestQueue(directory/'queue.json');s.webmon_stream_active=True
            s.webmon_stream_message=f'Following run {s.webmon_run} only. Each cumulative snapshot replaces the previous likelihood; nuclear/profile/normalization assumptions stay fixed. No counting advice.'
            self.webmon_stream_tick()
        except Exception as exc:s.webmon_stream_message=str(exc)
        s.flush()

    def webmon_stream_tick(self):
        from .stream import ReviewRequired
        s=self.server.state;stream=self._webmon_stream;queue=self._webmon_queue
        if not stream or not queue or queue.closed:return
        try:
            if str(s.powgen_ipts).removeprefix('IPTS-')!=stream.ipts.removeprefix('IPTS-'):
                raise ReviewRequired('Experiment changed; review a new stream')
            item=stream.capture();queue.offer(item)
            if item is None and not queue.pending and not queue.active and s.webmon_has_result:
                s.webmon_stale=(str(s.webmon_run)!=stream.run or s.webmon_revision!=s.webmon_analysis_revision)
            if item:
                s.webmon_stale=True
                if str(s.webmon_run)==stream.run:
                    from .snapshot import parse_live_ascii
                    pattern=parse_live_ascii(Path(item['binding']['source_dat']).read_bytes())
                    self._webmon_pattern=pattern;self._webmon_observed.update(plots.observed(pattern))
                    s.webmon_revision=item['receipt']['content_sha256'][:12]
                    self._webmon_last_preview=(str(s.powgen_ipts),stream.run,item['receipt']['content_sha256'])
            if queue.pending and not s.webmon_busy and (not self._webmon_task or self._webmon_task.done()):
                self._webmon_cancelled=False
                async def execute(update):
                    await self._webmon_execute(dict(sealed=(update['binding'],update['request']),webmon_run=stream.run))
                    if s.webmon_calculation_status!='Completed':raise RuntimeError(s.webmon_message)
                    lag=time.monotonic()-update['captured_monotonic']
                    s.webmon_stream_message=f'Run {stream.run}: latest completed analysis took {lag:.1f} s from capture. {queue.state["coalesced"]} superseded queued snapshots skipped. Fixed response; counting advice unavailable.'
                    s.webmon_stale=bool(queue.pending) or str(s.webmon_run)!=stream.run;s.flush()
                async def drain():
                    try:await queue.drain(execute)
                    except Exception as exc:
                        self.webmon_pause_stream();s.webmon_stream_message=f'Automatic analysis paused: {exc}';s.flush()
                self._webmon_task=asyncio.create_task(drain())
        except ReviewRequired as exc:
            self.webmon_pause_stream();s.webmon_stream_message=str(exc);s.webmon_accepted=False;s.webmon_stale=True
        except (ValueError,OSError,RuntimeError) as exc:
            s.webmon_stream_message=f'Waiting for a complete compatible snapshot: {exc}. Last completed result retained.'
            s.webmon_stale=bool(s.webmon_has_result)
        s.flush()

    def webmon_start(self,**_):
        if self.server.state.webmon_busy:return
        self.webmon_pause_stream()
        self._webmon_cancelled=False
        self.server.state.webmon_busy=True;self.server.state.flush()
        s=self.server.state
        launch={key:getattr(s,key) for key in ['powgen_ipts','webmon_run','webmon_k','webmon_ion','webmon_lower','webmon_upper','webmon_accepted']}
        if getattr(self,'_webmon',None):
            try:
                from fractions import Fraction
                k=[str(Fraction(x.strip())) for x in str(launch['webmon_k']).split(',')]
                launch['sealed']=self._webmon.request(launch['powgen_ipts'],launch['webmon_run'],k=k,ion=str(launch['webmon_ion']).strip(),lower=float(launch['webmon_lower']),upper=float(launch['webmon_upper']),accepted=launch['webmon_accepted'])
            except Exception as exc:
                s.webmon_busy=False;s.webmon_calculation_status='Rejected before launch';s.webmon_message=f'Analysis not started: {exc}';s.flush();return
        self._webmon_task=asyncio.create_task(self._webmon_execute(launch))

    async def _webmon_native_slot(self):
        from .resources import acquire
        return await acquire(self._webmon.config.get('slot_root',str(Path(self._webmon.config['jobs'])/'.slots')),self._webmon.config.get('max_active_jobs',1),self._webmon.config.get('job_timeout_seconds',3600))

    async def _webmon_execute(self,launch):
        if not self._webmon_is_local():
            return await self._webmon_execute_galaxy(launch)
        s=self.server.state;lease=None
        try:
            b,req=launch['sealed'];validate_request(req,b)
            if self._webmon_cancelled:raise ValueError('Analysis cancelled before launch')
            identity=request_identity(req)
            out=Path(self._webmon.config['jobs'])/(identity[:24]+'-'+uuid.uuid4().hex);out.mkdir(parents=True,exist_ok=False)
            # Reuse only per-subgroup native kernels; the worker checks source and catalogue identity.
            cache_roots=[Path(b['request']).parent/'models',*Path(self._webmon.config['jobs']).glob('*/models')]
            newest={}
            for cache in cache_roots:
                for kernel in cache.glob('*/kernel.json'):
                    if kernel.parent.name not in newest or kernel.stat().st_mtime_ns>newest[kernel.parent.name].stat().st_mtime_ns:newest[kernel.parent.name]=kernel
            for kernel in newest.values():
                target=out/'models'/kernel.parent.name;target.mkdir(parents=True,exist_ok=True)
                for name in ['kernel.json','kernel.npz','magnetic-refinement.gpx']:
                    if (kernel.parent/name).is_file() and not (target/name).exists():shutil.copy2(kernel.parent/name,target/name)
            request=out/'request.json';request.write_text(json.dumps(req,indent=2));
            if (out/'result.json').exists():(out/'result.json').replace(out/f'previous-result-{time.time_ns()}.json')
            self._webmon_model_paths=set()
            # Keep the last completed plot while the new revision is queued or
            # preparing. Its own caption/identity stays unchanged until a new
            # verified posterior is ready; it is visibly stale meanwhile.
            s.webmon_stale=bool(s.webmon_has_result)
            self._webmon_pending_binding=b;self._webmon_pending_request=req
            s.webmon_pending_target=str(launch['webmon_run']);s.webmon_busy=True
            s.webmon_message='Preparing native subgroup kernels and conditional Monte Carlo analysis…';s.flush();started=time.monotonic()
            lease=await self._webmon_native_slot()
            if self._webmon_cancelled:raise ValueError('Analysis cancelled while waiting for capacity')
            with (out/'worker.log').open('ab') as log:
                self._webmon_process=await spawn_process(*self._webmon_local_command('worker_command'),str(request),stdout=log,stderr=log)
                if self._webmon_cancelled:await stop_process(self._webmon_process)
                deadline=time.monotonic()+float(self._webmon.config.get('job_timeout_seconds',3600))
                while self._webmon_process.returncode is None:
                    if time.monotonic()>deadline:
                        await stop_process(self._webmon_process);raise TimeoutError('Analysis wall-time limit exceeded; candidates remain unresolved')
                    self._webmon_load_result(out)
                    s.webmon_latency=f'{time.monotonic()-started:.1f} s since request (includes preparation)';s.flush()
                    try:await asyncio.wait_for(self._webmon_process.wait(),timeout=1)
                    except asyncio.TimeoutError:pass
            self._webmon_load_result(out)
            s.webmon_calculation_status='Cancelled' if self._webmon_cancelled else ('Completed' if self._webmon_process.returncode==0 else 'Failed')
            s.webmon_message=f"Analysis {'cancelled' if self._webmon_cancelled else ('completed' if self._webmon_process.returncode==0 else 'failed')} for run {s.webmon_target}. Results remain conditional; inspect adequacy and unresolved candidates."
        except Exception as e:
            s.webmon_calculation_status='Cancelled' if self._webmon_cancelled else 'Failed'
            s.webmon_message=f'Analysis not started/completed: {e}'
        finally:
            if self._webmon_process and self._webmon_process.returncode is None:await stop_process(self._webmon_process)
            if lease:lease.release()
            self._webmon_process=None;s.webmon_busy=False;self.webmon_saved_list();s.flush()
    async def _webmon_execute_galaxy(self,launch):
        from .galaxy_runtime import GalaxyLiveRuntime
        from gpx_recipe.live.io import atomic_json
        import threading
        s=self.server.state
        try:
            b,req=launch['sealed'];validate_request(req,b)
            identity=request_identity(req)
            root=Path(self._webmon.config['jobs'])/'galaxy';root.mkdir(parents=True,exist_ok=True)
            runtime=GalaxyLiveRuntime(root,self.service)
            self._webmon_remote_cancel=threading.Event()
            self._webmon_pending_binding=b;self._webmon_pending_request=req
            s.webmon_busy=True;s.webmon_stale=bool(s.webmon_has_result)
            s.webmon_message='Submitting this snapshot to the Galaxy scientific worker. New snapshots will replace only pending work.';s.flush()
            loop=asyncio.get_running_loop()
            def submitted(job):
                def update():
                    s.webmon_message='Galaxy job '+str(job.get('galaxy_job_id',''))+' is queued or running.';s.flush()
                loop.call_soon_threadsafe(update)
            result=await asyncio.to_thread(runtime.call,req,root/identity[:24],dispatch_key=identity[:24],
                cache_dataset_id=getattr(self,'_webmon_response_cache',None),on_submitted=submitted,cancel_event=self._webmon_remote_cancel)
            if self._webmon_cancelled:return
            self._webmon_response_cache=result.get('response_cache_dataset_id')
            # Keep the original checked bundle intact. The UI view contains
            # hard-linked immutable models and explicitly rebased presentation
            # paths; it never changes scientific artifacts or their receipts.
            native=Path(result['directory']);view=Path(self._webmon.config['jobs'])/(identity[:24]+'-galaxy-view');view.mkdir(exist_ok=True)
            if not (view/'models').exists():shutil.copytree(native/'models',view/'models',copy_function=os.link)
            display=dict(result);display['rows']=[dict(row) for row in result['rows']]
            for row in display['rows']:
                for key in ('posterior','fitted_gpx'):
                    if row.get(key):row[key]=str(view/Path(row[key]).relative_to(native))
            atomic_json(view/'request.json',req)
            display['request_sha256']=digest(view/'request.json')
            display['verified_native_transport_sha256']=result['transport_sha256']
            atomic_json(view/'result.json',display)
            self._webmon_load_result(view)
            s.webmon_calculation_status='Completed';s.webmon_message='Galaxy snapshot analysis completed. Results remain conditional on the reviewed fixed nuclear response.'
        except Exception as exc:
            s.webmon_calculation_status='Stopped waiting' if self._webmon_cancelled else 'Failed'
            s.webmon_message=str(exc)
        finally:s.webmon_busy=False;s.flush()

    def _webmon_load_result(self,out):
        s=self.server.state;p=out/'result.json'
        if not p.exists():return
        req=json.loads((out/'request.json').read_text());r=json.loads(p.read_text())
        if r.get('request_sha256')!=digest(out/'request.json') or any(r.get(k)!=req.get(k) for k in ('source_revision','reference_revision')):raise ValueError('Worker result belongs to different scientific inputs')
        options=[dict(title=f"{a['name']} · {a['bns']} · {a['id'][-6:]}",value=a['posterior']) for a in r['rows'] if 'posterior' in a and a.get('artifacts')]
        options=[x for x in options if Path(x['value']).resolve().is_relative_to((out/'models').resolve())]
        progress=[dict(a,selectable='posterior' in a,status=STATUS.get(a['status'],a['status'].replace('_',' ')),setting=a['id'],**{k:round(a[k],3) for k in ['rwp','reduced_chi2'] if k in a}) for a in r['rows']]
        if not options and s.webmon_has_result:
            s.webmon_pending_rows=progress
            return
        if out != self._webmon_out:
            if req == getattr(self,'_webmon_pending_request',None):
                self._webmon_plot_binding=self._webmon_pending_binding
            self._webmon_request=req;self._webmon_payloads={};s.webmon_compared=[];s.webmon_table_selected=[];s.webmon_model=''
            s.webmon_target=req['source_revision'].split(':',1)[0].removeprefix('PG3_')
            s.webmon_analysis_revision=req['source_revision'].split(':')[-1][:12]
            s.webmon_stale=(str(s.webmon_run)!=s.webmon_target or s.webmon_revision!=s.webmon_analysis_revision)
        self._webmon_out=out
        s.webmon_rows=progress
        self._webmon_model_paths={x['value'] for x in options};s.webmon_models=options;s.webmon_has_result=bool(options)
        sampled=len(options);omitted=sum(a['status']=='policy_omitted' for a in r['rows']);incompatible=sum(a['status']=='constraint_incompatible' for a in r['rows'])
        s.webmon_count_summary=f'{sampled} sampled · {incompatible} assumption · {omitted} policy'
        self.webmon_filter_rows()
        if options and s.webmon_model not in self._webmon_model_paths:
            s.webmon_compared=[options[0]['value']];s.webmon_table_selected=s.webmon_compared;self.webmon_show_model(options[0]['value']);self.webmon_render()

    def webmon_control(self,name,value):
        allowed={'webmon_axis','webmon_region','webmon_errors','webmon_band','webmon_component','webmon_ticks','webmon_filter','webmon_pair_a','webmon_pair_b'}
        if name not in allowed:return
        self.server.state[name]=value
        if name=='webmon_filter':self.webmon_filter_rows()
        elif name in {'webmon_pair_a','webmon_pair_b'}:self.webmon_render_diagnostics()
        else:self.webmon_render()

    def webmon_filter_rows(self,*_):
        s=self.server.state;s.webmon_display_rows=[r for r in s.webmon_rows if s.webmon_filter=='All' or r['status']==s.webmon_filter];s.flush()

    def _webmon_post(self,value):
        if value not in self._webmon_model_paths:raise ValueError('Unrecognized analysis result')
        row=next(r for r in self.server.state.webmon_rows if r.get('posterior')==value)
        verify_posterior(value,row.get('artifacts'))
        if value not in self._webmon_payloads:
            if len(self._webmon_payloads)>=6:self._webmon_payloads.pop(next(iter(self._webmon_payloads)))
            self._webmon_payloads[value]=full_posterior(value)
        return self._webmon_payloads[value]

    def webmon_compare(self,values,from_table=False,**_):
        s=self.server.state;valid=list(dict.fromkeys(v for v in (values or []) if v in self._webmon_model_paths))
        if len(valid)>5:s.webmon_message='Compare at most five settings. Remove one before adding another.'
        s.webmon_compared=valid[:5]
        if not from_table or len(valid)>5:s.webmon_table_selected=valid[:5]
        self.webmon_render()

    def webmon_table_select(self,values,**_):
        previous=list(self.server.state.webmon_compared)
        self.webmon_compare(values,from_table=True)
        added=[v for v in self.server.state.webmon_compared if v not in previous]
        if added:self.webmon_show_model(added[-1])

    def webmon_render(self,*_):
        s=self.server.state;b=self._webmon_plot_binding or {};difc=b.get('display_export_difc')
        s.webmon_axes=[dict(title='TOF (µs)',value='tof')]+([dict(title='d (Å)',value='d'),dict(title='Q (Å⁻¹)',value='q')] if difc else [])
        if s.webmon_axis in {'d','q'} and not difc:s.webmon_axis='tof'
        models=[]
        self._webmon_colors={key:color for key,color in self._webmon_colors.items() if key in s.webmon_compared}
        for value in s.webmon_compared:
            if value not in self._webmon_model_paths:continue
            row=next(r for r in s.webmon_rows if r.get('posterior')==value)
            if value not in self._webmon_colors:self._webmon_colors[value]=next(c for c in plots.COLORS if c not in self._webmon_colors.values())
            models.append(dict(id=row['id'],label=f"{row['name']} · {row['bns']} [{row['id'][-6:]}]",color=self._webmon_colors[value],post=self._webmon_post(value),revision=s.webmon_analysis_revision))
        if not models:
            s.webmon_fit_metrics=[]
            s.webmon_fit_caption='Select one to five sampled settings.'
            self._webmon_fit.update(plots.comparison([]));s.flush();return
        self._webmon_fit.update(plots.comparison(models,axis=s.webmon_axis,difc=difc,errors=s.webmon_errors,ensemble=s.webmon_band,component=s.webmon_component,region=s.webmon_region,ticks=s.webmon_ticks))
        s.webmon_fit_metrics=[dict(id=m['id'],style=f"color:{m['color']}",text=f"M{i+1} · {m['label']} · Rwp {m['post']['rwp']:.2f}% · χ²/dof {m['post']['reduced_chi2']:.1f}") for i,m in enumerate(models)]
        s.webmon_fit_caption=f"Run {s.webmon_target} · {len(models)} settings · full saved bins · reference revision {str(self._webmon_request.get('reference_revision','')).split(':')[-1][:12]}"
        s.flush()

    def webmon_show_model(self,value,**_):
        s=self.server.state
        if value not in self._webmon_model_paths:return
        p=self._webmon_post(value);s.webmon_model=value
        lo=self._webmon_request.get('lower');hi=self._webmon_request.get('upper')
        s.webmon_lower_used=lo;s.webmon_upper_used=hi;s.webmon_moment_rows=moment_rows(p,lo,hi)
        s.webmon_validity='Exploratory only · baseline unqualified'
        s.webmon_boundary=f"{100*p.get('prior_edge_mass',0):.1f}% of draws near a moment bound. " + ('Constraint pressure: do not read narrow intervals as accuracy.' if p.get('prior_edge_mass',0)>.1 else 'Intervals still depend on the fixed baseline.')
        s.webmon_sampling=f"{STATUS.get(p['status'],p['status'])} · Rwp {p['rwp']:.2f}% · χ²/dof {p['reduced_chi2']:.1f}. Replica agreement is not proof of convergence."
        self._webmon_moments.update(plots.moments(p,lo,hi))
        s.webmon_site_options=[dict(title=label,value=i) for i,label in enumerate(p['labels'])];s.webmon_pair_a=0;s.webmon_pair_b=min(1,len(p['labels'])-1)
        s.webmon_sampler_summary=sampler_summary(p)
        self.webmon_render_diagnostics();self._webmon_sampler.update(plots.sampler(p))
        s.webmon_details=json.dumps(dict(prior_edge_mass=p.get('prior_edge_mass'),site_intervals_95pct=dict(zip(p['labels'],p['intervals'])),scope=p['scope'],forecast_enabled=False),indent=2);s.flush()

    def webmon_render_diagnostics(self,*_):
        s=self.server.state
        if s.webmon_model in self._webmon_model_paths:self._webmon_correlation.update(plots.correlation(self._webmon_post(s.webmon_model),(s.webmon_pair_a,s.webmon_pair_b)))

    def webmon_saved_list(self):
        s=self.server.state;self._webmon_saved_paths={};options=[]
        if self._webmon and s.webmon_run:
            for path in sorted(Path(self._webmon.config['jobs']).glob('*/result.json'),key=lambda p:p.stat().st_mtime_ns,reverse=True):
                if not path.resolve().is_relative_to(Path(self._webmon.config['jobs']).resolve()):continue
                try:
                    candidate_request=json.loads((path.parent/'request.json').read_text())
                    b=self._webmon.restore_binding(s.powgen_ipts,s.webmon_run,candidate_request,allow_archived=True)
                    req,r=verify_result(path.parent,binding=b,complete=True)
                    if any(row.get('posterior') and not row.get('artifacts') for row in r['rows']):continue
                    if r.get('status')!='complete' or req.get('source_revision')!=f"PG3_{s.webmon_run}:"+b['source_dat_sha256']:continue
                    if req.get('source_sha256')!=b['source_sha256'] or req.get('reference_revision','').split(':')[-1]!=b['reference_sha256']:continue
                    if r.get('request_sha256')!=digest(path.parent/'request.json'):continue
                    key=path.parent.name;self._webmon_saved_paths[key]=path.parent
                    options.append(dict(title=f"{req['lower']:g}–{req['upper']:g} µB · {len(r['rows'])} settings · {r['elapsed_seconds']:.0f} s",value=key))
                except (OSError,ValueError,KeyError,TypeError):continue
        s.webmon_saved=options;s.webmon_saved_job=options[0]['value'] if options else '';s.flush()

    def webmon_restore(self,**_):
        s=self.server.state
        if s.webmon_busy:return
        try:
            out=self._webmon_saved_paths[s.webmon_saved_job]
            b=self._webmon.restore_binding(s.powgen_ipts,s.webmon_run,json.loads((out/'request.json').read_text()),allow_archived=True)
            req,r=verify_result(out,binding=b,complete=True)
            for row in r['rows']:
                if row.get('posterior'):
                    if not Path(row['posterior']).resolve().is_relative_to((out/'models').resolve()):raise ValueError('Saved model path escapes the analysis')
                    verify_posterior(row['posterior'],row.get('artifacts'))
            if r['request_sha256']!=digest(out/'request.json') or req['source_sha256']!=b['source_sha256'] or req['source_revision']!=f"PG3_{s.webmon_run}:"+b['source_dat_sha256']:raise ValueError('Saved result no longer matches reviewed source')
            self._webmon_plot_binding=b;self._webmon_request=req;self._webmon_payloads={};s.webmon_model='';s.webmon_compared=[];s.webmon_table_selected=[];s.webmon_target=s.webmon_run;s.webmon_analysis_revision=req['source_revision'].split(':')[-1][:12]
            try:
                _,current,_=self._webmon.capture(s.powgen_ipts,s.webmon_run)
                s.webmon_stale=current['content_sha256'] != req['source_revision'].split(':')[-1]
            except (OSError,ValueError,RuntimeError):
                s.webmon_stale=True
            s.webmon_axis='d' if b.get('display_export_difc') else 'tof'
            s.webmon_latency=f"Saved calculation: {r['elapsed_seconds']:.1f} s; no new inference"
            s.webmon_calculation_status='Restored'
            self._webmon_response_cache=r.get('response_cache_dataset_id')
            self._webmon_load_result(out);s.webmon_message='Saved analysis restored. Results remain conditional; no counting recommendation is authorized.'
        except (OSError,ValueError,KeyError) as e:s.webmon_message='Cannot restore saved analysis: '+str(e)
        s.flush()

    async def webmon_cancel(self,**_):
        self.webmon_pause_stream()
        self._webmon_cancelled=True
        if getattr(self,'_webmon_remote_cancel',None):
            self._webmon_remote_cancel.set()
        if self._webmon_is_local():
            self.server.state.webmon_calculation_status='Cancelled'
        else:
            self.server.state.webmon_calculation_status='Stopped waiting'
            self.server.state.webmon_message='Stopped waiting. The submitted Galaxy job continues and remains recoverable; stop it in History if required.'
        await stop_process(self._webmon_process)

    async def webmon_shutdown(self):
        self.webmon_pause_stream()
        self._webmon_cancelled=True
        if getattr(self,'_webmon_remote_cancel',None):self._webmon_remote_cancel.set()
        if self._webmon_poll_task and not self._webmon_poll_task.done():
            self._webmon_poll_task.cancel()
            await asyncio.gather(self._webmon_poll_task,return_exceptions=True)
        if self._webmon_task and not self._webmon_task.done():
            self._webmon_task.cancel()
            await asyncio.gather(self._webmon_task,return_exceptions=True)
        await stop_process(self._webmon_process)
        self._webmon_process=None;self._webmon_payloads={};self._webmon_last_preview=None;self._webmon_start_scope=None;self._webmon_response_cache=None
        self.server.state.webmon_has_result=False;self.server.state.webmon_accepted=False;self.server.state.webmon_context_ready=False
        self.server.state.webmon_calculation_status='Not started'
