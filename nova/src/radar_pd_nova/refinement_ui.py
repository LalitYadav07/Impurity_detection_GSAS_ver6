"""A single integrated refinement workspace; credentials never enter UI state."""
from trame.widgets import client, html, plotly, vuetify3 as v


def select(label,field,items,**kwargs):
    return v.VSelect(label=label,v_model=(field,),items=(items,),density='compact',variant='outlined',hide_details=True,**kwargs)


def refinement_dashboard(app):
    with html.Section(classes='rf-workspace'):
        client.Style('''
        .rf-workspace{color:#183c31;padding:22px 0;max-width:1680px;margin:auto}
        .rf-heading,.rf-toolbar,.rf-actions,.rf-stats{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
        .rf-heading{justify-content:space-between;margin-bottom:18px}.rf-heading h2{font-size:27px;margin:0 0 5px}
        .rf-heading p,.rf-muted{color:#64766d;font-size:13px;line-height:1.6;margin:0}
        .rf-stats{gap:28px;margin:16px 0 20px}.rf-stat strong{font-size:24px;font-weight:650;display:block;color:#163d30}.rf-stat span{font-size:12px;color:#64766d}
        .rf-grid{display:grid;grid-template-columns:minmax(300px, .9fr) minmax(450px,1.6fr);gap:20px;align-items:start}
        .rf-card{border:1px solid #dbe5de;border-radius:10px;background:white;padding:22px;min-width:0;box-shadow:0 2px 8px #183c3105}
        .rf-card h3{margin:0 0 14px;font-size:18px}.rf-card h4{margin:0;font-size:15px}.rf-fields{display:grid;gap:16px}
        .rf-toolbar{margin:16px 0}.rf-toolbar>.v-input{min-width:175px;flex:1}.rf-actions{margin-top:18px}
        .rf-subtitle{font-size:11px;letter-spacing:1.2px;text-transform:uppercase;color:#61786a;margin-bottom:10px;font-weight:650}
        .rf-stage{border-left:3px solid #31876a;padding:0 0 18px 15px;margin:20px 0 0}.rf-stage ul{margin:8px 0 8px 17px;color:#28493b;font-size:13px;line-height:1.7}.rf-stage summary{font-size:12px;cursor:pointer;color:#577367}
        .rf-source{background:#f4f8f5;border-radius:6px;padding:12px;font-size:12px;line-height:1.7}.rf-source b{font-weight:600;color:#3d6251}
        .rf-results{display:grid;grid-template-columns:300px minmax(0,1fr);gap:20px;align-items:start}.rf-list{max-height:730px;overflow:auto;margin-top:14px}
        .rf-scan{display:block;width:100%;text-align:left;background:white;border:1px solid #dce7df;border-radius:7px;padding:13px;margin-bottom:9px;cursor:pointer;line-height:1.6;color:#214633}.rf-scan:hover{background:#f4f8f5}.rf-scan.active{border-color:#157347;background:#eef8f1;box-shadow:inset 3px 0 #157347}
        .rf-scan strong{font-size:13px;display:block}.rf-scan span{font-size:11px;display:block;color:#6c7c71}.rf-scan .rf-state{font-weight:600;color:#406451}
        .rf-result-title{display:flex;align-items:start;justify-content:space-between;gap:12px}.rf-result-title h3{margin:0 0 6px}.rf-plot{height:620px;width:100%}.rf-trend{height:450px;width:100%}
        .rf-empty{padding:54px 25px;text-align:center;border:1px dashed #c7d8cb;border-radius:8px;color:#667e6d;line-height:1.8}
        .rf-workspace .v-btn{letter-spacing:.3px}.rf-notice{font-size:12px;color:#62766a;line-height:1.6;margin:8px 0}.rf-tabs{margin:15px 0;border-bottom:1px solid #dce6df}
        .rf-overlay{position:fixed;inset:80px 18px 28px;z-index:2400;background:white;padding:22px;box-shadow:0 0 0 25px #102e2988;border-radius:10px;overflow:auto}
        @media(max-width:1100px){.rf-grid{grid-template-columns:1fr}.rf-results{grid-template-columns:240px minmax(0,1fr)}.rf-card{padding:17px}}
        @media(max-width:760px){.rf-results{grid-template-columns:1fr}.rf-list{max-height:220px}.rf-heading h2{font-size:23px}.rf-toolbar{gap:16px}}
        ''')
        with html.Div(classes='rf-heading'):
            with html.Div():
                html.H2('Refinement')
                html.P('Review a recipe once. Apply it to selected RADAR-PD projects as scans become available.')
            with html.Div(classes='rf-actions',style='margin-top:0'):
                v.VChip('{{ rf_provider }}',variant='tonal',color='#15543c',size='small')
                v.VBtn('Refresh projects',click=app.refresh_refinement_sources,loading=('rf_loading',),variant='text',prepend_icon='mdi-refresh',size='small')
        v.VProgressLinear(v_if='rf_loading || rf_busy',indeterminate=True,color='#287654',height=3,classes='mb-4')
        v.VAlert(v_if='!!rf_error',text=('rf_error',),type='error',variant='tonal',density='compact',classes='mb-3',closable=True)
        with html.Div(classes='rf-toolbar'):
            select('Experiment','rf_experiment','rf_experiments')
            with v.VBtnToggle(v_model=('rf_mode',),mandatory=True,color='#15543c',variant='outlined',divided=True):
                v.VBtn('Recipe',value='recipe',prepend_icon='mdi-playlist-edit')
                v.VBtn('Results',value='results',prepend_icon='mdi-chart-line')
        with html.Div(classes='rf-stats'):
            with html.Div(v_for='metric in rf_metrics',key='metric.label',classes='rf-stat'):
                html.Strong('{{ metric.value }}');html.Span('{{ metric.label }}')
            html.Span('{{ rf_transport }}',classes='rf-muted')
            v.VChip('Future matching scans enrolled',v_if='rf_future_armed',color='#157347',size='small',variant='tonal',prepend_icon='mdi-access-point')
        with html.Div(v_if='rf_jobs.length',classes='mb-4'):
            html.P('{{ rf_progress_text }}',classes='rf-notice')
            v.VProgressLinear(model_value=('rf_progress_percent',),height=6,color='#157347',rounded=True)
        with html.Div(v_if='!rf_experiments.length && !rf_loading',classes='rf-empty'):
            html.H3('Start with a completed RADAR-PD fit')
            html.P('Published GPX projects from this history will appear here, including scans recovered from an earlier session.')
        with html.Div(v_show="rf_mode === 'recipe' && rf_experiments.length",classes='rf-grid'):
            with html.Div(classes='rf-card'):
                html.Div('1 · Choose projects',classes='rf-subtitle');html.H3('Execution scope')
                with html.Div(classes='rf-fields'):
                    select('Apply recipe to','rf_scope',"[{title:'All available run results',value:'all'},{title:'Selected samples',value:'samples'},{title:'Selected scans',value:'scans'}]",update_modelValue=app.refinement_policy_edited)
                    select('Samples','rf_samples','rf_sample_options',v_if="rf_scope === 'samples'",multiple=True,chips=True,closable_chips=True,update_modelValue=app.refinement_policy_edited)
                    v.VAutocomplete(label='Scans',v_model=('rf_scans',),items=('rf_scan_options',),v_if="rf_scope === 'scans'",multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,update_modelValue=app.refinement_policy_edited)
                    select('Banks · empty means all','rf_banks','rf_bank_options',multiple=True,chips=True,closable_chips=True,clearable=True,update_modelValue=app.refinement_policy_edited)
                v.VCheckbox(label='Include available scans at approval',v_model=('rf_existing',),density='compact',hide_details=True,classes='mt-3',update_modelValue=app.refinement_policy_edited)
                v.VCheckbox(label='Enroll future matching scans',v_model=('rf_future',),density='compact',hide_details=True,update_modelValue=app.refinement_policy_edited)
                html.P('Enrollment follows this NOVA session. Submitted Galaxy jobs continue independently.',classes='rf-notice')
                html.P('{{ rf_scope_summary }}',classes='rf-notice')
                html.Hr(style='border:0;border-top:1px solid #e0e9e2;margin:20px 0')
                html.Div('2 · Describe the refinement',classes='rf-subtitle')
                select('Representative GPX','rf_source','rf_sources',update_modelValue=app.refinement_policy_edited)
                with html.Div(v_if='!!rf_source',classes='rf-source mt-3'):
                    with html.Div(v_for='item in rf_source_details',key='item.label'):
                        html.B('{{ item.label }}: ');html.Span('{{ item.value }}')
                v.VBtn('Inspect starting model',click=app.inspect_refinement_project,disabled=('rf_busy || !rf_source',),variant='text',size='small',prepend_icon='mdi-magnify',classes='mt-2')
                with html.Details(v_if='rf_context_atoms.length',classes='mt-2'):
                    html.Summary('Phases and atom labels ({{ rf_context_atoms.length }} sites)',style='cursor:pointer;font-size:13px')
                    v.VTextField(label='Find phase or atom',v_model=('rf_context_search',),density='compact',variant='outlined',hide_details=True,classes='my-3')
                    v.VDataTable(items=('rf_context_atoms',),search=('rf_context_search',),headers=("[{title:'Phase',key:'phase'},{title:'Atom',key:'atom'},{title:'Element',key:'element'},{title:'ADP',key:'adp'},{title:'Position freedoms',key:'position_dof'}]",),items_per_page=5,density='compact')
                v.VTextarea(label='Refinement request',v_model=('rf_request',),placeholder='Name the phases and exact atom labels. Refine background and scales, then cells and only symmetry-allowed positions. Keep occupancies fixed; release one Uiso site at a time if needed.',rows=7,auto_grow=True,variant='outlined',hide_details=True,classes='mt-4',update_modelValue=app.refinement_policy_edited)
                v.VBtn('Load staged atom + Uiso example',click=app.load_atomic_refinement_example,variant='text',size='small',prepend_icon='mdi-text-box-outline',classes='mt-2')
                html.P('Specify stages, phase names, atoms, constraints and anything that must remain fixed. Revisions replace approval and pause future enrollment.',classes='rf-notice')
                v.VBtn("{{ rf_cards.length ? 'Revise recipe' : 'Generate recipe' }}",click=app.generate_refinement_recipe,disabled=('rf_busy || !rf_source || !rf_request.trim()',),loading=('rf_busy',),color='#15543c',variant='flat',block=True,prepend_icon='mdi-auto-fix',classes='mt-4')
            with html.Div(classes='rf-card'):
                html.Div('3 · Review and approve',classes='rf-subtitle');html.H3('Refinement stages')
                with html.Div(v_if='rf_recipe_versions.length',classes='rf-toolbar'):
                    select('Saved recipe','rf_saved_recipe','rf_recipe_versions')
                    v.VBtn('Use saved recipe',click=app.reuse_refinement_recipe,disabled=('rf_busy || !rf_saved_recipe || !rf_source',),variant='outlined',size='small')
                html.P('{{ rf_active_summary }}',v_if='!!rf_active_summary',classes='rf-notice')
                html.P('{{ rf_message }}',v_if='!!rf_message',classes='rf-muted')
                with html.Div(v_if='!rf_cards.length && !rf_busy && !rf_questions.length',classes='rf-empty mt-4'):
                    html.H3('Your staged plan will appear here')
                    html.P('Inspect the exact parameters released at each stage before any scan is polished.')
                v.VAlert(v_for='question in rf_questions',key='question',text=('question',),type='warning',variant='tonal',density='compact',classes='mt-3')
                with html.Div(v_for='stage in rf_cards',key='stage.id',classes='rf-stage'):
                    html.H4('{{ stage.number }}. {{ stage.title }}')
                    html.P('{{ stage.cycles }} cycles · refine:',classes='rf-notice')
                    with html.Ul():html.Li('{{ target }}',v_for='target in stage.free',key='target')
                    with html.Details(v_if='stage.held.length || stage.edits.length'):
                        html.Summary('Parameters fixed since previous stage and explicit edits')
                        with html.Ul():html.Li('Fix: {{ target }}',v_for='target in stage.held',key='target')
                        html.P('{{ edit }}',v_for='edit in stage.edits',key='edit',classes='rf-notice')
                with html.Details(v_if='rf_assumptions.length',classes='mt-3'):
                    html.Summary('Plan assumptions',style='cursor:pointer;font-size:13px')
                    with html.Ul(classes='rf-notice'):html.Li('{{ item }}',v_for='item in rf_assumptions',key='item')
                html.P('Each stage declares its complete free parameter set. Other ordinary refinement flags are held; original constraints and restraints remain.',v_if='rf_cards.length',classes='rf-notice mt-4')
                html.P('{{ rf_plan_calls }}',v_if='!!rf_plan_calls',classes='rf-notice')
                with html.Div(v_if='!!rf_model_tokens',classes='rf-source mt-3'):
                    html.Div('Model usage',classes='rf-subtitle')
                    html.P('{{ rf_model_tokens }}',classes='rf-notice')
                    html.P('{{ rf_model_cost }}',v_if='!!rf_model_cost',classes='rf-notice')
                    with html.Details(v_if='rf_model_has_luna_price',classes='rf-notice'):
                        html.Summary('Estimate basis',style='cursor:pointer')
                        html.P('Public GPT-6 Luna Standard rates on 2026-10-28: $0.10 per million ordinary input tokens, $0.01 cached input, $0.125 cache writes, and $0.50 output. Requests above 272,000 input tokens use the published long-context multipliers. This is not an ORNL Azure billing record.')
                        html.A('Published Luna rates',href='https://developers.openai.com/api/docs/models/gpt-6-luna',target='_blank',rel='noopener noreferrer')
                v.VAlert(v_if='rf_approved',text=('rf_recipe_summary',),type='success',variant='tonal',density='compact',classes='mt-3')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Approve recipe and scope',click=(app.approve_refinement_recipe,'[rf_scope_token]'),disabled=('!rf_plan_ready || rf_busy || rf_approved',),variant='outlined',color='#15543c',prepend_icon='mdi-check-decagram-outline')
                    v.VBtn('Start refinement',click=app.start_refinement_queue,disabled=('!rf_approved || rf_busy',),variant='flat',color='#15543c',prepend_icon='mdi-play')
                html.P('Future GPXs use this same approved plan only when the model is compatible. Different phase or atom models need review. No model call is made for each scan.',classes='rf-notice mt-4')
        with html.Div(v_show="rf_mode === 'results' && rf_experiments.length",classes='rf-results'):
            with html.Aside(classes='rf-card'):
                html.Div('Scan queue',classes='rf-subtitle')
                v.VTextField(label='Find scan or sample',v_model=('rf_job_search',),density='compact',variant='outlined',hide_details=True,prepend_inner_icon='mdi-magnify')
                select('Recipe','rf_recipe_filter',"[{title:'All recipes',value:'all'}, ...rf_recipe_versions]",classes='mt-3')
                select('Status','rf_job_status',"[{title:'All statuses',value:'all'},{title:'Completed',value:'completed'},{title:'Queued',value:'queued'},{title:'Running',value:'running'},{title:'Needs review',value:'needs_review'},{title:'Different model',value:'needs_recipe'},{title:'Paused',value:'paused'},{title:'Interrupted',value:'interrupted'},{title:'Failed',value:'failed'}]",classes='mt-3')
                v.VCheckbox(label='Follow newly completed results',v_model=('rf_follow',),density='compact',hide_details=True,classes='mt-2')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Pause',click=app.pause_refinement_queue,variant='text',size='small',prepend_icon='mdi-pause')
                    v.VBtn('Resume queued',click=app.resume_refinement_queue,variant='text',size='small',prepend_icon='mdi-play')
                html.P('Pause stops new enrollment. The active job finishes. Resume runs queued scans; future enrollment requires Start refinement.',classes='rf-notice')
                with html.Div(classes='rf-list'):
                    with html.Button(v_for="job in rf_jobs.filter(j => (rf_job_status === 'all' || j.status === rf_job_status) && (rf_recipe_filter === 'all' || j.recipe_ids.includes(rf_recipe_filter)) && (j.scan + ' ' + j.sample).toLowerCase().includes(rf_job_search.toLowerCase()))",key='job.id',classes=("'rf-scan ' + (job.id === rf_selected_job ? 'active' : '')",),click=(app.select_refinement_job,'[job.id]')):
                        html.Strong('{{ job.scan }}');html.Span('{{ job.sample }}');html.Span('Recipe {{ job.recipe }} · {{ job.status.replaceAll("_", " ") }}',classes='rf-state')
                html.P('No refinements queued yet.',v_if='!rf_jobs.length',classes='rf-notice')
            with html.Div(classes='rf-card'):
                with html.Div(classes='rf-result-title'):
                    with html.Div():
                        html.H3('{{ rf_selected_title || "Select a scan" }}');html.P('{{ rf_selected_message }}',classes='rf-muted')
                    v.VBtn('Retry transport',v_if="rf_jobs.some(j => j.id === rf_selected_job && ['failed','interrupted'].includes(j.status))",click=app.retry_refinement_job,variant='outlined',size='small')
                with v.VTabs(v_model=('rf_result_tab',),color='#15543c',classes='rf-tabs'):
                    v.VTab('Fit',value='fit');v.VTab('Trends',value='trends');v.VTab('Parameters',value='parameters')
                with html.Div(v_show="rf_result_tab === 'fit'"):
                    with html.Div(v_if='rf_has_output'):
                        with html.Div(classes='rf-toolbar'):
                            select('Stage','rf_stage','rf_stages');select('Histogram','rf_histogram','rf_histograms')
                        with html.Div(classes='rf-toolbar'):
                            select('Coordinate','rf_axis',"[{title:'d-spacing (Å)',value:'d'},{title:'Q (Å⁻¹)',value:'Q'},{title:'Native coordinate',value:'native'}]")
                            select('Show fits','rf_display',"[{title:'Original and selected',value:'compare'},{title:'Selected refinement',value:'polished'},{title:'Original RADAR-PD',value:'original'}]")
                            v.VBtn('Reset zoom',click='rf_zoom += 1',variant='text',size='small');v.VBtn('Expand',click='rf_expanded = true',variant='text',size='small',prepend_icon='mdi-arrow-expand')
                        html.P('{{ rf_rwp_summary }}',style='font-size:14px;font-weight:600')
                        v.VAlert(v_for='warning in rf_parameter_warnings',key='warning',text=('warning',),type='warning',variant='tonal',density='compact',classes='mb-3')
                        with html.Div(classes='rf-plot'):
                            fig=plotly.Figure(display_mode_bar=True,responsive=True,style='height:100%;width:100%');fig.data=('rf_fit.data',);fig.layout=('rf_fit.layout',)
                        html.P('{{ rf_fit_notice }}',classes='rf-notice')
                    html.Div('Completed stages will appear here while original RADAR-PD results remain available in Results.',v_if='!rf_has_output',classes='rf-empty')
                with html.Div(v_show="rf_result_tab === 'trends'"):
                    with html.Div(classes='rf-toolbar'):
                        v.VAutocomplete(label='Parameter or plot',v_model=('rf_trend_metric',),items=('rf_trend_metrics',),density='compact',variant='outlined',hide_details=True)
                        select('Horizontal axis','rf_trend_axis','rf_trend_axes')
                    with html.Div(classes='rf-toolbar'):
                        select('Representation','rf_trend_display',"[{title:'Absolute values',value:'absolute'},{title:'Difference from first scan',value:'difference'},{title:'Relative cell change (%)',value:'relative'}]",disabled=("['rwp','phase_fractions'].includes(rf_trend_metric)",))
                    with html.Div(classes='rf-trend'):
                        fig=plotly.Figure(display_mode_bar=True,responsive=True,style='height:100%;width:100%');fig.data=('rf_trend.data',);fig.layout=('rf_trend.layout',)
                    html.P('{{ rf_trend_notice }}',classes='rf-notice')
                    html.P('Traces remain separate by sample, bank and recipe. Missing measured conditions leave gaps. Relative percent is supported for cell lengths and volume.',classes='rf-notice')
                with html.Div(v_show="rf_result_tab === 'parameters'"):
                    v.VTextField(label='Find phase, atom or parameter',v_model=('rf_parameter_search',),density='compact',variant='outlined',hide_details=True,classes='mb-3')
                    v.VDataTable(items=('rf_parameters',),search=('rf_parameter_search',),headers=("[{title:'Parameter',key:'label'},{title:'Value',key:'value_display'},{title:'Unit',key:'unit'},{title:'Conditional s.u.',key:'sigma_display'},{title:'Status',key:'status_display'},{title:'Stage',key:'stage'}]",),density='compact',items_per_page=12)
                    html.P('Phase HAP scales are not weight fractions. Crystalline weight values are shown only when derived by GSAS-II. Earlier-stage uncertainties retain their stage provenance.',classes='rf-notice')
                with html.Div(v_if='rf_downloads.length',classes='rf-toolbar',style='border-top:1px solid #e0e9e2;padding-top:18px'):
                    select('Verified output','rf_download','rf_downloads')
                    v.VBtn('Download',click=app.download_refinement_artifact,variant='outlined',size='small',prepend_icon='mdi-download')
                    v.VBtn('{{ rf_open_gsas_label }}',v_if='rf_can_open_gsas',click=app.open_refinement_in_gsasii,variant='outlined',size='small')
                html.P('{{ rf_output_provenance }}',v_if='!!rf_output_provenance',classes='rf-notice')
        with html.Details(v_if='rf_source_warnings.length',classes='mt-5'):
            html.Summary('Projects needing attention ({{ rf_source_warnings.length }})',style='cursor:pointer;color:#8b6028;font-size:13px')
            v.VDataTable(items=('rf_source_warnings',),headers=("[{title:'Scan',key:'scan'},{title:'Reason',key:'reason'}]",),items_per_page=5,density='compact')
        with html.Div(v_if='rf_expanded',classes='rf-overlay'):
            v.VBtn('Close expanded plot',click='rf_expanded = false',variant='text',prepend_icon='mdi-arrow-collapse')
            with html.Div(style='height:calc(100vh - 220px)'):
                fig=plotly.Figure(display_mode_bar=True,responsive=True,style='height:100%;width:100%');fig.data=('rf_fit.data',);fig.layout=('rf_fit.layout',)
