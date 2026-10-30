"""Compact portfolio navigation within the existing Refinement workspace."""
from trame.widgets import client,html,plotly,vuetify3 as v


def select(label,field,items,**kwargs):
    return v.VSelect(label=label,v_model=(field,),items=(items,),density='compact',variant='outlined',hide_details=True,**kwargs)


def model_exploration(app):
    from .symmetry_controller import MODEL_FIELDS
    form='[{' + ','.join(name+':'+name for name in (*MODEL_FIELDS,'sx_start_source','rf_experiment','sx_plan_token')) + '}]'
    with html.Div(v_show="rf_mode === 'models'"):
        client.Style('''
        .sx-layout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:18px;align-items:start}
        .sx-models{max-height:740px;overflow:auto;margin-top:12px}.sx-model{width:100%;text-align:left;border:1px solid #dce7df;border-radius:7px;padding:12px;margin:0 0 8px;background:white;color:#234839;cursor:pointer}
        .sx-model.active{background:#eff8f2;border-color:#277754;box-shadow:inset 3px 0 #277754}.sx-model b{display:block;font-size:14px}.sx-model span{font-size:12px;display:block;margin-top:4px;color:#64796b}
        .sx-fit{min-width:0;width:100%;height:570px}.sx-overview{min-width:0;width:100%;height:300px}.sx-evidence{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.sx-evidence p{font-size:12px;line-height:1.7;overflow-wrap:anywhere}
        .sx-settings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:18px 0}.sx-setup{margin-bottom:20px}.sx-status{padding:12px 16px;background:#f2f6f3;border-radius:6px;font-size:13px;line-height:1.8;margin-bottom:18px}
        .sx-choices .v-field__input{max-height:92px;overflow-y:auto}.sx-choices .v-chip{max-width:95%}
        @media(max-width:850px){.sx-layout,.sx-evidence,.sx-settings{grid-template-columns:1fr}.sx-models{max-height:260px}}
        ''')
        with html.Div(classes='rf-toolbar'):
            select('Model campaign','sx_campaign','sx_campaigns')
            select('Results to show','sx_result_scope',"[{title:'Selected experiment',value:'experiment'},{title:'Whole campaign',value:'campaign'}]")
            v.VBtn('Refresh evidence',click=app.refresh_symmetry,loading=('sx_busy',),variant='text',prepend_icon='mdi-refresh')
            v.VBtn('Continue saved campaign',click=app.continue_symmetry_campaign,disabled=('!sx_resumable || sx_busy || sx_campaign_running',),variant='text',prepend_icon='mdi-play')
            v.VBtn('Pause campaign',click=app.pause_model_campaign,disabled=('!sx_campaign_running',),variant='text',prepend_icon='mdi-pause')
            v.VBtn('New model search',v_if='!!sx_campaign',click='sx_setup = !sx_setup',variant='outlined',prepend_icon='mdi-molecule')
        v.VAlert(v_if='!!sx_error',text=('sx_error',),type='error',variant='tonal',density='compact',classes='mb-3')
        html.P('{{ sx_backend }}',classes='rf-notice')
        html.P('{{ sx_queue_notice }}',v_if='!!sx_queue_notice',classes='rf-notice')
        html.P('Pause or end any other viewer for this campaign before continuing here.',v_if='sx_resumable && !sx_campaign_running',classes='rf-notice')
        with html.Details(classes='rf-card mb-3'):
            html.Summary('Compute budget')
            with html.Div(classes='sx-settings'):
                select('Concurrent Galaxy batches','sx_parallel_jobs','[1,2]')
                select('Native workers requested per batch','sx_slots','[1,2,4]')
            html.P('Each batch contains many independent starts. Galaxy controls admission and CPU/RAM allocation; an unreported allocation permits one native worker per job. The batch limit includes model preparation jobs in this History. Lowering the limit lets accepted jobs finish before submitting more.',classes='rf-notice')
        v.VProgressLinear(v_if='sx_busy',indeterminate=True,color='#267653',height=3,classes='mb-3')
        html.P('{{ sx_message }}',classes='rf-notice',raw_attrs=['aria-live="polite"'])
        html.P('Qualified fits appear first when available. All failed, unresolved and physically rejected trials remain inspectable.',v_if='sx_groups.length',classes='rf-notice')
        with html.Div(v_if='sx_setup || !sx_campaign',classes='rf-card sx-setup'):
            html.H3('Explore models from a completed GPX')
            html.P('Enter a commensurate k vector, or use the magnetic k survey below. Review hypotheses before native refinement.',classes='rf-notice')
            with html.Div(classes='rf-toolbar'):
                select('Starting GPX · original or recipe checkpoint','sx_start_source','sx_start_sources')
                v.VBtn('Inspect exact parent',click=app.inspect_symmetry_source,disabled=('!sx_start_source || sx_busy',),variant='outlined')
            with html.Details(v_if='sx_has_context',classes='mt-3'):
                html.Summary('Describe a magnetic model search')
                html.P('Describe magnetic sites, ions and physical bounds. If k is unknown, request a data-based k survey; review its candidates before discovery. State whether moments may cant and whether to refine a shared nuclear/magnetic intensity scale.',classes='rf-notice')
                v.VTextarea(label='Magnetic refinement request',v_model=('sx_request',),rows=3,auto_grow=True,variant='outlined',hide_details=True,classes='mt-3')
                v.VBtn('Draft search settings',click=app.plan_symmetry_request,disabled=('sx_busy || !sx_request',),variant='outlined',classes='mt-3')
                html.P('{{ sx_planning_summary }}',v_if='!!sx_planning_summary',classes='rf-notice')
                html.P('{{ sx_planning_usage }}',v_if='!!sx_planning_usage',classes='rf-notice')
                html.P('{{ sx_planning_cost }}',v_if='!!sx_planning_cost',classes='rf-notice')
                html.P('{{ note }}',v_for='note in sx_planning_notes',classes='rf-notice')
            with html.Div(v_if='sx_has_context',classes='sx-settings'):
                select('Exploration','sx_kind',"[{title:'Magnetic models',value:'magnetic'},{title:'Structural displacement modes',value:'structural'}]")
                select('Moment model','sx_moment_model',"[{title:'Collinear moments',value:'collinear'},{title:'Symmetry-allowed components · can cant',value:'symmetry_allowed'}]",v_if="sx_kind === 'magnetic'")
                select('Nuclear parent','sx_parent','sx_parents')
                select('Histogram / bank','sx_histogram','sx_histograms')
                with html.Div():
                    v.VTextField(label='k in this parent basis · rational components',v_model=('sx_k',),placeholder='0, 0, 1/2',density='compact',variant='outlined',hide_details=True)
                    v.VBtn('Find candidate k-vectors',v_if="sx_kind === 'magnetic'",click=app.find_candidate_k,disabled=('sx_busy || !sx_sites.length',),variant='text',size='small',classes='mt-1',prepend_icon='mdi-magnify')
                select('Magnetic parent sites','sx_sites','sx_site_options',v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Ionic form factors · atom=ion; atom=ion',v_model=('sx_form_factors',),placeholder='Cr1=Cr+3',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Landé g · number or atom=value; atom=value',v_model=('sx_lande',),type='text',density='compact',variant='outlined',hide_details=True)
                select(("sx_moment_model === 'symmetry_allowed' ? 'Dominant parent-axis starts' : 'Finite child-axis starts'",),'sx_directions',"[{title:'x',value:'x'},{title:'y',value:'y'},{title:'z',value:'z'}]",v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                html.P('Each selected axis uses four relative-component sign combinations and moment starts at 25% and 75% of your ceiling. Only site-symmetry-allowed components vary. This finite set does not exhaust continuous orientations or signs between split child orbits.',v_if="sx_kind === 'magnetic' && sx_moment_model === 'symmetry_allowed'",classes='rf-notice')
                v.VCheckbox(v_if="sx_kind === 'magnetic'",label='Test relative initial signs between selected parent sites',v_model=('sx_sign_starts',),density='compact',hide_details=True)
                with html.Div(v_if="sx_kind === 'magnetic'"):
                    v.VCheckbox(label='Refine shared nuclear/magnetic intensity scale',v_model=('sx_refine_scale',),density='compact',hide_details=True)
                    v.VCheckbox(label='Refine existing background coefficients',v_model=('sx_refine_background',),density='compact',hide_details=True)
                    html.P('Consider these when the starting nuclear-only fit absorbed magnetic intensity. One common scale keeps phase amounts linked; the background basis and coefficient count stay fixed. The same scope applies to every candidate.',classes='rf-notice')
                v.VTextField(label='Experiment-specific adequate Rwp (%)',v_model=('sx_adequate',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Maximum moment (µB)',v_model=('sx_max_moment',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Magnetic d cutoff (Å) · 0 uses the full fit range',v_model=('sx_mag_d_min',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'structural'",label='Maximum normalized mode (Å)',v_model=('sx_max_mode',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'structural'",label='Signed mode start · ± amplitude (Å)',v_model=('sx_start_mode',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                select('Structural candidate scope','sx_opd_policy',"[{title:'Highest symmetry per irrep',value:'highest-symmetry-per-irrep'},{title:'All discovered OPDs',value:'all'}]",v_if="sx_kind === 'structural'")
                select('Maximum children to construct','sx_construct_limit','[16,50,100,250,500]',v_if="sx_kind === 'structural'")
                v.VTextField(label='Cycles per start',v_model=('sx_cycles',),type='text',inputmode='numeric',density='compact',variant='outlined',hide_details=True)
            with html.Details(v_if="sx_has_context && sx_kind === 'magnetic'",classes='mt-3'):
                html.Summary('K-survey scope and time budget')
                html.P('Select magnetic sites and ionic form factors first. The survey tests rational high-symmetry k points and a finite sample of supported collinear templates. Other commensurate vectors or structures can remain untested.',classes='rf-notice')
                with html.Div(classes='sx-settings'):
                    select('Survey budget · seconds','sx_k_budget','[30,60,120,300]')
                    select('Native templates per k','sx_k_limit','[3,6,12,24,48]')
                    select('K-survey requested workers','sx_k_workers','[1,2,4]')
                html.P('The time budget stops new work; cleanup of a running native process can add a few seconds. Catalogs are reused; profiles are bound to the exact GPX revision and settings.',classes='rf-notice')
            html.P('{{ sx_k_message }}',v_if='!!sx_k_message',classes='rf-notice mt-3')
            v.VBtn('Stop k survey',v_if='sx_k_busy',click=app.cancel_k_search,variant='outlined',size='small')
            with html.Div(v_if='sx_k_has_result',classes='mt-3'):
                html.H4('Candidate propagation vectors')
                v.VDataTable(items=('sx_k_rows',),headers=("[{title:'k · parent basis',key:'k'},{title:'Screening score',key:'score'},{title:'Profiles sampled',key:'sampled'},{title:'Catalog embeddings',key:'catalog'},{title:'Coverage',key:'status'}]",),items_per_page=8,density='compact')
                with html.Details(v_if='sx_k_notes.length',classes='mt-3'):
                    html.Summary('Survey coverage details')
                    html.P('{{ note }}',v_for='note in sx_k_notes',classes='rf-notice')
                with html.Div(classes='rf-toolbar mt-3'):
                    select('Inspect k family','sx_k_selected','sx_k_choices')
                    select('Pattern coordinate','sx_k_axis',"[{title:'Q (Å⁻¹)',value:'q'},{title:'d (Å)',value:'d'},{title:'Native coordinate',value:'native'}]")
                    v.VBtn('Reset zoom',click=app.reset_k_search_zoom,variant='text',size='small')
                    v.VBtn('Use selected k',click=app.use_candidate_k,disabled=('sx_busy || !sx_k_selected',),variant='outlined',color='#15543c')
                with html.Div(style='height:440px;width:100%;min-width:0'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,scroll_zoom=True,style='height:100%;width:100%');fig.data=('sx_k_plot.data',);fig.layout=('sx_k_plot.layout',)
                html.P('{{ sx_k_detail }}',classes='rf-notice')
                html.P('Screening overlay: the nuclear model is fixed and only one magnetic intensity amplitude is fitted. No Rietveld refinement runs in this survey. A leading k, a tie or a missing template does not establish magnetic order. Use a candidate to review a full refinement plan.',classes='rf-notice')
            html.P("{{ sx_kind === 'structural' ? 'Discovery sends the selected parent structure, without pattern arrays, to ISODISTORT. The selected OPD scope and construction budget are explicit; deferred or failed models remain visible. A budget smaller than the selected tier requires a narrower scope.' : 'Discovery sends parent symmetry operations and k to Bilbao. Refinement consumes the cached catalog offline.' }}",v_if='sx_has_context',classes='rf-notice')
            html.P('Review and use a k-survey candidate, or enter an exact rational k, before discovering models.',v_if="sx_has_context && !sx_k.trim()",classes='rf-notice')
            v.VBtn('Discover finite candidates',v_if='sx_has_context',click=app.discover_symmetry_candidates,disabled=('sx_busy || !sx_k.trim()',),variant='outlined',color='#15543c')
            with html.Div(v_if='sx_discovered',classes='mt-4'):
                html.P('{{ sx_discovery_scope }}',classes='rf-notice')
                v.VBtn('Include all catalog embeddings',click='sx_candidates = sx_candidate_options.map(c => c.value)',variant='text',size='small')
                v.VAutocomplete(label='Constructed models to explore',v_model=('sx_candidates',),items=('sx_candidate_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,classes='sx-choices',update_modelValue=app._symmetry_unapprove)
                with html.Details(v_if="sx_kind === 'structural'",classes='mt-3'):
                    html.Summary('Select native displacement modes')
                    v.VAutocomplete(label='Modes · every selected mode gets both start signs',v_model=('sx_modes',),items=('sx_mode_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,classes='mt-3 sx-choices',update_modelValue=app._symmetry_unapprove)
                with html.Div(classes='rf-toolbar mt-3'):
                    select('Acquisitions to enroll','sx_existing',"[{title:'Selected scan only',value:'selected'},{title:'Published scans for this sample and bank',value:'sample'}]")
                    app._number_field(label='Maximum acquisitions',v_model=('sx_max_scans',),min=1,max=2000,density='compact',variant='outlined')
                v.VCheckbox(label='Follow new matching GPXs while this session is open',v_model=('sx_follow_new',),density='compact')
                html.P('Every new acquisition is checked against the reviewed parent, sample, bank and recipe. Incompatible inputs wait for review.',classes='rf-notice')
                v.VBtn('Build finite plan',click=(app.build_symmetry_plan,form),disabled=('sx_busy || sx_campaign_running || !sx_candidates.length || !sx_adequate',),variant='outlined',color='#15543c',classes='mt-3')
            with html.Div(v_if='sx_plan_ready',classes='mt-4'):
                html.P('{{ sx_plan_summary }}',classes='rf-notice')
                with html.Details():
                    html.Summary('Review every embedding ({{ sx_plan_candidates.length }})')
                    v.VDataTable(items=('sx_plan_candidates',),headers=("[{title:'Model',key:'name'},{title:'BNS',key:'bns'},{title:'Basis',key:'basis'},{title:'Origin',key:'origin'},{title:'Free modes',key:'modes'}]",),items_per_page=8,density='compact')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Approve finite plan',click=(app.approve_symmetry,form),disabled=('sx_approved || sx_campaign_running',),variant='outlined',color='#15543c')
                    v.VBtn('Start approved campaign',click=(app.start_symmetry_campaign,form),disabled=('!sx_approved || sx_busy || sx_campaign_running',),variant='flat',color='#15543c')
                html.P('Each start uses a fresh native process. Galaxy jobs share immutable inputs and batch independent starts within allocated CPU/RAM. Missing allocation information limits the pool to one worker. Model calls run only during setup.',classes='rf-notice')
                html.P('Structural fit comparisons require a matched parent/nuisance refinement. A lower Rwp alone does not establish a transition.',v_if="sx_kind === 'structural'",classes='rf-notice')
        html.Div('{{ sx_progress }}',v_if='sx_results_current && !!sx_progress',classes='sx-status')
        html.P('Dispatched starts share a Galaxy job, which can be queued or running. The native worker count is reported after the first batch; it is separate from the number of dispatched starts.',
            v_if='sx_results_current && sx_campaign_running',classes='rf-notice')
        with html.Details(v_if='sx_enrollments.length',classes='rf-card mb-4'):
            html.Summary('Acquisition queue ({{ sx_enrollments.length }}) · {{ sx_following ? "following new scans" : "captured scans" }}')
            v.VDataTable(items=('sx_enrollments',),headers=("[{title:'Scan',key:'scan'},{title:'Bank',key:'bank'},{title:'State',key:'status'},{title:'Review needed',key:'reason'}]",),items_per_page=8,density='compact')
        with html.Details(v_if='sx_results_current && sx_coverage.length > 1',classes='rf-card mb-4'):
            html.Summary('Scans and search coverage ({{ sx_coverage.length }} hypotheses)')
            v.VDataTable(items=('sx_coverage',),headers=("[{title:'Scan',key:'scan'},{title:'Condition',key:'condition'},{title:'Bank',key:'bank'},{title:'Magnetic sites',key:'sites'},{title:'Terminal starts',key:'coverage'},{title:'Qualified starts',key:'qualified'},{title:'Best qualified Rwp',key:'best_qualified'},{title:'Best audited Rwp',key:'best_audited'},{title:'Status',key:'status'}]",),items_per_page=8,density='compact',classes='mt-3')
        with html.Div(v_if='sx_results_current && sx_source_options.length > 1',classes='rf-toolbar'):
            select('Compare models for this exact scan / bank','sx_source','sx_source_options',disabled=('sx_busy',))
        with html.Div(v_if='sx_results_current && sx_groups.length',classes='sx-layout'):
            with html.Aside(classes='rf-card'):
                html.H3('Calculated-pattern groups')
                html.P('Grouped only within the same source and fit domain. Expand a group to inspect each model and start.',classes='rf-notice')
                v.VTextField(label='Find model or status',v_model=('sx_search',),density='compact',variant='outlined',hide_details=True,classes='mt-3')
                with html.Div(classes='sx-models'):
                    with html.Button(v_for="group in sx_groups.filter(g => (g.title+' '+g.status).toLowerCase().includes(sx_search.toLowerCase())).filter((g,i) => i < sx_group_limit || g.id === sx_group)",key='group.id',classes=("['sx-model', {active: sx_group === group.id}]",),click=(app.select_symmetry_group,'[group.id]')):
                        html.B('{{ group.title }}')
                        html.Span('{{ group.rwp }} · {{ group.status }}')
                        html.Span('{{ group.count }} model / start records')
                v.VBtn('Show 12 more groups',v_if='sx_groups.length > sx_group_limit',click='sx_group_limit += 12',variant='text',size='small')
                html.P('{{ sx_groups.length }} groups; every model and disposition remains inspectable.',classes='rf-notice')
            with html.Div(classes='rf-card'):
                html.H3('Selected model fit')
                with html.Div(classes='rf-toolbar'):
                    v.VAutocomplete(label='Model / start in selected group',v_model=('sx_selected',),items=('sx_members',),density='compact',variant='outlined',hide_details=True,update_modelValue=app._symmetry_user_selected)
                with html.Div(classes='rf-toolbar'):
                    select('Horizontal coordinate','sx_axis',"[{title:'d spacing (Å)',value:'d'},{title:'Q (Å⁻¹)',value:'q'},{title:'Native coordinate',value:'native'}]")
                    select('Intensity display','sx_intensity_scale',"[{title:'Linear',value:'linear'},{title:'Log · positive values',value:'log'}]")
                    v.VAutocomplete(label='Pin alternatives · at most two',v_model=('sx_pins',),items=('sx_pin_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True)
                    v.VBtn('Reset zoom',click=app.symmetry_reset_zoom,variant='text',size='small')
                with html.Div(classes='sx-fit'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,scroll_zoom=True,style='height:100%;width:100%');fig.data=('sx_fit.data',);fig.layout=('sx_fit.layout',)
                with html.Div(classes='sx-evidence'):
                    with html.P(v_for='item in sx_details',key='item.label'):
                        html.B('{{ item.label }}: ');html.Span('{{ item.value }}')
                v.VAlert(v_if='!!sx_moment_note',text=('sx_moment_note',),type='warning',variant='tonal',density='compact',classes='my-3')
                v.VDataTable(v_if='sx_parameters.length',items=('sx_parameters',),headers=("[{title:'Phase',key:'phase'},{title:'Moment / normalized mode',key:'parameter'},{title:'Value',key:'value'},{title:'Conditional s.u.',key:'uncertainty'},{title:'Unit',key:'unit'}]",),items_per_page=6,density='compact')
                html.P('The first row for each magnetic atom is its magnitude; mx, my and mz use the candidate crystallographic basis shown in the embedding. Uncertainties are conditional on this model and its fixed nuclear parameters. A dash means fixed, dependent or unavailable; a magnitude uncertainty is not inferred from component errors without their covariance. A nonzero moment or qualified fit does not establish ordering.',v_if='sx_parameters.length',classes='rf-notice')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Open selected GPX in GSAS-II',click=app.open_symmetry_in_gsasii,
                        disabled=("!sx_has_gpx || gsasii_session_status === 'starting' || gsasii_session_status === 'ready'",),
                        loading=("gsasii_session_status === 'starting'",),variant='outlined',size='small')
                    v.VBtn('Open current GSAS-II desktop',href=('gsasii_launch_url',),target='_blank',rel='noopener noreferrer',
                        v_if="gsasii_session_status === 'ready' && !!gsasii_launch_url",variant='flat',color='#15543c',size='small')
                    v.VBtn('Download native GPX',click=(app.download_symmetry_artifact,"['gpx']"),disabled=('!sx_has_gpx',),variant='outlined',size='small',prepend_icon='mdi-download')
                    v.VBtn('Download trial receipt',click=(app.download_symmetry_artifact,"['receipt']"),disabled=('!sx_has_receipt',),variant='text',size='small')
                html.P('{{ gsasii_status_message }}',v_if="gsasii_session_status === 'starting' || gsasii_session_status === 'error'",classes='rf-notice',raw_attrs=['aria-live="polite"'])
                html.P('Desktop target: {{ sx_gsas_model || "another GPX opened in this viewer" }}. Save and exit that desktop before opening a different model.',
                    v_if="gsasii_session_status === 'ready'",classes='rf-notice')
                with html.Details(classes='mt-3'):
                    html.Summary('Trial control')
                    v.VBtn('Stop job containing selected trial',click=app.cancel_symmetry_selected,variant='text',size='small',classes='mt-2')
                    html.P('{{ sx_cancel_notice }}',classes='rf-notice')
                html.H3('Fit quality and model complexity',classes='mt-4')
                with html.Div(classes='sx-overview'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,style='height:100%;width:100%',click=(app.symmetry_overview_click,'[$event]'));fig.data=('sx_overview.data',);fig.layout=('sx_overview.layout',)
                html.P('Open diamonds are inadequate or unresolved fits. A lower residual is evidence to inspect, not a confirmed structure. Log intensity hides nonpositive values for display; the residual stays linear.',classes='rf-notice')
