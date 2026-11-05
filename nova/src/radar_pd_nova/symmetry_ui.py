"""Compact portfolio navigation within the existing Refinement workspace."""
from trame.widgets import client,html,plotly,vuetify3 as v


FIT_PRESENTATION_CSS='''
.sx-source-caveats{flex:0 0 auto;font-size:13px;line-height:1.5}.sx-source-caveats ul{margin:6px 0;padding-left:20px}.sx-source-caveats li+li{margin-top:4px}.sx-source-caveats summary{cursor:pointer;font-size:12px}
.sx-expanded-fit{display:flex;flex-direction:column}.sx-expanded-fit>.v-btn{align-self:flex-start;flex:0 0 auto}.sx-expanded-fit h3{flex:0 0 auto;font-size:16px;line-height:1.5}.sx-expanded-fit .sx-source-caveats{max-height:min(30vh,220px);overflow-y:auto}.sx-expanded-chart{flex:1;min-height:320px}
'''


def select(label,field,items,**kwargs):
    return v.VSelect(label=label,v_model=(field,),items=(items,),density='compact',variant='outlined',hide_details=True,**kwargs)


def scan_select(label,field,items,**kwargs):
    return v.VAutocomplete(label=label,v_model=(field,),items=(items,),
        density='compact',variant='outlined',hide_details=True,
        no_data_text='No matching scan or project',**kwargs)


def source_caveats():
    with v.VAlert(v_if='sx_source_display_notices.length',type='warning',color='#6b4b12',variant='tonal',density='compact',classes='sx-source-caveats my-2',role='note',tabindex=0):
        html.Strong('Source requires review')
        with html.Ul():
            html.Li('{{ notice }}',v_for='notice in sx_source_display_notices',key='notice')
        with html.Details():
            html.Summary('Original source notices')
            with html.Ul():
                html.Li('{{ notice }}',v_for='notice in sx_source_notices',key='notice')


def expanded_fit():
    with html.Div(v_if='sx_expanded',classes='rf-overlay sx-expanded-fit'):
        v.VBtn('Close expanded fit',click='sx_expanded = false',variant='text',prepend_icon='mdi-arrow-collapse')
        html.H3('{{ sx_expanded_title }}',v_if='sx_expanded_title',classes='mt-2 mb-2')
        source_caveats()
        with html.Div(classes='sx-expanded-chart'):
            fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,scroll_zoom=True,style='height:100%;width:100%');fig.data=('sx_fit.data',);fig.layout=('sx_expanded_layout',)


def model_exploration(app):
    from .symmetry_controller import MODEL_FIELDS
    form='[{' + ','.join(name+':'+name for name in (*MODEL_FIELDS,'sx_start_source','rf_experiment','sx_plan_token')) + '}]'
    with html.Div(v_show="rf_mode === 'models'"):
        client.Style('''
        .sx-layout{display:grid;grid-template-columns:290px minmax(0,1fr);gap:18px;align-items:start}
        .sx-models{max-height:740px;overflow:auto;margin-top:12px}.sx-model{width:100%;text-align:left;border:1px solid #dce7df;border-radius:7px;padding:12px;margin:0 0 8px;background:white;color:#234839;cursor:pointer}
        .sx-model.active{background:#eff8f2;border-color:#277754;box-shadow:inset 3px 0 #277754}.sx-model b{display:block;font-size:14px}.sx-model span{font-size:12px;display:block;margin-top:4px;color:#64796b}
        .sx-fit{min-width:0;width:100%;height:570px}.sx-overview{min-width:0;width:100%;height:300px}.sx-evidence{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}.sx-evidence p{font-size:12px;line-height:1.7;overflow-wrap:anywhere}
        .sx-settings{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:18px 0;align-items:start}.sx-settings>.v-input{min-width:0}.sx-bounds{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;align-items:start}.sx-bounds>.v-input{min-width:0}.sx-setup{margin-bottom:20px}.sx-status{padding:12px 16px;background:#f2f6f3;border-radius:6px;font-size:13px;line-height:1.8;margin-bottom:18px}
        .sx-choices .v-field__input{max-height:92px;overflow-y:auto}.sx-choices .v-chip{max-width:95%}
        @media(max-width:850px){.sx-layout,.sx-evidence,.sx-settings,.sx-bounds{grid-template-columns:1fr}.sx-models{max-height:260px}}
        ''')
        client.Style(FIT_PRESENTATION_CSS)
        with html.Div(classes='rf-toolbar'):
            select('Model campaign','sx_campaign','sx_campaigns')
            select('Results to show','sx_result_scope',"[{title:'Selected experiment',value:'experiment'},{title:'Whole campaign',value:'campaign'}]")
            v.VBtn('Refresh evidence',click=app.refresh_symmetry,loading=('sx_busy',),variant='text',prepend_icon='mdi-refresh')
            v.VBtn('Load saved campaigns',click=app.restore_saved_symmetry_campaigns,disabled=('sx_busy || sx_pause_pending',),variant='text',prepend_icon='mdi-history')
            v.VBtn('Continue saved campaign',click=app.continue_symmetry_campaign,disabled=('!sx_resumable || sx_busy || sx_campaign_running || sx_pause_pending',),variant='text',prepend_icon='mdi-play')
            v.VBtn('Pause campaign',click=app.pause_model_campaign,disabled=('!sx_campaign_running || sx_pause_pending',),loading=('sx_pause_pending',),variant='text',prepend_icon='mdi-pause')
            v.VBtn('New model search',v_if='!!sx_campaign',click=app.new_symmetry_search,disabled=('sx_busy || sx_pause_pending',),variant='outlined',prepend_icon='mdi-molecule')
            v.VBtn('Return to campaign',v_if='sx_setup && !!sx_campaign',click='sx_setup = false',variant='text',prepend_icon='mdi-arrow-left')
        html.P('Start a new search from a published GPX below. To inspect or continue an earlier model search, load its saved campaigns; this recovers prior work without submitting new refinements.',classes='rf-notice',v_if='!sx_campaign')
        v.VAlert(v_if='!!sx_error',text=('sx_error',),type='error',variant='tonal',density='compact',classes='mb-3')
        html.P('{{ sx_backend }}',classes='rf-notice')
        html.P('{{ sx_queue_notice }}',v_if='!sx_setup && !!sx_queue_notice',classes='rf-notice')
        html.P('Pause or end any other viewer for this campaign before continuing here.',v_if='!sx_setup && sx_resumable && !sx_campaign_running',classes='rf-notice')
        with html.Details(classes='rf-card mb-3'):
            html.Summary('Compute budget')
            with html.Div(classes='sx-settings'):
                select('Concurrent Galaxy batches','sx_parallel_jobs','[1,2,4,8]')
                select('Native workers requested per batch','sx_slots','[1,2,4]')
            html.P('Galaxy controls admission and CPU/RAM allocation. More batches can process different scans at once; they may remain queued when the cluster is busy. An unreported allocation permits one native worker per job. This History-wide limit includes model preparation jobs. Lowering it lets accepted jobs finish before submitting more.',classes='rf-notice')
        v.VProgressLinear(v_if='sx_busy',indeterminate=True,color='#267653',height=3,classes='mb-3')
        html.P('{{ sx_message }}',classes='rf-notice',raw_attrs=['aria-live="polite"'])
        html.P('Qualified fits appear first when available. All failed, unresolved and physically rejected trials remain inspectable.',v_if='!sx_setup && sx_groups.length',classes='rf-notice')
        with html.Div(v_if='sx_setup || !sx_campaign',classes='rf-card sx-setup'):
            html.H3('Explore models from a completed GPX')
            html.P('Enter a commensurate k vector, or use the magnetic k survey below. Review hypotheses before native refinement.',classes='rf-notice')
            with html.Div(classes='rf-toolbar'):
                scan_select('Starting GPX · original or recipe checkpoint','sx_start_source','sx_start_sources',
                    placeholder='Type a run number, temperature or recipe stage',persistent_placeholder=True)
                v.VBtn('Inspect exact parent',click=app.inspect_symmetry_source,disabled=('!sx_start_source || sx_busy',),variant='outlined')
            with html.Details(v_if="sx_has_context && sx_kind === 'magnetic'",classes='mt-3'):
                html.Summary('Describe a magnetic model search')
                html.P('Describe magnetic sites, ions and physical bounds. If k is unknown, request a data-based k survey; review its candidates before discovery. State whether moments may cant and whether to refine a shared nuclear/magnetic intensity scale.',classes='rf-notice')
                v.VTextarea(label='Magnetic refinement request',v_model=('sx_request',),rows=3,auto_grow=True,variant='outlined',hide_details=True,classes='mt-3')
                v.VBtn('Draft search settings',click=app.plan_symmetry_request,disabled=('sx_busy || !sx_request',),variant='outlined',classes='mt-3')
                v.VBtn('Download setup details',click=app.download_symmetry_setup_details,disabled=('sx_busy || !sx_has_setup_details',),variant='text',classes='mt-3',prepend_icon='mdi-download')
                html.P('{{ sx_planning_summary }}',v_if='!!sx_planning_summary',classes='rf-notice')
                html.P('{{ sx_planning_usage }}',v_if='!!sx_planning_usage',classes='rf-notice')
                html.P('{{ sx_planning_cost }}',v_if='!!sx_planning_cost',classes='rf-notice')
                html.P('{{ note }}',v_for='note in sx_planning_notes',classes='rf-notice')
            with html.Div(v_if='sx_has_context',classes='sx-settings'):
                select('Exploration','sx_kind',"[{title:'Magnetic models',value:'magnetic'},{title:'Structural displacement modes',value:'structural'}]")
                html.P('For structural distortions, enter the distortion k vector in the parent basis. Use 0, 0, 0 for a zone-center distortion with unchanged translation periodicity. The magnetic k survey and magnetic setup assistant do not configure structural searches.',v_if="sx_kind === 'structural'",classes='rf-notice')
                select('Moment model','sx_moment_model',"[{title:'Collinear moments',value:'collinear'},{title:'Symmetry-allowed components · can cant',value:'symmetry_allowed'}]",v_if="sx_kind === 'magnetic'")
                select('Magnetic optimizer','sx_magnetic_optimizer',"[{title:'Fast checked optimizer · fixed nuclear model',value:'profiled_quadratic'},{title:'GSAS-II Hessian',value:'gsas_hessian'}]",v_if="sx_kind === 'magnetic'")
                select('Parent reference GPX · optional','sx_parent_reference','sx_start_sources',v_if="sx_kind === 'structural'",clearable=True)
                html.P('The starting GPX is the target scan. An optional well-refined parent scan supplies fractional coordinates and ADPs for candidate construction. Target observations, instrument, fitted parent cell, background and secondary phases are retained. Both inputs must have the same named parent phase, site chemistry and setting. Clear the reference to use the target itself.',v_if="sx_kind === 'structural'",classes='rf-notice')
                html.P('Native geometry and starting curves are prepared for each exact input context. All independent displacement and lattice coordinates are fitted jointly, and the saved GPX is independently checked with GSAS-II. Fixed secondary nuclear phases are retained. Check the parent peak widths before interpreting lower Rwp as symmetry breaking. Unresolved optimization remains visible; a good fit alone does not identify a unique structure.',v_if="sx_kind === 'structural'",classes='rf-notice')
                html.P('Profiled optimization varies the allowed magnetic components and the selected common scale/background against the same GSAS-II model. It requires a commensurate constructed model, symmetry-allowed components and one neutron histogram. Unsupported inputs stop with a reason.',v_if="sx_kind === 'magnetic' && sx_magnetic_optimizer === 'profiled_quadratic'",classes='rf-notice')
                select('Nuclear parent','sx_parent','sx_parents')
                select('Histogram / bank','sx_histogram','sx_histograms')
                with html.Div():
                    v.VTextField(label='k in this parent basis · rational components',v_model=('sx_k',),placeholder='0, 0, 1/2',density='compact',variant='outlined',hide_details=True)
                select('Magnetic parent sites','sx_sites','sx_site_options',v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Ionic form factors · atom=ion; atom=ion',v_model=('sx_form_factors',),placeholder='Cr1=Cr+3',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Landé g · number or atom=value; atom=value',v_model=('sx_lande',),type='text',density='compact',variant='outlined',hide_details=True)
                select(("sx_moment_model === 'symmetry_allowed' ? 'Dominant parent-axis starts' : 'Finite child-axis starts'",),'sx_directions',"[{title:'x',value:'x'},{title:'y',value:'y'},{title:'z',value:'z'}]",v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                html.P('Each selected axis uses four relative-component sign combinations and two amplitude levels. Independent child-orbit signs and amplitudes vary between starts, preserving exact site-symmetry couplings. This finite portfolio is not exhaustive.',v_if="sx_kind === 'magnetic' && sx_moment_model === 'symmetry_allowed'",classes='rf-notice')
                v.VCheckbox(v_if="sx_kind === 'magnetic'",label='Test relative initial signs between selected parent sites',v_model=('sx_sign_starts',),density='compact',hide_details=True)
                with html.Div(v_if="sx_kind === 'magnetic'"):
                    v.VCheckbox(label='Refine shared nuclear/magnetic intensity scale',v_model=('sx_refine_scale',),density='compact',hide_details=True)
                    v.VCheckbox(label='Refine existing background coefficients',v_model=('sx_refine_background',),density='compact',hide_details=True)
                    html.P('Consider these when the starting nuclear-only fit absorbed magnetic intensity. One common scale keeps phase amounts linked; the background basis and coefficient count stay fixed. The same scope applies to every candidate.',classes='rf-notice')
                v.VTextField(label='Experiment-specific adequate Rwp (%)',v_model=('sx_adequate',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Maximum moment (µB)',v_model=('sx_max_moment',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Magnetic d cutoff (Å) · 0 uses the full fit range',v_model=('sx_mag_d_min',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                with html.Details(v_if="sx_kind === 'structural'",classes="mt-3"):
                    html.Summary("Search bounds and computation")
                    with html.Div(classes="sx-bounds mt-3"):
                        v.VTextField(label='Displacement bound (Å RMS)',v_model=('sx_structural_radius',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                        v.VTextField(label='Cell metric relative bound',v_model=('sx_structural_cells',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                        v.VTextField(label='Starts per model · 2–16',v_model=('sx_structural_starts',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                        v.VTextField(label='Fit time per model (s)',v_model=('sx_structural_seconds',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                        v.VTextField(label='Recovery time (s) · 0 disables',v_model=('sx_structural_recovery',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                    html.P('The displacement bound applies to each independent coordinate. The cell bound uses reciprocal-metric coordinates normalized by the mean parent diagonal metric; it is not a percentage of each cell length. Independent checks also limit cell lengths to 2% and angles to 3°. Native preparation and GPX verification take additional time. Time limits leave unresolved optimization visible.',classes='rf-notice mt-2')
                html.P('The empirical contact screen uses packaged MP/COD bond minima as declared assumptions, not universal bounds. Ambiguous partial-occupancy contacts and unsupported source constraints stop for review. The GPX can be continued in hosted GSAS-II; ordinary GUI refinement does not enforce this contact screen.',v_if="sx_kind === 'structural'",classes='rf-notice')
                select('Structural candidate scope','sx_opd_policy',"[{title:'Highest symmetry per irrep',value:'highest-symmetry-per-irrep'},{title:'All discovered OPDs',value:'all'}]",v_if="sx_kind === 'structural'")
                select('Maximum children to construct','sx_construct_limit','[16,50,100,250,500]',v_if="sx_kind === 'structural'")
                v.VTextField(v_if="sx_kind === 'magnetic' && sx_magnetic_optimizer === 'gsas_hessian'",label='GSAS-II cycles per start',v_model=('sx_cycles',),type='text',inputmode='numeric',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic' && sx_magnetic_optimizer === 'profiled_quadratic'",label='Profiled optimizer iterations per start · 1–200',v_model=('sx_optimizer_iterations',),type='text',inputmode='numeric',density='compact',variant='outlined',hide_details=True)
            with html.Details(v_if="sx_has_context && sx_kind === 'magnetic'",classes='mt-3'):
                html.Summary('K-survey coverage and workers')
                html.P('Select magnetic sites and ionic form factors first. The survey tests rational high-symmetry k points and a finite template sample using the selected moment model. Other commensurate vectors or structures can remain untested.',classes='rf-notice')
                with html.Div(classes='sx-settings'):
                    select('Native templates per k','sx_k_limit','[3,6,12,24,48]')
                    select('K-survey requested workers','sx_k_workers','[1,2,4]')
                html.P('The survey runs the finite selected coverage across all eligible k families. Stop cancels pending work. Individual native operations retain execution limits. Cached profiles are bound to the exact GPX and settings.',classes='rf-notice')
                v.VBtn('Find candidate k-vectors',click=app.find_candidate_k,disabled=('sx_busy || !sx_sites.length',),variant='outlined',prepend_icon='mdi-magnify')
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
                    select('Evidence view','sx_k_display',"[{title:'Full pattern and residual',value:'full'},{title:'Magnetic contribution and residual',value:'residual'}]")
                    v.VBtn('Reset zoom',click=app.reset_k_search_zoom,variant='text',size='small')
                    v.VBtn('Use selected k',click=app.use_candidate_k,disabled=('sx_busy || !sx_k_selected',),variant='outlined',color='#15543c')
                with html.Div(style='height:520px;width:100%;min-width:0'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,scroll_zoom=True,style='height:100%;width:100%');fig.data=('sx_k_plot.data',);fig.layout=('sx_k_plot.layout',)
                html.P('{{ sx_k_detail }}',classes='rf-notice')
                html.P('Screening overlay: the nuclear model is fixed and only one magnetic intensity amplitude is fitted. No Rietveld refinement runs in this survey. A leading k, a tie or a missing template does not establish magnetic order. Use a candidate to review a full refinement plan.',classes='rf-notice')
                html.P('Hover over Bragg ticks for the exact template reflection and indexing basis. Observed-minus-nuclear includes errors in the nuclear model; it is not a list of confirmed magnetic peaks.',classes='rf-notice')
            html.P("{{ sx_kind === 'structural' ? 'Discovery sends the selected parent structure, without pattern arrays, to ISODISTORT. The selected OPD scope and construction budget are explicit; deferred or failed models remain visible. A budget smaller than the selected tier requires a narrower scope.' : 'Discovery sends parent symmetry operations and k to Bilbao. Refinement consumes the cached catalog offline.' }}",v_if='sx_has_context',classes='rf-notice')
            html.P('Review and use a k-survey candidate, or enter an exact rational k, before discovering models.',v_if="sx_has_context && sx_kind === 'magnetic' && !sx_k.trim()",classes='rf-notice')
            v.VBtn('Discover finite candidates',v_if='sx_has_context',click=app.discover_symmetry_candidates,disabled=('sx_busy || !sx_k.trim()',),variant='outlined',color='#15543c')
            html.P('{{ sx_discovery_scope }}',v_if='sx_discovered && !!sx_discovery_scope',classes='rf-notice mt-4')
            with html.Details(v_if='sx_discovered && sx_discovery_rows.length',classes='mt-3'):
                html.Summary('Construction details ({{ sx_discovery_rows.length }})')
                v.VDataTable(items=('sx_discovery_rows',),headers=("[{title:'Model',key:'model'},{title:'State',key:'state'},{title:'Reason',key:'reason'}]",),items_per_page=10,density='compact')
            with html.Div(v_if='sx_discovered && sx_candidate_options.length',classes='mt-4'):
                v.VBtn('Include all catalog embeddings',click='sx_candidates = sx_candidate_options.map(c => c.value)',variant='text',size='small')
                v.VBtn('Clear model selection',click='sx_candidates = []',variant='text',size='small')
                v.VAutocomplete(label='Constructed models to explore',v_model=('sx_candidates',),items=('sx_candidate_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,classes='sx-choices',update_modelValue=app._symmetry_unapprove)
                with html.Div(classes='rf-toolbar mt-3'):
                    select('Scans to refine','sx_existing',"[{title:'Current scan only',value:'selected'},{title:'Select scans',value:'custom'},{title:'All matching scans',value:'sample'}]")
                    app._number_field(label='Maximum scans',v_model=('sx_max_scans',),min=1,max=2000,density='compact',variant='outlined')
                scan_select('Select scans · search by run or temperature','sx_selected_scans','sx_selectable_scans',v_if="sx_existing === 'custom'",multiple=True,chips=True,closable_chips=True,classes='mt-3')
                html.P('Only the same sample, bank and starting recipe stage are listed. The plan includes the exact scan selection; other samples and checkpoints remain separate.',classes='rf-notice')
                v.VCheckbox(label='Follow new matching GPXs while this session is open',v_model=('sx_follow_new',),density='compact')
                html.P('Each new scan is checked against the reviewed parent, sample, bank and recipe. Incompatible inputs wait for review.',classes='rf-notice')
                v.VBtn('Build finite plan',click=(app.build_symmetry_plan,form),disabled=('sx_busy || sx_campaign_running || !sx_candidates.length || !sx_adequate',),variant='outlined',color='#15543c',classes='mt-3')
            with html.Div(v_if='sx_plan_ready',classes='mt-4'):
                html.P('{{ sx_plan_summary }}',classes='rf-notice')
                with html.Details():
                    html.Summary('Review every embedding ({{ sx_plan_candidates.length }})')
                    v.VDataTable(items=('sx_plan_candidates',),headers=("[{title:'Model',key:'name'},{title:'BNS',key:'bns'},{title:'Basis',key:'basis'},{title:'Origin',key:'origin'},{title:'Free modes',key:'modes'}]",),items_per_page=8,density='compact')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Approve finite plan',click=(app.approve_symmetry,form),disabled=('sx_approved || sx_campaign_running',),variant='outlined',color='#15543c')
                    v.VBtn('Start approved campaign',click=(app.start_symmetry_campaign,form),disabled=('!sx_approved || sx_busy || sx_campaign_running',),variant='flat',color='#15543c')
                html.P('Fast checked magnetic searches prepare one surrogate per model and verify distinct fitted profiles in GSAS-II. Other searches use isolated native starts. Galaxy jobs share immutable inputs and use the allocated CPU/RAM. Missing allocation information limits the pool to one worker. Model calls run only during setup.',classes='rf-notice')
                html.P('Structural fit comparisons require a matched parent/nuisance refinement. A lower Rwp alone does not establish a transition.',v_if="sx_kind === 'structural'",classes='rf-notice')
        html.P('{{ sx_scope_progress }}',v_if='!sx_setup && sx_results_current && !!sx_scope_progress',classes='rf-notice')
        html.Div('{{ sx_progress }}',v_if='!sx_setup && sx_results_current && !!sx_progress',classes='sx-status')
        with html.Details(v_if='!sx_setup && sx_results_current && !!sx_execution_detail',classes='mb-3'):
            html.Summary('Start outcomes and detailed coverage')
            html.P('{{ sx_execution_detail }}',classes='rf-notice')
        with html.Details(v_if='!sx_setup && sx_results_current && sx_batches.length',classes='rf-card mb-4'):
            html.Summary('Galaxy batch status ({{ sx_batches.length }})')
            html.P('Elapsed time starts at submission and includes queueing. Last status check is in UTC. Assigned starts may still be waiting for a worker.',classes='rf-notice')
            v.VDataTable(items=('sx_batches',),headers=("[{title:'Batch',key:'batch'},{title:'Galaxy job',key:'job'},{title:'State',key:'state'},{title:'Starts',key:'starts'},{title:'Since submission',key:'elapsed'},{title:'Last status check (UTC)',key:'checked'}]",),items_per_page=8,density='compact')
        html.P('Starts assigned to a batch share a Galaxy job after its recovery checkpoint is ready. Jobs can be queued or running. The native worker count is reported after the first batch; it is separate from the number of starts.',
            v_if='!sx_setup && sx_results_current && sx_campaign_running',classes='rf-notice')
        with html.Details(v_if='!sx_setup && sx_enrollments.length',classes='rf-card mb-4'):
            html.Summary('Scan queue ({{ sx_enrollments.length }}) · {{ sx_following ? "following new scans" : "captured scans" }}')
            with html.Div(v_if='sx_results_current && sx_priority_options.length',classes='rf-toolbar mt-3'):
                select('Prepare this captured scan next','sx_priority_acquisition','sx_priority_options')
                v.VBtn('Prioritize scan',click=app.prioritize_symmetry_acquisition,disabled=('!sx_priority_acquisition',),variant='outlined',size='small')
            html.P('{{ sx_priority_notice }}',v_if='!!sx_priority_notice',classes='rf-notice',raw_attrs=['aria-live="polite"'])
            with html.Div(v_if='sx_results_current && sx_retry_options.length',classes='rf-toolbar mt-3'):
                select('Failed preparation to review','sx_retry_acquisition','sx_retry_options')
                v.VBtn('Retry preparation',click=app.retry_symmetry_preparation,
                    disabled=('!sx_retry_acquisition || sx_busy || sx_campaign_running',),variant='outlined',size='small')
            html.P('Retry is available only after a verified transport publication failure. It preserves the earlier attempt and queues this scan in the same approved campaign. Continue starts the queued work.',
                v_if='sx_retry_options.length',classes='rf-notice')
            html.P('{{ sx_retry_notice }}',v_if='!!sx_retry_notice',classes='rf-notice',raw_attrs=['aria-live="polite"'])
            v.VDataTable(items=('sx_enrollments',),headers=("[{title:'Scan',key:'scan'},{title:'Bank',key:'bank'},{title:'State',key:'status'},{title:'Review needed',key:'reason'}]",),items_per_page=8,density='compact')
        with html.Details(v_if='!sx_setup && sx_results_current && sx_coverage.length > 1',classes='rf-card mb-4'):
            html.Summary('Scans and search coverage ({{ sx_coverage.length }})')
            v.VDataTable(items=('sx_coverage',),headers=("[{title:'Scan',key:'scan'},{title:'Condition',key:'condition'},{title:'Bank',key:'bank'},{title:'Magnetic sites',key:'sites'},{title:'Terminal starts',key:'coverage'},{title:'Qualified starts',key:'qualified'},{title:'Best qualified Rwp',key:'best_qualified'},{title:'Best audited Rwp',key:'best_audited'},{title:'Status',key:'status'}]",),items_per_page=8,density='compact',classes='mt-3')
        with html.Div(v_if='!sx_setup && sx_source_options.length > 1',classes='rf-toolbar'):
            v.VBtn('Previous scan',click=(app.symmetry_adjacent_scan,'[-1]'),disabled=("sx_source_options.findIndex(x => x.value === sx_source) <= 0",),variant='text',prepend_icon='mdi-chevron-left')
            scan_select('Compare models for this exact scan / bank','sx_source','sx_source_options')
            v.VBtn('Next scan',click=(app.symmetry_adjacent_scan,'[1]'),disabled=("sx_source_options.findIndex(x => x.value === sx_source) >= sx_source_options.length - 1",),variant='text',append_icon='mdi-chevron-right')
        html.P('Saved results load for the selected scan. Switching scans reads and verifies their artifacts; it does not start new refinements.',v_if='!sx_setup && sx_source_options.length > 1',classes='rf-notice')
        with html.Div(v_if='!sx_setup && sx_results_current && sx_groups.length',classes='sx-layout'):
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
                html.P('{{ sx_fit_title }}',v_if='sx_fit_title',classes='rf-notice')
                source_caveats()
                with html.Div(classes='rf-toolbar'):
                    v.VAutocomplete(label='Model / start in selected group',v_model=('sx_selected',),items=('sx_members',),density='compact',variant='outlined',hide_details=True,update_modelValue=app._symmetry_user_selected)
                with html.Div(classes='rf-toolbar'):
                    select('Horizontal coordinate','sx_axis',"[{title:'d spacing (Å)',value:'d'},{title:'Q (Å⁻¹)',value:'q'},{title:'Native coordinate',value:'native'}]")
                    select('Intensity display','sx_intensity_scale',"[{title:'Linear',value:'linear'},{title:'Log · positive values',value:'log'}]")
                    v.VAutocomplete(label='Pin alternatives · at most two',v_model=('sx_pins',),items=('sx_pin_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True)
                    v.VBtn('Reset zoom',click=app.symmetry_reset_zoom,variant='text',size='small')
                    v.VBtn('Expand fit',click='sx_expanded = true',variant='text',size='small',prepend_icon='mdi-arrow-expand')
                with html.Div(classes='sx-fit'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,scroll_zoom=True,style='height:100%;width:100%');fig.data=('sx_fit.data',);fig.layout=('sx_fit.layout',)
                with html.Div(classes='sx-evidence'):
                    with html.P(v_for='item in sx_details',key='item.label'):
                        html.B('{{ item.label }}: ');html.Span('{{ item.value }}')
                v.VAlert(v_if='!!sx_moment_note',text=('sx_moment_note',),type='warning',variant='tonal',density='compact',classes='my-3')
                with html.Details(v_if='sx_embedding_evidence.length',classes='mt-3 mb-3'):
                    html.Summary('Compare fit quality and search repeatability')
                    with html.P(v_for='item in sx_fit_diagnostics',key='item.label',classes='rf-notice'):
                        html.B('{{ item.label }}: ');html.Span('{{ item.value }}')
                    html.P('{{ sx_embedding_summary }}',classes='rf-notice')
                    v.VDataTable(items=('sx_embedding_evidence',),headers=("[{title:'Exact model embedding',key:'model'},{title:'Terminal starts',key:'coverage',sortable:false},{title:'Qualified starts',key:'qualified'},{title:'Refined variables',key:'parameters'},{title:'Best qualified Rwp (%)',key:'best_rwp'},{title:'Qualified Rwp range',key:'rwp_range',sortable:false},{title:'Lowest unresolved Rwp (%)',key:'unresolved_rwp'},{title:'Selected',key:'selected'}]",),items_per_page=5,density='compact')
                    html.P('Each row belongs to this same starting GPX and approved fit protocol. The final identifier distinguishes embeddings with the same BNS number. Empty residual or variable-count cells are unavailable. Start counts describe optimization repeatability, not independent experimental replication or confidence in a structure. Compare residuals, model complexity, uncertainties and temperature evidence together; unresolved checkpoints remain visible.',classes='rf-notice')
                v.VDataTable(v_if='sx_parameters.length',items=('sx_parameters',),headers=("[{title:'Phase',key:'phase'},{title:'Moment / normalized mode',key:'parameter'},{title:'Value',key:'value'},{title:'Conditional s.u.',key:'uncertainty'},{title:'Unit',key:'unit'}]",),items_per_page=6,density='compact')
                html.P('The first row for each magnetic atom is its magnitude; mx, my and mz use the candidate crystallographic basis shown in the embedding. Uncertainties are conditional on this model and its fixed nuclear parameters. A dash means fixed, dependent or unavailable; a magnitude uncertainty is not inferred from component errors without their covariance. A nonzero moment or qualified fit does not establish ordering.',v_if='sx_parameters.length',classes='rf-notice')
                with html.Details(v_if='sx_parent_moments.length || !!sx_parent_moment_note',classes='mt-3 mb-3'):
                    html.Summary('Moment direction in parent frame')
                    select('Compare moment with parent axis','sx_parent_axis',"[{title:'a axis',value:'a'},{title:'b axis',value:'b'},{title:'c axis',value:'c'}]")
                    v.VDataTable(v_if='sx_parent_moments.length',items=('sx_parent_moments',),headers=("[{title:'Representative site',key:'label'},{title:'Magnitude (µB)',key:'magnitude'},{title:'Along +'+sx_parent_axis+' (µB)',key:'parallel_'+sx_parent_axis},{title:'Transverse to '+sx_parent_axis+' (µB)',key:'transverse_'+sx_parent_axis},{title:'Angle to '+sx_parent_axis+'-axis (°)',key:'angle_'+sx_parent_axis}]",),items_per_page=5,density='compact')
                    html.P('Coordinate conversion of asymmetric-site representative moments only; the fit status above still applies. Projection onto the selected direct axis is signed; the angle to that axis is 0–90° and unchanged by reversing every moment. In a nonorthogonal cell, this is a geometric projection rather than a raw fractional component. These are not net magnetization or proof of ordering or canting. Derived uncertainties are unavailable; raw component uncertainties remain in the table above.',classes='rf-notice')
                    html.P('{{ sx_parent_moment_note }}',v_if='!!sx_parent_moment_note',classes='rf-notice')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Open native-verified representative',v_if='!!sx_native_reference',
                        click=app.open_symmetry_representative,variant='outlined',size='small')
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
        expanded_fit()
