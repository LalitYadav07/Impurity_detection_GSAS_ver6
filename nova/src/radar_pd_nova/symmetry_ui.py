"""Compact portfolio navigation within the existing Refinement workspace."""
from trame.widgets import client,html,plotly,vuetify3 as v


def select(label,field,items,**kwargs):
    return v.VSelect(label=label,v_model=(field,),items=(items,),density='compact',variant='outlined',hide_details=True,**kwargs)


def model_exploration(app):
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
            v.VBtn('Continue original sealed queue',click=app.continue_symmetry_campaign,disabled=('!sx_resumable || sx_busy',),variant='text',prepend_icon='mdi-play')
            v.VBtn('Prepare exploration',click='sx_setup = !sx_setup',variant='outlined',prepend_icon='mdi-molecule')
        v.VAlert(v_if='!!sx_error',text=('sx_error',),type='error',variant='tonal',density='compact',classes='mb-3')
        v.VProgressLinear(v_if='sx_busy',indeterminate=True,color='#267653',height=3,classes='mb-3')
        with html.Div(v_if='sx_setup',classes='rf-card sx-setup'):
            html.H3('Explore models from a completed GPX')
            html.P('Choose a single commensurate k vector. Review displacement-mode or collinear magnetic hypotheses before native refinement.',classes='rf-notice')
            with html.Div(classes='rf-toolbar'):
                select('Starting GPX · original or recipe checkpoint','sx_start_source','sx_start_sources')
                v.VBtn('Inspect exact parent',click=app.inspect_symmetry_source,disabled=('!sx_start_source || sx_busy',),variant='outlined')
            with html.Details(v_if='sx_has_context',classes='mt-3'):
                html.Summary('Describe a magnetic model search')
                html.P('Supply k in the parent basis, magnetic sites and ions, and refinement limits. Review the draft below before discovering and approving the finite search.',classes='rf-notice')
                v.VTextarea(label='Magnetic refinement request',v_model=('sx_request',),rows=3,auto_grow=True,variant='outlined',hide_details=True,classes='mt-3')
                v.VBtn('Draft search settings',click=app.plan_symmetry_request,disabled=('sx_busy || !sx_request',),variant='outlined',classes='mt-3')
                html.P('{{ sx_planning_summary }}',v_if='!!sx_planning_summary',classes='rf-notice')
                html.P('{{ note }}',v_for='note in sx_planning_notes',classes='rf-notice')
            with html.Div(v_if='sx_has_context',classes='sx-settings'):
                select('Exploration','sx_kind',"[{title:'Collinear magnetic models',value:'magnetic'},{title:'Structural displacement modes',value:'structural'}]")
                select('Nuclear parent','sx_parent','sx_parents')
                select('Histogram / bank','sx_histogram','sx_histograms')
                v.VTextField(label='k in this parent basis · rational components',v_model=('sx_k',),placeholder='0, 0, 1/2',density='compact',variant='outlined',hide_details=True)
                select('Magnetic parent sites','sx_sites','sx_site_options',v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Ionic form factors · atom=ion; atom=ion',v_model=('sx_form_factors',),placeholder='Cr1=Cr+3',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Landé g · number or atom=value; atom=value',v_model=('sx_lande',),type='text',density='compact',variant='outlined',hide_details=True)
                select('Finite child-axis starts','sx_directions',"[{title:'x',value:'x'},{title:'y',value:'y'},{title:'z',value:'z'}]",v_if="sx_kind === 'magnetic'",multiple=True,chips=True)
                v.VCheckbox(v_if="sx_kind === 'magnetic'",label='Test relative initial signs between selected parent sites',v_model=('sx_sign_starts',),density='compact',hide_details=True)
                v.VTextField(label='Experiment-specific adequate Rwp (%)',v_model=('sx_adequate',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Maximum moment (µB)',v_model=('sx_max_moment',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'magnetic'",label='Magnetic d cutoff (Å) · 0 uses the full fit range',v_model=('sx_mag_d_min',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'structural'",label='Maximum normalized mode (Å)',v_model=('sx_max_mode',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(v_if="sx_kind === 'structural'",label='Signed mode start · ± amplitude (Å)',v_model=('sx_start_mode',),type='text',inputmode='decimal',density='compact',variant='outlined',hide_details=True)
                v.VTextField(label='Cycles per start',v_model=('sx_cycles',),type='text',inputmode='numeric',density='compact',variant='outlined',hide_details=True)
            html.P("{{ sx_kind === 'structural' ? 'Discovery sends the selected parent structure, without pattern arrays, to ISODISTORT. The first tier constructs one highest-symmetry OPD per discovered irrep; other OPDs remain deferred.' : 'Discovery sends parent symmetry operations and k to Bilbao. Refinement consumes the cached catalog offline.' }}",v_if='sx_has_context',classes='rf-notice')
            v.VBtn('Discover finite candidates',v_if='sx_has_context',click=app.discover_symmetry_candidates,disabled=('sx_busy',),variant='outlined',color='#15543c')
            with html.Div(v_if='sx_discovered',classes='mt-4'):
                html.P('{{ sx_discovery_scope }}',classes='rf-notice')
                v.VAutocomplete(label='Constructed models to explore',v_model=('sx_candidates',),items=('sx_candidate_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,classes='sx-choices',update_modelValue=app._symmetry_unapprove)
                with html.Details(v_if="sx_kind === 'structural'",classes='mt-3'):
                    html.Summary('Select native displacement modes')
                    v.VAutocomplete(label='Modes · every selected mode gets both start signs',v_model=('sx_modes',),items=('sx_mode_options',),multiple=True,chips=True,closable_chips=True,density='compact',variant='outlined',hide_details=True,classes='mt-3 sx-choices',update_modelValue=app._symmetry_unapprove)
                v.VBtn('Build finite plan',click=app.build_symmetry_plan,disabled=('sx_busy || !sx_candidates.length || !sx_adequate',),variant='outlined',color='#15543c',classes='mt-3')
            with html.Div(v_if='sx_plan_ready',classes='mt-4'):
                html.P('{{ sx_plan_summary }}',classes='rf-notice')
                with html.Details():
                    html.Summary('Review every embedding ({{ sx_plan_candidates.length }})')
                    v.VDataTable(items=('sx_plan_candidates',),headers=("[{title:'Model',key:'name'},{title:'BNS',key:'bns'},{title:'Basis',key:'basis'},{title:'Origin',key:'origin'},{title:'Free modes',key:'modes'}]",),items_per_page=8,density='compact')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Approve finite plan',click=app.approve_symmetry,disabled=('sx_approved',),variant='outlined',color='#15543c')
                    select('Concurrent native workers','sx_slots',"[1,2,4]",style='max-width:220px')
                    v.VBtn('Start approved campaign',click=app.start_symmetry_campaign,disabled=('!sx_approved || sx_busy',),variant='flat',color='#15543c')
                html.P('No model calls run during refinement. Each start has its own process, deadline and terminal record. This first interface starts one representative scan; enrollment of later scans still requires review.',classes='rf-notice')
        html.P('{{ sx_message }}',classes='rf-notice')
        html.Div('{{ sx_progress }}',v_if='sx_results_current && !!sx_progress',classes='sx-status')
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
                v.VDataTable(v_if='sx_parameters.length',items=('sx_parameters',),headers=("[{title:'Phase',key:'phase'},{title:'Moment / normalized mode',key:'parameter'},{title:'Value',key:'value'},{title:'Conditional s.u.',key:'uncertainty'},{title:'Unit',key:'unit'}]",),items_per_page=6,density='compact')
                html.P('Uncertainties are conditional on this model and its fixed nuclear parameters. A nonzero moment estimate or a numerically qualified fit does not establish ordering.',v_if='sx_parameters.length',classes='rf-notice')
                with html.Div(classes='rf-actions'):
                    v.VBtn('Download native GPX',click=(app.download_symmetry_artifact,"['gpx']"),disabled=('!sx_has_gpx',),variant='outlined',size='small',prepend_icon='mdi-download')
                    v.VBtn('Download trial receipt',click=(app.download_symmetry_artifact,"['receipt']"),disabled=('!sx_has_receipt',),variant='text',size='small')
                with html.Details(classes='mt-3'):
                    html.Summary('Trial control')
                    v.VBtn('Cancel selected active trial',click=app.cancel_symmetry_selected,variant='text',size='small',classes='mt-2')
                html.H3('Fit quality and model complexity',classes='mt-4')
                with html.Div(classes='sx-overview'):
                    fig=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,style='height:100%;width:100%',click=(app.symmetry_overview_click,'[$event]'));fig.data=('sx_overview.data',);fig.layout=('sx_overview.layout',)
                html.P('Open diamonds are inadequate or unresolved fits. A lower residual is evidence to inspect, not a confirmed structure. Log intensity hides nonpositive values for display; the residual stays linear.',classes='rf-notice')
