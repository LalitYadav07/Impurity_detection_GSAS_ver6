"""Finite model/start review and conditional residual comparison."""
from trame.widgets import html, plotly, vuetify3 as v


def portfolio_panel(app):
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H3('5 · Compare declared models')
        html.P('The sealed review below retains its displayed models and policy. Editing the form starts a new review.',v_if='battery_portfolio_ready',classes='text-body-2')
        with html.Details(classes='mt-4'):
            html.Summary('Recover a reviewed portfolio')
            v.VBtn('Refresh saved portfolios and results',click=app.refresh_battery_portfolios,disabled=('battery_busy',),variant='text',size='small')
            v.VSelect(label='Saved portfolio',v_model=('battery_portfolio_saved',),items=('battery_portfolio_options',),disabled=('battery_busy',),density='compact',variant='outlined')
            v.VBtn('Open saved portfolio',click=app.open_battery_portfolio,disabled=('battery_busy || !battery_portfolio_saved',),variant='outlined',size='small')
        with html.Details(open=('!battery_portfolio_ready',),classes='my-4'):
            html.Summary('Build or revise a comparison',style='cursor:pointer')
            _portfolio_form(app)
        v.VAlert('{{ battery_portfolio_message }}',type=('battery_portfolio_message_kind',),variant='tonal',density='compact',role='status',aria_live='polite',classes='text-body-2 my-3')
        with html.Div(v_if='battery_portfolio_ready'):
            html.Strong('{{ battery_portfolio_acquisition }}')
            html.P('{{ battery_portfolio_policy }}',classes='text-body-2')
            with html.Details(v_for='trial in battery_portfolio_review',key='trial.id',classes='my-2'):
                html.Summary('{{ trial.family }} · {{ trial.seed }} · {{ trial.parameters }} independent parameters',style='cursor:pointer')
                html.P('{{ trial.phases }}',classes='text-body-2')
                html.P('Reuses an existing verified fit. This review includes a new report using the corrected native fitted interval; no refinement is repeated.',v_if='trial.retained_report',classes='text-body-2')
                with html.Div(v_for='stage in trial.stages',key='stage.title'):
                    html.Strong('{{ stage.title }}')
                    html.P('{{ stage.variables }}',classes='text-body-2')
                    html.P('{{ stage.edits }}',v_if='stage.edits',classes='text-body-2')
                with html.Details():
                    html.Summary('Exact starting cells, phase scales, widths, backgrounds and atomic policy')
                    html.Pre('{{ trial.values }}',style='max-height:350px;overflow:auto;font-size:12px')
                    html.Pre('{{ trial.backgrounds }}',style='max-height:250px;overflow:auto;font-size:12px')
            v.VBtn('Approve and queue reviewed trials',click=app.start_battery_portfolio,disabled=('battery_busy || battery_portfolio_started',),variant='flat',color='#15543c',classes='my-3')
            with html.Div(v_if='battery_portfolio_started',classes='d-flex flex-wrap ga-2 mb-3'):
                v.VBtn('Pause queued trials',click=(app.pause_battery_portfolio,'[true]'),variant='text',size='small')
                v.VBtn('Resume paused trials',click=(app.pause_battery_portfolio,'[false]'),variant='text',size='small')
                html.Span('An active native job finishes and retains its verified checkpoints.',classes='text-body-2')
            v.VDataTable(items=('battery_portfolio_trials',),headers=("[{title:'Family',key:'family'},{title:'Start',key:'seed'},{title:'Execution',key:'execution_display'},{title:'Parameters',key:'parameters'},{title:'Weighted residual / point',key:'score'},{title:'By physical bank',key:'bank_scores'},{title:'Adequacy screen',key:'adequacy'},{title:'Convergence screen',key:'convergence_display'},{title:'Detail',key:'message'}]",),density='compact',items_per_page=24,hide_default_footer=True)
            v.VDataTable(items=('battery_portfolio_families',),headers=("[{title:'Family',key:'family'},{title:'Declared starts',key:'declared_starts'},{title:'Completed starts',key:'completed_starts'},{title:'Minimum residual / point',key:'minimum_display'},{title:'Spread across starts',key:'spread_display'},{title:'Start check',key:'start_check'}]",),density='compact',items_per_page=24,hide_default_footer=True,classes='mt-3')
            with html.Div(v_if='battery_li_result_ready',classes='mt-4'):
                html.H4('Held-Li sensitivity under the reviewed assumptions')
                v.VAlert('{{ battery_li_notice }}',type='warning',variant='tonal',density='compact',classes='my-3')
                with html.Div(style='height:350px;width:100%'):
                    figure=plotly.Figure(display_mode_bar=True,responsive=True,display_logo=False,style='height:100%;width:100%')
                    figure.data=('battery_li_figure.data',);figure.layout=('battery_li_figure.layout',)
                v.VDataTable(items=('battery_li_points',),headers=("[{title:'Held Li occupancy',key:'occupancy'},{title:'Declared starts',key:'declared_starts'},{title:'Completed starts',key:'completed_starts'},{title:'Minimum residual / point',key:'minimum_display'},{title:'Increase above grid minimum',key:'delta_display'},{title:'Checks',key:'reason_display'}]",),density='compact',items_per_page=12,hide_default_footer=True)
            html.P('Incomplete and failed trials remain in the denominator. Extra parameters can lower residuals. No model probability, phase assignment or Li composition is inferred from this table. Use Refinement results to inspect each verified checkpoint and download its GPX.',classes='text-body-2')
            with html.Div(classes='d-flex flex-wrap ga-2 my-3'):
                with v.VBtn(v_for='trial in battery_portfolio_trials.filter(t => t.job_id)',key='trial.id',click=(app.open_battery_trial_result,'[trial.job_id]'),variant='outlined',size='small'):
                    html.Span('Inspect {{ trial.family }} / {{ trial.seed }}')
                with v.VBtn(v_for="trial in battery_portfolio_trials.filter(t => t.status === 'failed' || t.status === 'interrupted')",key="'retry-'+trial.id",click=(app.retry_battery_trial,'[trial.job_id]'),variant='outlined',size='small'):
                    html.Span('Retry {{ trial.family }} / {{ trial.seed }}')
            v.VBtn('Download portfolio review and results',click=app.download_battery_portfolio,variant='outlined',size='small')


def _portfolio_form(app):
    html.P('Compare phase families and explicit starts for one cell exposure. Prepare each model above, then name its family and start here. Ordinary comparisons refine phase scales and shared cells under one reviewed background and sample-width policy.',classes='text-body-2')
    v.VAlert('Atomic positions, occupancies, displacement parameters, texture and the instrument profile stay fixed. A different fixed input is a different family. Duplicate starts and differences in observations, profiles or fit limits stop review.',v_if='!battery_li_enabled && !battery_portfolio_ready',type='info',variant='tonal',density='compact',classes='my-3')
    v.VBtn('Find prepared models for comparison',click=app.refresh_saved_battery_models,disabled=('battery_busy',),variant='text',size='small')
    v.VSelect(label='Prepared model to add',v_model=('battery_saved_model',),items=('battery_saved_models',),disabled=('battery_busy',),density='compact',variant='outlined')
    with v.VRow():
        with v.VCol(cols=12,md=6):
            v.VTextField(label='Phase family label',v_model=('battery_portfolio_family',),disabled=('battery_busy',),density='compact',variant='outlined',hint='For example, layered cathode + graphite + collectors',persistent_hint=True)
        with v.VCol(cols=12,md=6):
            v.VTextField(label='Explicit start label',v_model=('battery_portfolio_seed',),disabled=('battery_busy',),density='compact',variant='outlined',hint='For example, starting phase scales A; independent starts need different values',persistent_hint=True)
    with html.Div(classes='d-flex ga-2 mb-3'):
        v.VBtn('Add declared trial',click=app.add_battery_trial,disabled=('battery_busy || !battery_saved_model || !battery_portfolio_family.trim() || !battery_portfolio_seed.trim()',),variant='outlined')
        v.VBtn('Clear trial set',click=app.clear_battery_trials,disabled=('battery_busy || !battery_portfolio_entries.length',),variant='text')
    v.VDataTable(items=('battery_portfolio_entries',),headers=("[{title:'Family',key:'family'},{title:'Start',key:'seed'},{title:'Prepared revision',key:'model'}]",),density='compact',items_per_page=24,hide_default_footer=True)
    with html.Div(classes='d-flex flex-wrap ga-2'):
        with v.VBtn(v_for='trial in battery_portfolio_entries',key='trial.index',click=(app.remove_battery_trial,'[trial.index]'),disabled=('battery_busy',),variant='text',size='small'):
            html.Span('Remove {{ trial.family }} / {{ trial.seed }}')
    html.H4('Background and sample-width policy',classes='mt-4')
    html.P('Use identical choices across this trial set. Compare a different policy in a separate reviewed portfolio; extra flexibility can absorb real diffraction signal.',classes='text-body-2')
    v.VSelect(label='Background policy',v_model=('battery_portfolio_background',),items=("[{title:'Preserve starting background models',value:'preserve'},{title:'Set Chebyshev terms in every bank',value:'chebyshev'}]",),disabled=('battery_busy',),density='compact',variant='outlined')
    v.VTextField(v_if="battery_portfolio_background === 'chebyshev'",label='Background coefficients per bank',v_model=('battery_portfolio_bg_coefficients',),disabled=('battery_busy',),inputmode='numeric',density='compact',variant='outlined',hint='1–36 coefficients. Existing background support points are retained.',persistent_hint=True)
    v.VSelect(label='Refine sample size for',v_model=('battery_portfolio_sizes',),items=('battery_portfolio_width_options',),multiple=True,chips=True,disabled=('battery_busy || !battery_portfolio_entries.length',),density='compact',variant='outlined')
    v.VSelect(label='Refine sample microstrain for',v_model=('battery_portfolio_strains',),items=('battery_portfolio_width_options',),multiple=True,chips=True,disabled=('battery_busy || !battery_portfolio_entries.length',),density='compact',variant='outlined')
    html.P('Only phase/bank links shared by every trial are offered. Native preflight requires an existing isotropic model. Other widths remain fixed. These are separate bank parameters, not an automatically shared physical size. Changing the trial set removes choices it no longer shares.',classes='text-body-2')
    with html.Details(classes='my-4'):
        html.Summary('Optional held-Li sensitivity',style='cursor:pointer')
        v.VCheckbox(label='Review a held-Li occupancy grid',v_model=('battery_li_enabled',),disabled=('battery_busy',),density='compact',hide_details=True)
        with html.Div(v_if='battery_li_enabled'):
            html.P('Prepare 3–12 different Li occupancy values for the same structural model. Add at least two distinct starts per value to test start sensitivity, within the 24-trial limit. Values are read from the native models; the family or start labels do not set occupancy. Other fixed assumptions must match across the grid.',classes='text-body-2')
            with v.VRow():
                with v.VCol(cols=12,md=6):
                    v.VTextField(label='Held-Li phase name',v_model=('battery_li_phase',),disabled=('battery_busy',),density='compact',variant='outlined')
                with v.VCol(cols=12,md=6):
                    v.VTextField(label='Held-Li atom label',v_model=('battery_li_atom',),disabled=('battery_busy',),density='compact',variant='outlined')
            v.VTextField(label='Optional co-located mixing atom to refine',v_model=('battery_li_mixing_atom',),disabled=('battery_busy',),density='compact',variant='outlined',hint='Blank holds all other occupancies fixed. An exact non-Li atom label is required; bounds use the remaining site capacity and permit vacancies.',persistent_hint=True)
            v.VCheckbox(label='Refine shared phase cells in this Li grid',v_model=('battery_li_refine_cells',),disabled=('battery_busy',),density='compact',hide_details=True)
            v.VAlert('This tests conditional site sensitivity. Positions, ADPs, texture, unselected occupancies and instrument profiles remain fixed. Review other assumptions separately. A low residual or an interior grid minimum does not identify Li composition.',type='warning',variant='tonal',density='compact',classes='my-3')
    v.VTextField(label='Maximum cycles per stage',v_model=('battery_portfolio_cycles',),disabled=('battery_busy',),inputmode='numeric',density='compact',variant='outlined',hint='1–30. Native stopping criteria remain active; more allowed cycles do not guarantee convergence.',persistent_hint=True)
    with v.VRow(classes='mt-2'):
        with v.VCol(cols=12,md=6):
            v.VTextField(label='Close-fit screening band (weighted residual per point)',v_model=('battery_portfolio_band',),disabled=('battery_busy',),density='compact',variant='outlined',hint='Flags sensitivity to starts; this is not a confidence interval or significance test.',persistent_hint=True)
        with v.VCol(cols=12,md=6):
            v.VTextField(label='Optional per-bank residual screening ceiling',v_model=('battery_portfolio_maximum',),disabled=('battery_busy',),density='compact',variant='outlined',hint='Blank leaves adequacy unevaluated. Passing a ceiling does not establish scientific acceptance.',persistent_hint=True)
    v.VTextField(label='Final-cycle shift/sigma screening ceiling',v_model=('battery_portfolio_shift_maximum',),disabled=('battery_busy',),density='compact',variant='outlined',type='text',inputmode='decimal',hint='Requires final-cycle shifts for every independent parameter and a native convergence report. Total-run displacement is not a substitute. This is a diagnostic threshold, not scientific acceptance.',persistent_hint=True,classes='mt-2')
    v.VTextarea(label='Portfolio review note',v_model=('battery_portfolio_note',),disabled=('battery_busy',),density='compact',variant='outlined',rows=2,auto_grow=True,maxlength=2000,counter=2000,classes='mt-3',hint='Record the comparison purpose and unresolved assumptions (1–2000 characters).',persistent_hint=True)
    v.VBtn('Preflight all declared trials',click=app.review_battery_portfolio,disabled=('battery_busy || !battery_native_available || !battery_portfolio_entries.length || !battery_portfolio_note.trim()',),variant='outlined')
