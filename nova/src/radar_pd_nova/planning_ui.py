"""Guided planning inside the NDIP workspace; calculations live in shared services."""
from trame.widgets import html, plotly, vuetify3 as v
from .uploads import NamedFileUpload
from .planning_bundle import MAX_BYTES as PLANNER_BUNDLE_LIMIT


def field(name, label, **kwargs):
    return v.VTextField(v_model=(name,), label=label, density='comfortable', variant='outlined',
                        hide_details='auto', classes='mb-3', **kwargs)


def number(name, label, **kwargs):
    return field(name, label, type='text', inputmode='decimal', **kwargs)


def select(name, label, items, **kwargs):
    return v.VSelect(v_model=(name,), label=label, items=(items,), item_title='title', item_value='value',
                     density='comfortable', variant='outlined', hide_details='auto', classes='mb-3', **kwargs)


def note(text, **kwargs):
    return html.P(text, classes='planner-help', **kwargs)


def methodology():
    with html.Div(classes='planner-method'):
        html.H3('How the prediction is calculated')
        for title, text in [
            ('1 · Structure and peak shape', 'GSAS-II calculates nuclear and magnetic diffraction from your structure, isotope choices and selected instrument profile. Transition controls use the prepared GSAS-II phase and its symmetry constraints.'),
            ('2 · Scale and instrument response', 'The measured cycle response connects scattering to accepted neutron counts. Sample mass, density, packing, container and beam dimensions set the assumed illuminated amount and absorption. The result is conditional on those inputs.'),
            ('3 · Exposure and counting noise', 'Time × average delivered current gives proton charge. Expected counts grow with charge; Poisson counting is simulated cumulatively. In the normalized pattern, counting noise falls approximately as 1/√charge.'),
            ('4 · Error bars and their limits', 'The default ±1σ bars include counting noise and finite reference statistics. You can isolate sample counting noise or plot σ directly. Shared reference errors remain correlated; the bars do not include every uncertainty in sample physics or geometry.')]:
            with html.Div(classes='planner-method-step'):
                html.Strong(title);html.P(text)
        note('Measured container/reference contributions and the implemented scattering terms provide a baseline. Unknown sample-specific diffuse or inelastic scattering, disorder and unentered broadening can still change the measured pattern. A custom profile without a measured count response gives a shape preview only.')


def site_table(state, magnetic=False):
    with html.Div(v_for=f'site in {state}', key=('site.label',), classes='planner-site-card'):
        with html.Div(classes='planner-site-title'):
            html.Strong('{{ site.label }} · {{ site.ion }}')
            if magnetic:html.Span('|M| = {{ site.magnitude }} μB', classes='planner-magnitude')
        if magnetic:
            note('Parent {{ site.parent_label }} · fractional position ({{ site.position_text }}) · site symmetry {{ site.site_symmetry }} · multiplicity {{ site.multiplicity }}')
        else:
            note('Site symmetry {{ site.site_symmetry }} · multiplicity {{ site.multiplicity }} · fractional coordinates')
        with html.Div(classes='planner-component-grid'):
            with html.Div(v_for='component in site.components', key=('component.axis',), classes='planner-component'):
                html.Span('{{ component.axis }}', classes='planner-component-label')
                html.Strong('{{ component.value }}')
                html.Small('{{ component.equation }}', title=('component.status',))


def planning_dashboard(app):
    with html.Div(classes='planner-page'):
        with html.Div(classes='planner-heading'):
            with html.Div():
                html.Div('BEFORE YOU MEASURE', classes='planner-eyebrow')
                html.H1('Plan your experiment')
                note('Explore instrument settings, counting time and possible transitions. No IPTS or measured data is needed.')
            with html.Div(classes='planner-beamline'):
                select('plan_beamline', 'Beamline', "[{title:'POWGEN · SNS powder diffraction',value:'PG3'}]",
                       disabled=('plan_busy',))
        with html.Div(classes='planner-save-bar'):
            field('plan_name', 'Plan name', maxlength=100)
            v.VBtn('Save plan', click=app.save_planning_plan, disabled=('plan_busy || !plan_cif',),
                   prepend_icon='mdi-content-save-outline', variant='outlined')
            select('plan_saved_id', 'Saved plans and History', 'plan_saved_options', clearable=True)
            v.VBtn('Open', click=app.restore_planning_plan, disabled=('plan_busy || !plan_saved_id',), variant='text')
        note('{{ plan_save_message }}', v_if='!!plan_save_message', role='status')
        with html.Details(classes='planner-recovery mb-4'):
            html.Summary('Saved work, downloads and reconnecting', style='cursor:pointer;font-weight:600')
            note('Save plan keeps a snapshot in Galaxy History. Reopen it in a fresh session, or download a portable copy. Returning to a running calculation does not submit it again.')
            with html.Div(classes='d-flex flex-wrap', style='gap:8px'):
                v.VBtn('Refresh saved plans', click=app.refresh_planning_plans,
                       disabled=('plan_busy',), variant='text', prepend_icon='mdi-refresh')
                v.VBtn('Download plan', click=app.export_planning_plan,
                       disabled=('plan_busy || !plan_cif',), variant='outlined', prepend_icon='mdi-download')
            NamedFileUpload('plan_bundle_upload', label='Saved experiment plan', extensions=['.zip'],
                            optional=True, key='planner-portable-import', max_bytes=PLANNER_BUNDLE_LIMIT)
            v.VBtn('Open uploaded plan', click=app.import_planning_plan,
                   disabled=('plan_busy || !plan_bundle_upload',), variant='outlined')
            note('An uploaded plan keeps its settings and saved curves. Native transition models must be prepared again unless their original Galaxy job can be verified.')
            v.VBtn('Find previous calculations', click=app.refresh_planning_jobs,
                   disabled=('plan_busy',), variant='text', prepend_icon='mdi-history', classes='mt-3')
            select('plan_recover_job_id', 'Calculation to reconnect', 'plan_recoverable_jobs',
                   disabled=('plan_busy',), no_data_text='Find previous calculations in this Galaxy History first.')
            v.VBtn('Reconnect to calculation', click=app.recover_planning_job,
                   disabled=('plan_busy || !plan_recover_job_id',), variant='outlined')
        v.VAlert(v_if='!!plan_error', text=('plan_error',), type='error', variant='tonal', classes='mb-4', role='alert')
        if hasattr(app, 'use_planning_inputs_for_preparation'):
            note('{{ plan_handoff_message }}', v_if='!!plan_handoff_message', role='status')
        with v.VAlert(v_if='!!plan_job_state', type='info', variant='tonal', classes='mb-4'):
            html.Strong('{{ plan_job_operation.replaceAll("planning-", "").replaceAll("-", " ") }} · {{ plan_job_state }}')
            note('Galaxy job {{ plan_job_id }}', v_if='!!plan_job_id')
        with html.Div(classes='planner-grid'):
            with html.Div(classes='planner-input-card'):
                with v.VTabs(v_model=('plan_step_ui',), color='#15543c', grow=True, show_arrows=True,
                             classes='planner-step-tabs'):
                    for index, label in enumerate(['Instrument', 'Sample', 'Transitions', 'Exposure']):
                        v.VTab(f'{index + 1}. {label}', value=(str(index),))
                with html.Div(classes='planner-form-body'):
                    with html.Fieldset(disabled=('plan_busy',), classes='planner-fields'):
                        with html.Div(v_show='plan_step_ui === 0'):
                            html.H2('Choose the instrument')
                            select('plan_cycle', 'Calibration cycle', 'plan_cycles')
                            select('plan_configuration', 'Instrument configuration', 'plan_visible_configurations')
                            note('{{ plan_configuration_note }}')
                            select('plan_response_mode', 'Prediction', 'plan_response_options')
                            with v.VExpansionPanels(variant='accordion'):
                                with v.VExpansionPanel(title='Custom instrument file and peak shape'):
                                    with v.VExpansionPanelText():
                                        NamedFileUpload('plan_custom_profile', label='Instrument parameter file', extensions=['.instprm','.prm'], optional=True, key='plan-custom-profile', max_bytes=2*1024*1024)
                                        number('plan_profile_bank', 'Bank within the uploaded file')
                                        select('plan_resolution_model', 'Peak-shape model', 'plan_resolution_options')
                                        note('A custom instrument file provides a shape preview. Count predictions require a matching measured response.')
                        with html.Div(v_show='plan_step_ui === 1'):
                            html.H2('Describe your sample')
                            NamedFileUpload('plan_cif', label='Nuclear structure · CIF', extensions=['.cif'], key='plan-nuclear-cif', max_bytes=20*1024*1024)
                            v.VBtn('Try silicon example', click=app.load_planning_example, variant='text', size='small', classes='mb-3')
                            with html.Div(classes='planner-field-pair'):
                                number('plan_mass', 'Powder mass (g)')
                                number('plan_radius', 'Powder radius (cm)', hint='0.30 cm is a 6 mm diameter.', persistent_hint=True)
                            number('plan_packing', 'Packing fraction', hint='Fraction of the filled volume occupied by solid. 0.5 means 50%.', persistent_hint=True)
                            note('These are your sample assumptions. Mass, crystal density and packing determine the powder fill height.')
                            select('plan_container', 'Sample container', "[{title:'6 mm PAC can · measured background',value:'PAC06'},{title:'No can · instrument blank',value:'none'}]")
                            with v.VExpansionPanels(variant='accordion', multiple=True):
                                with v.VExpansionPanel(title='Illumination'):
                                    with v.VExpansionPanelText():
                                        number('plan_beam_width', 'Beam width at sample (cm)')
                                        number('plan_beam_height', 'Beam height at sample (cm)')
                                        note('Assumed rectangular illumination of a cylindrical powder sample.')
                                with v.VExpansionPanel(title='Isotope composition'):
                                    with v.VExpansionPanelText():
                                        note('Natural abundance is used unless you choose an isotope or mixture.')
                                        v.VBtn('Read structure and isotope choices', click=app.prepare_planning_parent, disabled=('!plan_cif',), variant='outlined', size='small', classes='mb-3')
                                        with html.Div(v_for='entry in plan_isotope_options', key=('entry.element',)):
                                            v.VSelect(model_value=('entry.selected',), items=("[...entry.options, 'Atom fractions']",), label=("entry.element + ' isotope'",),
                                                update_modelValue=(app.update_planning_isotope,'[entry.element, $event]'), density='comfortable', variant='outlined')
                                            with html.Div(v_if="entry.selected === 'Atom fractions'"):
                                                note('Percent of atoms; the total must be 100%.')
                                                with html.Div(v_for='fraction in entry.abundances', key=('fraction.label',)):
                                                    v.VTextField(model_value=('fraction.percent',), label=("fraction.label + ' atoms (%)'",),
                                                        update_modelValue=(app.update_planning_isotope_fraction,'[entry.element, fraction.label, $event]'),
                                                        type='text', inputmode='decimal', density='comfortable', variant='outlined')
                                with v.VExpansionPanel(title='Additional crystalline phases'):
                                    with v.VExpansionPanelText():
                                        note('Set mass percentages for coexisting phases. The remaining mass belongs to the primary CIF. Transitions apply to the primary phase.')
                                        for index in range(1,4):
                                            NamedFileUpload(f'plan_phase_{index}_cif', label=f'Phase {index + 1} · CIF', extensions=['.cif'], optional=True, key=f'plan-additional-{index}', max_bytes=20*1024*1024)
                                            number(f'plan_phase_{index}_percent', 'Mass fraction (%)')
                                            number(f'plan_phase_{index}_size', 'Optional domain size (μm)')
                                            number(f'plan_phase_{index}_microstrain', 'Optional microstrain (10⁻⁶)')
                        with html.Div(v_show='plan_step_ui === 2'):
                            html.H2('Explore a possible transition')
                            select('plan_transition', 'Scenario', "[{title:'Nuclear structure only',value:'none'},{title:'Magnetic ordering · discover subgroups',value:'magnetic'},{title:'Structural distortion · discover modes',value:'structural'},{title:'Known magnetic structure · upload MCIF',value:'mcif'},{title:'Compare another nuclear structure',value:'comparison'}]")
                            note('Optional. Start with the nuclear pattern, or prepare a state to compare with it.', v_if="plan_transition === 'none'")
                            with html.Div(v_show="['magnetic','structural'].includes(plan_transition)"):
                                html.H3('Propagation vector k')
                                with html.Div(classes='planner-k-fields'):
                                    for i, axis in enumerate(['h', 'k', 'l']):
                                        v.VTextField(model_value=(f"plan_k.split(',')[{i}] || '0'",), label=axis,
                                            update_modelValue=(app.update_planning_k_component, f'[{i}, $event]'),
                                            density='comfortable', variant='outlined', hide_details=True)
                                note('Use exact fractions, for example 0, 0, 1/2. Components refer to the current parent cell.')
                                note('{{ plan_parent_note }}', v_if='plan_promoted_parent')
                                v.VBtn('Return to original nuclear parent', v_if='plan_promoted_parent', click=app.reset_planning_nuclear_parent, variant='text', size='small')
                            with html.Div(v_show="plan_transition === 'magnetic'"):
                                note('Bilbao receives the parent symmetry and propagation vector. The parent is prepared automatically.')
                                v.VBtn('Find magnetic subgroups', click=app.discover_planning_magnetic_groups, disabled=('!plan_cif',), variant='outlined', classes='mb-3')
                                note('{{ plan_magnetic_message }}', role='status')
                                select('plan_candidate', 'Magnetic subgroup', 'plan_candidates', disabled=('!plan_candidates.length',))
                                select('plan_sites', 'Sites that may order', 'plan_site_options', multiple=True, chips=True)
                                with html.Div(v_for='entry in plan_site_rows', key=('entry.label',), classes='planner-site-row'):
                                    html.H3('{{ entry.label }}')
                                    v.VSelect(model_value=('entry.form_factor',), items=('entry.options',), label='Magnetic ion / form factor',
                                        update_modelValue=(app.update_planning_site, "[entry.label, 'form_factor', $event]"), density='comfortable', variant='outlined')
                                note('Ion choices come from GSAS-II’s magnetic form-factor table. Other magnetic scattering settings use the prepared project or native defaults.', v_if='plan_site_rows.length > 0')
                                select('plan_site_policy', 'Ordering assumption', "[{title:'Every selected orbit can order',value:'all'},{title:'Some selected orbits may remain unordered',value:'any'}]")
                                v.VBtn('Prepare allowed moments', click=app.prepare_planning_magnetic_moments, disabled=('!plan_candidate || !plan_sites.length',), variant='outlined', classes='mb-3')
                                with html.Div(v_if='plan_parameter_rows.length > 0', classes='planner-transition-controls'):
                                    html.H3('Set the independent moments')
                                    note('{{ plan_coordinate_note }}')
                                    with html.Div(v_for='entry in plan_parameter_rows', key=('entry.name',), classes='planner-parameter'):
                                        v.VTextField(model_value=('entry.value',), label=('entry.label',), type='text', inputmode='decimal',
                                            update_modelValue=(app.update_planning_magnetic_parameter,'[entry.name, $event]'), density='comfortable', variant='outlined', hide_details='auto')
                                        note('{{ entry.symbol }} · {{ entry.description }}')
                                    html.H3('Resulting moments on each magnetic site')
                                    note('Read-only components follow the selected subgroup. Expressions use the q values above; a fixed zero is forbidden by this prepared model.')
                                    site_table('plan_moment_sites', magnetic=True)
                            with html.Div(v_show="plan_transition === 'structural'"):
                                note('ISODISTORT receives the parent CIF. Discovery lists the supported distortion candidates and any incomplete branches.')
                                v.VBtn('Find structural subgroups', click=app.discover_planning_structural_groups, disabled=('!plan_cif',), variant='outlined', classes='mb-3')
                                note('{{ plan_structural_message }}', role='status')
                                select('plan_structural_candidate', 'Structural child', 'plan_structural_candidates', disabled=('!plan_structural_candidates.length',))
                                v.VBtn('Prepare displacement modes', click=app.prepare_planning_structural_modes, disabled=('!plan_structural_candidate',), variant='outlined', classes='mb-3')
                                with html.Div(v_if='plan_displacement_sites.length > 0', classes='planner-transition-controls'):
                                    html.H3('Set the distortion amplitudes')
                                    note('{{ plan_structural_note }}')
                                    with html.Div(v_for='entry in plan_structural_parameter_rows', key=('entry.name',), classes='planner-parameter'):
                                        v.VTextField(model_value=('entry.value',), label=('entry.label',), type='text', inputmode='decimal',
                                            update_modelValue=(app.update_planning_structural_parameter,'[entry.name, $event]'), density='comfortable', variant='outlined', hide_details='auto')
                                        note('{{ entry.description }}')
                                        html.Code('{{ entry.mode_label }}', classes='planner-mode-name')
                                    with v.VExpansionPanels(variant='accordion', classes='my-3'):
                                        with v.VExpansionPanel(title='Resulting atomic positions'):
                                            with v.VExpansionPanelText():site_table('plan_displacement_sites')
                                with v.VExpansionPanels(variant='accordion'):
                                    with v.VExpansionPanel(title='Optional child lattice'):
                                        with v.VExpansionPanelText():
                                            note('{{ plan_child_cell_note }}', v_if='!!plan_child_cell_note')
                                            note('Leave all six fields blank to inherit the parent metric. A supplied cell must preserve the selected child symmetry.')
                                            v.VBtn('Edit inherited cell', click=app.edit_planning_child_cell, disabled=('!plan_child_cell_note',), size='small', variant='text', classes='mb-3')
                                            v.VBtn('Reset to inherited cell', click=app.inherit_planning_cell, size='small', variant='text', classes='mb-3')
                                            with html.Div(classes='planner-field-pair'):
                                                for i,label in enumerate(['a (Å)','b (Å)','c (Å)','α (°)','β (°)','γ (°)']):
                                                    v.VTextField(model_value=(f'plan_cell_values[{i}]',), label=label, type='text', inputmode='decimal',
                                                        update_modelValue=(app.update_planning_cell, f'[{i}, $event]'), density='comfortable', variant='outlined')
                                v.VBtn('Add magnetic ordering to this result', click=app.promote_planning_structural_parent, disabled=('!plan_can_promote_parent',), variant='outlined', classes='mt-3')
                                note('First simulate the structural state. Then use it as the parent for magnetic ordering.')
                            with html.Div(v_show="plan_transition === 'mcif'"):
                                NamedFileUpload('plan_mcif', label='Magnetic structure · MCIF', extensions=['.mcif','.cif'], optional=True, key='plan-magnetic-cif', max_bytes=20*1024*1024)
                                number('plan_moment_multiplier', 'Moment multiplier', hint='1 uses the moments in the file.', persistent_hint=True)
                            with html.Div(v_show="plan_transition === 'comparison'"):
                                NamedFileUpload('plan_comparison_cif', label='Comparison nuclear structure · CIF', extensions=['.cif'], optional=True, key='plan-comparison-cif', max_bytes=20*1024*1024)
                                note('Compare structures at the same assumed total sample mass and exposure.')
                        with html.Div(v_show='plan_step_ui === 3'):
                            html.H2('Choose how long to measure')
                            v.VAlert(v_if="plan_response_mode === 'profile_only'", text='This configuration provides a shape preview. Select a configuration with a measured response to compare counting times.', type='info', variant='tonal', classes='mb-3')
                            with html.Div(v_show="plan_response_mode !== 'profile_only'"):
                                select('plan_exposure_basis', 'Specify exposure using', "[{title:'Time and average delivered current',value:'current_time'},{title:'Cumulative proton charge',value:'charge'},{title:'Time, beam power and proton energy',value:'power_time'}]")
                                field('plan_charge', 'Cumulative charges (μAh)', v_if="plan_exposure_basis === 'charge'", hint='Increasing values separated by commas, e.g. 25, 125, 750.', persistent_hint=True)
                                field('plan_seconds', 'Cumulative counting times (s)', v_if="plan_exposure_basis !== 'charge'", hint='Increasing values separated by commas, e.g. 60, 300, 1800.', persistent_hint=True)
                                number('plan_current_ua', 'Average delivered current (μA)', v_if="plan_exposure_basis === 'current_time'", hint='Use an average that includes beam interruptions.', persistent_hint=True)
                                number('plan_power', 'Average beam power (MW)', v_if="plan_exposure_basis === 'power_time'")
                                number('plan_energy', 'Proton energy (GeV)', v_if="plan_exposure_basis === 'power_time' && plan_response_mode !== 'current_cycle'")
                                note('The 2026B response uses 1.3 GeV protons.', v_if="plan_exposure_basis === 'power_time' && plan_response_mode === 'current_cycle'")
                            with v.VExpansionPanels(variant='accordion'):
                                with v.VExpansionPanel(title='Plot range, binning and reproducibility'):
                                    with v.VExpansionPanelText():
                                        number('plan_d_min', 'Minimum d spacing (Å)')
                                        number('plan_d_max', 'Maximum d spacing (Å)')
                                        number('plan_step', 'Relative d bin width')
                                        number('plan_seed', 'Counting-noise seed')
                                        note('The same inputs and seed reproduce the same noisy realization. Longer exposures are cumulative.')
                    with html.Div(classes='planner-step-actions'):
                        v.VBtn('Back', v_if='plan_step_ui > 0', click='plan_step_ui -= 1', variant='text', disabled=('plan_busy',))
                        v.VSpacer()
                        v.VBtn('Continue', v_if='plan_step_ui < 3', click='plan_step_ui += 1', variant='tonal', color='#15543c', disabled=('plan_busy',))
                with html.Div(classes='planner-run-bar'):
                    v.VBtn('Simulate pattern', click=app.run_planning_simulation, color='#15543c', block=True,
                        loading=('plan_busy',), disabled=('plan_busy || !plan_cif',), prepend_icon='mdi-chart-bell-curve')
                    v.VBtn('Cancel calculation', v_if='plan_busy && plan_can_cancel', click=app.cancel_planning_simulation, variant='text', block=True)
                    note('{{ plan_message }}', role='status')
            with html.Div(classes='planner-results-card'):
                html.H2('Predicted pattern')
                with html.Div(v_if='!plan_current', classes='planner-empty-plot'):
                    v.VIcon('mdi-chart-bell-curve', size=48, color='#15543c')
                    html.H3("{{ plan_busy ? 'Preparing your prediction…' : 'See your experiment before you measure' }}")
                    note('Choose a structure and configuration, then simulate. Compare the expected signal with a noisy realization at each exposure.')
                    methodology()
                    v.VProgressLinear(v_if='plan_busy', indeterminate=True, color='#15543c', classes='mt-4')
                with html.Div(v_show='plan_current'):
                    v.VAlert(text=('plan_qualification',), type='info', variant='tonal', classes='mb-3')
                    with html.Div(classes='planner-field-pair'):
                        select('plan_look', 'Exposure to display', 'plan_look_options', v_if='plan_look_options.length > 1')
                        select('plan_pattern_view', 'Intensity view', "[{title:'Normalized pattern',value:'reduced'},{title:'Raw neutron counts',value:'counts'}]", v_if="plan_response_mode === 'current_cycle'")
                    with html.Div(v_if="plan_response_mode !== 'profile_only' && plan_look_options.length > 0", classes='planner-field-pair'):
                        select('plan_plot_mode', 'Plot', "[{title:'Pattern with ±1σ bars',value:'pattern'},{title:'Error-bar size (σ)',value:'uncertainty'}]")
                        select('plan_error_model', 'Error bars include', "[{title:'Counting + calibration references',value:'predictive'},{title:'Sample counting only',value:'counting'}]", v_if="plan_plot_mode === 'pattern'")
                    with html.Div(classes='planner-plot'):
                        figure=plotly.Figure(display_mode_bar=True, responsive=True, display_logo=False, style='height:100%;width:100%')
                        figure.data=('plan_figure.data',);figure.layout=('plan_figure.layout',)
                    note('Click a legend entry to show or hide a component. Drag to zoom; double-click to reset.')
                    note('{{ plan_error_note }}')
                    with v.VExpansionPanels(variant='accordion', classes='mb-3'):
                        with v.VExpansionPanel(title='How the prediction and error bars are calculated'):
                            with v.VExpansionPanelText():methodology()
                        with v.VExpansionPanel(title='Assumptions and calculation details'):
                            with v.VExpansionPanelText():
                                with html.Div(v_for='item in plan_summary', key=('item.label',), classes='planner-detail'):
                                    html.Strong('{{ item.label }}');html.Span('{{ item.value }}')
                                with html.Div(v_for='warning in plan_warnings', key=('warning',)):
                                    note('{{ warning }}')
                    v.VBtn('Download patterns and provenance', click=app.export_planning_simulation, disabled=('!plan_current || !plan_result_ready',),
                           variant='outlined', prepend_icon='mdi-download', classes='mr-2 mb-2')
                    v.VBtn('Save this plan', click=app.save_planning_plan, disabled=('plan_busy',), variant='text', prepend_icon='mdi-content-save-outline', classes='mb-2')
        if hasattr(app, 'use_planning_inputs_for_preparation'):
            with html.Div(classes='planner-next-step mt-4'):
                html.H2('Use this setup')
                note('Carry the reviewed nuclear structure and profile into the next workflow. You will review exactly what is copied; no analysis or model search starts automatically.')
                with html.Div(classes='d-flex flex-wrap', style='gap:12px'):
                    v.VBtn('Use inputs for preparation', click=app.use_planning_inputs_for_preparation,
                           disabled=('plan_busy || !plan_cif',), variant='outlined', prepend_icon='mdi-database-clock-outline')
                    v.VBtn('Use inputs for analysis setup', click=app.use_planning_inputs_for_analysis,
                           disabled=('plan_busy || !plan_cif',), variant='outlined', prepend_icon='mdi-chart-line')


def planning_handoff_dialog(app):
    with v.VDialog(v_model=('plan_handoff_open',), max_width=800, scrollable=True):
        with v.VCard():
            v.VCardTitle('Review inputs for {{ plan_handoff_target }}')
            with v.VCardText():
                with html.Dl():
                    with html.Div(v_for='row in plan_handoff_rows', key='row.label', classes='my-3'):
                        html.Dt('{{ row.label }}', classes='font-weight-bold')
                        html.Dd('{{ row.value }}', style='overflow-wrap:anywhere')
                with html.Div(v_for='warning in plan_handoff_warnings', key='warning'):
                    note('{{ warning }}')
                number('plan_handoff_source_bank', 'Physical source bank',
                       v_if='plan_handoff_source_bank_required')
                v.VCheckbox(v_if='plan_handoff_parent_choice_required',
                    v_model=('plan_handoff_use_original_parent',),
                    label='Use the original nuclear parent as the starting setup; retain the transition as pending intent',
                    hide_details=True, classes='my-3')
                v.VAlert(v_if='!!plan_handoff_error', text=('plan_handoff_error',),
                         type='warning', variant='tonal', classes='mt-3')
            with v.VCardActions(classes='d-flex flex-wrap'):
                v.VBtn('Download full plan intent', click=app.export_planning_handoff,
                       disabled=('!plan_handoff_intent_available',), variant='text')
                v.VSpacer()
                v.VBtn('Cancel', click=app.cancel_planning_handoff, variant='text')
                v.VBtn('Apply reviewed inputs', click=app.confirm_planning_handoff,
                       disabled=('!plan_handoff_can_apply',), color='#15543c', variant='flat')


def planning_css():
    return '''
    .radar-primary-tabs {background:#fff;border-bottom:1px solid #dce5df;margin-bottom:16px;}
    .radar-primary-tabs .v-tab {text-transform:none;font-weight:650;letter-spacing:0;}
    .planner-page {max-width:1660px;margin:0 auto;padding:8px 8px 28px;min-width:0;}
    .planner-heading {display:flex;align-items:flex-start;justify-content:space-between;gap:20px;margin:8px 0 20px;}
    .planner-beamline {flex:0 0 320px;max-width:100%;padding-top:8px;}
    .planner-eyebrow {font-size:.72rem;letter-spacing:.13em;color:#557066;font-weight:700;margin-bottom:8px;}
    .planner-heading h1 {font-size:1.9rem;line-height:1.25;letter-spacing:-.025em;color:#17362c;}
    .planner-help {font-size:.86rem;line-height:1.55;color:#56675f;margin:8px 0 16px;overflow-wrap:anywhere;}
    .planner-save-bar {display:grid;grid-template-columns:minmax(160px,1fr) auto minmax(180px,1fr) auto;gap:12px;align-items:start;margin-bottom:16px;}
    .planner-save-bar>.v-btn {margin-top:8px;}
    .planner-grid {display:grid;grid-template-columns:minmax(360px, .8fr) minmax(0,1.2fr);gap:22px;align-items:start;}
    .planner-input-card,.planner-results-card {min-width:0;background:white;border:1px solid #dce5df;border-radius:14px;box-shadow:0 4px 18px #17362c05;overflow:hidden;}
    .planner-form-body {padding:22px;}
    .planner-fields {border:0;padding:0;min-width:0;}
    .planner-fields:disabled {opacity:.75;pointer-events:none;}
    .planner-page h2 {font-size:1.12rem;font-weight:650;color:#17362c;margin:0 0 20px;}
    .planner-page h3 {font-size:.95rem;font-weight:600;margin:8px 0 12px;color:#294a3b;}
    .planner-step-tabs {border-bottom:1px solid #e3ebe6;}
    .planner-step-tabs .v-tab {min-width:0;text-transform:none;letter-spacing:0;font-size:.8rem;padding:0 10px;}
    .planner-field-pair {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;}
    .planner-k-fields {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;}
    .planner-step-actions {display:flex;align-items:center;margin-top:24px;}
    .planner-run-bar {padding:18px 22px 8px;border-top:1px solid #e3ebe6;background:#f7faf8;}
    .planner-run-bar .v-btn {text-transform:none;letter-spacing:0;font-weight:600;}
    .planner-results-card {padding:22px;position:sticky;top:12px;}
    .planner-empty-plot {min-height:410px;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:36px;background:linear-gradient(150deg,#f0f7f3,#fbfcfb);border:1px dashed #cbdcd2;border-radius:10px;}
    .planner-empty-plot h3 {font-size:1.15rem;margin-top:20px;}
    .planner-plot {height:520px;width:100%;min-width:0;}
    .planner-site-row {padding:14px;background:#f7faf8;border:1px solid #e3ebe6;border-radius:8px;margin-bottom:14px;}
    .planner-method {text-align:left;width:100%;margin-top:10px;}
    .planner-method-step {border-left:3px solid #bed7c8;padding:4px 0 4px 14px;margin:16px 0;font-size:.85rem;line-height:1.55;color:#40594c;}
    .planner-method-step strong {display:block;color:#204c38;margin-bottom:4px;}
    .planner-method-step p {margin:0;}
    .planner-transition-controls {border-top:1px solid #dce5df;padding-top:12px;margin-top:8px;}
    .planner-parameter {margin:16px 0 22px;}
    .planner-parameter .planner-help {margin:6px 0;}
    .planner-mode-name {display:block;font-size:.73rem;color:#66786e;overflow-wrap:anywhere;}
    .planner-site-card {padding:14px;border:1px solid #dce5df;border-radius:8px;margin-bottom:12px;background:#fbfcfb;}
    .planner-site-title {display:flex;flex-wrap:wrap;justify-content:space-between;gap:8px;font-size:.88rem;}
    .planner-magnitude {color:#15543c;font-variant-numeric:tabular-nums;}
    .planner-site-card .planner-help {font-size:.76rem;margin:8px 0 10px;}
    .planner-component-grid {display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;}
    .planner-component {display:flex;flex-direction:column;gap:4px;padding:8px;background:#edf4ef;border-radius:5px;overflow-wrap:anywhere;font-size:.85rem;}
    .planner-component-label {color:#587363;font-size:.75rem;}
    .planner-component small {color:#587363;font-size:.72rem;}
    .planner-detail {display:grid;grid-template-columns:minmax(100px,.6fr) 1fr;gap:12px;font-size:.85rem;line-height:1.5;border-bottom:1px solid #edf1ee;padding:10px 0;overflow-wrap:anywhere;}
    .planner-page .v-expansion-panel-title {font-size:.88rem;line-height:1.4;}
    .planner-page .v-label {font-size:.9rem;}
    @media(max-width:1100px) {.planner-grid {grid-template-columns:1fr;}.planner-results-card {position:static;}.planner-empty-plot {min-height:240px;}}
    @media(max-width:650px) {.planner-page {padding:4px;}.planner-heading {flex-direction:column;gap:0;}.planner-beamline {flex:auto;width:100%;}.planner-heading h1 {font-size:1.5rem;}.planner-save-bar {grid-template-columns:1fr auto;}.planner-form-body,.planner-results-card {padding:16px;}.planner-field-pair {grid-template-columns:1fr;gap:0;}.planner-plot {height:450px;}}
    '''
