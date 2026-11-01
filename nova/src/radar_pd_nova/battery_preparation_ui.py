"""Explicit bank pairing and phase-hypothesis setup, before refinement."""
from trame.widgets import html, vuetify3 as v
from .uploads import NamedFileUpload


def preparation_panel(app):
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H3('4 · Prepare a joint-bank model')
        html.P('Choose one exposure, its complementary banks and a reviewed starting model. Preparation preserves the measurements and keeps every refinement parameter fixed.',classes='text-body-2')
        with html.Details(classes='my-3'):
            html.Summary('Reopen a saved prepared model',style='cursor:pointer')
            v.VBtn('Find saved models',click=app.refresh_saved_battery_models,disabled=('battery_busy',),variant='text',size='small')
            v.VSelect(label='Saved prepared model',v_model=('battery_saved_model',),items=('battery_saved_models',),disabled=('battery_busy',),density='compact',variant='outlined')
            html.P('{{ battery_saved_notice }}',classes='text-body-2')
            v.VBtn('Open saved model in refinement',click=app.open_saved_battery_model,disabled=('battery_busy || !battery_saved_model',),variant='outlined',size='small')
        v.VAlert(v_if="battery_instrument === 'PG3'",text='POWGEN profiles can be matched to each run’s recorded reduction settings. Preparation supports individual SLOG/FXYE products with recorded bin-width scaling. Legacy lookup-table profiles require a reviewed compatible profile.',type='info',variant='tonal',density='compact')
        with html.Div():
            with v.VRow(classes='mt-2'):
                with v.VCol(cols=12,md=6):
                    v.VSelect(label='Exposure to prepare',v_model=('battery_model_run',),items=('battery_model_runs',),disabled=('battery_busy',),density='compact',variant='outlined',no_data_text='Inspect resolved acquisition metadata first')
                with v.VCol(cols=12,md=6):
                    v.VSelect(label='Reduced observation file',v_model=('battery_model_data',),items=('battery_model_sources',),disabled=('battery_busy',),density='compact',variant='outlined',no_data_text='Choose an exposure from the inspected family')
            v.VBtn('Inspect physical banks',click=app.inspect_battery_banks,disabled=("battery_busy || !battery_model_data",),variant='outlined',prepend_icon='mdi-magnify')
            v.VDataTable(v_if='battery_model_bank_rows.length',items=('battery_model_bank_rows',),headers=("[{title:'Physical bank',key:'bank'},{title:'Angle (degrees)',key:'angle'},{title:'Records',key:'points'},{title:'Nominal d range (Å)',key:'nominal_d'},{title:'Data check',key:'quality'}]",),density='compact',hide_default_footer=True,items_per_page=6,classes='mt-3')
            html.P('{{ note }}',v_for='note in battery_model_mapping_notes',key='note',classes='text-body-2')
            v.VSelect(label='Banks fitted jointly',v_model=('battery_model_banks',),items=('battery_model_bank_options',),multiple=True,chips=True,disabled=('battery_busy',),density='compact',variant='outlined',classes='mt-3')
            with html.Details(classes='mb-3'):
                html.Summary("Excluded edge-bin policy · {{ battery_allow_invalid_edges ? 'enabled' : 'off' }}",style='cursor:pointer')
                v.VCheckbox(label='Allow invalid intensity or uncertainty bins only outside the selected fit limits',v_model=('battery_allow_invalid_edges',),disabled=('battery_busy',),density='compact',hide_details=True)
                html.P('With this option, preparation must verify that every excluded bin lies outside the fitted interval. Invalid coordinates and invalid bins inside the fit still stop preparation. Original records are retained.',classes='text-body-2')
            for bank in range(1,7):
                with v.VCard(v_show=f'battery_model_banks.includes({bank})',variant='tonal',classes='pa-3 mb-3'):
                    html.H4(f'Physical bank {bank}')
                    v.VSelect(label=f'Peak profile for bank {bank}',v_model=(f'battery_b{bank}_profile',),items=('battery_profile_options',),disabled=('battery_busy',),density='compact',variant='outlined')
                    with html.Div(v_show=f"battery_b{bank}_profile === 'upload'"):
                        app._battery_profile_uploads[bank]=NamedFileUpload(f'battery_b{bank}_path',label=f'Bank {bank} instrument profile',extensions=['.instprm','.prm','.inst','.ins'],max_bytes=1024*1024,key=f'battery-profile-{bank}',help_text='GSAS-II or legacy GSAS peak profile, maximum 1 MiB. Legacy files undergo a checked native conversion. Detector calibration HDF5 files cannot substitute for this file.')
                    with v.VRow(classes='mt-2'):
                        for field,label in [('profile_bank','Bank label inside profile'),('d_min','Fit d minimum (Å)'),('d_max','Fit d maximum (Å)')]:
                            with v.VCol(cols=12,md=4):
                                disabled=f"battery_busy || battery_b{bank}_profile === 'powgen_catalog'" if field=='profile_bank' else 'battery_busy'
                                v.VTextField(label=label,v_model=(f'battery_b{bank}_{field}',),type='text',inputmode='decimal',disabled=(disabled,),density='compact',variant='outlined')
            html.H4('Starting phase hypotheses',classes='mt-4')
            html.P('Each added phase links to every selected bank. HAP scale is a starting intensity coefficient, not a mass fraction. Use separate prepared models to test competing phase families.',classes='text-body-2')
            app._battery_phase_upload=NamedFileUpload('battery_phase_path',label='Phase CIF',extensions=['.cif'],max_bytes=1024*1024,key='battery-phase',help_text='One structural CIF, maximum 1 MiB. The native importer checks its symmetry setting and elemental inventory.')
            with v.VRow(classes='mt-2'):
                with v.VCol(cols=12,md=4):
                    v.VTextField(label='Phase hypothesis name',v_model=('battery_phase_name',),disabled=('battery_busy',),density='compact',variant='outlined')
                with v.VCol(cols=12,md=4):
                    v.VSelect(label='Physical role',v_model=('battery_phase_role',),items=("[{title:'Cathode',value:'cathode'},{title:'Anode',value:'anode'},{title:'Current collector',value:'current_collector'},{title:'Electrolyte',value:'electrolyte'},{title:'Cell component',value:'cell_component'},{title:'Other hypothesis',value:'other'}]",),disabled=('battery_busy',),density='compact',variant='outlined')
                with v.VCol(cols=12,md=4):
                    v.VTextField(label='Starting HAP scale in each bank',v_model=('battery_phase_scale',),type='text',inputmode='decimal',disabled=('battery_busy',),density='compact',variant='outlined')
            with html.Details(classes='mb-3'):
                html.Summary('Optional sample-width starting values',style='cursor:pointer')
                html.P('Apply these isotropic starting values to this phase in every selected bank. The banks remain independent during refinement. Blank fields retain the native imported values, which are recorded in the prepared model. These inputs are hypotheses, not measured particle size or strain.',classes='text-body-2')
                with v.VRow(classes='mt-2'):
                    with v.VCol(cols=12,md=6):
                        v.VTextField(label='Starting sample size in each bank (µm)',v_model=('battery_phase_size_um',),type='text',inputmode='decimal',disabled=('battery_busy',),density='compact',variant='outlined',hint='0.001–10 µm; blank retains the imported value',persistent_hint=True)
                    with v.VCol(cols=12,md=6):
                        v.VTextField(label='Starting sample microstrain in each bank (µstrain)',v_model=('battery_phase_microstrain_ustrain',),type='text',inputmode='decimal',disabled=('battery_busy',),density='compact',variant='outlined',hint='Nonnegative; blank retains the imported value',persistent_hint=True)
            v.VBtn('Add phase hypothesis',click=app.add_battery_phase,disabled=("battery_busy || !battery_phase_path || !battery_phase_name.trim() || !battery_phase_role",),variant='outlined',prepend_icon='mdi-plus')
            with html.Div(v_for='item in battery_model_phases',key='item.key',classes='d-flex align-center ga-2 mt-2'):
                html.Span('{{ item.name }} · {{ item.role.replaceAll("_", " ") }} · starting HAP scale {{ item.hap_scale }} · {{ item.filename }}')
                html.Span('Size: {{ item.sample_widths?.size_um ?? "imported" }} {{ item.sample_widths?.size_um !== undefined ? "µm" : "" }}; microstrain: {{ item.sample_widths?.microstrain_ustrain ?? "imported" }} {{ item.sample_widths?.microstrain_ustrain !== undefined ? "µstrain" : "" }}',classes='text-body-2')
                v.VBtn('Remove',click=(app.remove_battery_phase,'[item.key]'),disabled=('battery_busy',),variant='text',size='small')
            v.VTextField(label='Cell identity for this model',v_model=('battery_model_cell',),disabled=('battery_busy',),density='compact',variant='outlined',classes='mt-4')
            v.VCheckbox(label='Include the reviewed cycler record when available',v_model=('battery_model_include_echem',),disabled=('battery_busy',),density='compact',hide_details=True)
            html.P('The sample and cell must match. The bundle retains the original cycler bytes, clock mapping and a reproducible join over this exposure. Without a bound record, electrical conditions remain unavailable.',classes='text-body-2')
            v.VTextarea(label='Profile mapping and model review notes',v_model=('battery_model_review',),disabled=('battery_busy',),variant='outlined',rows=2,hint='Record why these profiles match the selected banks and acquisition, and any unresolved model assumptions. This does not certify calibration.',persistent_hint=True)
            html.P('{{ battery_model_message }}',role='status',aria_live='polite',classes='text-body-2 mt-3')
            v.VAlert(v_for='note in battery_model_profile_notes',key='note',text=('note',),type='warning',variant='tonal',density='compact',classes='my-2')
            v.VAlert(v_if='!battery_native_available',text='Native preparation requires a configured local worker or signed-in Galaxy History.',type='info',variant='tonal',density='compact',classes='mb-3')
            with html.Div(classes='d-flex flex-wrap ga-2'):
                v.VBtn('Prepare and verify model',click=app.prepare_battery_model,disabled=("battery_busy || !battery_native_available || !battery_model_banks.length || !battery_model_phases.length || !battery_model_cell.trim() || !battery_model_review.trim()",),variant='flat',color='#15543c')
                v.VBtn('Download model and provenance',click=app.download_battery_model,disabled=('battery_busy || !battery_model_ready',),variant='outlined')
                v.VBtn('Review refinement recipe',click=app.open_battery_refinement,disabled=('battery_busy || !battery_model_ready',),variant='outlined')
