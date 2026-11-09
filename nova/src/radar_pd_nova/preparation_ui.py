"""Before-measurement model preparation, outside the scientific-results tabs."""
from trame.widgets import html,vuetify3 as v
from .uploads import NamedFileUpload


def preparation_dashboard(app):
    with html.Div(id='radar-preparation-workspace',style='max-width:1400px;margin:auto;padding:12px'):
        v.VBtn('Back to analysis',click=app.close_preparation_workspace,prepend_icon='mdi-arrow-left',variant='text',size='small')
        html.H1('Experiment preparation',classes='mt-2')
        html.P('Prepare reusable magnetic and structural model calculations from a nuclear CIF and bank profiles before the experiment starts.',classes='text-body-1')
        html.P('No measured scan is needed. Prepared calculations are checked against the actual structure and instrument context before they can support a fit.',classes='text-body-2')
        v.VAlert(v_if='!!prep_error',text=('prep_error',),type='error',variant='tonal',classes='my-3')
        html.P('{{ prep_message }}',role='status',aria_live='polite',classes='text-body-2')
        v.VProgressLinear(v_if='prep_busy',indeterminate=True,aria_label='Preparation operation in progress',classes='mb-3')
        with v.VTabs(v_model=('prep_tab',),color='#15543c',classes='mb-4'):
            v.VTab('New preparation',value='new');v.VTab('Libraries and recovery',value='libraries')
        with html.Div(v_show="prep_tab === 'new'"):
            inputs_panel(app)
            scope_panel(app)
        with html.Div(v_show="prep_tab === 'libraries'"):
            libraries_panel(app)


def inputs_panel(app):
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H2('1 · Inspect the parent and profiles',classes='text-h6')
        html.P('Use one nuclear structure and add each bank or detector group you expect to analyze. The native worker reads its exact symmetry and checks the profile type.',classes='text-body-2')
        v.VTextField(label='Preparation name',v_model=('prep_name',),maxlength=160,disabled=('prep_busy',),variant='outlined',density='compact')
        app._prep_parent_upload=NamedFileUpload('prep_parent_path',label='Nuclear parent CIF',extensions=['.cif'],
            key='preparation-parent-upload',max_bytes=10*1024*1024,help_text='One unmodulated nuclear structure, up to 10 MiB. Magnetic candidates are generated from its exact parent setting.')
        with html.Details(classes='my-3',open=True):
            html.Summary('Add an instrument-profile context',style='cursor:pointer')
            app._prep_profile_upload=NamedFileUpload('prep_profile_path',label='GSAS-II instrument profile',
                extensions=['.instprm','.prm','.ins','.inst'],key='preparation-profile-upload',max_bytes=1024*1024,
                help_text='A reviewed profile for this physical bank, up to 1 MiB. Several banks can use different blocks of the same file.')
            with v.VRow(classes='mt-2'):
                with v.VCol(cols=12,md=5):v.VTextField(label='Bank or detector-group name',v_model=('prep_bank_label',),placeholder='POWGEN bank 3',variant='outlined',density='compact',disabled=('prep_busy',))
                with v.VCol(cols=12,md=3):v.VTextField(label='Profile block / bank number',v_model=('prep_profile_bank',),type='text',inputmode='numeric',variant='outlined',density='compact',disabled=('prep_busy',))
                with v.VCol(cols=12,md=4):v.VSelect(label='Profile type',v_model=('prep_grid_kind',),items=("[{title:'Time of flight',value:'PNT'},{title:'Constant wavelength',value:'PNC'}]",),variant='outlined',density='compact',disabled=('prep_busy',))
            with html.Details(classes='mb-3'):
                html.Summary('Expected calculation range and grid',style='cursor:pointer')
                html.P('This defines a reference calculation, not measured counts. Changing the actual range or instrument later can require new rendering. TOF values are microseconds; constant-wavelength values are degrees 2θ.',classes='text-body-2')
                with v.VRow():
                    with v.VCol(cols=12,md=4):v.VTextField(label='Lower limit',v_model=('prep_lower',),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',))
                    with v.VCol(cols=12,md=4):v.VTextField(label='Upper limit',v_model=('prep_upper',),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',))
                    with v.VCol(cols=12,md=4):v.VTextField(label='Reference bins',v_model=('prep_bins',),type='text',inputmode='numeric',variant='outlined',density='compact',disabled=('prep_busy',))
            v.VBtn('Add bank context',click=app.add_preparation_bank,disabled=('prep_busy || !prep_profile_path',),variant='outlined',prepend_icon='mdi-plus')
        with html.Div(v_for='bank in prep_bank_rows',key='bank.label',classes='d-flex align-center my-2',style='gap:12px;flex-wrap:wrap'):
            html.Span('{{ bank.label }} · {{ bank.profile }} · profile bank {{ bank.bank }} · {{ bank.kind }} · {{ bank.range }}',classes='text-body-2')
            v.VBtn('Remove',click=(app.remove_preparation_bank,'[bank.label]'),disabled=('prep_busy',),variant='text',size='small')
        v.VBtn('Inspect parent and profiles',id='preparation-inspect',click=app.inspect_preparation_inputs,
            disabled=('prep_busy || !prep_parent_path || !prep_bank_rows.length',),loading=("prep_busy && prep_operation === 'Inspecting the parent and bank profiles'",),color='#15543c',variant='flat',prepend_icon='mdi-magnify')
        html.P('Inspection runs as a Galaxy job. Existing analyses and their Results tabs stay available.',classes='text-caption mt-2')


def scope_panel(app):
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H2('2 · Choose the model scope',classes='text-h6')
        v.VSelect(label='Completed parent/profile inspection',v_model=('prep_reference',),items=('prep_reference_options',),
            disabled=('prep_busy',),variant='outlined',density='compact',no_data_text='Inspect inputs above, or recover an existing inspection in Libraries.')
        html.P('{{ prep_parent_title }}',v_if='prep_context_ready',classes='text-body-1 font-weight-bold')
        html.P('The available k vectors and sites appear after native inspection finishes.',v_if='!prep_context_ready',classes='text-body-2')
        with html.Div(v_show='prep_context_ready'):
            with v.VRow():
                with v.VCol(cols=12,md=6):v.VSelect(label='Models to prepare',v_model=('prep_kind',),items=("[{title:'Commensurate magnetic models',value:'magnetic'},{title:'Structural displacement models',value:'structural'}]",),variant='outlined',density='compact',disabled=('prep_busy',))
                with v.VCol(cols=12,md=6):v.VSelect(label='Banks in this preparation',v_model=('prep_selected_banks',),items=('prep_inspected_banks',),multiple=True,chips=True,variant='outlined',density='compact',disabled=('prep_busy',))
            v.VSelect(label='k vectors in the inspected parent basis',v_model=('prep_selected_k',),items=('prep_k_options',),multiple=True,chips=True,variant='outlined',density='compact',disabled=('prep_busy',),hint='This is a finite commensurate scope. It does not cover every rational vector, multi-k state or incommensurate structure.',persistent_hint=True)
            with html.Details(classes='my-3'):
                html.Summary('Add another exact rational k vector',style='cursor:pointer')
                with html.Div(classes='d-flex align-center',style='gap:12px;flex-wrap:wrap'):
                    v.VTextField(label='Parent-basis k',v_model=('prep_custom_k',),placeholder='0, 0, 1/3',variant='outlined',density='compact',disabled=('prep_busy',),style='max-width:360px')
                    v.VBtn('Add k',click=app.add_preparation_k,disabled=('prep_busy || !prep_custom_k.trim()',),variant='outlined',size='small')
            with html.Div(v_show="prep_kind === 'magnetic'"):
                v.VSelect(label='Magnetic parent sites',v_model=('prep_sites',),items=('prep_site_options',),multiple=True,chips=True,variant='outlined',density='compact',disabled=('prep_busy',))
                html.P('Review the ionic form factor and g for each selected site. An uncharged CIF atom label does not uniquely determine its oxidation state. The initial g = 2 assumption should be reviewed, especially for rare-earth ions.',classes='text-body-2')
                with v.VRow(v_for='site in prep_site_rows',key='site.label'):
                    with v.VCol(cols=12,md=4):html.P('{{ site.label }} · {{ site.element }}',classes='mt-2')
                    with v.VCol(cols=12,md=4):v.VTextField(label='Ionic form factor',model_value=('site.form_factor',),update_modelValue=(app.set_preparation_ion,'[site.label, $event]'),placeholder='Ni+2',variant='outlined',density='compact',disabled=('prep_busy',))
                    with v.VCol(cols=12,md=4):v.VTextField(label='Landé g',model_value=('site.lande_g',),update_modelValue=(app.set_preparation_g,'[site.label, $event]'),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',))
                v.VTextField(label='Physical moment ceiling (μB per atom)',v_model=('prep_maximum_moment',),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',),style='max-width:420px')
                html.P('Bilbao receives the parent symmetry operations and requested k vectors. The service key remains in this session’s backend.',classes='text-caption')
            with html.Div(v_show="prep_kind === 'structural'"):
                html.P('ISODISTORT receives the parent structure—cell, symmetry and atomic sites—to generate displacement-mode children. No measured diffraction pattern is uploaded.',classes='text-body-2')
                html.P('Preparation also generates native patterns along 64 independent distortion/lattice trajectories per model and bank, then trains a compound-specific ranking model. Pattern generation, training and any failures appear separately in Libraries. The model orders candidates for native refinement; it does not establish a structure or discard alternatives.',classes='text-body-2')
                v.VSelect(label='Structural candidate scope',v_model=('prep_opd_policy',),items=("[{title:'Highest symmetry per irrep',value:'highest-symmetry-per-irrep'},{title:'All returned order-parameter directions',value:'all'}]",),variant='outlined',density='compact',disabled=('prep_busy',))
                with html.Details(classes='my-3'):
                    html.Summary('Structural construction and refinement limits',style='cursor:pointer')
                    html.P('The construction limit must fit the declared scope; the workflow will not silently drop candidates to meet it. The displacement and cell limits bound the later continuous refinement as well as its starting models. All selected modes remain free within those limits. Starting curves help initialization; nonlinear fitting still uses the actual scan.',classes='text-body-2')
                    with v.VRow():
                        with v.VCol(cols=12,md=4,style='box-sizing:border-box;min-width:0'):v.VTextField(label='Maximum constructed candidates',v_model=('prep_construct_limit',),type='text',inputmode='numeric',variant='outlined',density='compact',disabled=('prep_busy',))
                        with v.VCol(cols=12,md=4,style='box-sizing:border-box;min-width:0'):v.VTextField(label='Displacement-coordinate bound (Å RMS)',v_model=('prep_mode_radius',),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',),hint='Per independent normalized displacement coordinate. Several coordinates can contribute to an atom’s displacement.',persistent_hint=True)
                        with v.VCol(cols=12,md=4,style='box-sizing:border-box;min-width:0'):v.VTextField(label='Reciprocal-metric search radius',v_model=('prep_cell_radius',),type='text',inputmode='decimal',variant='outlined',density='compact',disabled=('prep_busy',),hint='Dimensionless reciprocal-cell bound. 0.01 does not mean ±1% of each lattice constant; accepted cells are checked separately.',persistent_hint=True)
            v.VBtn('Discover model families',id='preparation-discover',click=app.discover_preparation_models,disabled=('prep_busy || !prep_context_ready || !prep_selected_k.length',),variant='outlined',prepend_icon='mdi-file-search-outline')
            v.VDataTable(v_if='prep_discovery_rows.length',items=('prep_discovery_rows',),headers=("[{title:'Parent-basis k',key:'k'},{title:'Constructed candidates',key:'count'},{title:'Discovery status',key:'state'}]",),density='compact',items_per_page=10,classes='my-3')
            html.P('{{ prep_candidate_count }} model/bank contexts in the selected scope. Structural training tasks are added automatically.',v_if='prep_discovery_ready',classes='text-body-1 font-weight-bold')
            v.VSelect(label='Concurrent Galaxy preparation jobs',v_model=('prep_parallel_jobs',),items=([1,2,4,8],),variant='outlined',density='compact',disabled=('prep_busy',),style='max-width:420px',hint='Native process pools respect the CPUs and memory actually granted by NDIP. This setting does not reserve resources.',persistent_hint=True)
            v.VBtn('Prepare selected models',id='preparation-start',click=app.start_preparation_models,disabled=('prep_busy || !prep_discovery_ready || !prep_selected_banks.length',),color='#15543c',variant='flat',prepend_icon='mdi-play',classes='mt-3')


def libraries_panel(app):
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H2('Prepared libraries',classes='text-h6')
        v.VSelect(label='Library or input inspection',v_model=('prep_library',),items=('prep_library_options',),disabled=('prep_busy',),variant='outlined',density='compact')
        html.P('{{ prep_library_kind }} · {{ prep_coverage }}',v_if='!!prep_library',classes='text-body-2')
        html.P('{{ prep_active_jobs }} active Galaxy jobs attached to this workspace.',v_if='!!prep_library',classes='text-caption')
        with html.Div(v_if='!!prep_library',classes='d-flex flex-wrap my-3',style='gap:8px'):
            v.VBtn('Pause new batches',click=app.pause_preparation,disabled=('prep_busy || !prep_can_pause',),variant='outlined',size='small',prepend_icon='mdi-pause')
            v.VBtn('Resume pending work',click=app.resume_preparation,disabled=('prep_busy || !prep_can_resume',),variant='outlined',size='small',prepend_icon='mdi-play')
            v.VBtn('Inspect captured inputs',click=app.inspect_preparation_library,disabled=('prep_busy',),variant='text',size='small',prepend_icon='mdi-magnify')
            v.VBtn('Save portable library to History',click=app.save_preparation_download,disabled=('prep_busy',),variant='outlined',size='small',prepend_icon='mdi-download')
        with html.A(v_if='!!prep_download_href',href=('prep_download_href',),target='_blank',rel='noopener noreferrer',classes='radar-guide-link'):
            html.Span('Download the saved preparation ZIP')
        html.P('Prepared counts describe reference calculations, not accepted experimental fits. Actual-scan compatibility and native qualification are required before scientific use.',classes='text-body-2')
        with html.Div(classes='d-flex flex-wrap my-3',style='gap:8px'):
            v.VChip('{{ stage.title }}: {{ stage.prepared }} / {{ stage.total }}',v_for='stage in prep_stage_rows',
                key='stage.kind',variant='tonal',color='#15543c')
        with v.VRow():
            with v.VCol(cols=12,md=4):v.VTextField(label='Find a model, bank or message',v_model=('prep_entry_search',),variant='outlined',density='compact')
            with v.VCol(cols=12,md=4):v.VSelect(label='Preparation stage',v_model=('prep_stage_filter',),
                items=("[{title:'All stages',value:'all'},...prep_stage_rows.map(row=>({title:row.title,value:row.kind}))]",),
                variant='outlined',density='compact')
            with v.VCol(cols=12,md=4):v.VSelect(label='Entry status',v_model=('prep_entry_filter',),items=([dict(title='All entries',value='all')]+[dict(title=x.capitalize(),value=x) for x in ['pending','preparing','prepared','failed','cancelled','unsupported']],),variant='outlined',density='compact')
        with v.VTable(density='compact',style='overflow-x:auto'):
            with html.Thead():
                with html.Tr():
                    for title in ['Bank','Model','k','Status','Message','Action']:html.Th(title)
            with html.Tbody():
                with html.Tr(v_for='item in prep_entries',key='item.id'):
                    html.Td('{{ item.bank }}')
                    with html.Td():
                        html.Div('{{ item.model }}')
                        html.Div('{{ item.coordinates }}',v_if='!!item.coordinates',classes='text-caption')
                        with html.Details(v_if='item.validation && item.validation.length',classes='text-caption my-2'):
                            html.Summary('Held-out model validation',style='cursor:pointer')
                            html.P('Top-five recall: at least one compatible space group appears among the first five. These are simulation results, not measured experimental accuracy.',classes='my-2',style='max-width:560px')
                            with html.Div(v_for='metric in item.validation',key='metric.title',classes='my-1'):
                                html.Strong('{{ metric.title }}: {{ metric.value }}')
                                html.Span(' · {{ metric.count }}')
                                html.Div('{{ metric.detail }}',v_if='!!metric.detail')
                            html.P('Class-balanced recall gives each represented class equal weight; the lowest-class value exposes weakly recovered classes.',classes='my-2',style='max-width:560px')
                            html.P('The model prioritizes native refinement. Every reviewed candidate remains available.',classes='my-2',style='max-width:560px')
                        with html.Details(v_if='!!item.candidate',classes='text-caption'):
                            html.Summary('Model identity',style='cursor:pointer')
                            html.Div('{{ item.candidate }}',style='overflow-wrap:anywhere')
                            html.Div('Basis: {{ item.basis }}')
                            html.Div('Origin: {{ item.origin }}')
                    html.Td('{{ item.k || "Not recorded" }}');html.Td('{{ item.state }}');html.Td('{{ item.reason }}')
                    with html.Td():v.VBtn('Retry',v_if="['failed','cancelled','unsupported'].includes(item.state)",click=(app.retry_preparation_entry,'[item.id]'),disabled=('prep_busy',),variant='text',size='small')
        html.P('{{ prep_table_count }} matching entries · 25 per page',classes='text-caption mt-2')
        v.VPagination(v_model=('prep_page',),length=('prep_pages',),total_visible=7)
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H2('Recover the same work',classes='text-h6')
        html.P('Use the latest saved checkpoint in this Galaxy History to reconnect original job identities and recover completed results without submitting duplicates.',classes='text-body-2')
        v.VBtn('Find saved preparation',click=app.refresh_preparation_saved,disabled=('prep_busy',),variant='outlined',size='small')
        v.VSelect(label='Saved preparation checkpoint',v_model=('prep_saved',),items=('prep_saved_options',),variant='outlined',density='compact',disabled=('prep_busy',),classes='mt-3')
        v.VBtn('Recover original work',click=app.recover_preparation,disabled=('prep_busy || !prep_saved',),variant='outlined',size='small')
    with v.VCard(variant='outlined',classes='pa-4 mb-4'):
        html.H2('Import a portable copy',classes='text-h6')
        html.P('Completed calculations travel with the ZIP. Running jobs do not: importing makes an independent copy. Use recovery above when you want to reconnect the original work.',classes='text-body-2')
        v.VSelect(label='Portable library source',v_model=('prep_import_source',),items=("[{title:'Upload from this computer · up to 64 MiB',value:'upload'},{title:'ZIP already in this Galaxy History · up to 2 GiB',value:'history'}]",),variant='outlined',density='compact',disabled=('prep_busy',))
        with html.Div(v_show="prep_import_source === 'upload'"):
            app._prep_library_upload=NamedFileUpload('prep_import_path',label='Preparation library ZIP',extensions=['.zip'],key='preparation-library-upload',max_bytes=64*1024*1024,help_text='For larger libraries, upload the ZIP using Galaxy and select it from this History.')
        v.VSelect(v_show="prep_import_source === 'history'",label='ZIP in this History',v_model=('prep_history_archive',),items=('prep_history_archives',),variant='outlined',density='compact',disabled=('prep_busy',),no_data_text='Use Find saved preparation above to refresh available ZIPs.')
        v.VBtn('Import portable library',click=app.import_preparation,disabled=("prep_busy || (prep_import_source === 'upload' ? !prep_import_path : !prep_history_archive)",),variant='outlined',size='small')
        html.P('Keep this session open to admit future batches. Acknowledged Galaxy jobs can finish after closing; a fresh session can recover their receipts and resume the remaining finite plan.',classes='text-caption mt-3')
