"""Battery experiment input review in the existing NOVA workspace."""
from trame.widgets import html, plotly, vuetify3 as v

from .uploads import NamedFileUpload
from .battery.electrochemistry import MAX_BYTES
from .battery_preparation_ui import preparation_panel
from .battery_campaign_ui import portfolio_panel


def battery_dashboard(app):
    with html.Div(classes="radar-battery", style="padding:12px;max-width:1500px;margin:auto"):
        with html.Div(classes="radar-section-heading"):
            with html.Div():
                html.H2("Battery experiments")
                html.P("Bind a cell and its exposure history before comparing evolving electrode models.", classes="text-body-2")
            v.VBtn("Export input evidence", click=app.export_battery_evidence, disabled=("!battery_export_ready || battery_busy",), variant="outlined", size="small", prepend_icon="mdi-download")
        v.VBtn("Back to analysis setup", click=app.close_battery_workspace, variant="text", size="small", prepend_icon="mdi-arrow-left", classes="mb-3")
        v.VAlert("Input qualification is separate from refinement acceptance. A folder name, voltage channel or low Rwp does not establish a valid battery model.", type="info", variant="tonal", density="compact", classes="mb-3")
        v.VAlert(v_if="!!battery_error", text=("battery_error",), type="error", variant="tonal", density="compact", classes="mb-3")
        html.P("{{ battery_message }}", role="status", aria_live="polite", classes="text-body-2")
        v.VProgressLinear(v_if="battery_busy", indeterminate=True, color="#15543c", aria_label="Battery operation in progress", classes="mb-3")
        with v.VCard(variant="outlined", classes="pa-4 mb-4"):
            html.H3("1 · Find the experiment")
            with v.VRow(classes="mt-2"):
                with v.VCol(cols=12, md=3):
                    v.VSelect(label="Battery beamline", v_model=("battery_instrument",), items=("[{title:'NOMAD',value:'NOM'},{title:'POWGEN',value:'PG3'}]",), density="compact", variant="outlined", disabled=("battery_busy",), hide_details=True)
                with v.VCol(cols=12, md=3):
                    v.VTextField(label="Battery IPTS", v_model=("battery_ipts",), placeholder="IPTS-21886", density="compact", variant="outlined", disabled=("battery_busy",), hide_details=True)
                with v.VCol(cols=12, md=6):
                    v.VBtn("Discover proposal sources", click=(app.discover_battery_proposal,"[{instrument:battery_instrument,ipts:battery_ipts}]"), disabled=("battery_busy || !battery_ipts.trim()",), loading=("battery_busy && battery_operation === 'discover'",), color="#15543c", variant="flat", prepend_icon="mdi-folder-search-outline")
            html.P("The archive is read only. Discovery checks known layouts and shallow auxiliary folders; it does not load detector events.", classes="text-caption mt-2")
            html.P('{{ battery_catalog_message }}',role='status',aria_live='polite',classes='text-body-2 mt-2')
            html.P('{{ battery_catalog_proposal_message }}',v_if='battery_catalog_proposal_message',classes='text-caption')
            with html.Details(v_if='!!battery_catalog_title || !!battery_catalog_abstract'):
                html.Summary('Catalog proposal context · {{ battery_catalog_title }}')
                html.P('{{ battery_catalog_abstract || "Abstract unavailable with current access." }}',classes='text-body-2',style='white-space:pre-wrap')
                html.P('{{ battery_catalog_source }}',classes='text-caption')
                html.P('Recorded proposal purpose and registered chemistry are context. They do not select phases, establish measured temperature, bind a cell or qualify calibration.',classes='text-body-2')
            v.VAlert(v_for='item in battery_catalog_issues',key='item',text=('item',),type='warning',variant='tonal',density='compact')
            v.VAlert(v_for="item in battery_issues", key="item.code + item.path", text=("item.message + (item.path ? ' · ' + item.path : '')",), type="warning", variant="tonal", density="compact", classes="mt-2")
            v.VSelect(label="Diffraction reduction family", v_model=("battery_family",), items=("battery_families",), density="compact", variant="outlined", disabled=("battery_busy || !battery_families.length",), no_data_text="Discover a proposal to list its reduction families", hint="Choose one reduction family. Regenerated products are kept separate.", persistent_hint=True, classes="mt-4")
            v.VTextField(label="Exact run number (optional)", v_model=("battery_exact_run",), type="text", inputmode="numeric", density="compact", variant="outlined", disabled=("battery_busy || !battery_family",), hint="The run must have an individual reduced observation in the selected folder. Leave blank for representative sampling. This reads metadata only; it does not prepare or submit an analysis.", persistent_hint=True, classes="mt-3")
            v.VBtn("Inspect acquisition metadata", click=app.inspect_battery_acquisitions, disabled=("battery_busy || !battery_family",), loading=("battery_busy && battery_operation === 'inspect'",), variant="outlined", prepend_icon="mdi-magnify")
            with html.Details(classes="my-3"):
                html.Summary("Inspection sample · up to {{ battery_inspection_limit }} representative runs", style="cursor:pointer")
                v.VTextField(label="Representative runs to inspect", v_model=("battery_inspection_limit",), type="text", inputmode="decimal", min=1, max=200, density="compact", variant="outlined", disabled=("battery_busy || !!battery_exact_run.trim()",), hint="Used when Exact run number is blank. Evenly sampled across the selected family, up to 200.", persistent_hint=True, classes="mt-3")
            with html.Details(v_if="battery_metadata_sources.length"):
                html.Summary("Metadata and calibration candidates ({{ battery_metadata_sources.length }})")
                v.VDataTable(items=("battery_metadata_sources",), headers=("[{title:'Kind',key:'kind'},{title:'Source within IPTS',key:'path'},{title:'Recorded setup references',key:'summary'}]",), density="compact", items_per_page=5)
        with v.VCard(variant="outlined", classes="pa-4 mb-4"):
            html.H3("2 · Review samples and exposure intervals")
            html.P("References, empty cells and several batteries can share an IPTS or reduction folder. Review titles as well as sample identifiers.", classes="text-body-2")
            html.P("Inspected {{ battery_rows.length }} of {{ battery_total_runs }} available runs. This table is the selected metadata sample, not a complete series analysis.", v_if="battery_rows.length", classes="text-caption")
            v.VDataTable(items=("battery_rows",), headers=("[{title:'Run',key:'run'},{title:'Sample ID',key:'sample_id'},{title:'Recorded title',key:'title'},{title:'Exposure start (UTC)',key:'start'},{title:'Seconds',key:'duration'},{title:'State / identity',key:'issues'},{title:'Catalog comparison',key:'catalog'},{title:'Mean voltage',key:'voltage'},{title:'Electrical coverage',key:'coverage'},{title:'Branch',key:'branch'}]",), density="compact", items_per_page=10, no_data_text="Select a reduction family and inspect its acquisition metadata.", style="overflow-x:auto")
            with html.Details(v_if='battery_catalog_rows.length',classes='mt-3'):
                html.Summary('Compare catalog and file metadata')
                html.P('Disagreements remain visible. Catalog-only fields do not satisfy the verified sample or exposure requirements. Full captured values and retrieval sources are retained in Export input evidence.',classes='text-body-2')
                v.VDataTable(items=('battery_catalog_rows',),headers=("[{title:'Run',key:'run'},{title:'Field',key:'field'},{title:'File metadata',key:'archive'},{title:'Catalog context',key:'catalog'},{title:'Comparison',key:'status'}]",),density='compact',items_per_page=10)
        with v.VCard(variant="outlined", classes="pa-4 mb-4"):
            html.H3("3 · Bind the electrochemical record")
            html.P("Choose the file and record which sample, cell and channel it measures. Missing electrical data leaves the diffraction timeline usable without inventing SOC.", classes="text-body-2")
            v.VSelect(label="Cycler source", v_model=("battery_echem_source",), items=("[{title:'Discovered in this IPTS',value:'archive'},{title:'Upload a cycler file',value:'upload'}]",), density="compact", variant="outlined", disabled=("battery_busy",))
            v.VSelect(v_show="battery_echem_source === 'archive'", label="Cycler file or ZIP member", v_model=("battery_cycler",), items=("battery_cycler_options",), density="compact", variant="outlined", disabled=("battery_busy",), no_data_text="No cycler records found in the inspected folders. Upload a file or inspect its known location.", hint="Each ZIP member remains a separate, unbound source.", persistent_hint=True)
            with html.Div(v_show="battery_echem_source === 'upload'"):
                app._battery_upload = NamedFileUpload("battery_cycler_path", label="Electrochemistry file", extensions=[".csv", ".tsv", ".mpt", ".mpr"], help_text="CSV with named units, EC-Lab MPT or BioLogic MPR. Maximum 64 MiB.", key="battery-cycler-upload", max_bytes=MAX_BYTES)
            with v.VRow(classes="mt-2"):
                with v.VCol(cols=12, md=4):
                    v.VSelect(label="Verified diffraction sample ID", v_model=("battery_sample",), items=("battery_sample_options",), density="compact", variant="outlined", disabled=("battery_busy",), no_data_text="Inspect acquisition metadata first")
                with v.VCol(cols=12, md=4):
                    v.VTextField(label="Cell identity", v_model=("battery_cell",), density="compact", variant="outlined", disabled=("battery_busy",), hint="A cell label from your experiment records", persistent_hint=True)
                with v.VCol(cols=12, md=4):
                    v.VTextField(label="Cycler channel", v_model=("battery_channel",), density="compact", variant="outlined", disabled=("battery_busy",), hint="Exact channel if the table contains several channels", persistent_hint=True)
            with v.VRow():
                with v.VCol(cols=12, md=6):
                    v.VSelect(label="Positive recorded current means", v_model=("battery_positive",), items=("[{title:'Charge',value:'charge'},{title:'Discharge',value:'discharge'}]",), density="compact", variant="outlined", disabled=("battery_busy",))
                with v.VCol(cols=12, md=6):
                    v.VTextField(label="Cycler start with UTC offset", v_model=("battery_origin",), placeholder="2019-12-09T12:00:26-05:00", density="compact", variant="outlined", disabled=("battery_busy",), hint="Required for elapsed-time files. A recorded local clock alone does not establish its timezone.", persistent_hint=True)
            with html.Details(classes="mb-3"):
                html.Summary("Calendar-date exports instead of elapsed seconds")
                html.P("Some EC-Lab exports put calendar dates in a column labelled time/s. Select the recorded convention and timezone; clear the elapsed-time start above. Daylight-saving ambiguity requires explicit UTC timestamps.", classes="text-body-2")
                with v.VRow():
                    with v.VCol(cols=12, md=6):
                        v.VSelect(label="Recorded date format", v_model=("battery_date_format",), items=("[{title:'Elapsed seconds or timezone-qualified ISO timestamps',value:''},{title:'Month/day/year hours:minutes:seconds',value:'month/day/year'},{title:'Day/month/year hours:minutes:seconds',value:'day/month/year'},{title:'Year-month-day hours:minutes:seconds',value:'year-month-day'}]",), density="compact", variant="outlined", disabled=("battery_busy",))
                    with v.VCol(cols=12, md=6):
                        v.VTextField(label="Recorded clock timezone", v_model=("battery_timezone",), placeholder="America/New_York", density="compact", variant="outlined", disabled=("battery_busy",))
            with html.Details(classes="mb-3"):
                html.Summary("Clock alignment and gaps · correction {{ battery_offset }} s · uncertainty {{ battery_clock_uncertainty === null || battery_clock_uncertainty === '' ? 'unknown' : battery_clock_uncertainty + ' s' }} · gap limit {{ battery_gap }} s", style="cursor:pointer")
                with v.VRow(classes="mt-2"):
                    with v.VCol(cols=12, md=4):
                        v.VTextField(label="Clock correction (seconds)", v_model=("battery_offset",), type="text", inputmode="decimal", density="compact", variant="outlined", disabled=("battery_busy",), hint="Added to the cycler timestamps", persistent_hint=True)
                    with v.VCol(cols=12, md=4):
                        v.VTextField(label="Clock uncertainty (seconds)", v_model=("battery_clock_uncertainty",), type="text", inputmode="decimal", min=0, density="compact", variant="outlined", disabled=("battery_busy",), hint="Leave blank when unknown; zero means an explicitly exact mapping.", persistent_hint=True)
                    with v.VCol(cols=12, md=4):
                        v.VTextField(label="Largest joined log interval (seconds)", v_model=("battery_gap",), type="text", inputmode="decimal", min=0, max=86400, density="compact", variant="outlined", disabled=("battery_busy",), hint="Longer gaps and protocol changes are excluded", persistent_hint=True)
            v.VBtn("Bind and join exposure intervals", click=(app.join_battery_cycler,"[{sample:battery_sample,cell:battery_cell,channel:battery_channel,positive:battery_positive,origin:battery_origin,offset:battery_offset,clock_uncertainty:battery_clock_uncertainty,gap:battery_gap,date_format:battery_date_format,timezone:battery_timezone}]"), disabled=("battery_busy || !battery_rows.length || !battery_sample || !battery_cell.trim() || !battery_channel.trim() || !battery_positive || (battery_echem_source === 'archive' ? !battery_cycler : !battery_cycler_path)",), loading=("battery_busy && battery_operation === 'join'",), variant="flat", color="#15543c", prepend_icon="mdi-link-variant")
            v.VAlert(v_if="!!battery_echem_summary", text=("battery_echem_summary",), type="info", variant="tonal", density="compact", classes="mt-3")
            with html.Div(v_if="!!battery_echem_summary", style="height:440px;width:100%", classes="mt-3"):
                figure = plotly.Figure(display_mode_bar=True, responsive=True, display_logo=False,
                    scroll_zoom=True, style="height:100%;width:100%")
                figure.data = ("battery_timeline.data",)
                figure.layout = ("battery_timeline.layout",)
            with html.Details(classes="mt-3"):
                html.Summary("Supported CSV columns and coverage meaning")
                html.P("Use timestamp (ISO 8601 with UTC offset) and voltage_v; or elapsed_seconds plus a qualified start. Add current_a or current_ma when recorded. Optional cell_id, channel, cycle and step columns preserve their recorded identities. Missing current leaves charge and branch unresolved; control setpoints and power are not silently treated as measured current.", classes="text-body-2")
                html.P("Coverage is the fraction of each diffraction exposure supported by this electrical record. Voltage means and signed charge use only covered segments. Long gaps and intervals spanning a protocol change are excluded. No extrapolation is made before or after the log.", classes="text-body-2")
        preparation_panel(app)
        portfolio_panel(app)
