"""Compact presentation of the unchanged Live run-exclusion selection."""

from trame.widgets import html, vuetify3 as vuetify


def run_exclusions_control(*, label, disabled, closable_chips=False, hint=None):
    options = "powgen_run_options.filter(r => powgen_processing_scope === 'all' || powgen_processing_sample_ids.includes(r.sample_id))"
    kwargs = dict(hint=hint, persistent_hint=True) if hint else {}
    with vuetify.VAutocomplete(
        label=label, v_model=("powgen_excluded_runs",), items=(options,),
        item_title="title", item_value="value", multiple=True, chips=True,
        closable_chips=closable_chips, density="compact", variant="outlined",
        disabled=(disabled,), **kwargs,
    ):
        with html.Template(raw_attrs=['v-slot:chip="{ props, item, index }"']):
            with vuetify.VChip(v_if="index < 2", size="small", style="max-width:24ch",
                              raw_attrs=['v-bind="props"', ':aria-label="item.title"']):
                html.Span("{{ item.title }}", style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap")
            html.Span("+{{ powgen_excluded_runs.length - 2 }} more", v_if="index === 2",
                      classes="text-caption ml-1")
    with html.Details(v_if="powgen_excluded_runs.length > 0", classes="radar-soft-panel mb-3"):
        html.Summary("Show all {{ powgen_excluded_runs.length }} excluded entries")
        with html.Div(style="max-height:16rem;overflow:auto", tabindex=0,
                      raw_attrs=['role="region"', 'aria-label="All excluded runs or reduced products"']):
            with html.Ol():
                html.Li("{{ (powgen_run_options.find(r => r.value === run) || {}).title || run }}",
                        v_for="run in powgen_excluded_runs", key="run", style="overflow-wrap:anywhere")
