"""Shared conditional-mode inspection in Model exploration and Live."""
from trame.widgets import html,vuetify3 as v


def mode_table(*,prefix,on_select,arguments):
    with html.Details(v_if=f'{prefix}.mode_rows && {prefix}.mode_rows.length',classes='mb-3'):
        html.Summary('Conditional mode estimates')
        html.P('Learned ranges from prepared simulations (Å RMS in the recorded native gauge), '
               'not refined values or calibrated confidence intervals. Native confirmation is required. '
               'Changing this view does not change the model search.',classes='rf-notice')
        v.VSelect(label='Symmetry to inspect',model_value=(prefix+'.mode_group',),
            items=(prefix+'.mode_groups',),update_modelValue=(on_select,arguments),
            variant='outlined',density='compact',hide_details=True,classes='my-2',style='max-width:360px')
        html.P('{{ '+prefix+'.mode_summary }}',classes='rf-notice')
        with html.Div(style='overflow-x:auto'):
            v.VDataTable(items=(prefix+'.mode_visible_rows',),headers=(
                "[{title:'Distortion family',key:'family'},{title:'k',key:'k'},"
                "{title:'Learned range (Å RMS)',key:'amplitude'},{title:'Evidence',key:'status'}]",),
                items_per_page=8,density='compact')
        with html.Details(v_if=prefix+'.mode_notes.length'):
            html.Summary('How to interpret these estimates')
            with html.Ul():
                html.Li('{{ note }}',v_for='note in '+prefix+'.mode_notes',key='note')
