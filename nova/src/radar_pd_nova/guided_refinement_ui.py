from trame.widgets import html, vuetify3 as v
from .uploads import NamedFileUpload


def guided_setup(app):
    with html.Details(classes='mt-3'):
        html.Summary('Correct the phase model',style='cursor:pointer')
        html.P('Inspect the GPX first. Each correction creates a separate starting model and requires a new recipe. Replacement retains the component name and bank links; review the new structure’s width and texture settings.',classes='rf-notice')
        v.VSelect(label='Correction',v_model=('rf_edit_operation',),items=("[{title:'Replace phase from CIF',value:'replace'},{title:'Add phase from CIF',value:'add'},{title:'Remove phase',value:'remove'}]",),density='compact')
        v.VSelect(label='Existing phase',v_model=('rf_edit_phase',),items=('rf_edit_phase_options',),v_if="rf_edit_operation !== 'add'",density='compact')
        with html.Div(v_show="rf_edit_operation !== 'remove'"):
            NamedFileUpload('rf_edit_cif_path',label='Phase CIF',extensions=['.cif'],max_bytes=1024*1024,help_text='One structure per CIF; maximum 1 MiB.',key='guided-phase-cif')
            with html.Details():
                html.Summary('Or paste CIF text (clear the uploaded file first)')
                v.VTextarea(label='CIF text',v_model=('rf_edit_cif_text',),rows=4)
        with html.Div(v_if="rf_edit_operation === 'add'"):
            v.VTextField(label='Distinct component name',v_model=('rf_edit_name',))
            v.VSelect(label='Link to banks',v_model=('rf_edit_banks',),items=('rf_edit_bank_options',),multiple=True,chips=True)
            v.VTextField(label='Initial phase scale in each selected bank',v_model=('rf_edit_scale',),type='text',inputmode='decimal')
        v.VTextField(label='Reason for correction',v_model=('rf_edit_reason',))
        v.VBtn('Create corrected model',click=(app.correct_refinement_phase,
            '[{source:rf_source,source_sha256:rf_edit_source_sha,operation:rf_edit_operation,phase:rf_edit_phase,name:rf_edit_name,cif_path:rf_edit_cif_path,cif_text:rf_edit_cif_text,banks:rf_edit_banks,scale:rf_edit_scale,reason:rf_edit_reason}]'),
            disabled=('rf_busy || rf_guided_busy || !rf_edit_phase_options.length',),variant='outlined',size='small')
    v.VSelect(label='Recipe setup',v_model=('rf_setup_mode',),items=("[{title:'Guided recipes',value:'guided'},{title:'Expert request',value:'expert'}]",),density='compact',classes='mt-4')
    with html.Div(v_show="rf_setup_mode === 'guided'",classes='rf-source'):
        html.P('Set up once, then apply the approved stages to compatible selected scans and optionally future arrivals. Occupancy recipes ask whether the held reference applies to the series. Direct answers need no model call.',classes='rf-notice')
        v.VBtn('Start / restart guided setup',click=app.start_guided_refinement,disabled=('rf_busy || rf_guided_busy || !rf_source',),variant='outlined',size='small')
        v.VProgressLinear(v_if='rf_guided_busy',indeterminate=True,color='#287654',classes='my-3')
        with html.Div(v_if='!!rf_guide_question',classes='mt-3'):
            html.P('{{ rf_guide_question.title }}')
            v.VSelect(v_if="rf_guide_question.kind === 'choice' || rf_guide_question.kind === 'multiple'",label='Your choice',
                v_model=('rf_guide_answer',),items=('rf_guide_question.options',),multiple=("rf_guide_question.kind === 'multiple'",),
                key=('rf_guide_question.id',),density='compact',variant='outlined')
            v.VTextField(v_if="rf_guide_question.kind === 'number'",label='Ions per unit cell',v_model=('rf_guide_answer',),type='text',inputmode='decimal',density='compact')
            v.VTextarea(v_if="rf_guide_question.kind === 'text'",label='Reference and applicability',v_model=('rf_guide_answer',),rows=3)
            v.VBtn('Use this answer',click=(app.answer_guided_refinement,'[{token:rf_guide_token,value:rf_guide_answer}]'),disabled=('rf_busy || rf_guided_busy',),color='#15543c',size='small')
            with html.Details(classes='mt-3'):
                html.Summary('Describe the answer to Luna instead')
                v.VTextarea(label='Answer this question in your own words',v_model=('rf_guide_text',),rows=2)
                v.VBtn('Interpret answer',click=(app.interpret_guided_answer,'[{token:rf_guide_token,text:rf_guide_text}]'),disabled=('rf_busy || rf_guided_busy || !rf_guide_text.trim()',),variant='text',size='small')
                html.P('{{ rf_guide_proposal }}',v_if='!!rf_guide_proposal',classes='rf-notice')
                v.VBtn('Use suggested answer',v_if='rf_guide_can_accept',click=(app.accept_guided_answer,'[{token:rf_guide_token,text:rf_guide_text}]'),disabled=('rf_busy || rf_guided_busy',),variant='outlined',size='small')
        with html.Details(v_if='rf_guide_summary.length',classes='mt-3'):
            html.Summary('Recorded setup decisions')
            html.P('{{ rf_guide_summary_notice }}',v_if='!!rf_guide_summary_notice',classes='rf-notice')
            with html.Ul():html.Li('{{ item.question }}: {{ item.answer }}',v_for='item in rf_guide_summary',key='item.question')
        v.VBtn('Build and check recipe',v_if='rf_guide_complete',click=app.compile_guided_refinement,disabled=('rf_busy || rf_guided_busy',),color='#15543c',classes='mt-3')
