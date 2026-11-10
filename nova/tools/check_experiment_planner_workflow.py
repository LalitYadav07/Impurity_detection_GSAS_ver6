"""Exercise real Trame callbacks, public symmetry services and native workers.

The deployment supplies BCS_API_KEY. Credentials are never written to reports.
Run in the integrated image; this makes no Galaxy submissions or refinements.
"""
import argparse
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import uuid
from unittest.mock import patch
from contextlib import ExitStack
import hashlib
import shutil
import numpy as np

from trame.app import get_server
from radar_pd_nova.planning_controller import PlanningMixin
from radar_pd_nova.uploads import store_browser_upload


class Harness(PlanningMixin):
    def __init__(self,root):
        self.server=get_server('planner-check-'+uuid.uuid4().hex,client_type='vue3')
        self.service=SimpleNamespace(output_root=root)
        self.tasks=[]
        self._init_planning();self.server.state.ready()

    def _refinement_schedule(self,coroutine,name):
        task=asyncio.create_task(coroutine,name=name);self.tasks.append(task);return task

    async def settle(self):
        while self.tasks:
            pending,self.tasks=self.tasks,[]
            await asyncio.gather(*pending)
        self.server.state.flush()
        assert not self.server.state.plan_busy
        if self.server.state.plan_error:raise AssertionError(self.server.state.plan_error)


def input_file(path):return str(store_browser_upload(path.read_bytes(),path.name))


def verify_control_readback(app, receipt, kind):
    """Compare displayed affine controls with the saved native GPX atoms."""
    from gpx_recipe.snapshot import gsas
    state=receipt[kind+'_state'];controls=state['controls']
    project=gsas().G2Project(state['gpx'])
    phases=[p.data for p in project.phases() if (
        p.data['General']['Type']=='magnetic' if kind=='magnetic' else bool(p.data.get('ISODISTORT')))]
    assert len(phases)==1
    phase=phases[0];cx,ct,_,_=phase['General']['AtomPtrs']
    atoms={str(a[ct-1]):a for a in phase['Atoms']}
    values=({p['name']:p['value'] for p in state['parameters']} if kind=='magnetic' else state['modes'])
    q=np.asarray([values[p['name']] for p in controls['parameters']])
    shown=app.server.state.plan_moment_sites if kind=='magnetic' else app.server.state.plan_displacement_sites
    assert {r['label'] for r in shown}==set(atoms)
    for site in controls['sites']:
        calculated=np.asarray(site['offset'])+np.asarray(site['basis'])@q
        start=cx+4 if kind=='magnetic' else cx
        actual=np.asarray(atoms[site['label']][start:start+3],float)
        np.testing.assert_allclose(calculated,actual,rtol=1e-9,atol=1e-9)
        display=next(r for r in shown if r['label']==site['label'])
        np.testing.assert_allclose([float(c['value']) for c in display['components']],actual,rtol=5e-5,atol=5e-6)


def recorded_services(fixtures):
    """Replay declared reference candidates; forbid all HTTP in this process."""
    def magnetic(parent,k,**kwargs):
        for name in ['magnetic-candidates.json','combined-k0-catalog.json']:
            catalog=json.loads((fixtures/name).read_text())
            if catalog['parent_setting_sha256']==parent['setting_sha256'] and catalog['k']==k:return catalog
        raise AssertionError('No recorded catalog matches this exact parent and k')
    def structural(parent_cif,parent,k,directory,*,inventory,**kwargs):
        assert k==['1/3','1/3','0']
        root=Path(directory);root.mkdir(parents=True)
        source=fixtures/'structural-child.cif';target=root/'child.cif';shutil.copyfile(source,target)
        candidate=json.loads((fixtures/'manifest.json').read_text())['structural']['candidate']
        assert hashlib.sha256(target.read_bytes()).hexdigest()==candidate['child']['sha256']
        candidate['modes']=inventory(target)['modes']
        return dict(schema='radar-structural-catalog/v1',parent_setting_sha256=parent['setting_sha256'],k=k,
            candidates=[candidate],coverage={'discovered_count':1,'constructed_count':1,'opd_policy':'recorded-reference',
            'candidates':[{'state':'constructed'}],'irrep_failures':[]},
            provenance={'service':'Previously captured ISODISTORT child; no live discovery'})
    stack=ExitStack()
    stack.enter_context(patch('gpx_recipe.symmetry.discovery.discover_magnetic',magnetic))
    stack.enter_context(patch('gpx_recipe.symmetry.isodistort_discovery.discover_structural',structural))
    stack.enter_context(patch('requests.sessions.Session.request',side_effect=AssertionError('Recorded mode forbids HTTP')))
    return stack


async def run(root,fixtures,recorded=False):
    root.mkdir(parents=True,exist_ok=False)
    report={'schema':'ndip-planner-workflow-check/v1','status':'running','steps':[],
            'experimental_validation':False,'galaxy_submissions':0,
            'services':'recorded reference candidates; HTTP forbidden' if recorded else 'live Bilbao and ISODISTORT'}
    def mark(step,**details):
        report['steps'].append(dict(step=step,**details))
        (root/'workflow.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(dict(passed=step,**details)),flush=True)
    try:
        for kind in ('magnetic','structural'):
            app=Harness(root/kind);s=app.server.state
            s.plan_cif=input_file(fixtures/(kind+'-parent.cif'))
            s.plan_transition=kind;s.plan_response_mode='profile_only';s.plan_resolution_model='profile'
            s.plan_d_min=1.2 if kind=='magnetic' else .55;s.plan_d_max=5.;s.plan_step=.003
            s.plan_k='0, 0, 0' if kind=='magnetic' else '1/3, 1/3, 0';s.flush()
            if kind=='magnetic':
                app.discover_planning_magnetic_groups();await app.settle()
                assert app._planning_parent_output is not None
                catalog=app._planning_catalog[0]
                candidate=next(c for c in catalog['candidates'] if c['bns_number']=='113.271')
                s.plan_candidate=candidate['id'];s.plan_sites=['Er1'];s.flush()
                app.update_planning_site('Er1','form_factor','Er+3')
                s.flush()
                app.prepare_planning_magnetic_moments();await app.settle()
                assert s.plan_parameter_rows
                assert s.plan_moment_sites
                assert all('Controls ' in p['description'] for p in s.plan_parameter_rows)
                assert all(row['lande_g_source']=='GSAS-II SetupGeneral default' for row in app._planning_prepared['magnetic_ions'])
                for row in s.plan_parameter_rows:
                    app.update_planning_magnetic_parameter(row['name'], '2.25')
                s.flush()
                mark('magnetic catalog and shared preparation',subgroups=len(catalog['candidates']),
                     parameters=len(s.plan_parameter_rows),parent_prepared_automatically=True)
            else:
                app.discover_planning_structural_groups();await app.settle()
                catalog=app._planning_structural_catalog[0]
                reference=json.loads((fixtures/'manifest.json').read_text())['structural']['candidate']
                matches=[c for c in catalog['candidates'] if all(c[key]==reference[key] for key in
                         ('irrep','transformation_matrix','origin_shift'))]
                assert len(matches)==1, 'Expected one K1 reference embedding in the returned catalog'
                candidate=matches[0]
                s.plan_structural_candidate=candidate['id'];s.flush()
                app.prepare_planning_structural_modes();await app.settle()
                assert s.plan_displacement_sites and s.plan_child_cell_note
                if s.plan_error:raise AssertionError(s.plan_error)
                for row in s.plan_structural_parameter_rows:
                    app.update_planning_structural_parameter(row['name'],'.01')
                s.flush()
                mark('structural catalog and shared mode preparation',coverage=catalog['coverage'])
            await app._run_planning_simulation();await app.settle()
            assert s.plan_current and s.plan_result_ready
            receipt=app._planning_receipt(app._planning_output,app._planning_receipt_sha256)
            assert receipt[kind+'_state']['no_refinement']
            verify_control_readback(app,receipt,kind)
            mark(kind+' controller simulation and native site readback',result=str(app._planning_output/'result.json'))
            s.plan_name='Er₂Be₂GeO₇ · magnetic k=0' if kind=='magnetic' else 'YMnO₃ · structural K1'
            s.plan_step_ui=2;s.flush()
            app.save_planning_plan();assert not s.plan_error
            s.plan_mass=2.;s.flush();assert not s.plan_current
            app.restore_planning_plan();assert not s.plan_error and s.plan_current
            mark(kind+' saved result restored')
            if kind=='structural':
                multiplier=receipt['structural_state']['nuclear_cell_multiplier']
                app.promote_planning_structural_parent();s.flush()
                assert s.plan_transition=='magnetic' and s.plan_promoted_parent and not s.plan_current
                assert s.plan_k=='0, 0, 0'
                assert app._planning_parent_state['nuclear_cell_multiplier']==multiplier
                parent=app._planning_parent_request(app._planning_parent_receipt())
                assert parent['nuclear_reference']['nuclear_cell_multiplier']==multiplier
                mark('structural child promoted for magnetic planning',nuclear_cell_multiplier=multiplier)
                if recorded:
                    app.discover_planning_magnetic_groups();await app.settle()
                    candidate=next(c for c in app._planning_catalog[0]['candidates'] if c['bns_number']=='193.261')
                    s.plan_candidate=candidate['id'];s.plan_sites=['Mn1_1'];s.flush()
                    app.update_planning_site('Mn1_1','form_factor','Mn+3');s.flush()
                    app.prepare_planning_magnetic_moments();await app.settle()
                    await app._run_planning_simulation();await app.settle()
                    combined=app._planning_receipt(app._planning_output,app._planning_receipt_sha256)
                    assert combined['magnetic_state']['nuclear_cell_multiplier']==multiplier
                    assert combined['magnetic_state']['structural_parent']['structural_state']['modes']==receipt['structural_state']['modes']
                    mark('combined structural and magnetic controller simulation',result=str(app._planning_output/'result.json'))
        report['status']='passed'
    except Exception as exc:
        report.update(status='failed',error_type=type(exc).__name__,error=str(exc))
        raise
    finally:(root/'workflow.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--fixtures',type=Path,default=Path(__file__).resolve().parents[1]/'tests/data/planning-transitions')
    parser.add_argument('--recorded',action='store_true')
    args=parser.parse_args()
    with recorded_services(args.fixtures.resolve()) if args.recorded else ExitStack():
        asyncio.run(run(args.output.resolve(),args.fixtures.resolve(),args.recorded))
