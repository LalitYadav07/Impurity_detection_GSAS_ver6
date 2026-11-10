"""Standalone entry point for the same no-IPTS NOVA planning workspace.

Run one application process/container per user session. Scientific jobs run in
bounded child processes using the packaged, pinned GSAS-II runtime.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from types import SimpleNamespace

from nova.trame import ThemedApp
from trame.app import get_server
from trame.widgets import client

from .planning_controller import PlanningMixin
from .planning_ui import planning_dashboard, planning_css


class PowgenPlanner(PlanningMixin, ThemedApp):
    def __init__(self, work_dir):
        self.server=get_server(None,client_type='vue3')
        super().__init__(server=self.server)
        root=Path(work_dir).expanduser().resolve();root.mkdir(parents=True,exist_ok=True)
        self.service=SimpleNamespace(output_root=root)
        self._tasks=set()
        self._init_planning()
        self.server.state.plan_standalone=True
        self.server.state.workspace_view='planning'
        self.server.state.trame__title='POWGEN Experiment Planner'
        self.create_ui()

    def _refinement_schedule(self, coroutine, name):
        try:loop=asyncio.get_running_loop()
        except RuntimeError:
            coroutine.close()
            raise RuntimeError('Start the planner server before submitting a calculation') from None
        task=loop.create_task(coroutine,name=name)
        self._tasks.add(task);task.add_done_callback(self._tasks.discard)
        return task

    def create_ui(self):
        self.set_theme('CompactTheme')
        with super().create_ui() as layout:
            layout.toolbar_title.set_text('POWGEN Experiment Planner')
            client.Style('''
                .radar-workspace-view {width:100%;max-width:1600px;box-sizing:border-box;margin:0 auto;padding:24px;}
                .radar-section-heading {display:flex;align-items:center;justify-content:space-between;margin-bottom:20px;gap:16px;}
                .radar-section-heading h1 {font-size:1.6rem;font-weight:600;margin-bottom:6px;}
                .radar-help-copy {font-size:.9rem;line-height:1.5;margin:8px 0 14px;color:#4a5564;}
                .radar-upload-card {margin:8px 0;}
                @media(max-width:700px) {.radar-workspace-view {padding:12px;}}
            ''')
            client.Style(planning_css())
            with layout.content:planning_dashboard(self)
        return layout


def main(argv=None):
    parser=argparse.ArgumentParser(description='POWGEN experiment planning; no IPTS or Galaxy account required')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8080)
    parser.add_argument('--work-dir',default='./powgen-plans')
    args=parser.parse_args(argv)
    app=PowgenPlanner(args.work_dir)
    app.server.start(host=args.host,port=args.port,open_browser=False,timeout=0)


if __name__=='__main__':main()
