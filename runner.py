"""Compatibility shim for the canonical runner implementation.

The maintained implementation lives in `scripts/runner.py`. Re-export it here so
older import paths do not silently drift out of sync.
"""

from scripts.runner import PipelineRunner, stop_process_tree, watch_events

__all__ = ["PipelineRunner", "stop_process_tree", "watch_events"]
