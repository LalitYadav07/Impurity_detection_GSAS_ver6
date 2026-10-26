"""Browser-owned cursor and serialized Plotly previews for the live map."""
from pathlib import Path
from hashlib import sha256

from trame_client.widgets.core import AbstractElement


class TransitionCursor(AbstractElement):
    def __init__(self, **kwargs):
        super().__init__("radar-transition-cursor", **kwargs)
        self._attr_names += ["figure", "frames", "cursor", "scope", "epoch", "ack",
                             ("run_ids", "runIds"), "top", "height"]
        self._event_names += [(name.lower(), name) for name in
                              ("Begin", "Commit", "Click", "Relayout", "Reset", "Preview")]
        directory=Path(__file__).with_name("web")
        route="radar_cursor_"+sha256((directory/"transition_cursor.js").read_bytes()).hexdigest()[:12]
        self.server.enable_module({
            "serve": {route: str(directory)},
            "scripts": [route+"/transition_cursor.js"],
            "vue_use": ["radarTransitionCursor"],
        })
