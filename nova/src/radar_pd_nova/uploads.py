"""Upload controls that preserve scientific filenames from the browser."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from nova.trame._internal.utils import get_state_param
from nova.trame.view.components import FileUpload
from nova.trame.view.components.remote_file_input import RemoteFileInput
from trame.widgets import vuetify3 as vuetify


def safe_client_filename(value: Any) -> str:
    """Reduce a browser-provided filename to one safe local path component."""

    candidate = str(value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    candidate = candidate.replace("\x00", "")
    candidate = re.sub(r"[^\w.() +\-]", "_", candidate, flags=re.UNICODE)
    candidate = candidate.strip(". ")
    return candidate if candidate and candidate not in {".", ".."} else "upload"


def store_browser_upload(contents: bytes, original_name: Any) -> Path:
    """Persist an uploaded blob under its sanitized original basename."""

    directory = Path(tempfile.mkdtemp(prefix="radar_pd_upload_"))
    target = directory / safe_client_filename(original_name)
    target.write_bytes(contents)
    return target


class NamedFileUpload(FileUpload):
    """NOVA FileUpload variant that retains the laptop filename and suffix.

    The upstream component sends only ``File.arrayBuffer()`` to Python and
    writes it into an extensionless ``NamedTemporaryFile``. RADAR-PD also
    sends ``File.name`` and stores the bytes in a private temporary directory
    under that sanitized basename. Server-file selection continues to use the
    standard NOVA RemoteFileInput behavior.
    """

    def create_ui(self) -> None:
        self.local_file_input = vuetify.VFileInput(
            v_model=self._v_model,
            __properties=["accept"],
            accept=",".join(
                get_state_param(self.state, (self._extensions,))
                if isinstance(self._extensions, str)
                else self._extensions
            ),
            classes="d-none",
            ref=self._ref_name,
            key=f"{self._ref_name}_local_input",
            update_modelValue=(
                f"{self._v_model} && {self._v_model}.arrayBuffer().then((contents) => {{"
                f"  trigger('decode_named_blob_{self._id}', [contents, {self._v_model}.name]); "
                "});"
            ),
        )
        self.remote_file_input = RemoteFileInput(
            v_model=self._v_model,
            base_paths=self._base_paths,
            extensions=self._extensions,
            input_props={"classes": "d-none", "key": f"{self._ref_name}_remote_input"},
            return_contents=self._return_contents,
            use_bytes=self._use_bytes,
        )

        with self:
            with vuetify.VMenu(v_if=self._show_server_files.expression, activator="parent"):
                with vuetify.VList():
                    vuetify.VListItem(
                        "From Local Machine",
                        click=f"trame.refs.{self._ref_name}.click()",
                        key=f"{self._ref_name}_local_choice",
                    )
                    vuetify.VListItem(
                        "From Server",
                        click=self.remote_file_input.open_dialog,
                        key=f"{self._ref_name}_remote_choice",
                    )

        @self.server.controller.trigger(f"decode_named_blob_{self._id}")
        def _decode_named_blob(contents: bytes, original_name: str = "upload") -> None:
            if get_state_param(self.state, self._return_contents):
                self.remote_file_input.decode_file(contents, True)
                return
            target = store_browser_upload(contents, original_name)
            self.remote_file_input.set_v_model(str(target))

