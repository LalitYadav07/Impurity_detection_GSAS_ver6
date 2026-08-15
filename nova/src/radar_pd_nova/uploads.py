"""Stable browser-upload controls for scientific input files.

The NOVA ``FileUpload`` component wraps its button in a parent-activated menu
and creates a second server-file input. In conditionally displayed forms that
menu can be re-attached by Vue, duplicating the control and causing sibling
upload models to be replaced. RADAR-PD uses explicit Galaxy History and SNS
sources, so this component only handles direct browser uploads.
"""

from __future__ import annotations

import itertools
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

from trame.app import get_server
from trame.widgets import html, vuetify3 as vuetify


_UPLOAD_IDS = itertools.count(1)


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


def display_filename(value: Any) -> str:
    """Return only the basename of a local, browser, or Windows-style path."""

    text = str(value or "").replace("\\", "/")
    return text.rsplit("/", 1)[-1] if text else ""


class NamedFileUpload:
    """Render one direct, stable local-file upload card.

    ``v_model`` is the persisted server-side path consumed by RADAR-PD. A
    private client model holds the browser ``File`` object, so choosing this
    file cannot overwrite a sibling upload. The browser input is mounted once
    and is never recreated when surrounding setup sections are collapsed.
    """

    def __init__(
        self,
        v_model: str,
        *,
        label: str,
        extensions: Iterable[str] | None = None,
        help_text: str = "",
        optional: bool = False,
        return_contents: bool = False,
        show_server_files: bool = False,
        color: str = "#15543c",
        key: str | None = None,
        **_: Any,
    ) -> None:
        if return_contents:
            raise ValueError("NamedFileUpload stores a path; return_contents=True is unsupported")
        # Kept in the signature for source compatibility. Pod filesystem
        # browsing is deliberately not a user-facing data source.
        del show_server_files

        self.server = get_server(None, client_type="vue3")
        self.v_model = v_model
        self.extensions = list(extensions or [])
        self.instance_id = next(_UPLOAD_IDS)
        self.key = key or f"radar-upload-{self.instance_id}"
        key_slug = self.key.replace("-", "_")
        self.client_model = f"{key_slug}_browser_file"
        self.display_model = f"{key_slug}_display_name"
        self.ref_name = f"{key_slug}_input"
        self.decode_trigger = f"decode_named_blob_{self.instance_id}"
        self.clear_trigger = f"clear_named_blob_{self.instance_id}"

        state = self.server.state
        setattr(state, self.client_model, None)
        setattr(state, self.display_model, display_filename(getattr(state, v_model, "")))

        @self.server.controller.trigger(self.decode_trigger)
        def _decode_named_blob(contents: bytes, original_name: str = "upload") -> None:
            target = store_browser_upload(contents, original_name)
            setattr(state, self.v_model, str(target))
            setattr(state, self.display_model, target.name)
            setattr(state, self.client_model, None)
            state.flush()

        @self.server.controller.trigger(self.clear_trigger)
        def _clear_named_blob() -> None:
            setattr(state, self.v_model, "")
            setattr(state, self.display_model, "")
            setattr(state, self.client_model, None)
            state.flush()

        @state.change(v_model)
        def _sync_display_name(**kwargs: Any) -> None:
            value = kwargs.get(v_model, getattr(state, v_model, ""))
            setattr(state, self.display_model, display_filename(value))

        accepted = ",".join(self.extensions)
        file_expr = self.client_model
        decode_js = (
            f"{file_expr} && {file_expr}.arrayBuffer().then((contents) => {{"
            f" trigger('{self.decode_trigger}', [contents, {file_expr}.name]);"
            "})"
        )
        empty_label = "Optional" if optional else "No file selected"

        # Trame treats ``key`` as a JavaScript expression. Quote constant keys
        # so Vue receives stable string identities instead of unresolved names.
        with html.Div(classes="radar-upload-card", key=repr(self.key)):
            vuetify.VFileInput(
                v_model=(self.client_model,),
                __properties=["accept"],
                accept=accepted,
                classes="radar-hidden-file-input",
                ref=self.ref_name,
                key=repr(f"{self.key}-native"),
                update_modelValue=decode_js,
            )
            with html.Div(classes="radar-upload-copy"):
                html.Div(label, classes="radar-upload-label")
                html.Div(help_text or f"Accepted: {accepted}", classes="radar-upload-help")
                html.Div(
                    f"{{{{ {self.display_model} || '{empty_label}' }}}}",
                    classes=(
                        f"{self.display_model} ? 'radar-upload-filename is-ready' : 'radar-upload-filename'",
                    ),
                )
            with html.Div(classes="radar-upload-actions"):
                vuetify.VBtn(
                    text=(f"{self.display_model} ? 'Replace' : 'Choose file'",),
                    size="small",
                    variant="outlined",
                    color=color,
                    prepend_icon="mdi-upload-outline",
                    click=f"trame.refs.{self.ref_name}.click()",
                    key=repr(f"{self.key}-choose"),
                )
                vuetify.VBtn(
                    icon="mdi-close",
                    title="Clear selected file",
                    size="x-small",
                    variant="text",
                    color="#7c2d2d",
                    v_show=f"!!{self.display_model}",
                    click=f"trigger('{self.clear_trigger}')",
                    key=repr(f"{self.key}-clear"),
                )


__all__ = [
    "NamedFileUpload",
    "display_filename",
    "safe_client_filename",
    "store_browser_upload",
]
