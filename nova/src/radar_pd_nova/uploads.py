"""Stable browser-upload controls for scientific input files.

The NOVA ``FileUpload`` component wraps its button in a parent-activated menu
and creates a second server-file input. In conditionally displayed forms that
menu can be re-attached by Vue, duplicating the control and causing sibling
upload models to be replaced. RADAR-PD uses explicit Galaxy History and SNS
sources, so this component only handles direct browser uploads.
"""

from __future__ import annotations

import itertools
import hashlib
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


def inspect_cif_upload(contents: bytes, original_name: Any) -> dict[str, Any]:
    """Validate one browser CIF before it is uploaded to Galaxy.

    This is intentionally a fast structural preflight rather than a diffraction
    simulation. The database builder remains the scientific authority, but an
    empty, mislabeled, duplicated, or obviously incomplete file should be
    rejected while the user can still correct the selection.
    """

    name = safe_client_filename(original_name)
    if Path(name).suffix.lower() != ".cif":
        raise ValueError(f"{name}: expected a .cif file")
    if not contents:
        raise ValueError(f"{name}: file is empty")
    if len(contents) > 20 * 1024 * 1024:
        raise ValueError(f"{name}: file exceeds the 20 MB CIF limit")
    text = contents.decode("utf-8-sig", errors="replace")
    lowered = text.lower()
    if not re.search(r"(?m)^\s*data_\S*", text, flags=re.IGNORECASE):
        raise ValueError(f"{name}: no CIF data block was found")
    required_cell_tags = (
        "_cell_length_a",
        "_cell_length_b",
        "_cell_length_c",
        "_cell_angle_alpha",
        "_cell_angle_beta",
        "_cell_angle_gamma",
    )
    missing = [tag for tag in required_cell_tags if tag not in lowered]
    if missing:
        raise ValueError(f"{name}: missing unit-cell fields ({', '.join(missing)})")

    def _tag_value(*tags: str) -> str:
        for tag in tags:
            match = re.search(
                rf"(?im)^\s*{re.escape(tag)}\s+(?:'([^']+)'|\"([^\"]+)\"|(\S+))",
                text,
            )
            if match:
                return next((value for value in match.groups() if value), "")
        return ""

    return {
        "name": name,
        "size": len(contents),
        "digest": hashlib.sha256(contents).hexdigest(),
        "formula": _tag_value("_chemical_formula_sum", "_chemical_formula_structural"),
        "space_group": _tag_value("_space_group_it_number", "_symmetry_int_tables_number"),
    }


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


class NamedMultiCifUpload:
    """Render a multi-file CIF picker with immediate, per-file preflight."""

    def __init__(
        self,
        v_model: str,
        rows_model: str,
        *,
        key: str = "radar-cif-library-upload",
        color: str = "#15543c",
    ) -> None:
        self.server = get_server(None, client_type="vue3")
        self.v_model = v_model
        self.rows_model = rows_model
        self.key = key
        key_slug = key.replace("-", "_")
        self.client_model = f"{key_slug}_browser_files"
        self.ref_name = f"{key_slug}_input"
        self.decode_trigger = f"decode_{key_slug}"
        self.remove_trigger = f"remove_{key_slug}"
        self.clear_trigger = f"clear_{key_slug}"

        state = self.server.state
        setattr(state, self.client_model, None)
        if getattr(state, v_model, None) is None:
            setattr(state, v_model, [])
        if getattr(state, rows_model, None) is None:
            setattr(state, rows_model, [])

        @self.server.controller.trigger(self.decode_trigger)
        def _decode_cif(contents: bytes, original_name: str = "candidate.cif") -> None:
            rows = list(getattr(state, self.rows_model, []) or [])
            paths = list(getattr(state, self.v_model, []) or [])
            try:
                metadata = inspect_cif_upload(contents, original_name)
                duplicate = next((row for row in rows if row.get("digest") == metadata["digest"]), None)
                if duplicate:
                    raise ValueError(f"{metadata['name']}: duplicates {duplicate.get('name', 'an existing selection')}")
                target = store_browser_upload(contents, metadata["name"])
                metadata.update(
                    {
                        "path": str(target),
                        "status": "Ready",
                        "detail": " / ".join(
                            value
                            for value in (
                                metadata.get("formula"),
                                f"SG {metadata['space_group']}" if metadata.get("space_group") else "",
                            )
                            if value
                        )
                        or "CIF structure preflight passed",
                    }
                )
                paths.append(str(target))
            except Exception as exc:
                metadata = {
                    "name": safe_client_filename(original_name),
                    "path": "",
                    "digest": "",
                    "status": "Rejected",
                    "detail": str(exc),
                }
            rows.append(metadata)
            setattr(state, self.v_model, paths)
            setattr(state, self.rows_model, rows)
            setattr(state, self.client_model, None)
            state.flush()

        @self.server.controller.trigger(self.remove_trigger)
        def _remove_cif(path: str = "", name: str = "") -> None:
            rows = list(getattr(state, self.rows_model, []) or [])
            paths = list(getattr(state, self.v_model, []) or [])
            setattr(
                state,
                self.rows_model,
                [row for row in rows if not (str(row.get("path") or "") == str(path) and str(row.get("name") or "") == str(name))],
            )
            setattr(state, self.v_model, [item for item in paths if str(item) != str(path)])
            state.flush()

        @self.server.controller.trigger(self.clear_trigger)
        def _clear_cifs() -> None:
            setattr(state, self.v_model, [])
            setattr(state, self.rows_model, [])
            setattr(state, self.client_model, None)
            state.flush()

        decode_js = (
            f"Array.from({self.client_model} || []).forEach((file) => "
            f"file.arrayBuffer().then((contents) => trigger('{self.decode_trigger}', [contents, file.name])));"
        )
        with html.Div(classes="radar-multi-upload", key=repr(key)):
            vuetify.VFileInput(
                v_model=(self.client_model,),
                multiple=True,
                __properties=["accept"],
                accept=".cif,chemical/x-cif",
                classes="radar-hidden-file-input",
                ref=self.ref_name,
                key=repr(f"{key}-native"),
                update_modelValue=decode_js,
            )
            with html.Div(classes="radar-button-row"):
                vuetify.VBtn(
                    "Add CIF files",
                    size="small",
                    variant="outlined",
                    color=color,
                    prepend_icon="mdi-file-plus-outline",
                    click=f"trame.refs.{self.ref_name}.click()",
                )
                vuetify.VBtn(
                    "Clear",
                    size="small",
                    variant="text",
                    color="#7c2d2d",
                    v_show=f"{self.rows_model}.length > 0",
                    click=f"trigger('{self.clear_trigger}')",
                )
            with html.Div(v_for=f"item in {self.rows_model}", key="item.name + item.digest", classes="radar-cif-file-row"):
                with html.Div(classes="radar-file-copy"):
                    html.Strong("{{ item.name }}")
                    html.Span("{{ item.status }} / {{ item.detail }}")
                vuetify.VBtn(
                    icon="mdi-close",
                    title="Remove CIF",
                    size="x-small",
                    variant="text",
                    color="#7c2d2d",
                    click=f"trigger('{self.remove_trigger}', [item.path, item.name])",
                )


__all__ = [
    "NamedFileUpload",
    "NamedMultiCifUpload",
    "display_filename",
    "inspect_cif_upload",
    "safe_client_filename",
    "store_browser_upload",
]
