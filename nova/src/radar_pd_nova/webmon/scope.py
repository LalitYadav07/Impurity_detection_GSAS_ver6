"""Apply the saved Live watch's identities to acquisition previews and actions."""
from dataclasses import dataclass
import threading

from ..facility_metadata import sample_identifier


def sample_id(value):
    if isinstance(value, bool):
        return ""
    value = sample_identifier(value)
    return "" if value in {"4294967295", "18446744073709551615"} else value


def physical_bank(value):
    if isinstance(value, bool):
        return None
    text = str(value or "").strip()
    return int(text) if text.isdigit() and int(text) > 0 else None


@dataclass(frozen=True)
class WatchScope:
    history_id: str
    ipts: str
    sample_ids: tuple[str, ...] | None
    banks: tuple[int, ...] | None
    excluded_runs: tuple[str, ...] = ()

    @classmethod
    def from_settings(cls, settings):
        if str(getattr(settings, "instrument", "PG3")).upper() != "PG3":
            raise ValueError("Live acquisition is available for the saved POWGEN watch only")
        ids = getattr(settings, "selected_sample_ids", None)
        if ids is not None:
            parsed = tuple(sample_id(value) for value in ids)
            if not parsed or not all(parsed):
                raise ValueError("The saved Live sample selection is unresolved")
            ids = tuple(sorted(set(parsed)))
        banks = getattr(settings, "selected_banks", None)
        if banks is not None:
            parsed = tuple(physical_bank(value) for value in banks)
            if not parsed or None in parsed:
                raise ValueError("The saved Live bank selection is unresolved")
            banks = tuple(sorted(set(parsed)))
        ipts = str(settings.ipts).upper().removeprefix("IPTS-")
        if not ipts.isdigit() or int(ipts) < 1 or not settings.history_id:
            raise ValueError("The saved Live experiment/History selection is unresolved")
        return cls(str(settings.history_id), "IPTS-" + ipts, ids, banks,
                   tuple(sorted(str(value) for value in getattr(settings, "excluded_run_numbers", ()))))

    @property
    def label(self):
        samples = ", ".join(self.sample_ids) if self.sample_ids is not None else "all recorded sample IDs"
        banks = ", ".join(map(str, self.banks)) if self.banks is not None else "all banks"
        return f"Saved Live scope: sample {samples}; physical bank {banks}."

    def row_error(self, row):
        identifier = sample_id(row.get("sample_id"))
        if not identifier or row.get("identity_status", "resolved") != "resolved":
            return "The acquisition has no unambiguous recorded sample ID; it is not eligible for this Live scope"
        if self.sample_ids is not None and identifier not in self.sample_ids:
            return f"Acquisition sample {identifier} is outside the saved Live sample selection"
        if str(row.get("run", "")) in self.excluded_runs:
            return "The acquisition is excluded by the saved Live watch"
        bank = physical_bank(row.get("physical_bank"))
        if bank is not None and self.banks is not None and bank not in self.banks:
            return f"Acquisition bank {bank} is outside the saved Live bank selection"
        return ""

    def source_error(self, source, row=None):
        from ..refinement_sources import experiment_key
        if source.get("experiment") != experiment_key(self.history_id, "PG3", self.ipts):
            return "The published GPX is not from this Live History and experiment"
        metadata = source.get("metadata") or {}
        if metadata.get("identity_status") in {"conflict", "unidentified"}:
            return "The published GPX sample identity requires review"
        key = str(source.get("sample_key") or "")
        values = [sample_id(value) for value in (
            key[3:] if key.startswith("id:") else None, metadata.get("sample_id"))]
        identifiers = {value for value in values if value}
        if len(identifiers) != 1:
            return "The published GPX has no unambiguous recorded sample ID"
        identifier = next(iter(identifiers))
        if self.sample_ids is not None and identifier not in self.sample_ids:
            return f"Published GPX sample {identifier} is outside the saved Live sample selection"
        if row is not None:
            error = self.row_error(row)
            if error:
                return error
            if identifier != sample_id(row.get("sample_id")):
                return "The published GPX and viewed acquisition have different sample IDs"
        bank = physical_bank(source.get("bank"))
        if bank is None:
            return "The published GPX physical bank is unresolved"
        if self.banks is not None and bank not in self.banks:
            return f"Published GPX bank {bank} is outside the saved Live bank selection"
        if row is not None and physical_bank(row.get("physical_bank")) not in {None, bank}:
            return "The published GPX and acquisition have different physical banks"
        return ""


class ScopeCancellation(threading.Event):
    """Use existing admission checks without cancelling an accepted Galaxy job."""
    def __init__(self, current):
        super().__init__()
        self.current = current

    def is_set(self):
        try:current=self.current()
        except Exception:current=False
        if not current:super().set()
        return super().is_set()

    def wait(self, timeout=None):
        return self.is_set() or super().wait(timeout)
