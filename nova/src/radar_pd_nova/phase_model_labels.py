"""Display native symmetry separately from an inherited catalog label."""
import re


def native_phase_label(label, native_space_group):
    """Keep matching labels byte-for-byte; never infer symmetry from chemistry."""
    if not isinstance(label, str) or not isinstance(native_space_group, str):
        return label, None
    match = re.fullmatch(r"(.*?)\s*\(SG\s+(.+)\)\s*", label)
    native = native_space_group.strip()
    if not match or not native or native.lower() in {"unknown", "-", "?"}:
        return label, None
    base, catalog = match.groups()
    def symbol(value):
        value = re.sub(r"(?:,\s*\d+|\s*\(\d+\))$", "", value.strip())
        return value, re.sub(r"[\s_]", "", value).lower()
    catalog_symbol, catalog_key = symbol(catalog)
    native_symbol, native_key = symbol(native)
    if not catalog_key or native_key == catalog_key:
        return label, None
    return f"{base.strip()} · {native_symbol} (catalog {catalog_symbol})", catalog


def lattice_display_rows(rows):
    """Copy native-bound rows for presentation/context without changing IDs."""
    result = []
    for original in rows or []:
        if not isinstance(original, dict):
            continue
        row = dict(original)
        if row.get("source_gpx"):
            display, catalog = native_phase_label(row.get("phase"), row.get("space_group"))
            if catalog is not None:
                row.update(phase=display, catalog_phase_label=row["phase"],
                           catalog_space_group=catalog, native_space_group=row["space_group"])
        result.append(row)
    return result
