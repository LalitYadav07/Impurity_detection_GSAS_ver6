"""Source-checkout bridge; the single implementation ships in the NOVA wheel."""
from pathlib import Path
_core = Path(__file__).resolve().parents[2] / "nova" / "src" / "gpx_recipe"
if not _core.is_dir():
    raise ImportError("The packaged gpx_recipe core is missing from this image")
__path__.append(str(_core))
exec(compile((_core / "__init__.py").read_text(), str(_core / "__init__.py"), "exec"))
