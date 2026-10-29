"""Native-only GPX context export. No arrays or Python objects cross into NOVA."""
from pathlib import Path
import sys
from ..execution import write_json
from .native_audit import inspect_source

if __name__=='__main__':write_json(sys.argv[2],inspect_source(Path(sys.argv[1]).resolve()))
