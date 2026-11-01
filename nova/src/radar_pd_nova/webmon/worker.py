"""Compatibility import; scientific implementation lives in gpx_recipe.live."""
import importlib
import sys
_module = importlib.import_module("gpx_recipe.live.worker")
if __name__ == "__main__":
    _module.run(sys.argv[1], sys.argv[3] if len(sys.argv)>3 and sys.argv[2]=="--candidate" else None)
else:
    sys.modules[__name__] = _module
