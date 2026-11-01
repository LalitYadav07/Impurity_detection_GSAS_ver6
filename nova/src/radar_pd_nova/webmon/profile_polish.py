"""Compatibility import; scientific implementation lives in gpx_recipe.live."""
import importlib
import sys
_module = importlib.import_module("gpx_recipe.live.profile_polish")
if __name__ == "__main__":
    _module.polish(sys.argv[1])
else:
    sys.modules[__name__] = _module
