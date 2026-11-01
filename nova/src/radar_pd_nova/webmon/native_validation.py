"""Compatibility import; scientific implementation lives in gpx_recipe.live."""
import importlib
import sys
_module = importlib.import_module("gpx_recipe.live.native_validation")
sys.modules[__name__] = _module
