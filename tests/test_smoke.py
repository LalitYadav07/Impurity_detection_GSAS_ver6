def test_import_app():
    """Smoke test: importing the top-level `app` module must succeed."""
    import importlib

    mod = importlib.import_module('app')
    assert mod is not None
