import importlib
import unittest


class SmokeTests(unittest.TestCase):
    """Minimal import smoke checks for the top-level app package."""

    def test_import_app(self):
        mod = importlib.import_module("app")
        self.assertIsNotNone(mod)


if __name__ == "__main__":
    unittest.main()
