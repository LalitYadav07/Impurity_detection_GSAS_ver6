"""Interactive NDIP/NOVA client for RADAR-PD."""

__all__ = ["AnalysisConfig", "InputSelection", "RunRecord"]


def __getattr__(name):
    # The native worker also uses pure beamline file adapters from this package.
    # Importing those must not load the interactive client's optional UI/models.
    if name in __all__:
        from . import models
        value = getattr(models, name)
        globals()[name] = value
        return value
    raise AttributeError(name)
