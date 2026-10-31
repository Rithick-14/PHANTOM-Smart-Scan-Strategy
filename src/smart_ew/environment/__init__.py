"""
RF Environment simulation module.
"""

from smart_ew.environment.schema import FrequencyBand, EmitterConfig, SimConfig, GroundTruthFrame


def __getattr__(name: str):
    """Load the simulator lazily to avoid the emitter/schema import cycle."""
    if name == "RFEnvironment":
        from smart_ew.environment.simulator import RFEnvironment

        return RFEnvironment
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FrequencyBand",
    "EmitterConfig",
    "SimConfig",
    "GroundTruthFrame",
    "RFEnvironment",
]
