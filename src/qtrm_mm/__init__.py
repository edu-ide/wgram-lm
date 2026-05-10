from .config import QTRMConfig, DonorConfig, TrainConfig, FullConfig, load_config

__all__ = [
    "QTRMConfig",
    "DonorConfig",
    "TrainConfig",
    "FullConfig",
    "load_config",
    "QTRMCoreCarry",
    "QTRMMultimodalModel",
    "JepaWorldModelHead",
]


def __getattr__(name):
    if name == "QTRMCoreCarry":
        from .core import QTRMCoreCarry

        return QTRMCoreCarry
    if name == "QTRMMultimodalModel":
        from .qtrm_model import QTRMMultimodalModel

        return QTRMMultimodalModel
    if name == "JepaWorldModelHead":
        from .world_model import JepaWorldModelHead

        return JepaWorldModelHead
    raise AttributeError(name)
