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
    "QwenBackboneQTRM",
    "QwenBackboneQTRMReport",
    "QwenLayerWrappedRecursiveCore",
    "QwenLayerWrappedStack",
    "OuroLayerWrappedStack",
    "OuroWeightWrappedRecursiveCore",
    "build_qtrm_core_config_from_qwen",
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
    if name in {
        "QwenBackboneQTRM",
        "QwenBackboneQTRMReport",
        "QwenLayerWrappedRecursiveCore",
        "QwenLayerWrappedStack",
        "OuroLayerWrappedStack",
        "OuroWeightWrappedRecursiveCore",
        "build_qtrm_core_config_from_qwen",
    }:
        from .qwen_backbone_qtrm import (
            OuroLayerWrappedStack,
            OuroWeightWrappedRecursiveCore,
            QwenBackboneQTRM,
            QwenBackboneQTRMReport,
            QwenLayerWrappedRecursiveCore,
            QwenLayerWrappedStack,
            build_qtrm_core_config_from_qwen,
        )

        return {
            "QwenBackboneQTRM": QwenBackboneQTRM,
            "QwenBackboneQTRMReport": QwenBackboneQTRMReport,
            "QwenLayerWrappedRecursiveCore": QwenLayerWrappedRecursiveCore,
            "QwenLayerWrappedStack": QwenLayerWrappedStack,
            "OuroLayerWrappedStack": OuroLayerWrappedStack,
            "OuroWeightWrappedRecursiveCore": OuroWeightWrappedRecursiveCore,
            "build_qtrm_core_config_from_qwen": build_qtrm_core_config_from_qwen,
        }[name]
    raise AttributeError(name)
