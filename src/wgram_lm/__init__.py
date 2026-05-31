from .config import QTRMConfig, WGRAMConfig, DonorConfig, TrainConfig, FullConfig, load_config

__all__ = [
    "WGRAMConfig",
    "QTRMConfig",
    "DonorConfig",
    "TrainConfig",
    "FullConfig",
    "load_config",
    "WGRAMMultimodalModel",
    "QTRMCoreCarry",
    "QTRMMultimodalModel",
    "JepaWorldModelHead",
    "QwenBackboneWGRAM",
    "QwenBackboneWGRAMReport",
    "QwenBackboneQTRM",
    "QwenBackboneQTRMReport",
    "QwenLayerWrappedRecursiveCore",
    "QwenLayerWrappedStack",
    "OuroLayerWrappedStack",
    "OuroWeightWrappedRecursiveCore",
    "build_wgram_core_config_from_qwen",
    "build_qtrm_core_config_from_qwen",
]


def __getattr__(name):
    if name == "QTRMCoreCarry":
        from .core import QTRMCoreCarry

        return QTRMCoreCarry
    if name in {"WGRAMMultimodalModel", "QTRMMultimodalModel"}:
        from .wgram_model import QTRMMultimodalModel, WGRAMMultimodalModel

        return {
            "WGRAMMultimodalModel": WGRAMMultimodalModel,
            "QTRMMultimodalModel": QTRMMultimodalModel,
        }[name]
    if name == "JepaWorldModelHead":
        from .world_model import JepaWorldModelHead

        return JepaWorldModelHead
    if name in {
        "QwenBackboneWGRAM",
        "QwenBackboneWGRAMReport",
        "QwenBackboneQTRM",
        "QwenBackboneQTRMReport",
        "QwenLayerWrappedRecursiveCore",
        "QwenLayerWrappedStack",
        "OuroLayerWrappedStack",
        "OuroWeightWrappedRecursiveCore",
        "build_wgram_core_config_from_qwen",
        "build_qtrm_core_config_from_qwen",
    }:
        from .qwen_backbone_wgram import (
            OuroLayerWrappedStack,
            OuroWeightWrappedRecursiveCore,
            QwenBackboneWGRAM,
            QwenBackboneWGRAMReport,
            QwenBackboneQTRM,
            QwenBackboneQTRMReport,
            QwenLayerWrappedRecursiveCore,
            QwenLayerWrappedStack,
            build_wgram_core_config_from_qwen,
            build_qtrm_core_config_from_qwen,
        )

        return {
            "QwenBackboneWGRAM": QwenBackboneWGRAM,
            "QwenBackboneWGRAMReport": QwenBackboneWGRAMReport,
            "QwenBackboneQTRM": QwenBackboneQTRM,
            "QwenBackboneQTRMReport": QwenBackboneQTRMReport,
            "QwenLayerWrappedRecursiveCore": QwenLayerWrappedRecursiveCore,
            "QwenLayerWrappedStack": QwenLayerWrappedStack,
            "OuroLayerWrappedStack": OuroLayerWrappedStack,
            "OuroWeightWrappedRecursiveCore": OuroWeightWrappedRecursiveCore,
            "build_wgram_core_config_from_qwen": build_wgram_core_config_from_qwen,
            "build_qtrm_core_config_from_qwen": build_qtrm_core_config_from_qwen,
        }[name]
    raise AttributeError(name)
