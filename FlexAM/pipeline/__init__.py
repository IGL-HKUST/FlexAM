from .pipeline_wan2_2_fun_control_FlexAM import Wan2_2FunControlPipeline_FlexAM

import importlib.util

if importlib.util.find_spec("paifuser") is not None:
    # --------------------------------------------------------------- #
    #   Sparse Attention
    # --------------------------------------------------------------- #
    from paifuser.ops import sparse_reset

    # Wan2.1
    WanFunInpaintPipeline.__call__ = sparse_reset(WanFunInpaintPipeline.__call__)
    WanFunPipeline.__call__ = sparse_reset(WanFunPipeline.__call__)
    WanFunControlPipeline.__call__ = sparse_reset(WanFunControlPipeline.__call__)
    WanI2VPipeline.__call__ = sparse_reset(WanI2VPipeline.__call__)
    WanPipeline.__call__ = sparse_reset(WanPipeline.__call__)

    # Phantom
    WanFunPhantomPipeline.__call__ = sparse_reset(WanFunPhantomPipeline.__call__)

    # Wan2.2
    Wan2_2FunInpaintPipeline.__call__ = sparse_reset(Wan2_2FunInpaintPipeline.__call__)
    Wan2_2FunPipeline.__call__ = sparse_reset(Wan2_2FunPipeline.__call__)
    Wan2_2FunControlPipeline.__call__ = sparse_reset(Wan2_2FunControlPipeline.__call__)
    Wan2_2Pipeline.__call__ = sparse_reset(Wan2_2Pipeline.__call__)
    Wan2_2I2VPipeline.__call__ = sparse_reset(Wan2_2I2VPipeline.__call__)
    Wan2_2TI2VPipeline.__call__ = sparse_reset(Wan2_2TI2VPipeline.__call__)