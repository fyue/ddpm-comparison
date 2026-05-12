"""
samplers/__init__.py — サンプラーパッケージの公開 API
"""

from .base import BaseSampler, NoiseSchedule, SchedulerOutput, predict_x0, predict_eps
from .ddpm import DDPMSampler
from .ddim import DDIMSampler
from .euler import EulerSampler
from .heun import HeunSampler
from .lms2 import LMS2Sampler
from .dpm_solver import (
    DPMSolver1,
    DPMSolver2Singlestep,
    DPMSolverMultistep1,
    DPMSolverMultistep2,
)

__all__ = [
    "BaseSampler",
    "NoiseSchedule",
    "SchedulerOutput",
    "predict_x0",
    "predict_eps",
    "DDPMSampler",
    "DDIMSampler",
    "EulerSampler",
    "HeunSampler",
    "LMS2Sampler",
    "DPMSolver1",
    "DPMSolver2Singlestep",
    "DPMSolverMultistep1",
    "DPMSolverMultistep2",
]

# ----------------------------------------------------------------
# 全サンプラーの設定レジストリ
# pipeline.py / experiment.py から参照する
# ----------------------------------------------------------------
SAMPLER_REGISTRY = {
    "ddpm":               {"cls": DDPMSampler,  "kwargs": {}},
    "ddim":               {"cls": DDIMSampler,  "kwargs": {"eta": 0.0}},
    "euler":              {"cls": EulerSampler, "kwargs": {}},
    "heun":               {"cls": HeunSampler,  "kwargs": {}},
    "lms2":               {"cls": LMS2Sampler,  "kwargs": {}},
    "dpm_solver_1":        {"cls": DPMSolver1,           "kwargs": {}},
    "dpm_solver_2_single": {"cls": DPMSolver2Singlestep, "kwargs": {}},
    "dpm_solver_2_multi":  {"cls": DPMSolverMultistep2,  "kwargs": {}},
}

# NFE (Network Function Evaluations) per step の情報
NFE_PER_STEP = {
    "ddpm":                1,
    "ddim":                1,
    "euler":               1,
    "heun":                2,
    "lms2":                1,
    "dpm_solver_1":        1,
    "dpm_solver_2_single": 2,
    "dpm_solver_2_multi":  1,
}


def build_sampler(name: str) -> BaseSampler:
    """
    名前からサンプラーインスタンスを生成する。

    使用例:
      sampler = build_sampler("dpm_solver_2_multi")

    Args:
      name: SAMPLER_REGISTRY のキー

    Returns:
      BaseSampler インスタンス
    """
    if name not in SAMPLER_REGISTRY:
        raise ValueError(
            f"未知のサンプラー名: '{name}'\n"
            f"利用可能: {list(SAMPLER_REGISTRY.keys())}"
        )
    entry = SAMPLER_REGISTRY[name]
    return entry["cls"](**entry["kwargs"])
