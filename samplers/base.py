"""
samplers/base.py — 全サンプラー共通の基底クラスとユーティリティ
=============================================================================

このモジュールが提供するもの:
  1. BaseSampler   — 全サンプラーが継承する抽象基底クラス (ABC)
                     diffusers Scheduler インターフェース準拠
  2. NoiseSchedule — SD 2.1 (scaled_linear) のノイズスケジュール計算
  3. v_prediction ヘルパー関数 — predict_x0 / predict_eps

SD 2.1 のノイズスケジュール (scaled_linear):
  β_t = ( √β_start + t/T * (√β_end − √β_start) )²
  ᾱ_t = Π_{s=1}^{t} (1 − β_s)        ... 累積積
  α_t = √ᾱ_t                           ... signal scale
  σ_t = √(1 − ᾱ_t)                     ... noise scale
  Σ_t = σ_t  (連続時間では σ に対応)

v-prediction (SD 2.1 の予測タイプ):
  UNet は「velocity」v_θ(x_t, t) を出力する。
  v_θ は以下のように定義される:
    v_t = α_t * ε − σ_t * x_0          (Eq.2 in [Salimans & Ho 2022])

  ここから x_0 と ε を復元できる:
    x̂_0 = α_t * x_t − σ_t * v_θ       (predict_x0)
    ε̂   = σ_t * x_t + α_t * v_θ       (predict_eps)

  証明:
    x_t = α_t * x_0 + σ_t * ε より
    α_t * x_t − σ_t * v_θ
      = α_t*(α_t*x_0 + σ_t*ε) − σ_t*(α_t*ε − σ_t*x_0)
      = α_t²*x_0 + α_t*σ_t*ε − α_t*σ_t*ε + σ_t²*x_0
      = (α_t² + σ_t²)*x_0 = x_0   (∵ α_t² + σ_t² = 1)

参考文献:
  [DDPM]   Ho et al., 2020. https://arxiv.org/abs/2006.11239
  [DDIM]   Song et al., 2020. https://arxiv.org/abs/2010.02502
  [v-pred] Salimans & Ho, 2022. https://arxiv.org/abs/2202.00512
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


# =============================================================================
# ノイズスケジュール計算
# =============================================================================

class NoiseSchedule:
    """
    SD 2.1 の scaled_linear ノイズスケジュールを計算し保持するクラス。

    scaled_linear スケジュール (diffusers 実装準拠):
      β_t = ( √β_start + t/(T-1) * (√β_end − √β_start) )²

    ここでは 0-indexed: t ∈ {0, 1, ..., T-1}

    主要なテンソル (shape: [T]):
      betas        : β_t
      alphas       : α_t = 1 − β_t
      alphas_cumprod : ᾱ_t = Π_{s=0}^{t} α_s
      sqrt_alphas_cumprod : √ᾱ_t  = α_t (signal scale)
      sqrt_one_minus_alphas_cumprod : √(1−ᾱ_t) = σ_t (noise scale)
      sigmas_for_ode : σ_t^ODE = √(1−ᾱ_t) / √ᾱ_t  (Euler/Heun/LMS/DPM で使用)
      log_snr      : λ_t = log(ᾱ_t / (1−ᾱ_t)) = log(α_t² / σ_t²)
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps

        # ----------------------------------------------------------
        # [数式] scaled_linear ベータスケジュール:
        #
        #   β_t = ( √β_start + t/(T-1) * (√β_end − √β_start) )²
        #
        # これは β の線形補間ではなく √β の線形補間であることに注意。
        # ----------------------------------------------------------
        betas = (
            torch.linspace(
                math.sqrt(beta_start),
                math.sqrt(beta_end),
                num_train_timesteps,
                dtype=torch.float64,
            )
            ** 2
        )
        self.betas: torch.Tensor = betas.float()

        # ----------------------------------------------------------
        # [数式] α_t と累積積 ᾱ_t:
        #
        #   α_t  = 1 − β_t
        #   ᾱ_t  = Π_{s=0}^{t} α_s
        # ----------------------------------------------------------
        alphas = 1.0 - self.betas
        self.alphas_cumprod: torch.Tensor = torch.cumprod(alphas, dim=0)

        # ----------------------------------------------------------
        # [数式] signal scale と noise scale:
        #
        #   √ᾱ_t  = alpha_t    (signal を x_0 に掛ける係数)
        #   √(1−ᾱ_t) = sigma_t (noise を ε に掛ける係数)
        #
        # x_t = √ᾱ_t * x_0 + √(1−ᾱ_t) * ε   (forward process)
        # ----------------------------------------------------------
        self.sqrt_alphas_cumprod: torch.Tensor = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod: torch.Tensor = torch.sqrt(
            1.0 - self.alphas_cumprod
        )

        # ----------------------------------------------------------
        # [数式] ODE sigma (Karras/Euler 空間で使用):
        #
        #   σ_t^ODE = √(1−ᾱ_t) / √ᾱ_t
        #
        # これは連続時間 ODE の「ノイズレベル」に対応し、
        # Euler / Heun / LMS / DPM-Solver が使う σ の定義。
        # ----------------------------------------------------------
        self.sigmas_for_ode: torch.Tensor = (
            self.sqrt_one_minus_alphas_cumprod / self.sqrt_alphas_cumprod
        )

        # ----------------------------------------------------------
        # [数式] log-SNR (DPM-Solver で使用):
        #
        #   λ_t = log(ᾱ_t / (1−ᾱ_t))
        #       = log(α_t² / σ_t²)  (α_t = √ᾱ_t, σ_t = √(1−ᾱ_t) と定義)
        #
        # DPM-Solver は λ を変数として ODE を解く。
        # ----------------------------------------------------------
        self.log_snr: torch.Tensor = torch.log(
            self.alphas_cumprod / (1.0 - self.alphas_cumprod)
        )

    def get_at(self, t_indices: torch.Tensor) -> dict:
        """
        タイムステップインデックス t_indices に対応するスケジュール値を返す。

        引数:
          t_indices: shape [B] の整数テンソル (0 <= t < T)

        戻り値 (dict):
          alpha_t    : √ᾱ_t   shape [B, 1, 1, 1] (ブロードキャスト用)
          sigma_t    : √(1−ᾱ_t)
          sigma_ode  : √(1−ᾱ_t) / √ᾱ_t
          log_snr    : λ_t
        """
        def _reshape(x: torch.Tensor) -> torch.Tensor:
            return x[t_indices].view(-1, 1, 1, 1)

        return {
            "alpha_t": _reshape(self.sqrt_alphas_cumprod),
            "sigma_t": _reshape(self.sqrt_one_minus_alphas_cumprod),
            "sigma_ode": _reshape(self.sigmas_for_ode),
            "log_snr": _reshape(self.log_snr),
        }


# =============================================================================
# v-prediction ヘルパー関数
# =============================================================================

def predict_x0(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """
    v-prediction の UNet 出力から x_0 を推定する。

    SD 2.1 UNet は v_θ(x_t, t) を出力する。
    ここで v は「velocity」と呼ばれ、以下で定義:
      v_t = α_t * ε − σ_t * x_0     [Salimans & Ho 2022, Eq.2]

    x_0 の復元式 (predict_x0):
    -------------------------------------------------------
    [数式] x̂_0 の導出:

      x_t = α_t * x_0 + σ_t * ε  より
      x_0 = (x_t − σ_t * ε) / α_t

      v_θ = α_t * ε − σ_t * x_0 を ε について解くと
      ε = (v_θ + σ_t * x_0) / α_t

      これを x_0 式に代入して整理:
        x̂_0 = α_t * x_t − σ_t * v_θ

    変数対応:
      v_pred  ← v_θ(x_t, t)   UNet の出力
      x_t     ← x_t           現在のノイズ付き潜在変数
      alpha_t ← α_t = √ᾱ_t   signal scale
      sigma_t ← σ_t = √(1−ᾱ_t) noise scale
    -------------------------------------------------------

    引数:
      v_pred  : shape [B, C, H, W]
      x_t     : shape [B, C, H, W]
      alpha_t : shape [B, 1, 1, 1] または スカラー
      sigma_t : shape [B, 1, 1, 1] または スカラー

    戻り値:
      x0_hat  : shape [B, C, H, W]
    """
    return alpha_t * x_t - sigma_t * v_pred


def predict_eps(
    v_pred: torch.Tensor,
    x_t: torch.Tensor,
    alpha_t: torch.Tensor,
    sigma_t: torch.Tensor,
) -> torch.Tensor:
    """
    v-prediction の UNet 出力から ε (ノイズ) を推定する。

    -------------------------------------------------------
    [数式] ε̂ の導出:

      x_t = α_t * x_0 + σ_t * ε  より
      ε = (x_t − α_t * x_0) / σ_t

      predict_x0 から x̂_0 = α_t * x_t − σ_t * v_θ を代入:
        ε̂ = (x_t − α_t * (α_t * x_t − σ_t * v_θ)) / σ_t
           = (x_t*(1 − α_t²) + α_t*σ_t*v_θ) / σ_t
           = σ_t * x_t + α_t * v_θ   (∵ 1 − α_t² = σ_t²)

    変数対応:
      v_pred  ← v_θ(x_t, t)
      x_t     ← x_t
      alpha_t ← α_t = √ᾱ_t
      sigma_t ← σ_t = √(1−ᾱ_t)
    -------------------------------------------------------
    """
    return sigma_t * x_t + alpha_t * v_pred


# =============================================================================
# スケジューラ出力データクラス
# =============================================================================

@dataclass
class SchedulerOutput:
    """
    各サンプラーの step() が返す出力。
    diffusers の SchedulerOutput と互換。

    Attributes:
      prev_sample : x_{t-1}、次の推論ステップへ渡す潜在変数
      pred_original_sample : x̂_0 の推定値 (オプション、デバッグ用)
    """
    prev_sample: torch.Tensor
    pred_original_sample: Optional[torch.Tensor] = None


# =============================================================================
# 抽象基底クラス
# =============================================================================

class BaseSampler(ABC):
    """
    全サンプラーの抽象基底クラス。

    diffusers Scheduler インターフェースに準拠:
      - set_timesteps(num_inference_steps, device)
      - scale_model_input(sample, timestep) -> scaled_sample
      - step(model_output, timestep, sample) -> SchedulerOutput

    サブクラスが必ず実装すること:
      - step()

    NoiseSchedule を内包し、サブクラスから self.schedule で参照可能。
    """

    # diffusers との互換性のために必要な属性
    config: dict = field(default_factory=dict)

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        self.num_train_timesteps = num_train_timesteps
        self.schedule = NoiseSchedule(num_train_timesteps, beta_start, beta_end)

        # set_timesteps() で設定される推論用タイムステップ
        self.timesteps: Optional[torch.Tensor] = None
        self.num_inference_steps: Optional[int] = None

        # prediction_type は pipeline.py が set() で注入する
        # デフォルト v_prediction (SD 2.x); epsilon の場合は pipeline が上書き
        self.prediction_type: str = "v_prediction"

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        推論用タイムステップ列を設定する。

        デフォルト実装: 訓練ステップ T を num_inference_steps に
        均等サブサンプリングして降順で返す。

        DDPM: [999, 979, 959, ...] のような等間隔離散化
        サブクラスで sigma 空間のスケジュールが必要な場合は override する。

        引数:
          num_inference_steps : 推論ステップ数
          device              : タイムステップを置くデバイス
        """
        self.num_inference_steps = num_inference_steps

        # ----------------------------------------------------------
        # [数式] 離散タイムステップのサブサンプリング:
        #
        #   t_i = round( T - 1 - i * (T-1)/(N-1) )  for i=0,...,N-1
        #
        # 例: T=1000, N=20 → t = [999, 947, 894, ..., 52, 0]
        # ----------------------------------------------------------
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (
            torch.arange(0, num_inference_steps, dtype=torch.long) * step_ratio
        ).flip(0)

        self.timesteps = timesteps.to(device)

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        UNet への入力をスケーリングする。

        デフォルト実装: 恒等変換 (DDPM / DDIM で使用)
        Euler / Heun / LMS / DPM-Solver はサブクラスで override し、
          x_scaled = x / √(σ_ODE² + 1)
        を適用する。

        この正規化は、EDM (Karras et al. 2022) の c_in 係数に相当する。
        UNet がスケール不変な入力分布を期待するため必要。
        """
        return sample

    @abstractmethod
    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        1 サンプリングステップを実行する。

        引数:
          model_output : UNet の出力  v_θ(x_t, t)   shape [B, C, H, W]
          timestep     : 現在のタイムステップ t      スカラーまたは shape [B]
          sample       : 現在の潜在変数 x_t          shape [B, C, H, W]

        戻り値:
          SchedulerOutput.prev_sample = x_{t-1}
        """
        ...

    # ------------------------------------------------------------------
    # 共通ヘルパー: v_prediction → x̂_0 / ε̂
    # ------------------------------------------------------------------

    def _predict_x0_eps(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        timestep_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        タイムステップインデックスから α_t, σ_t を取得し
        x̂_0 と ε̂ を同時に計算する。

        戻り値: (x0_hat, eps_hat)
        """
        # schedule テンソルを model_output と同じデバイスへ
        alpha_t = self.schedule.sqrt_alphas_cumprod[timestep_idx].to(
            model_output.device
        ).view(1, 1, 1, 1)
        sigma_t = self.schedule.sqrt_one_minus_alphas_cumprod[timestep_idx].to(
            model_output.device
        ).view(1, 1, 1, 1)

        if self.prediction_type == "v_prediction":
            x0_hat = predict_x0(model_output, sample, alpha_t, sigma_t)
            eps_hat = predict_eps(model_output, sample, alpha_t, sigma_t)
        else:  # epsilon prediction (SD 1.x)
            # [数式] epsilon 予測: x̂_0 = (x_t − σ_t * ε_θ) / α_t
            x0_hat = (sample - sigma_t * model_output) / alpha_t
            eps_hat = model_output
        return x0_hat, eps_hat
