"""
samplers/lms2.py — LMS-2 (2次 Linear Multistep Method) サンプラー
=============================================================================

論文:
  Liu, L., Ren, Y., Lin, Z., & Zhao, Z. (2022).
  "Pseudo Numerical Methods for Diffusion Models on Manifolds"
  ICLR 2022. https://arxiv.org/abs/2202.09778

  ※ LMS (Linear Multistep) 自体は古典的な ODE 数値解法。
     Adams-Bashforth 法として知られる explicit multistep 法。

概要:
  LMS-2 は前ステップの傾きを再利用することで、Euler (1 次) より
  高精度な 2 次精度を NFE=1/step で達成する。

  ODE の傾き d を複数ステップ分のバッファに保持し、
  Lagrange 補間多項式の積分で重み付き平均を計算する。

  Heun と違い、1 ステップあたり UNet フォワードは 1 回のみ。
  代わりに前ステップ (i-1) の傾き d_{i-1} を再利用する。

特性:
  タイプ : ODE (決定論的)
  NFE/step: 1 (前ステップ再利用)
  推奨ステップ数: 20–40
  ⚠️ 最初のステップは履歴なし → Euler フォールバック (order=1)

2 次 Adams-Bashforth 式 (等間隔近似):
  x_{i+1} = x_i + (σ_{i+1} − σ_i) * [ (3/2)*d_i − (1/2)*d_{i-1} ]

  等間隔でない場合 (実用的な一般式):
    x_{i+1} = x_i + ∫_{σ_i}^{σ_{i+1}} P(σ) dσ

  P(σ) は d_i, d_{i-1} を通る Lagrange 補間多項式:
    P(σ) = d_i * (σ − σ_{i-1})/(σ_i − σ_{i-1})
          + d_{i-1} * (σ − σ_i)/(σ_{i-1} − σ_i)

  積分を解析的に実行すると LMS 係数 β_0, β_1 が得られる:
    β_0 = ∫_{σ_i}^{σ_{i+1}} (σ − σ_{i-1})/(σ_i − σ_{i-1}) dσ
    β_1 = ∫_{σ_i}^{σ_{i+1}} (σ − σ_i)/(σ_{i-1} − σ_i) dσ
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Union

import torch

from .base import BaseSampler, SchedulerOutput, predict_x0
from .euler import EulerSampler  # sigma スケジュール / scale_model_input を継承


class LMS2Sampler(EulerSampler):
    """
    2 次 Linear Multistep (Adams-Bashforth) サンプラー。

    EulerSampler を継承し、set_timesteps / scale_model_input は流用。
    step() のみ 2 次 LMS にオーバーライドする。

    Args:
      num_train_timesteps : 訓練ステップ数 T
      beta_start, beta_end: ノイズスケジュールパラメータ
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        super().__init__(num_train_timesteps, beta_start, beta_end)
        # 前ステップの傾き d を保存するリングバッファ (order=2 なので最大 2 個)
        self._derivative_buffer: Deque[torch.Tensor] = deque(maxlen=2)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        set_timesteps をオーバーライド: バッファをリセット。
        """
        super().set_timesteps(num_inference_steps, device)
        self._derivative_buffer.clear()  # 前実験の履歴をリセット

    # ----------------------------------------------------------------
    # step の実装
    # ----------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        LMS-2 の 1 ステップを実行する。

        ---------------------------------------------------------------
        [アルゴリズム] 2 次 Linear Multistep (Adams-Bashforth):

        入力: x_i, σ_i, v_θ(x_i/c_in, t_i)
        出力: x_{i+1}

        Step 1 — 現在の傾き d_i を計算 (Euler と同じ):
          c = √(σ_i² + 1)
          x̂_0 = predict_x0(v_θ, x_i/c, 1/c, σ_i/c)
          d_i = (x_i − x̂_0) / σ_i

        Step 2 — 積分重み (LMS 係数) を計算:

          バッファに d_{i-1} が存在する場合 (order=2):
            Lagrange 補間多項式を σ_i → σ_{i+1} で積分

            β_0 = ∫_{σ_i}^{σ_{i+1}} (σ − σ_{i-1})/(σ_i − σ_{i-1}) dσ
            β_1 = ∫_{σ_i}^{σ_{i+1}} (σ − σ_i)/(σ_{i-1} − σ_i) dσ

            解析解:
              h    = σ_{i+1} − σ_i
              Δ_{01} = σ_i − σ_{i-1}
              β_0 = h + h²/(2 * Δ_{01})
              β_1 = −h²/(2 * Δ_{01})

            等間隔 (Δ_{01} = h) のとき: β_0 = 3h/2, β_1 = −h/2

          バッファが空 (最初のステップ、order=1):
            β_0 = σ_{i+1} − σ_i = h   (Euler フォールバック)

        Step 3 — x_{i+1} を更新:
          x_{i+1} = x_i + β_0 * d_i + β_1 * d_{i-1}

        変数対応:
          d_i    ← _derivative_buffer[-1] (現在、バッファ追加後)
          d_{i-1}← _derivative_buffer[-2] (前ステップ)
          σ_{i-1}← sigma_prev (バッファ追加前の最後の sigma)
        ---------------------------------------------------------------
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx = self._timestep_to_idx[t]

        dev = sample.device
        sigma_i    = self._sigmas[idx].to(dev)
        sigma_next = self._sigmas[idx + 1].to(dev)

        # ----------------------------------------------------------
        # Step 1: 現在の傾き d_i を計算
        #
        # [数式]
        #   c_i = √(σ_i² + 1)
        #   α_t = 1/c_i,  σ_t = σ_i/c_i
        #   x̂_0 = predict_x0(v_θ, x_i/c_i, α_t, σ_t)
        #   d_i = (x_i − x̂_0) / σ_i
        # ----------------------------------------------------------
        c_i = (sigma_i ** 2 + 1.0).sqrt()
        x_scaled = sample / c_i

        x0_hat = self._sigma_to_x0(model_output, sample, sigma_i)
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        d_i = (sample - x0_hat) / sigma_i

        # ----------------------------------------------------------
        # Step 2: LMS 係数を計算し更新
        #
        # バッファ長に応じて order を自動決定する
        # ----------------------------------------------------------
        order = len(self._derivative_buffer)  # 0 (Euler) or 1 (2 次 LMS)

        if order == 0:
            # --------------------------------------------------------
            # 最初のステップ: Euler フォールバック (1 次)
            #
            # [数式]
            #   x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_i
            # --------------------------------------------------------
            h = sigma_next - sigma_i
            x_next = sample + h * d_i

        else:
            # --------------------------------------------------------
            # 2 次 LMS (Adams-Bashforth 2次):
            #
            # バッファ: [d_{i-1}]  (len=1)
            # 使用 sigma: σ_{i-1} をバッファに保存しておく必要あり
            #
            # [数式] Lagrange 補間の積分:
            #
            #   h    = σ_{i+1} − σ_i   (現ステップ幅)
            #   Δ_01 = σ_i − σ_{i-1}  (前ステップ幅)
            #
            #   β_0 = h + h² / (2 * Δ_01)     (d_i の係数)
            #   β_1 = −h² / (2 * Δ_01)         (d_{i-1} の係数)
            #
            # 等間隔 (Δ_01 = h) なら β_0 = 3h/2, β_1 = −h/2
            # --------------------------------------------------------
            d_prev = self._derivative_buffer[-1]
            sigma_prev = self._sigma_prev  # 前ステップの sigma

            h     = sigma_next - sigma_i
            delta = sigma_i - sigma_prev

            # 数値安定化: delta がほぼ 0 の場合は Euler にフォールバック
            if abs(float(delta)) < 1e-6:
                x_next = sample + h * d_i
            else:
                beta_0 = h + h ** 2 / (2.0 * delta)
                beta_1 = -(h ** 2) / (2.0 * delta)
                x_next = sample + beta_0 * d_i + beta_1 * d_prev

        # 次のステップのために σ_i を保存
        self._sigma_prev = sigma_i.clone()

        # バッファを更新 (maxlen=2 で自動的に古いものが削除される)
        self._derivative_buffer.append(d_i)

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=x0_hat)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        set_timesteps のオーバーライド: バッファと sigma_prev をリセット。
        """
        # EulerSampler の set_timesteps を呼ぶ
        EulerSampler.set_timesteps(self, num_inference_steps, device)
        self._derivative_buffer.clear()
        self._sigma_prev: Optional[torch.Tensor] = None
