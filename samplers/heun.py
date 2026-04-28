"""
samplers/heun.py — Heun 2次 ODE サンプラー (sigma 空間)
=============================================================================

論文:
  Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.
  (2022). "Elucidating the Design Space of Diffusion-Based Generative Models"
  NeurIPS 2022. https://arxiv.org/abs/2206.00364  (EDM)

概要:
  Heun 法は台形公式 (Trapezoidal Rule) を使った 2 次精度の ODE 積分法。
  各ステップで UNet を 2 回呼ぶ (NFE=2/step) が、
  同じ NFE 数なら Euler より高品質な画像が得られる。

  Euler 法 (1 次) が「現在点の傾き」だけを使うのに対し、
  Heun 法は「現在点の傾き d_1 と、Euler 予測点の傾き d_2 の平均」を使う。

特性:
  タイプ : ODE (決定論的)
  NFE/step: 2 (UNet フォワード 2 回/ステップ)
  推奨ステップ数: 15–35 (NFE ベースでは Euler 30–70 相当)

  ⚠️ 比較グラフでは「ステップ数」ではなく「NFE 数 = steps × 2」で
     他のサンプラーと公平比較すること。

Heun 法の更新式 (EDM Algorithm 1):
  [Predictor — Euler step]
    d_1 = (x_i − D_θ(x_i, σ_i)) / σ_i
    x̃_{i+1} = x_i + (σ_{i+1} − σ_i) * d_1    ... Euler 予測

  [Corrector — 2 次補正]
    d_2 = (x̃_{i+1} − D_θ(x̃_{i+1}, σ_{i+1})) / σ_{i+1}
    x_{i+1} = x_i + (σ_{i+1} − σ_i) * (d_1 + d_2) / 2

  ここで D_θ は v-prediction から復元した x̂_0 (= denoiser 出力)。

最終ステップ (σ_{i+1} = 0):
  σ_{i+1} = 0 のとき d_2 が定義できない (÷0) ため、
  Heun ではなく Euler step のみ実行する。
"""

from __future__ import annotations

from typing import Optional, Union

import torch

from .base import BaseSampler, SchedulerOutput, predict_x0
from .euler import EulerSampler  # sigma スケジュール / scale_model_input を継承


class HeunSampler(EulerSampler):
    """
    Heun 2 次 ODE サンプラー。

    EulerSampler を継承し、set_timesteps / scale_model_input をそのまま使う。
    step() のみ 2 次 Heun 法にオーバーライドする。

    Args:
      num_train_timesteps : 訓練ステップ数 T
      beta_start, beta_end: ノイズスケジュールパラメータ
    """

    # Heun は step 内で UNet を 2 回呼ぶため、2 回目の呼び出しには
    # 外部から UNet が必要。推論ループから unet を渡す設計を採用。
    # (pipeline.py が step_with_unet() を使う)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        Heun 2 次 ODE の 1 ステップを実行する。

        ⚠️ このメソッドは Predictor (1 回目の UNet 出力) のみを使う
           Euler step を返す。2 次補正 (Corrector) は
           step_heun() を使うこと。

        pipeline.py では step_heun() を使う実装を推奨。
        このメソッドは BaseSampler ABC の要件を満たすための Euler フォールバック。
        """
        # Euler step として実行 (BaseSampler の実装を流用)
        return super().step(model_output, timestep, sample)

    def step_heun(
        self,
        model_output_1: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        unet_forward_fn,  # Callable[[Tensor, Tensor], Tensor]
        encoder_hidden_states: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> SchedulerOutput:
        """
        Heun 法の完全な 1 ステップ (Predictor + Corrector) を実行する。

        ---------------------------------------------------------------
        [アルゴリズム] Heun Method (EDM Algorithm 1, Karras et al. 2022):

        入力: x_i, σ_i, σ_{i+1}, v_θ(x_i/c_in, t_i)  ← 1 回目フォワード済み
        出力: x_{i+1}

        --- Predictor (Euler step) ---

        Step 1 — 1 回目の denoiser D_θ(x_i, σ_i) を計算:
          α_t = 1/√(σ_i²+1), σ_t = σ_i/√(σ_i²+1)
          D_θ(x_i, σ_i) = x̂_0 = α_t * x_scaled − σ_t * v_θ1

        Step 2 — 傾き d_1 を計算:
          d_1 = (x_i − D_θ(x_i, σ_i)) / σ_i

        Step 3 — Euler 予測ステップ:
          x̃_{i+1} = x_i + (σ_{i+1} − σ_i) * d_1

        --- Corrector (Heun correction, σ_{i+1} > 0 のとき) ---

        Step 4 — x̃_{i+1} で 2 回目の UNet フォワード:
          v_θ2 = UNet(x̃_{i+1} / √(σ_{i+1}²+1), t_{i+1})

        Step 5 — 2 回目の denoiser D_θ(x̃_{i+1}, σ_{i+1}):
          D_θ(x̃_{i+1}, σ_{i+1}) = predict_x0(v_θ2, ...)

        Step 6 — 傾き d_2 を計算:
          d_2 = (x̃_{i+1} − D_θ(x̃_{i+1}, σ_{i+1})) / σ_{i+1}

        Step 7 — Heun (台形) 補正:
          x_{i+1} = x_i + (σ_{i+1} − σ_i) * (d_1 + d_2) / 2

          台形則: 両端の傾きの平均を使って台形で面積を近似。
          → Euler (左端のみ) より 2 次精度が高い。

        σ_{i+1} = 0 (最終ステップ): Corrector は省略し Euler 結果を返す。

        変数対応:
          sample          ← x_i
          model_output_1  ← v_θ(x_i/c_in, t_i)  (1 回目フォワード)
          sigma_i         ← σ_i
          sigma_next      ← σ_{i+1}
        ---------------------------------------------------------------

        引数:
          model_output_1        : 1 回目の UNet 出力 v_θ  shape [B, C, H, W]
          timestep              : 現在のタイムステップ t_i
          sample                : x_i                      shape [B, C, H, W]
          unet_forward_fn       : UNet の推論関数
                                  fn(x_scaled, t, encoder_hidden_states) -> v_θ
          encoder_hidden_states : テキスト埋め込み (CFG 込み)
          guidance_scale        : CFG スケール (デフォルト 7.5)
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx = self._timestep_to_idx[t]

        dev = sample.device
        sigma_i    = self._sigmas[idx].to(dev)
        sigma_next = self._sigmas[idx + 1].to(dev)

        # ----------------------------------------------------------
        # Step 1: 1 回目の denoiser D_θ(x_i, σ_i) を計算
        #
        # [数式]
        #   c = √(σ_i² + 1)   = 1 / α_t
        #   α_t = 1/c
        #   σ_t^orig = σ_i / c
        #   x̂_0 = α_t * (x_i/c) − σ_t^orig * v_θ1
        # ----------------------------------------------------------
        c_i = (sigma_i ** 2 + 1.0).sqrt()
        x_scaled_i = sample / c_i

        x0_hat_1 = self._sigma_to_x0(model_output_1, sample, sigma_i)
        x0_hat_1 = x0_hat_1.clamp(-1.0, 1.0)

        # ----------------------------------------------------------
        # Step 2 & 3: 傾き d_1 と Euler 予測
        #
        # [数式]
        #   d_1 = (x_i − D_θ(x_i, σ_i)) / σ_i
        #   x̃_{i+1} = x_i + (σ_{i+1} − σ_i) * d_1
        # ----------------------------------------------------------
        d_1 = (sample - x0_hat_1) / sigma_i
        dt = sigma_next - sigma_i
        x_euler = sample + dt * d_1

        # σ_{i+1} = 0 (最終ステップ) → Corrector 不要
        if sigma_next == 0.0:
            return SchedulerOutput(prev_sample=x_euler, pred_original_sample=x0_hat_1)

        # ----------------------------------------------------------
        # Step 4: x̃_{i+1} で 2 回目の UNet フォワード
        #
        # x̃_{i+1} を c_in でスケーリングして UNet に入力する。
        # timestep は σ_{i+1} に対応する t_{i+1} を使う。
        # ----------------------------------------------------------
        t_next = int(self.timesteps[idx + 1].item())
        t_next_tensor = torch.tensor([t_next], device=dev)

        c_next = (sigma_next ** 2 + 1.0).sqrt()
        x_scaled_euler = x_euler / c_next

        # 2 回目の UNet フォワード
        # unet_forward_fn は pipeline から渡される CFG 適用済み関数
        # (内部で uncond/cond の repeat と CFG 適用を行う)
        model_output_2 = unet_forward_fn(
            x_scaled_euler,
            t_next_tensor,
            encoder_hidden_states,
        )

        # ----------------------------------------------------------
        # Step 5 & 6: 2 回目の denoiser と傾き d_2
        # ----------------------------------------------------------
        x0_hat_2 = self._sigma_to_x0(model_output_2, x_euler, sigma_next)
        x0_hat_2 = x0_hat_2.clamp(-1.0, 1.0)

        d_2 = (x_euler - x0_hat_2) / sigma_next

        # ----------------------------------------------------------
        # Step 7: Heun (台形) 補正
        #
        # [数式]
        #   d_avg = (d_1 + d_2) / 2          ... 台形平均
        #   x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_avg
        #
        # Euler の 2 倍の計算コストで 2 次精度を達成する。
        # ----------------------------------------------------------
        d_avg = (d_1 + d_2) / 2.0
        x_next = sample + dt * d_avg

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=x0_hat_2)
