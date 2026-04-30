"""
samplers/dpm_solver.py — DPM-Solver (order=1/2/3, singlestep/multistep)
=============================================================================

論文:
  Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).
  "DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling
   in Around 10 Steps"
  NeurIPS 2022. https://arxiv.org/abs/2206.00927

  Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., & Zhu, J. (2022).
  "DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic
   Models"
  https://arxiv.org/abs/2211.01095

概要:
  DPM-Solver は拡散 ODE を log-SNR 空間 (λ 空間) で解析解を求め、
  Taylor 展開により高次精度を達成するサンプラー。

  連続時間 ODE (DPM-Solver++ の data-prediction 形式):
    dx/dλ = x − D_θ(x, t)    ... D_θ は x̂_0 の推定値

  log-SNR λ_t = log(ᾱ_t / (1−ᾱ_t)) = log(α_t² / σ_t²)

  各 order の NFE/step:
    Order 1 (singlestep = multistep) : NFE = 1
    Order 2 singlestep               : NFE = 2
    Order 2 multistep                : NFE = 1 (前ステップ再利用)
    Order 3 singlestep               : NFE = 3
    Order 3 multistep                : NFE = 1 (前 2 ステップ再利用)

  ⚠️ Order 1 は DDIM (η=0) と数学的に等価。

=============================================================================
DPM-Solver++ の更新式 (data-prediction form):
=============================================================================

[Order 1] — Taylor 展開 0 次項のみ (= DDIM)

  x_{t-1} = (α_{t-1}/α_t) * x_t
             − α_{t-1} * (e^h − 1) * D_θ(x_t, t)

  ここで h = λ_{t-1} − λ_t > 0

[Order 2 singlestep] — 1 次補正項追加 (NFE=2)

  中間点 t_mid: λ_mid = (λ_t + λ_{t-1}) / 2
  u = step(x_t → t_mid) using Order 1

  x_{t-1} = (α_{t-1}/α_t) * x_t
             − α_{t-1} * (e^h − 1) * D_θ(x_t, t)
             − α_{t-1}/2 * (e^h − 1) * [D_θ(u, t_mid) − D_θ(x_t, t)]

  整理すると:
  x_{t-1} = Order1_term
             − α_{t-1}/2 * (e^h − 1) * ΔD

  ΔD = D_θ(u, t_mid) − D_θ(x_t, t)   (1 次差分)

[Order 2 multistep] — 前ステップ D_θ を再利用 (NFE=1)

  ΔD = D_θ(x_t, t) − D_θ(x_{t_prev}, t_prev)

  補正係数:
    r = (λ_t − λ_{t_prev}) / (2 * h)   ... 前ステップ幅の比

  x_{t-1} = Order1_term + α_{t-1} * (e^h−1)/h * r * ΔD * h

[Order 3 singlestep] — 2 次補正項追加 (NFE=3)

  中間点 t_1: λ_1 = λ_t + h/3
  中間点 t_2: λ_2 = λ_t + 2h/3

  各中間点で Order 1/2 step を実行して u_1, u_2 を取得。
  3 点の D_θ から 2 次 Taylor 展開係数を計算して更新。

[Order 3 multistep] — 前 2 ステップ再利用 (NFE=1)

  D_0 = D_θ(x_t, t)          (現在)
  D_1 = D_θ(x_{t-1}, t_1)    (前ステップ)
  D_2 = D_θ(x_{t-2}, t_2)    (前々ステップ)

  1 次差分: ΔD_0 = (D_0 − D_1) / r_0
  2 次差分: ΔΔD  = (ΔD_0 − ΔD_1) / r_1

  x_{t-1} = Order1_term
             + 補正1 * ΔD_0
             + 補正2 * ΔΔD
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional, Union

import torch
import math

from .base import BaseSampler, SchedulerOutput, predict_x0
from .euler import EulerSampler  # sigma スケジュール / scale_model_input を継承


class DPMSolver(EulerSampler):
    """
    DPM-Solver++ サンプラー (order=1/2/3, singlestep/multistep)。

    EulerSampler を継承し sigma スケジュールを共有。
    step() を DPM-Solver++ 公式で実装する。

    Args:
      order        : 1, 2, 3 のいずれか
      solver_mode  : "singlestep" または "multistep"
                     - singlestep: 各ステップで order 回 UNet フォワード
                     - multistep : NFE=1/step、前ステップ結果を再利用
    """

    def __init__(
        self,
        order: int = 2,
        solver_mode: str = "multistep",
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        super().__init__(num_train_timesteps, beta_start, beta_end)
        assert order in (1, 2, 3), f"order は 1, 2, 3 のいずれかを指定してください: {order}"
        assert solver_mode in ("singlestep", "multistep"), (
            f"solver_mode は 'singlestep' または 'multistep' を指定してください: {solver_mode}"
        )
        self.order = order
        self.solver_mode = solver_mode

        # multistep 用: 前ステップの D_θ 履歴 (maxlen = order-1)
        self._model_output_buffer: Deque[torch.Tensor] = deque(maxlen=max(order - 1, 1))
        # multistep 用: 対応する λ 値の履歴
        self._lambda_buffer: Deque[float] = deque(maxlen=max(order - 1, 1))

        # log-SNR (λ) テンソルはスケジュールから参照
        # self.schedule.log_snr[t] = λ_t

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        タイムステップ設定とバッファのリセット。
        """
        super().set_timesteps(num_inference_steps, device)
        self._model_output_buffer.clear()
        self._lambda_buffer.clear()

    # ----------------------------------------------------------------
    # 共通ヘルパー
    # ----------------------------------------------------------------

    def _get_alpha_sigma(self, t_idx: int, device: torch.device):
        """
        タイムステップインデックス t_idx から α_t, σ_t を取得。

        戻り値: (alpha_t, sigma_t, lambda_t)  全てスカラーテンソル
        """
        alpha_t = self.schedule.sqrt_alphas_cumprod[t_idx].to(device)
        sigma_t = self.schedule.sqrt_one_minus_alphas_cumprod[t_idx].to(device)
        lambda_t = self.schedule.log_snr[t_idx].to(device)
        return alpha_t, sigma_t, lambda_t

    def _get_denoiser(
        self,
        v_pred: torch.Tensor,
        x: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        v_pred と x から D_θ (= x̂_0) を計算する。

        ---------------------------------------------------------------
        [数式] DPM-Solver++ の data-prediction form:
          D_θ(x, t) = x̂_0 = α_t * x_t^orig − σ_t * v_θ

          ただし UNet 入力は x_scaled = x / √(σ_ODE² + 1) なので
          x_t^orig = x_scaled = x / c (c = √(σ_ODE²+1))

          sigma_ODE[t_idx] から α_t, σ_t を再計算:
            alpha_t = 1/√(σ_ODE²+1)
            sigma_t = σ_ODE/√(σ_ODE²+1)
        ---------------------------------------------------------------
        """
        sigma_ode = self.schedule.sigmas_for_ode[t_idx].to(x.device)
        x0_hat = self._sigma_to_x0(v_pred, x, sigma_ode)
        return x0_hat.clamp(-1.0, 1.0)

    # ----------------------------------------------------------------
    # Order 1 の基本更新 (= DDIM と等価)
    # ----------------------------------------------------------------

    def _order1_step(
        self,
        x_t: torch.Tensor,
        d_theta_t: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_tm1: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_tm1: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver++ Order 1 の更新式。

        ---------------------------------------------------------------
        [数式] DPM-Solver++ Order 1 (Lu et al. 2022, Eq.14):

          h = λ_{t-1} − λ_t   (> 0, λ は単調増加方向)

          x_{t-1} = (α_{t-1}/α_t) * x_t
                    − α_{t-1} * (e^h − 1) * D_θ(x_t, t)

          これは DDIM (η=0) と数学的に等価であることが証明されている。

        変数対応:
          x_t        ← x_t
          d_theta_t  ← D_θ(x_t, t) = x̂_0
          alpha_t    ← α_t = √ᾱ_t
          alpha_tm1  ← α_{t-1} = √ᾱ_{t-1}
          lambda_t   ← λ_t = log(α_t²/σ_t²)
          lambda_tm1 ← λ_{t-1}
        ---------------------------------------------------------------
        """
        # h = λ_{t-1} − λ_t
        h = lambda_tm1 - lambda_t
        # φ_1 = e^h − 1
        phi1 = torch.expm1(h)  # numerically stable than exp(h)-1

        # x_{t-1} = (α_{t-1}/α_t)*x_t − α_{t-1}*(e^h−1)*D_θ
        return (alpha_tm1 / alpha_t) * x_t - alpha_tm1 * phi1 * d_theta_t

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
        DPM-Solver の 1 ステップを実行する。

        multistep の場合はこのメソッドのみで完結 (NFE=1)。
        singlestep の order >= 2 の場合は追加の UNet フォワードが
        必要なため、step_singlestep() を pipeline.py から呼ぶこと。

        このメソッドは:
          - Order 1: singlestep / multistep 両方対応
          - Order 2 multistep: 前ステップ D_θ を再利用
          - Order 3 multistep: 前 2 ステップ D_θ を再利用
          - Order 2/3 singlestep: Euler フォールバック (要 step_singlestep)
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx = self._timestep_to_idx[t]

        dev = sample.device
        t_next_idx = idx + 1
        # 次のタイムステップのインデックス (最後は 0 に近づく)
        t_next = int(self.timesteps[t_next_idx].item()) if t_next_idx < len(self.timesteps) else 0

        # スケジュール値の取得
        alpha_t,  sigma_t,  lambda_t  = self._get_alpha_sigma(t, dev)
        alpha_tm1, sigma_tm1, lambda_tm1 = self._get_alpha_sigma(t_next, dev)

        # D_θ (= x̂_0) を計算
        d_theta = self._get_denoiser(model_output, sample, t)

        if self.solver_mode == "multistep":
            x_next = self._multistep_update(
                sample, d_theta,
                alpha_t, alpha_tm1, lambda_t, lambda_tm1,
            )
        else:
            # singlestep の order >= 2 は Euler フォールバック
            # 本来は step_singlestep() を使うこと
            x_next = self._order1_step(
                sample, d_theta,
                alpha_t, alpha_tm1, lambda_t, lambda_tm1,
            )

        # バッファを更新
        self._model_output_buffer.append(d_theta)
        self._lambda_buffer.append(float(lambda_t))

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta)

    def _multistep_update(
        self,
        x_t: torch.Tensor,
        d_theta_t: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_tm1: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_tm1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multistep モードの更新。現在のバッファ長に応じて order を自動決定。

        ---------------------------------------------------------------
        バッファ長 0 → order-1 (Euler) フォールバック
        バッファ長 1 → order 2 multistep
        バッファ長 2 → order 3 multistep (self.order==3 のとき)
        ---------------------------------------------------------------
        """
        effective_order = min(self.order, len(self._model_output_buffer) + 1)

        if effective_order == 1:
            # --------------------------------------------------------
            # Order 1 (= Euler / DDIM):
            # [数式] x_{t-1} = (α_{t-1}/α_t)*x_t − α_{t-1}*(e^h−1)*D_θ
            # --------------------------------------------------------
            return self._order1_step(
                x_t, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )

        elif effective_order == 2:
            # --------------------------------------------------------
            # Order 2 multistep (Adams-Bashforth 風):
            #
            # [数式] (DPM-Solver++ Eq.17):
            #
            #   h = λ_{t-1} − λ_t
            #   r_0 = (λ_t − λ_{prev}) / h   (前ステップの相対幅)
            #
            #   D_1 = (D_θ(x_t,t) − D_θ(x_{prev},t_{prev})) / r_0
            #       = ΔD / r_0
            #
            #   x_{t-1} = order1_term + α_{t-1} * φ_1 / (2*r_0) * h * ΔD
            #           = order1_term + α_{t-1} * (e^h−1) / (2*r_0) * ΔD
            #
            # ここで ΔD = D_θ_now − D_θ_prev (1 次差分)
            # --------------------------------------------------------
            d_prev = self._model_output_buffer[-1]
            lambda_prev = self._lambda_buffer[-1]

            h = lambda_tm1 - lambda_t
            h_float = float(h)
            if abs(h_float) < 1e-6:
                # 最終ステップ (h≈0): order-1 にフォールバック
                return self._order1_step(x_t, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1)
            h_prev = float(lambda_t) - lambda_prev    # > 0
            r_0 = h_prev / h_float

            phi1 = torch.expm1(h)
            order1 = self._order1_step(
                x_t, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )

            # ΔD = D_θ_now − D_θ_prev
            delta_d = d_theta_t - d_prev
            # 補正項: α_{t-1} * (e^h−1) / (2*r_0) * ΔD
            correction = alpha_tm1 * phi1 / (2.0 * r_0) * delta_d

            return order1 + correction

        else:
            # --------------------------------------------------------
            # Order 3 multistep:
            #
            # [数式] (DPM-Solver++ Eq.18):
            #
            #   h = λ_{t-1} − λ_t
            #   r_0 = h_1 / h   (1 つ前のステップ幅比)
            #   r_1 = h_2 / h   (2 つ前のステップ幅比)
            #
            #   1 次差分:
            #     D1_0 = (D_0 − D_1) / r_0
            #     D1_1 = (D_1 − D_2) / r_1
            #
            #   2 次差分:
            #     D2 = (D1_0 − D1_1) / (r_0 + r_1)
            #
            #   x_{t-1} = order1_term
            #             + α_{t-1} * φ_1 * D1_0 / (2*r_0) * h
            #             + α_{t-1} * φ_2 * D2        (3次項)
            #
            #   φ_2 = (e^h − 1 − h) / h   ... DPM-Solver Eq.4
            # --------------------------------------------------------
            d_prev1 = self._model_output_buffer[-1]  # D_θ_{t-1}
            d_prev2 = self._model_output_buffer[-2]  # D_θ_{t-2}
            lambda_1 = self._lambda_buffer[-1]       # λ_{t-1}
            lambda_2 = self._lambda_buffer[-2]       # λ_{t-2}

            h  = float(lambda_tm1 - lambda_t)
            if abs(h) < 1e-6:
                # 最終ステップ (h≈0): order-1 にフォールバック
                return self._order1_step(x_t, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1)
            h1 = float(lambda_t) - lambda_1
            h2 = lambda_1 - lambda_2

            r_0 = h1 / h
            r_1 = h2 / h

            # 1 次差分
            D1_0 = (d_theta_t - d_prev1) / r_0
            D1_1 = (d_prev1   - d_prev2) / r_1
            # 2 次差分
            D2 = (D1_0 - D1_1) / (r_0 + r_1)

            phi1 = torch.expm1(torch.tensor(h, device=x_t.device, dtype=x_t.dtype))
            # φ_2 = (e^h − 1 − h) / h
            phi2 = (phi1 - h) / h

            order1 = self._order1_step(
                x_t, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )
            correction1 = alpha_tm1 * phi1 / (2.0 * r_0) * (d_theta_t - d_prev1)
            correction2 = alpha_tm1 * phi2 * D2

            return order1 + correction1 + correction2

    # ----------------------------------------------------------------
    # Singlestep 用の完全ステップ (pipeline.py から呼ぶ)
    # ----------------------------------------------------------------

    def step_singlestep(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        unet_forward_fn: Callable,
        encoder_hidden_states: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> SchedulerOutput:
        """
        DPM-Solver++ singlestep の完全ステップ。
        order に応じて 1〜3 回の UNet フォワードを実行する。

        ---------------------------------------------------------------
        Order 2 singlestep (NFE=2):

        [数式] DPM-Solver++ Eq.16:

          λ_mid = (λ_t + λ_{t-1}) / 2
          t_mid = λ^{-1}(λ_mid)   (λ の逆関数でタイムステップを求める)

          Step 1: u = Order1_step(x_t → t_mid)
          Step 2: D_θ_mid = D_θ(u, t_mid)

          補正項:
            h = λ_{t-1} − λ_t
            x_{t-1} = Order1(x_t → t_1)
                      − α_{t-1}*(e^h−1)/2 * (D_θ_mid − D_θ_t)

        Order 3 singlestep (NFE=3):

        [数式] DPM-Solver++ Eq.19:

          λ_1 = λ_t + h/3,  λ_2 = λ_t + 2h/3
          u_1 = Order1(x_t → t_1)
          u_2 = Order2_singlestep(u_1 → t_2)

          3 点の D_θ から 2 次 Taylor 展開でより精密に更新
        ---------------------------------------------------------------
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx = self._timestep_to_idx[t]
        dev = sample.device

        t_next_idx = idx + 1
        t_next = int(self.timesteps[t_next_idx].item()) if t_next_idx < len(self.timesteps) else 0

        alpha_t,   sigma_t,   lambda_t   = self._get_alpha_sigma(t, dev)
        alpha_tm1, sigma_tm1, lambda_tm1 = self._get_alpha_sigma(t_next, dev)
        d_theta_t = self._get_denoiser(model_output, sample, t)

        # 最終ステップ (h≈0) は order-1 で処理
        if abs(float(lambda_tm1 - lambda_t)) < 1e-6:
            order1_result = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )
            return SchedulerOutput(prev_sample=order1_result, pred_original_sample=d_theta_t)

        if self.order == 1:
            # Order 1 は singlestep == multistep
            x_next = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )
            return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta_t)

        elif self.order == 2:
            # ----------------------------------------------------------
            # Order 2 singlestep:
            # 中間点 λ_mid を求め 2 回目のフォワードを実行
            # ----------------------------------------------------------
            lambda_mid = (lambda_t + lambda_tm1) / 2.0

            # λ_mid に最も近いタイムステップインデックスを探す
            log_snr = self.schedule.log_snr.to(dev)
            t_mid_idx = int(torch.argmin(torch.abs(log_snr - lambda_mid)).item())

            alpha_mid, sigma_mid, _ = self._get_alpha_sigma(t_mid_idx, dev)

            # Order 1 step: x_t → u (at t_mid)
            u = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_mid, lambda_t, lambda_mid
            )

            # 2 回目の UNet フォワード
            sigma_ode_mid = self.schedule.sigmas_for_ode[t_mid_idx].to(dev)
            c_mid = (sigma_ode_mid ** 2 + 1.0).sqrt()
            u_scaled = u / c_mid

            t_mid_tensor = torch.tensor([t_mid_idx], device=dev)
            v_mid = unet_forward_fn(u_scaled, t_mid_tensor, encoder_hidden_states)
            d_theta_mid = self._get_denoiser(v_mid, u, t_mid_idx)

            # ----------------------------------------------------------
            # [数式] Order 2 singlestep 最終更新:
            #
            #   h = λ_{t-1} − λ_t
            #   x_{t-1} = order1_term
            #             − α_{t-1} * (e^h−1) / 2 * (D_θ_mid − D_θ_t)
            # ----------------------------------------------------------
            h = lambda_tm1 - lambda_t
            phi1 = torch.expm1(h)
            order1 = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )
            x_next = order1 - alpha_tm1 * phi1 / 2.0 * (d_theta_mid - d_theta_t)

            return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta_t)

        else:
            # ----------------------------------------------------------
            # Order 3 singlestep (NFE=3):
            # λ_1 = λ_t + h/3,  λ_2 = λ_t + 2h/3 の 2 中間点を使う
            # ----------------------------------------------------------
            h = float(lambda_tm1 - lambda_t)
            lambda_1 = lambda_t + h / 3.0
            lambda_2 = lambda_t + 2.0 * h / 3.0
            log_snr = self.schedule.log_snr.to(dev)

            t_1_idx = int(torch.argmin(torch.abs(log_snr - lambda_1)).item())
            t_2_idx = int(torch.argmin(torch.abs(log_snr - lambda_2)).item())

            alpha_1, _, lambda_1_actual = self._get_alpha_sigma(t_1_idx, dev)
            alpha_2, _, lambda_2_actual = self._get_alpha_sigma(t_2_idx, dev)

            # --- 中間点 u_1 ---
            u_1 = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_1, lambda_t, lambda_1_actual
            )
            # UNet フォワード (2 回目)
            sig_ode_1 = self.schedule.sigmas_for_ode[t_1_idx].to(dev)
            c1 = (sig_ode_1 ** 2 + 1.0).sqrt()
            v1 = unet_forward_fn(
                u_1 / c1,
                torch.tensor([t_1_idx], device=dev),
                encoder_hidden_states,
            )
            d_theta_1 = self._get_denoiser(v1, u_1, t_1_idx)

            # --- 中間点 u_2 (Order 2 で計算) ---
            u_2_order1 = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_2, lambda_t, lambda_2_actual
            )
            h_01 = float(lambda_1_actual - lambda_t)
            h_02 = float(lambda_2_actual - lambda_t)
            r = h_01 / h_02
            phi1_02 = torch.expm1(torch.tensor(h_02, device=dev, dtype=sample.dtype))
            u_2 = u_2_order1 - alpha_2 * phi1_02 / 2.0 * (d_theta_1 - d_theta_t)

            # UNet フォワード (3 回目)
            sig_ode_2 = self.schedule.sigmas_for_ode[t_2_idx].to(dev)
            c2 = (sig_ode_2 ** 2 + 1.0).sqrt()
            v2 = unet_forward_fn(
                u_2 / c2,
                torch.tensor([t_2_idx], device=dev),
                encoder_hidden_states,
            )
            d_theta_2 = self._get_denoiser(v2, u_2, t_2_idx)

            # ----------------------------------------------------------
            # [数式] Order 3 singlestep 最終更新 (簡略形):
            #
            #   h = λ_{t-1} − λ_t
            #   D1 = (D_θ_1 − D_θ_t) / (h/3)
            #   D2 = (D_θ_2 − D_θ_t) / (2h/3)
            #   D2nd = (D2 − D1) / (2h/3 − h/3) = (D2 − D1) / (h/3)
            #
            #   x_{t-1} = order1_term
            #             − α_{t-1}*(e^h−1)/2 * (D_θ_2 − D_θ_t)
            #             + α_{t-1}*(φ_1 - 1) / 3 * D2nd * h
            # ----------------------------------------------------------
            h_t = torch.tensor(h, device=dev, dtype=sample.dtype)
            phi1 = torch.expm1(h_t)
            phi2 = (phi1 - h_t) / h_t  # (e^h − 1 − h) / h

            order1 = self._order1_step(
                sample, d_theta_t, alpha_t, alpha_tm1, lambda_t, lambda_tm1
            )

            D1   = (d_theta_1 - d_theta_t) / (h / 3.0)
            D2   = (d_theta_2 - d_theta_t) / (2.0 * h / 3.0)
            D2nd = (D2 - D1) / (h / 3.0)

            x_next = (
                order1
                - alpha_tm1 * phi1 / 2.0 * (d_theta_2 - d_theta_t)
                + alpha_tm1 * phi2 / 3.0 * D2nd * h_t
            )

            return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta_t)
