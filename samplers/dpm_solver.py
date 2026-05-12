"""
samplers/dpm_solver.py — DPM-Solver++ (4 クラス構成)
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
  DPM-Solver は拡散 ODE を log-SNR 空間 (λ 空間) で解析的に解き、
  Taylor 展開により高次精度を達成するサンプラー。
  本ファイルは役割を明確にするため 4 つのクラスに分割している。

  クラス構成:
    _DPMSolverBase       — 共通基底 (EulerSampler 継承 + ヘルパー)
    DPMSolver1           — Order 1 singlestep (DDIM η=0 と等価)
    DPMSolver2Singlestep — Order 2 singlestep (NFE=2、中間点で 2 回目フォワード)
    DPMSolverMultistep1  — Order 1 multistep  (NFE=1、DPMSolver1 と数式同一)
    DPMSolverMultistep2  — Order 2 multistep  (NFE=1、前ステップ D_θ を再利用)

  log-SNR と VP-SDE パラメータの関係:
    λ_t = log(ᾱ_t / (1−ᾱ_t)) = log(α_t² / σ_t²)
    α_t = √ᾱ_t  (signal scale)
    σ_t = √(1−ᾱ_t)  (noise scale)
    α_t² + σ_t² = 1   (VP-SDE の分散保存条件)

=============================================================================
DPM-Solver++ の更新式 (data-prediction form, Eq.14/16/17):
=============================================================================

  共通定義:
    h = λ_{t-1} − λ_t  (> 0、ノイズ減少方向)
    φ_1 = e^h − 1

  [Order 1]:
    x_{t-1} = (α_{t-1}/α_t) * x_t − α_{t-1} * φ_1 * D_θ(x_t, t)

  [Order 2 singlestep] (中間点 t_mid を使用):
    λ_mid = (λ_t + λ_{t-1}) / 2
    u = Order1(x_t → t_mid)              ... 中間点への Order 1 ステップ
    x_{t-1} = Order1(x_t → t_{t-1})
              − α_{t-1} * φ_1 / 2 * (D_θ(u, t_mid) − D_θ(x_t, t))

  [Order 2 multistep] (前ステップの D_θ を再利用):
    r_0 = h_prev / h                     ... 前ステップ幅の相対比
          h_prev = λ_t − λ_{prev}
    ΔD  = D_θ(x_t, t) − D_θ(x_{prev}, t_{prev})
    x_{t-1} = Order1(x_t → t_{t-1})
              + α_{t-1} * φ_1 / (2 * r_0) * ΔD
"""

from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Union

import torch

from .base import BaseSampler, SchedulerOutput
from .euler import EulerSampler


# =============================================================================
# 共通基底クラス
# =============================================================================

class _DPMSolverBase(EulerSampler):
    """
    DPM-Solver 各クラスが継承する共通基底。

    実装上は EulerSampler を継承しているが、DPM-Solver++ の更新式は
    VP 空間の x_t = α_t x_0 + σ_t ε を前提にする。
    そのため set_timesteps() / scale_model_input() は BaseSampler と同じ
    VP 用の挙動に戻し、Euler の EDM sigma スケーリングは使わない。

    本クラスが追加するヘルパー:
      - _get_lambda_alpha()  : t_idx → (λ_t, α_t)
      - _get_denoiser()      : model_output + x → D_θ (= x̂_0)
      - _order1_step()       : Order 1 の 1 ステップ更新
      - _resolve_t_and_next(): timestep テンソル → (t, idx, t_next) インデックス
    """

    # ----------------------------------------------------------------
    # スケジュール値取得
    # ----------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        VP 空間用のタイムステップ列を設定する。

        EulerSampler の sigma 列を作ると pipeline 側で初期 latent が
        sigma_max 倍されるため、DPM-Solver++ では BaseSampler と同じ
        N(0, I) latent を使う。
        """
        BaseSampler.set_timesteps(self, num_inference_steps, device)
        self._sigmas = None
        self._timestep_to_idx = {int(t): i for i, t in enumerate(self.timesteps)}

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """VP 空間では DDIM/DDPM と同じく UNet 入力をそのまま渡す。"""
        return BaseSampler.scale_model_input(self, sample, timestep)

    def _get_lambda_alpha(self, t_idx: int, device: torch.device):
        """
        タイムステップインデックス t_idx から (λ_t, α_t) を返す。

        λ_t = log(α_t / σ_t) = 0.5 * log(ᾱ_t / (1−ᾱ_t))
        α_t = sqrt(ᾱ_t)             (signal scale)

        戻り値: (lambda_t, alpha_t)  いずれもスカラーテンソル
        """
        lambda_t = 0.5 * self.schedule.log_snr[t_idx].to(device)
        alpha_t  = self.schedule.sqrt_alphas_cumprod[t_idx].to(device)
        return lambda_t, alpha_t

    # ----------------------------------------------------------------
    # D_θ (denoiser = x̂_0) の計算
    # ----------------------------------------------------------------

    def _get_denoiser(
        self,
        model_output: torch.Tensor,
        x: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """
        UNet 出力 model_output と潜在変数 x から D_θ (= x̂_0) を計算する。

        ---------------------------------------------------------------
        x は VP 空間の潜在変数 (= pipeline の latents)。

        [v_prediction] (SD 2.x):
          x̂_0 = α_t x_t − σ_t v_θ

        [epsilon] (SD 1.x):
          x̂_0 = (x_t − σ_t ε_θ) / α_t
        ---------------------------------------------------------------

        Args:
          model_output : UNet 出力テンソル (v_θ または ε_θ)
          x            : VP 空間の潜在変数
          t_idx        : タイムステップインデックス (整数)
        Returns:
          x̂_0
        """
        x0_hat, _ = self._predict_x0_eps(model_output, x, t_idx)
        return x0_hat

    # ----------------------------------------------------------------
    # Order 1 の 1 ステップ更新
    # ----------------------------------------------------------------

    def _order1_step(
        self,
        x_t: torch.Tensor,
        d_theta_t: torch.Tensor,
        lambda_t: torch.Tensor,
        lambda_tm1: torch.Tensor,
        alpha_t: torch.Tensor,
        alpha_tm1: torch.Tensor,
    ) -> torch.Tensor:
        """
        DPM-Solver++ Order 1 の 1 ステップを実行する。

        ---------------------------------------------------------------
        [数式] DPM-Solver++ Eq.14:

          h   = λ_{t-1} − λ_t          (> 0)

          x_{t-1} = (α_{t-1} / α_t) * e^{-h} * x_t
                    + α_{t-1} * (1 − e^{-h}) * D_θ(x_t, t)

        DDIM (η=0) と数学的に等価 (Lu et al. 2022, Appendix B.1)。

        Args:
          x_t       : 現在の潜在変数 x_i
          d_theta_t : D_θ(x_t, t) = x̂_0
          lambda_t  : λ_t  (現在の log-SNR)
          lambda_tm1: λ_{t-1} (次ステップの log-SNR、λ_t より大きい)
          alpha_t   : α_t = sqrt(ᾱ_t)
          alpha_tm1 : α_{t-1} = sqrt(ᾱ_{t-1})
        Returns:
          x_{t-1}
        ---------------------------------------------------------------
        """
        h = lambda_tm1 - lambda_t          # > 0
        exp_neg_h = torch.exp(-h)

        return (
            (alpha_tm1 / alpha_t) * exp_neg_h * x_t
            + alpha_tm1 * (1.0 - exp_neg_h) * d_theta_t
        )

    # ----------------------------------------------------------------
    # タイムステップインデックスの解決
    # ----------------------------------------------------------------

    def _resolve_t_and_next(self, timestep: torch.Tensor):
        """
        timestep テンソルから現在・次のタイムステップ値と idx を返す。

        Returns:
          (t, idx, t_next)
            t      : 現在のタイムステップ値 (整数)
            idx    : self.timesteps 上のインデックス
            t_next : 次のタイムステップ値 (最後のステップは 0)
        """
        t     = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx   = self._timestep_to_idx[t]
        t_next = int(self.timesteps[idx + 1].item()) if idx + 1 < len(self.timesteps) else 0
        return t, idx, t_next


# =============================================================================
# DPM-Solver-1 (Order 1 singlestep)
# =============================================================================

class DPMSolver1(_DPMSolverBase):
    """
    DPM-Solver-1: Order 1 singlestep サンプラー。

    ---------------------------------------------------------------
    特性:
      タイプ   : ODE (決定論的)
      NFE/step : 1
      等価性   : DDIM (η=0) と数学的に等価
      推奨ステップ数: 10–20 (品質は低め)

    更新式 (DPM-Solver++ Eq.14):
      h   = λ_{t-1} − λ_t
      x_{t-1} = (α_{t-1}/α_t) * x_t − α_{t-1} * (e^h−1) * D_θ(x_t, t)

    pipeline からは step() を呼ぶ (step_singlestep は不要)。
    ---------------------------------------------------------------
    """

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        DPM-Solver-1 の 1 ステップを実行する。

        Args:
          model_output : UNet 出力 (v_θ または ε_θ)
          timestep     : 現在のタイムステップテンソル
          sample       : EDM 空間の潜在変数 x_t (未スケール)
        Returns:
          SchedulerOutput (prev_sample=x_{t-1}, pred_original_sample=x̂_0)
        """
        dev = sample.device
        t, idx, t_next = self._resolve_t_and_next(timestep)

        # スケジュール値を取得
        lambda_t,   alpha_t   = self._get_lambda_alpha(t,      dev)
        lambda_tm1, alpha_tm1 = self._get_lambda_alpha(t_next, dev)

        # D_θ = x̂_0 を計算
        d_theta = self._get_denoiser(model_output, sample, t)

        # Order 1 更新式を適用
        x_next = self._order1_step(
            sample, d_theta,
            lambda_t, lambda_tm1,
            alpha_t, alpha_tm1,
        )

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta)


# =============================================================================
# DPM-Solver-2 singlestep (Order 2 singlestep)
# =============================================================================

class DPMSolver2Singlestep(_DPMSolverBase):
    """
    DPM-Solver-2 singlestep: Order 2 singlestep サンプラー。

    ---------------------------------------------------------------
    特性:
      タイプ   : ODE (決定論的)
      NFE/step : 2  (中間点で 2 回目の UNet フォワードを実行)
      推奨ステップ数: 10–20

    アルゴリズム概要 (DPM-Solver++ Eq.16):
      Step 1: 中間点 t_mid を λ_mid = (λ_t + λ_{t-1})/2 で定義し、
              Order 1 で x_t → u (at t_mid) を計算
      Step 2: u を UNet に通して D_θ(u, t_mid) を取得
      Step 3: 2 点の D_θ から 1 次 Taylor 補正を適用して x_{t-1} を計算

    pipeline からは step_singlestep() を呼ぶこと。
    (pipeline.py は isinstance(sampler, DPMSolver2Singlestep) で分岐)
    ---------------------------------------------------------------
    """

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        フォールバック用 Order 1 ステップ。

        pipeline から step_singlestep() が呼ばれるため、
        このメソッドが推論ループで呼ばれることは通常ない。
        万が一直接呼ばれた場合は Order 1 として動作する。
        """
        dev = sample.device
        t, idx, t_next = self._resolve_t_and_next(timestep)
        lambda_t,   alpha_t   = self._get_lambda_alpha(t,      dev)
        lambda_tm1, alpha_tm1 = self._get_lambda_alpha(t_next, dev)
        d_theta = self._get_denoiser(model_output, sample, t)
        x_next = self._order1_step(sample, d_theta, lambda_t, lambda_tm1, alpha_t, alpha_tm1)
        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta)

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
        DPM-Solver-2 singlestep の完全ステップ (NFE=2)。

        pipeline.py から呼ばれる。1 ステップで 2 回 UNet を呼ぶ。

        ---------------------------------------------------------------
        [アルゴリズム] DPM-Solver++ Eq.16:

        入力: x_t, 1 回目の UNet 出力 model_output (pipeline 側で計算済み)
        出力: x_{t-1}

        Step 1 — 中間点 t_mid を決定:
          λ_mid = (λ_t + λ_{t-1}) / 2
          t_mid = argmin_s |λ_s − λ_mid|   (離散スケジュールで最近傍)

        Step 2 — x_t から t_mid への Order 1 ステップ:
          u = (α_mid/α_t)*x_t − α_mid*(e^{h_mid}−1)*D_θ(x_t, t)
          ここで h_mid = λ_mid − λ_t

        Step 3 — u を UNet に通して 2 点目の D_θ を取得:
          u_scaled = u / sqrt(σ_ODE_mid² + 1)   (EDM→VP スケーリング)
          v_mid = UNet(u_scaled, t_mid, enc_hs)
          D_θ_mid = _get_denoiser(v_mid, u, t_mid)

        Step 4 — Order 2 補正を適用して x_{t-1} を計算:
          h   = λ_{t-1} − λ_t
          φ_1 = e^h − 1
          ΔD  = D_θ(u, t_mid) − D_θ(x_t, t)   (1 次差分)

          x_{t-1} = Order1(x_t → t_{t-1})
                    − α_{t-1} * φ_1 / 2 * ΔD

          直感: Order 1 の誤差は D_θ の変化量 ΔD に比例するので、
          中間点での 2 回目評価で ΔD を推定し補正する。

        Args:
          model_output          : 1 回目の UNet 出力 (pipeline が計算済み)
          timestep              : 現在のタイムステップ
          sample                : EDM 空間の潜在変数 x_t
          unet_forward_fn       : UNet フォワード fn(x_scaled, t, enc_hs) → v
          encoder_hidden_states : テキスト埋め込み
          guidance_scale        : CFG スケール (unet_forward_fn が内包)
        ---------------------------------------------------------------
        """
        dev = sample.device
        t, idx, t_next = self._resolve_t_and_next(timestep)

        lambda_t,   alpha_t   = self._get_lambda_alpha(t,      dev)
        lambda_tm1, alpha_tm1 = self._get_lambda_alpha(t_next, dev)

        # 1 回目の D_θ = x̂_0
        d_theta_t = self._get_denoiser(model_output, sample, t)

        # ----------------------------------------------------------
        # 最終ステップ (h ≈ 0) は Order 1 にフォールバック
        # λ_{t-1} ≈ λ_t となるのは t が最後のタイムステップの時
        # ----------------------------------------------------------
        h = float(lambda_tm1 - lambda_t)
        if abs(h) < 1e-6:
            x_next = self._order1_step(
                sample, d_theta_t, lambda_t, lambda_tm1, alpha_t, alpha_tm1
            )
            return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta_t)

        # ----------------------------------------------------------
        # Step 1: 中間点 t_mid を λ 空間の中点で決定
        # λ_mid = (λ_t + λ_{t-1}) / 2
        # 離散スケジュールなので log_snr テーブルから最近傍インデックスを探す
        # ----------------------------------------------------------
        lambda_mid = (lambda_t + lambda_tm1) / 2.0
        log_snr    = 0.5 * self.schedule.log_snr.to(dev)
        t_mid_idx  = int(torch.argmin(torch.abs(log_snr - lambda_mid)).item())

        lambda_mid_actual, alpha_mid = self._get_lambda_alpha(t_mid_idx, dev)

        # ----------------------------------------------------------
        # Step 2: x_t → u  (t → t_mid の Order 1 ステップ)
        # h_mid = λ_mid − λ_t  (全ステップ幅 h の約半分)
        # ----------------------------------------------------------
        u = self._order1_step(
            sample, d_theta_t,
            lambda_t, lambda_mid_actual,
            alpha_t, alpha_mid,
        )

        # ----------------------------------------------------------
        # Step 3: u を UNet に通して D_θ(u, t_mid) を取得
        # u は EDM 空間なので、UNet 入力用に /c でスケールする
        # c = sqrt(σ_ODE_mid² + 1)
        # ----------------------------------------------------------
        sigma_ode_mid = self.schedule.sigmas_for_ode[t_mid_idx].to(dev)
        c_mid         = (sigma_ode_mid ** 2 + 1.0).sqrt()
        u_scaled      = u / c_mid

        v_mid       = unet_forward_fn(
            u_scaled,
            torch.tensor([t_mid_idx], device=dev),
            encoder_hidden_states,
        )
        d_theta_mid = self._get_denoiser(v_mid, u, t_mid_idx)

        # ----------------------------------------------------------
        # Step 4: Order 2 補正を適用
        #
        # x_{t-1} = Order1(x_t → t_{t-1})
        #           − α_{t-1} * φ_1 / 2 * (D_θ_mid − D_θ_t)
        #
        # 第 2 項は D_θ の変化率 ΔD を使った 1 次 Taylor 補正。
        # ΔD > 0 ならノイズ推定が大きくなっているので x_{t-1} を補正。
        # ----------------------------------------------------------
        phi1    = torch.expm1(torch.tensor(h, device=dev, dtype=sample.dtype))
        order1  = self._order1_step(
            sample, d_theta_t, lambda_t, lambda_tm1, alpha_t, alpha_tm1
        )
        delta_d = d_theta_mid - d_theta_t                      # ΔD (1 次差分)
        x_next  = order1 - alpha_tm1 * phi1 / 2.0 * delta_d

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta_t)


# =============================================================================
# DPM-Solver multistep Order 1
# =============================================================================

class DPMSolverMultistep1(_DPMSolverBase):
    """
    DPM-Solver multistep Order 1 サンプラー。

    ---------------------------------------------------------------
    特性:
      タイプ   : ODE (決定論的)
      NFE/step : 1
      等価性   : DPMSolver1 および DDIM (η=0) と数学的に等価
                 (multistep Order 1 は前ステップを再利用しないので
                  singlestep Order 1 と同一の更新式になる)

    更新式 (DPM-Solver++ Eq.14):
      h   = λ_{t-1} − λ_t
      x_{t-1} = (α_{t-1}/α_t) * x_t − α_{t-1} * (e^h−1) * D_θ(x_t, t)

    DPMSolver1 と同じ計算だが、コードの対称性のため独立クラスとして定義。
    ---------------------------------------------------------------
    """

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        DPM-Solver multistep Order 1 の 1 ステップを実行する。

        Args:
          model_output : UNet 出力
          timestep     : 現在のタイムステップ
          sample       : EDM 空間の潜在変数 x_t
        Returns:
          SchedulerOutput
        """
        dev = sample.device
        t, idx, t_next = self._resolve_t_and_next(timestep)

        lambda_t,   alpha_t   = self._get_lambda_alpha(t,      dev)
        lambda_tm1, alpha_tm1 = self._get_lambda_alpha(t_next, dev)

        d_theta = self._get_denoiser(model_output, sample, t)

        x_next = self._order1_step(
            sample, d_theta,
            lambda_t, lambda_tm1,
            alpha_t, alpha_tm1,
        )

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta)


# =============================================================================
# DPM-Solver multistep Order 2
# =============================================================================

class DPMSolverMultistep2(_DPMSolverBase):
    """
    DPM-Solver multistep Order 2 サンプラー。

    ---------------------------------------------------------------
    特性:
      タイプ   : ODE (決定論的)
      NFE/step : 1  (前ステップの D_θ を再利用して 2 次精度を達成)
      推奨ステップ数: 10–20

    アルゴリズム概要 (DPM-Solver++ Eq.17, Adams-Bashforth 風):
      - 各ステップで D_θ(x_t, t) を計算し deque に保存
      - 最初のステップ (バッファ空) は Order 1 にフォールバック
      - 2 ステップ目以降は前ステップの D_θ を再利用して補正を適用

    更新式:
      h    = λ_{t-1} − λ_t
      h_prev = λ_t − λ_{prev}         (前ステップの λ 幅)
      r_0  = h_prev / h                (前ステップ幅の相対比)
      ΔD   = D_θ(x_t, t) − D_θ(x_{prev}, t_{prev})   (1 次差分)

      x_{t-1} = Order1(x_t → t_{t-1})
                + α_{t-1} * (e^h−1) / (2 * r_0) * ΔD

    バッファ管理:
      _d_theta_prev : 前ステップの D_θ (deque maxlen=1)
      _lambda_prev  : 前ステップの λ_t (deque maxlen=1)
    ---------------------------------------------------------------
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
    ) -> None:
        super().__init__(num_train_timesteps, beta_start, beta_end)
        # 前ステップの D_θ と λ を保持するバッファ (maxlen=1)
        self._d_theta_prev: Deque[torch.Tensor] = deque(maxlen=1)
        self._lambda_prev:  Deque[float]         = deque(maxlen=1)

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        タイムステップの設定とバッファのリセット。

        set_timesteps() はエピソード (generate() 呼び出し) の開始時に
        呼ばれるため、バッファをクリアして前エピソードの状態を破棄する。
        """
        super().set_timesteps(num_inference_steps, device)
        self._d_theta_prev.clear()
        self._lambda_prev.clear()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        DPM-Solver multistep Order 2 の 1 ステップを実行する。

        バッファが空の場合 (最初のステップ) は Order 1 にフォールバックし、
        次のステップ以降で Order 2 補正が有効になる。

        ---------------------------------------------------------------
        [アルゴリズム]:

        Case A: バッファ空 (最初のステップ)
          → Order 1 を適用し、D_θ と λ_t をバッファに保存

        Case B: バッファあり (2 ステップ目以降)
          h      = λ_{t-1} − λ_t
          h_prev = λ_t − λ_{prev}
          r_0    = h_prev / h

          ΔD = D_θ_now − D_θ_prev   (前後 2 ステップの D_θ の差分)

          補正項 = α_{t-1} * φ_1 / (2*r_0) * ΔD
            ここで φ_1 = e^h − 1

          x_{t-1} = Order1(x_t → t_{t-1}) + 補正項

          直感: ΔD は D_θ の「変化速度」の推定。これを使って
          Order 1 で無視していた高次項を補正する。
          r_0 は前後のステップ幅が不均一な場合の正規化係数。

        最後にバッファを現在の D_θ と λ_t で更新する。
        ---------------------------------------------------------------

        Args:
          model_output : UNet 出力
          timestep     : 現在のタイムステップ
          sample       : EDM 空間の潜在変数 x_t
        Returns:
          SchedulerOutput
        """
        dev = sample.device
        t, idx, t_next = self._resolve_t_and_next(timestep)

        lambda_t,   alpha_t   = self._get_lambda_alpha(t,      dev)
        lambda_tm1, alpha_tm1 = self._get_lambda_alpha(t_next, dev)

        # 現ステップの D_θ = x̂_0
        d_theta = self._get_denoiser(model_output, sample, t)

        if len(self._d_theta_prev) == 0:
            # --------------------------------------------------------
            # Case A: 最初のステップ → Order 1 フォールバック
            # バッファが空なので前ステップ情報がなく補正不可
            # --------------------------------------------------------
            x_next = self._order1_step(
                sample, d_theta,
                lambda_t, lambda_tm1,
                alpha_t, alpha_tm1,
            )
        else:
            # --------------------------------------------------------
            # Case B: 2 ステップ目以降 → Order 2 補正を適用
            # --------------------------------------------------------
            d_theta_prev = self._d_theta_prev[-1]
            lambda_prev  = self._lambda_prev[-1]

            h      = lambda_tm1 - lambda_t
            h_float = float(h)

            if abs(h_float) < 1e-6:
                # 最終ステップ (h≈0): Order 1 にフォールバック
                x_next = self._order1_step(
                    sample, d_theta,
                    lambda_t, lambda_tm1,
                    alpha_t, alpha_tm1,
                )
            else:
                h_prev = float(lambda_t) - lambda_prev   # > 0
                r_0    = h_prev / h_float                # 前ステップ幅の相対比

                # Order 1 ベース項
                order1 = self._order1_step(
                    sample, d_theta,
                    lambda_t, lambda_tm1,
                    alpha_t, alpha_tm1,
                )

                # 1 次差分 ΔD = D_θ_now − D_θ_prev
                delta_d = d_theta - d_theta_prev

                # 補正項: α_{t-1} * (e^h−1) / (2*r_0) * ΔD
                phi1       = torch.expm1(h)
                correction = alpha_tm1 * phi1 / (2.0 * r_0) * delta_d

                x_next = order1 + correction

        # バッファを現在のステップで更新
        self._d_theta_prev.append(d_theta)
        self._lambda_prev.append(float(lambda_t))

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=d_theta)
