"""
samplers/euler.py — Euler ODE サンプラー (sigma 空間)
=============================================================================

論文:
  Karras, T., Laine, S., Aittala, M., Hellsten, J., Lehtinen, J., & Aila, T.
  (2022). "Elucidating the Design Space of Diffusion-Based Generative Models"
  NeurIPS 2022. https://arxiv.org/abs/2206.00364  (EDM)

概要:
  拡散 ODE を sigma 空間に変換し、単純な 1 次 Euler 法で数値積分する。
  DDPM/DDIM が discrete timestep (t ∈ {0,...,T}) を使うのに対し、
  Euler/Heun/LMS は σ (ノイズレベル) を基準に積分する。

  EDM の確率フロー ODE:
    dx/dσ = x/σ − D_θ(x, σ) / σ    (Eq.5 in Karras et al. 2022)

  ここで D_θ(x, σ) は denoiser 出力 = x̂_0 の推定値。

特性:
  タイプ : ODE (決定論的)
  NFE/step: 1
  推奨ステップ数: 20–50

sigma 空間への変換:
  SD の discrete timestep t と σ の対応:
    σ_t = √(1−ᾱ_t) / √ᾱ_t    (= sigmas_for_ode in NoiseSchedule)

  UNet への入力スケーリング (c_in, EDM Eq.7):
    x_scaled = x / √(σ² + 1)

  UNet 出力から x̂_0 への変換 (v-prediction):
    x̂_0 = α_t * x_t − σ_t * v_θ  (base.predict_x0 参照)

Euler 法の更新式 (EDM Eq.4):
  d_i = (x_i − D_θ(x_i, σ_i)) / σ_i    ... スコア方向
  x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_i

  d_i は「現在位置 x_i から denoiser 予測 D_θ への方向を σ_i で正規化」
  したベクトルで、ODE の傾きに相当する。
"""

from __future__ import annotations

from typing import List, Optional, Union

import torch

from .base import BaseSampler, SchedulerOutput, predict_x0


class EulerSampler(BaseSampler):
    """
    Euler ODE サンプラー。

    sigma 空間で 1 次 Euler 法を適用する。
    シンプルで安定した ODE 積分の基準実装。

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
        # sigma 列 (降順) と対応する timestep を推論時に設定
        self._sigmas: Optional[torch.Tensor] = None  # shape [N+1], σ_N,...,σ_0, 0
        self._timestep_to_idx: dict = {}

    # ----------------------------------------------------------------
    # set_timesteps のオーバーライド (sigma 空間用)
    # ----------------------------------------------------------------

    def set_timesteps(
        self,
        num_inference_steps: int,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        """
        推論用 sigma 列を設定する。

        ---------------------------------------------------------------
        [数式] sigma のサブサンプリング:

        訓練時の全 sigma 列から N 点を等間隔でサブサンプリングし、
        降順 (大→小) に並べる。最後に σ=0 を付加。

        σ_{i} = sigmas_for_ode[t_i]
        t_i = round( T-1 - i*(T-1)/(N-1) )   for i=0,...,N-1

        推論ループでは [σ_0, σ_1, ..., σ_{N-1}, 0] を順に使う。
        ---------------------------------------------------------------
        """
        self.num_inference_steps = num_inference_steps

        # 等間隔でタイムステップをサンプリング (両端含む, 降順)
        timesteps = torch.linspace(
            self.num_train_timesteps - 1, 0, num_inference_steps
        ).round().long()
        self.timesteps = timesteps.to(device)

        # sigma 列を取得 (降順, 末尾に 0 を付加)
        sigmas = self.schedule.sigmas_for_ode[timesteps].to(device)
        self._sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        # timestep → sigma インデックスの対応マップ
        self._timestep_to_idx = {int(t): i for i, t in enumerate(timesteps)}

    # ----------------------------------------------------------------
    # scale_model_input のオーバーライド
    # ----------------------------------------------------------------

    def scale_model_input(
        self, sample: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        UNet への入力を sigma に応じてスケーリングする。

        ---------------------------------------------------------------
        [数式] EDM の c_in スケーリング (Karras et al. 2022, Eq.7):

          x_scaled = x / √(σ² + 1)

        これにより UNet への入力の分散を約 1 に正規化する。
        DDIM/DDPM は x_t の分散がすでに 1 に近いため不要だが、
        sigma 空間では σ が大きい (ノイズが多い) ステップで
        |x| が大きくなるため正規化が必要。
        ---------------------------------------------------------------
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        if t in self._timestep_to_idx:
            idx = self._timestep_to_idx[t]
            sigma = self._sigmas[idx]
            return sample / (sigma ** 2 + 1.0).sqrt()
        return sample  # フォールバック

    # ----------------------------------------------------------------
    # sigma 空間共通ヘルパー (Heun / LMS-2 / DPM-Solver が継承)
    # ----------------------------------------------------------------

    def _sigma_to_x0(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        sigma_ode: torch.Tensor,
    ) -> torch.Tensor:
        """
        sigma 空間での model_output → x̂_0 変換。
        prediction_type に応じて v-prediction または epsilon を処理する。

        ---------------------------------------------------------------
        [v-prediction] (SD 2.x):
          c = √(σ²+1)
          x̂_0 = predict_x0(v_θ, x_t/c, 1/c, σ/c)

        [epsilon-prediction] (SD 1.x):
          x̂_0 = x_t * c − σ * ε_θ   where c = √(σ²+1)

          導出: x_t = α_t*x0 + σ_t*ε, α_t=1/c, σ_t=σ/c より
            x0 = (x_t − (σ/c)*ε) * c = x_t*c − σ*ε
        ---------------------------------------------------------------
        """
        c = (sigma_ode ** 2 + 1.0).sqrt()
        if self.prediction_type == "v_prediction":
            alpha_t = (1.0 / c).view(1, 1, 1, 1)
            sigma_t = (sigma_ode / c).view(1, 1, 1, 1)
            return predict_x0(model_output, x_t / c, alpha_t, sigma_t)
        else:  # epsilon (SD 1.x)
            # x̂_0 = x_t − σ_ODE * ε_θ
            # (x_t は sigma 空間の未スケール潜在変数、c 倍は不要)
            return x_t - sigma_ode * model_output

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
        Euler ODE の 1 ステップを実行する。

        ---------------------------------------------------------------
        [アルゴリズム] Euler Method (EDM Algorithm 1 の inner loop):

        入力: x_i, σ_i, v_θ(x_i/c_in, t_i)
        出力: x_{i+1}

        Step 1 — UNet 出力から D_θ (= x̂_0) を復元:
          x_t = scale_model_input で既にスケーリング済みの入力
          x̂_0 = α_t * x_t − σ_t * v_θ   (v-prediction 変換)

          注意: UNet には x_i/√(σ²+1) が入力されているため、
          v_θ は元のスケール x_i に対する推定値に変換が必要。

        Step 2 — ODE の傾き d を計算 (EDM Eq.4):
          d_i = (x_i − D_θ(x_i, σ_i)) / σ_i

          直感: x_i がノイズ方向から x̂_0 方向への「信号成分比」を
          σ_i で正規化したもの。

        Step 3 — Euler 更新:
          x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_i

          σ_{i+1} < σ_i なので (σ_{i+1} − σ_i) < 0 → ノイズが減少する方向

        変数対応:
          sample        ← x_i   (スケーリング前の潜在変数)
          model_output  ← v_θ   (UNet 出力)
          sigma_i       ← σ_i   (現在の sigma)
          sigma_next    ← σ_{i+1} (次の sigma, 末端では 0)
        ---------------------------------------------------------------
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        idx = self._timestep_to_idx[t]

        dev = sample.device
        sigma_i    = self._sigmas[idx].to(dev)
        sigma_next = self._sigmas[idx + 1].to(dev)

        # ----------------------------------------------------------
        # Step 1: D_θ (= x̂_0) を復元
        #
        # UNet には x_scaled = x_i / √(σ²+1) が入力されている。
        # v_θ は x_scaled 基準なので、x_i 基準に戻す。
        #
        # [数式] sigma 空間での x̂_0:
        #   ᾱ_t = 1 / √(σ_ODE² + 1)   (定義から逆算)
        #   σ_t^orig = σ_ODE / √(σ_ODE² + 1)
        #
        # denoiser D_θ は x̂_0 に相当するため:
        #   D_θ(x_i, σ_i) = (x_i − σ_i * ε̂) / α_t
        #
        # v-prediction の場合は predict_x0 を直接適用できる。
        # alpha と sigma を sigma_ode から再計算:
        #   alpha_t = 1 / √(σ_i² + 1)
        #   sigma_t_orig = σ_i / √(σ_i² + 1)
        # ----------------------------------------------------------
        x0_hat = self._sigma_to_x0(model_output, sample, sigma_i)
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        # ----------------------------------------------------------
        # Step 2: ODE の傾き d を計算 (EDM Eq.4)
        #
        # [数式]
        #   D_θ(x_i, σ_i) = x̂_0
        #   d_i = (x_i − D_θ) / σ_i
        # ----------------------------------------------------------
        d = (sample - x0_hat) / sigma_i

        # ----------------------------------------------------------
        # Step 3: Euler 更新
        #
        # [数式]
        #   x_{i+1} = x_i + (σ_{i+1} − σ_i) * d_i
        # ----------------------------------------------------------
        dt = sigma_next - sigma_i
        x_next = sample + dt * d

        return SchedulerOutput(prev_sample=x_next, pred_original_sample=x0_hat)
