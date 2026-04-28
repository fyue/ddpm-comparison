"""
samplers/ddpm.py — DDPM (Denoising Diffusion Probabilistic Models) サンプラー
=============================================================================

論文:
  Ho, J., Jain, A., & Abbeel, P. (2020).
  "Denoising Diffusion Probabilistic Models"
  NeurIPS 2020. https://arxiv.org/abs/2006.11239

概要:
  DDPM は拡散過程の逆過程を学習した確率的 (SDE ベース) サンプラー。
  各ステップでガウスノイズを注入するため、同じ初期ノイズでも
  毎回異なる出力を生成する (stochastic)。

  論文では T=1000 ステップで学習・推論するが、サブサンプリングにより
  任意のステップ数で推論可能 (品質は低下する)。

特性:
  タイプ : SDE (確率的微分方程式)
  NFE/step: 1 (UNet フォワード 1 回/ステップ)
  推奨ステップ数: 1000 (少ないと品質低下が顕著)

逆過程の遷移カーネル (DDPM Eq.6/7):
  q(x_{t-1} | x_t, x_0) = N(x_{t-1}; μ̃_t(x_t, x_0), β̃_t * I)

  後験平均 μ̃_t:
    μ̃_t(x_t, x_0) = (√ᾱ_{t-1} * β_t) / (1−ᾱ_t) * x_0
                   + (√α_t * (1−ᾱ_{t-1})) / (1−ᾱ_t) * x_t

  後験分散 β̃_t (fixed_small):
    β̃_t = (1−ᾱ_{t-1}) / (1−ᾱ_t) * β_t
"""

from __future__ import annotations

from typing import Union

import torch

from .base import BaseSampler, SchedulerOutput, predict_x0, predict_eps


class DDPMSampler(BaseSampler):
    """
    DDPM 確率的サンプラー。

    Args:
      num_train_timesteps : 訓練時のタイムステップ数 T (デフォルト 1000)
      beta_start          : β スケジュールの開始値
      beta_end            : β スケジュールの終了値
      variance_type       : 後験分散の種類
                            "fixed_small"  → β̃_t  (推奨)
                            "fixed_large"  → β_t
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        variance_type: str = "fixed_small",
    ) -> None:
        super().__init__(num_train_timesteps, beta_start, beta_end)
        assert variance_type in ("fixed_small", "fixed_large"), (
            f"variance_type は 'fixed_small' または 'fixed_large' を指定してください: {variance_type}"
        )
        self.variance_type = variance_type

        # ----------------------------------------------------------
        # [数式] 後験分散 β̃_t の事前計算 (DDPM Eq.7):
        #
        #   β̃_t = (1−ᾱ_{t-1}) / (1−ᾱ_t) * β_t
        #
        # t=0 では ᾱ_{-1} = 1 (定義) とするため β̃_0 = 0
        # ----------------------------------------------------------
        alphas_cumprod = self.schedule.alphas_cumprod
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0]), alphas_cumprod[:-1]]
        )  # ᾱ_{t-1}、t=0 のときは 1
        self._posterior_variance: torch.Tensor = (
            self.schedule.betas
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - alphas_cumprod)
        )
        # 数値安定化: log は 0 を避けるため clamp
        self._log_posterior_variance: torch.Tensor = torch.log(
            self._posterior_variance.clamp(min=1e-20)
        )

    # ----------------------------------------------------------------
    # BaseSampler のインターフェース実装
    # ----------------------------------------------------------------

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
    ) -> SchedulerOutput:
        """
        DDPM 逆拡散の 1 ステップを実行する。

        ---------------------------------------------------------------
        [アルゴリズム] DDPM Algorithm 2 (Ho et al. 2020):

        入力: x_t, t, v_θ(x_t, t)
        出力: x_{t-1}

        Step 1 — x̂_0 の推定 (v-prediction 変換):
          x̂_0 = α_t * x_t − σ_t * v_θ      (base.predict_x0 参照)

        Step 2 — 後験平均 μ̃_t の計算 (DDPM Eq.7):
          μ̃_t = coef_x0 * x̂_0 + coef_xt * x_t

          ここで:
            coef_x0 = √ᾱ_{t-1} * β_t / (1−ᾱ_t)
            coef_xt = √α_t * (1−ᾱ_{t-1}) / (1−ᾱ_t)

        Step 3 — ノイズ付加 (t > 0 のとき):
          x_{t-1} = μ̃_t + √β̃_t * z,  z ~ N(0, I)

          t = 0 のときはノイズなし: x_{t-1} = μ̃_t

        変数対応:
          sample       ← x_t
          model_output ← v_θ(x_t, t)
          alpha_bar_t  ← ᾱ_t  = alphas_cumprod[t]
          alpha_bar_tm1← ᾱ_{t-1} = alphas_cumprod[t-1]
          alpha_t      ← √ᾱ_t
          sigma_t      ← √(1−ᾱ_t)
          beta_t       ← β_t = betas[t]
        ---------------------------------------------------------------

        引数:
          model_output : v_θ(x_t, t)  shape [B, C, H, W]
          timestep     : t  スカラー整数テンソル
          sample       : x_t           shape [B, C, H, W]
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())
        t_prev = t - (self.num_train_timesteps // self.num_inference_steps)
        t_prev = max(t_prev, 0)

        dev = sample.device

        # ----------------------------------------------------------
        # スケジュール値の取得
        # ᾱ_t と ᾱ_{t-1} を取得する
        # ----------------------------------------------------------
        alpha_bar_t   = self.schedule.alphas_cumprod[t].to(dev)
        alpha_bar_tm1 = (
            self.schedule.alphas_cumprod[t_prev].to(dev)
            if t_prev >= 0
            else torch.tensor(1.0, device=dev)
        )
        beta_t = self.schedule.betas[t].to(dev)

        alpha_t  = alpha_bar_t.sqrt()           # √ᾱ_t
        sigma_t  = (1.0 - alpha_bar_t).sqrt()   # √(1−ᾱ_t)

        # ----------------------------------------------------------
        # Step 1: x̂_0 を推定 (v-prediction 変換)
        #
        # [数式] x̂_0 = α_t * x_t − σ_t * v_θ
        # ----------------------------------------------------------
        x0_hat, _ = self._predict_x0_eps(model_output, sample, t)
        # x_0 を [-1, 1] にクリップ (訓練時の想定範囲)
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        # ----------------------------------------------------------
        # Step 2: 後験平均 μ̃_t を計算 (DDPM Eq.7)
        #
        # [数式]
        #   coef_x0 = √ᾱ_{t-1} * β_t / (1−ᾱ_t)
        #   coef_xt = √α_t * (1−ᾱ_{t-1}) / (1−ᾱ_t)
        #   μ̃_t    = coef_x0 * x̂_0 + coef_xt * x_t
        # ----------------------------------------------------------
        sqrt_alpha_bar_tm1 = alpha_bar_tm1.sqrt()
        sqrt_alpha_t       = (1.0 - beta_t).sqrt()  # √α_t = √(1−β_t)
        one_minus_alpha_bar_t = 1.0 - alpha_bar_t

        coef_x0 = sqrt_alpha_bar_tm1 * beta_t / one_minus_alpha_bar_t
        coef_xt = sqrt_alpha_t * (1.0 - alpha_bar_tm1) / one_minus_alpha_bar_t

        mu_posterior = coef_x0 * x0_hat + coef_xt * sample

        # ----------------------------------------------------------
        # Step 3: ノイズを付加して x_{t-1} を生成 (DDPM Algorithm 2)
        #
        # [数式]
        #   x_{t-1} = μ̃_t + √β̃_t * z,  z ~ N(0, I)   (t > 0)
        #   x_{t-1} = μ̃_t                              (t = 0)
        #
        # β̃_t = posterior_variance[t] (事前計算済み)
        # ----------------------------------------------------------
        if t > 0:
            # MPS 互換: CPU Generator でノイズ生成 → デバイス転送
            noise = torch.randn_like(sample, device="cpu").to(dev)

            if self.variance_type == "fixed_small":
                # β̃_t = (1−ᾱ_{t-1}) / (1−ᾱ_t) * β_t
                variance = self._posterior_variance[t].to(dev)
            else:
                # fixed_large: β_t をそのまま使う
                variance = beta_t

            x_prev = mu_posterior + variance.sqrt() * noise
        else:
            # t = 0: ノイズなし (最終ステップ)
            x_prev = mu_posterior

        return SchedulerOutput(prev_sample=x_prev, pred_original_sample=x0_hat)
