"""
samplers/ddim.py — DDIM (Denoising Diffusion Implicit Models) サンプラー
=============================================================================

論文:
  Song, J., Meng, C., & Ermon, S. (2020).
  "Denoising Diffusion Implicit Models"
  ICLR 2021. https://arxiv.org/abs/2010.02502

概要:
  DDIM は DDPM のマルコフ連鎖を非マルコフ過程に拡張することで、
  同一の訓練済み UNet を使いながら少ステップで高品質な画像を生成する。
  η=0 のとき完全決定論的 (ODE ベース) になり、再現性が保証される。

  任意のサブセットタイムステップ {τ_1, ..., τ_S} ⊂ {0,...,T-1} に適用でき、
  S << T でも機能する (DDPM は S=T が必要)。

特性:
  タイプ : ODE (η=0、決定論的) または SDE (η>0、確率的)
  NFE/step: 1
  推奨ステップ数: 20–250

更新式 (DDIM Eq.12):
  x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} * x̂_0
              + √(1−ᾱ_{τ_{i-1}} − σ²) * ε_θ(x_{τ_i}, τ_i)
              + σ * z

  ここで:
    x̂_0 = (x_{τ_i} − √(1−ᾱ_{τ_i}) * ε_θ) / √ᾱ_{τ_i}
    σ = η * √( (1−ᾱ_{τ_{i-1}}) / (1−ᾱ_{τ_i}) ) * √(1 − ᾱ_{τ_i}/ᾱ_{τ_{i-1}})
    z ~ N(0, I)

  η=0 のとき σ=0 で決定論的 ODE になる (「DDIM」の名前の所以)。
"""

from __future__ import annotations

from typing import Union

import torch

from .base import BaseSampler, SchedulerOutput, predict_x0, predict_eps


class DDIMSampler(BaseSampler):
    """
    DDIM サンプラー (η=0 で決定論的 ODE、η>0 で確率的)。

    Args:
      num_train_timesteps : 訓練ステップ数 T
      beta_start          : β スケジュール開始値
      beta_end            : β スケジュール終了値
      eta                 : 確率性パラメータ η ∈ [0, 1]
                            0 = 完全決定論的 (推奨)
                            1 = DDPM と同じノイズレベル
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        eta: float = 0.0,
    ) -> None:
        super().__init__(num_train_timesteps, beta_start, beta_end)
        assert 0.0 <= eta <= 1.0, f"eta は [0, 1] の範囲で指定してください: {eta}"
        self.eta = eta

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
        DDIM 逆拡散の 1 ステップを実行する。

        ---------------------------------------------------------------
        [アルゴリズム] DDIM Eq.12 (Song et al. 2020):

        入力: x_{τ_i}, τ_i, v_θ(x_{τ_i}, τ_i), η
        出力: x_{τ_{i-1}}

        Step 1 — ε̂ の推定 (v-prediction 変換):
          ε̂ = σ_{τ_i} * x_{τ_i} + α_{τ_i} * v_θ    (predict_eps)

        Step 2 — x̂_0 の推定:
          x̂_0 = (x_{τ_i} − σ_{τ_i} * ε̂) / α_{τ_i}
               = α_{τ_i} * x_{τ_i} − σ_{τ_i} * v_θ   (predict_x0)
          ※ x̂_0 を [-1, 1] にクリップ

        Step 3 — DDIM ノイズレベル σ を計算:
          σ = η * √( (1−ᾱ_{τ_{i-1}}) / (1−ᾱ_{τ_i}) * (1 − ᾱ_{τ_i}/ᾱ_{τ_{i-1}}) )
          η=0 なら σ=0 (決定論的)

        Step 4 — x_{τ_{i-1}} を計算 (DDIM Eq.12):
          direction = √(1−ᾱ_{τ_{i-1}} − σ²) * ε̂
          x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} * x̂_0 + direction + σ * z

          z ~ N(0, I)  (η=0 なら z の項はゼロ)

        変数対応:
          sample        ← x_{τ_i}
          model_output  ← v_θ(x_{τ_i}, τ_i)
          alpha_bar_t   ← ᾱ_{τ_i}
          alpha_bar_tm1 ← ᾱ_{τ_{i-1}}
          alpha_t       ← √ᾱ_{τ_i}
          sigma_t       ← √(1−ᾱ_{τ_i})
        ---------------------------------------------------------------
        """
        t = int(timestep.item()) if timestep.ndim == 0 else int(timestep[0].item())

        # 前のタイムステップ τ_{i-1} を計算
        step_size = self.num_train_timesteps // self.num_inference_steps
        t_prev = t - step_size
        t_prev = max(t_prev, 0)

        dev = sample.device

        # スケジュール値を取得 (デバイスへ転送)
        alpha_bar_t   = self.schedule.alphas_cumprod[t].to(dev)
        alpha_bar_tm1 = (
            self.schedule.alphas_cumprod[t_prev].to(dev)
            if t_prev >= 0
            else torch.tensor(1.0, device=dev)
        )

        alpha_t = alpha_bar_t.sqrt()           # √ᾱ_{τ_i}
        sigma_t = (1.0 - alpha_bar_t).sqrt()   # √(1−ᾱ_{τ_i})

        # ----------------------------------------------------------
        # Step 1 & 2: x̂_0, ε̂ を計算 (prediction_type に応じて変換)
        # ----------------------------------------------------------
        x0_hat, eps_hat = self._predict_x0_eps(model_output, sample, t)
        x0_hat = x0_hat.clamp(-1.0, 1.0)

        # ----------------------------------------------------------
        # Step 3: DDIM ノイズレベル σ を計算 (DDIM Eq.16)
        #
        # [数式]
        #   σ = η * √( (1−ᾱ_{τ_{i-1}}) / (1−ᾱ_{τ_i})
        #              * (1 − ᾱ_{τ_i} / ᾱ_{τ_{i-1}}) )
        #
        # η=0 のとき σ=0 → 決定論的 ODE になる (DDIM の主要な特性)
        # η=1 のとき DDPM の posterior variance に近似する
        # ----------------------------------------------------------
        sigma_ddim = 0.0
        if self.eta > 0.0 and t_prev > 0:
            # ᾱ_{τ_i} / ᾱ_{τ_{i-1}} の計算 (< 1)
            ratio = alpha_bar_t / alpha_bar_tm1
            # 内部の平方根引数
            variance_ratio = (
                (1.0 - alpha_bar_tm1) / (1.0 - alpha_bar_t) * (1.0 - ratio)
            )
            sigma_ddim = self.eta * variance_ratio.sqrt()

        # ----------------------------------------------------------
        # Step 4: x_{τ_{i-1}} を組み立てる (DDIM Eq.12)
        #
        # [数式]
        #   direction   = √(1−ᾱ_{τ_{i-1}} − σ²) * ε̂
        #   x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} * x̂_0 + direction + σ * z
        #
        # σ=0 (η=0) のとき: x_{τ_{i-1}} = √ᾱ_{τ_{i-1}} * x̂_0
        #                                 + √(1−ᾱ_{τ_{i-1}}) * ε̂
        # これは x_t の空間を x_{t-1} の空間へ「方向転換」する解釈ができる。
        # ----------------------------------------------------------
        sqrt_alpha_bar_tm1 = alpha_bar_tm1.sqrt()

        # direction ベクトルの係数: √(1−ᾱ_{τ_{i-1}} − σ²)
        dir_coef = (
            (1.0 - alpha_bar_tm1 - sigma_ddim ** 2).clamp(min=0.0).sqrt()
        )

        x_prev = (
            sqrt_alpha_bar_tm1 * x0_hat
            + dir_coef * eps_hat
        )

        # η > 0 のとき確率的ノイズを加える
        if self.eta > 0.0 and t_prev > 0:
            # MPS 互換: CPU Generator でノイズ生成 → デバイス転送
            noise = torch.randn_like(sample, device="cpu").to(dev)
            x_prev = x_prev + sigma_ddim * noise

        return SchedulerOutput(prev_sample=x_prev, pred_original_sample=x0_hat)
