"""
pipeline.py — Stable Diffusion 2.1 推論パイプライン
=============================================================================

このモジュールは:
  1. SD 2.1 の UNet / VAE / TextEncoder をロードする
  2. カスタムサンプラーを受け取り、テキスト条件付き画像を生成する
  3. Heun / DPM-Solver singlestep のような「中間でもう一度 UNet を呼ぶ」
     サンプラーに対応した推論ループを提供する

モデル:
  stabilityai/stable-diffusion-2-1
    - UNet2DConditionModel (v_prediction, 512x512 latent)
    - AutoencoderKL (VAE, downscale factor=8, scale=0.18215)
    - CLIPTextModel + CLIPTokenizer (OpenCLIP ViT-H/14)

デバイス対応:
  - CUDA (NVIDIA GPU)
  - MPS  (Apple Silicon M1/M2/M3/M4/M5)
  - CPU  (フォールバック)

CFG (Classifier-Free Guidance):
  Guidance Scale g で以下を適用:
    ε_guided = ε_uncond + g * (ε_cond − ε_uncond)

  uncond と cond をバッチに concat して 1 回の UNet フォワードで処理する。
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer

from samplers import BaseSampler, HeunSampler, DPMSolver, SAMPLER_REGISTRY


# =============================================================================
# デバイス選択ユーティリティ
# =============================================================================

def get_device() -> torch.device:
    """
    利用可能な最良のデバイスを自動選択する。

    優先順位: CUDA → MPS (Apple Silicon) → CPU

    戻り値:
      torch.device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =============================================================================
# モデルロード
# =============================================================================

class SDPipeline:
    """
    Stable Diffusion 2.1 推論パイプライン。

    使用例:
      pipe = SDPipeline.from_pretrained()
      image = pipe.generate("a photo of a cat", sampler=DDIMSampler(), steps=20)

    Args:
      unet          : UNet2DConditionModel
      vae           : AutoencoderKL
      text_encoder  : CLIPTextModel
      tokenizer     : CLIPTokenizer
      device        : 推論デバイス
      torch_dtype   : float16 (MPS/CUDA 推奨) または float32
    """

    # VAE のスケール係数 (SD 2.1 の固定値)
    VAE_SCALE_FACTOR = 0.18215
    # UNet が期待する latent の空間サイズ (512px 画像に対して 64x64)
    LATENT_H = 64
    LATENT_W = 64
    LATENT_C = 4  # latent チャンネル数

    def __init__(
        self,
        unet: UNet2DConditionModel,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        device: torch.device,
        torch_dtype: torch.dtype = torch.float16,
    ) -> None:
        self.unet         = unet.to(device)
        self.vae          = vae.to(device)
        self.text_encoder = text_encoder.to(device)
        self.tokenizer    = tokenizer
        self.device       = device
        self.torch_dtype  = torch_dtype
        # UNet 設定から prediction_type を取得 (epsilon: SD 1.x, v_prediction: SD 2.x)
        self.prediction_type: str = getattr(unet.config, "prediction_type", "epsilon")

        # 全モデルを eval モードにして gradient 計算を無効化
        self.unet.eval()
        self.vae.eval()
        self.text_encoder.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "CompVis/stable-diffusion-v1-4",
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> "SDPipeline":
        """
        Hugging Face Hub からモデルをロードする。

        Args:
          model_id    : HF Hub のモデル ID
          device      : None の場合は get_device() で自動選択
          torch_dtype : None の場合は MPS/CUDA → float16, CPU → float32
        """
        if device is None:
            device = get_device()

        if torch_dtype is None:
            # MPS と CPU では float16 が不安定なため float32 を使う
            # CUDA のみ float16 を使用
            torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

        print(f"[SDPipeline] デバイス: {device}, dtype: {torch_dtype}")
        print(f"[SDPipeline] モデルをロード中: {model_id}")

        unet = UNet2DConditionModel.from_pretrained(
            model_id, subfolder="unet", torch_dtype=torch_dtype
        )
        vae = AutoencoderKL.from_pretrained(
            model_id, subfolder="vae", torch_dtype=torch_dtype
        )
        text_encoder = CLIPTextModel.from_pretrained(
            model_id, subfolder="text_encoder", torch_dtype=torch_dtype
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            model_id, subfolder="tokenizer"
        )

        print("[SDPipeline] モデルロード完了")
        return cls(unet, vae, text_encoder, tokenizer, device, torch_dtype)

    # ----------------------------------------------------------------
    # テキストエンコード
    # ----------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        テキストプロンプトを CLIP エンコードし、CFG 用の concat テンソルを返す。

        ---------------------------------------------------------------
        [数式] Classifier-Free Guidance (CFG):

          v_guided = v_uncond + g * (v_cond − v_uncond)

          ここで g は guidance_scale。

        実装:
          uncond と cond を dim=0 で concat し、
          バッチサイズ 2 で 1 回 UNet フォワードを実行する。
          → 後で .chunk(2) で分割して CFG を適用する。
        ---------------------------------------------------------------

        引数:
          prompt        : 生成プロンプト文字列
          guidance_scale: CFG スケール (デフォルト 7.5)

        戻り値:
          encoder_hidden_states: shape [2, seq_len, hidden_dim]
                                  [0] = uncond, [1] = cond
        """
        # 空文字列で uncond (negative prompt) をエンコード
        batch = [("", prompt)]

        input_ids_list = []
        for text in [b[0] for b in batch] + [b[1] for b in batch]:
            ids = self.tokenizer(
                text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(self.device)
            input_ids_list.append(ids)

        # shape: [2, seq_len]
        input_ids = torch.cat(input_ids_list, dim=0)

        # CLIPTextModel でエンコード → shape: [2, seq_len, hidden_dim]
        encoder_hidden_states = self.text_encoder(input_ids).last_hidden_state

        return encoder_hidden_states

    # ----------------------------------------------------------------
    # VAE エンコード / デコード
    # ----------------------------------------------------------------

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        latent → pixel 空間に変換する。

        ---------------------------------------------------------------
        [数式] VAE デコード:

          z_scaled = z / VAE_SCALE_FACTOR     (= latents / 0.18215)
          x_pixel  = VAE.decode(z_scaled)     → [-1, 1] 範囲
          x_uint8  = ((x_pixel + 1) / 2).clamp(0, 1)  → [0, 1]
        ---------------------------------------------------------------

        引数:
          latents: shape [B, 4, H//8, W//8], float16 or float32

        戻り値:
          images : shape [B, 3, H, W], float32, [0, 1]
        """
        # スケールを元に戻す
        latents = latents / self.VAE_SCALE_FACTOR

        # VAE デコード (VAE と同じ dtype に変換してから実行)
        latents_vae = latents.to(dtype=self.torch_dtype)
        decoded = self.vae.decode(latents_vae).sample

        # [-1, 1] → [0, 1]
        images = (decoded / 2.0 + 0.5).clamp(0.0, 1.0)
        return images

    # ----------------------------------------------------------------
    # UNet フォワード関数 (CFG 付き)
    # ----------------------------------------------------------------

    def _unet_forward_cfg(
        self,
        x_scaled: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        guidance_scale: float = 7.5,
    ) -> torch.Tensor:
        """
        CFG を適用した UNet フォワードを 1 回実行する。

        ---------------------------------------------------------------
        [数式] CFG (Ho & Salimans 2022):

          v_guided = v_uncond + guidance_scale * (v_cond − v_uncond)

          バッチに [uncond, cond] を concat して 1 フォワードで計算。
        ---------------------------------------------------------------

        引数:
          x_scaled              : スケーリング済み latent  shape [B, 4, H, W]
          timestep              : 現在のタイムステップ       shape [B]
          encoder_hidden_states : CFG 用テキスト埋め込み   shape [2B, seq, dim]
          guidance_scale        : CFG スケール g

        戻り値:
          v_guided: shape [B, 4, H, W]
        """
        B = x_scaled.shape[0]
        # uncond と cond を concat してバッチサイズ 2B で実行
        x_input = x_scaled.repeat(2, 1, 1, 1).to(dtype=self.torch_dtype)  # [2B, 4, H, W]
        t_input = timestep.repeat(2)            # [2B]
        enc_hs = encoder_hidden_states.to(dtype=self.torch_dtype)

        with torch.no_grad():
            v_both = self.unet(
                x_input,
                t_input,
                encoder_hidden_states=enc_hs,
            ).sample  # [2B, 4, H, W]

        v_uncond, v_cond = v_both.chunk(2, dim=0)  # 各 [B, 4, H, W]

        # CFG: v_guided = v_uncond + g * (v_cond - v_uncond)
        v_guided = v_uncond + guidance_scale * (v_cond - v_uncond)
        return v_guided

    # ----------------------------------------------------------------
    # メイン生成関数
    # ----------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        sampler: BaseSampler,
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 512,
        width: int = 512,
    ) -> Tuple[torch.Tensor, float]:
        """
        テキストプロンプトから画像を生成する。

        Args:
          prompt              : 生成プロンプト
          sampler             : 使用するサンプラーインスタンス
          num_inference_steps : 推論ステップ数
          guidance_scale      : CFG スケール
          seed                : 乱数シード (再現性確保)
          height, width       : 生成画像サイズ (8 の倍数)

        Returns:
          (image_tensor, elapsed_time)
          image_tensor : shape [1, 3, H, W], float32, [0, 1]
          elapsed_time : 生成時間 (秒)
        """
        device = self.device
        h_latent = height // 8
        w_latent = width // 8

        # ----------------------------------------------------------
        # [1] テキストエンコード
        # ----------------------------------------------------------
        encoder_hidden_states = self.encode_prompt(prompt, guidance_scale)

        # ----------------------------------------------------------
        # [2] 初期ノイズ生成 (MPS 互換: CPU Generator → デバイス転送)
        #
        # MPS デバイスでは torch.Generator("mps") が不安定な場合があるため、
        # CPU で生成して .to(device) で転送する。
        # ----------------------------------------------------------
        generator = torch.Generator().manual_seed(seed)  # CPU generator
        latents = torch.randn(
            (1, self.LATENT_C, h_latent, w_latent),
            generator=generator,
            dtype=self.torch_dtype,
        ).to(device)

        # ----------------------------------------------------------
        # [3] サンプラーのタイムステップを設定
        # ----------------------------------------------------------
        # prediction_type をサンプラーに伝える (epsilon: SD 1.x, v_prediction: SD 2.x)
        sampler.prediction_type = self.prediction_type
        sampler.set_timesteps(num_inference_steps, device=device)

        # sigma 空間サンプラー (Euler/Heun/LMS) は sigma_max でスケール
        if hasattr(sampler, '_sigmas') and sampler._sigmas is not None:
            latents = latents * sampler._sigmas[0]  # sigma_max ≈ 14.6

        # UNet フォワード関数 (Heun / DPM-Solver singlestep から呼ばれる)
        def unet_forward_fn(x, t, enc_hs):
            return self._unet_forward_cfg(x, t, enc_hs, guidance_scale)

        # ----------------------------------------------------------
        # [4] 推論ループ
        # ----------------------------------------------------------
        start_time = time.perf_counter()

        for i, t in enumerate(sampler.timesteps):
            t_tensor = torch.tensor([int(t)], device=device)

            # UNet 入力のスケーリング (sigma 空間サンプラーでは x / sqrt(σ²+1))
            latents_scaled = sampler.scale_model_input(latents, t_tensor)

            # UNet フォワード (CFG 適用)
            model_output = self._unet_forward_cfg(
                latents_scaled, t_tensor, encoder_hidden_states, guidance_scale
            )

            # サンプラー固有のステップを実行
            if isinstance(sampler, HeunSampler):
                # Heun: 2 回目の UNet フォワードが必要
                output = sampler.step_heun(
                    model_output_1=model_output,
                    timestep=t_tensor,
                    sample=latents,
                    unet_forward_fn=unet_forward_fn,
                    encoder_hidden_states=encoder_hidden_states,
                    guidance_scale=guidance_scale,
                )
            elif isinstance(sampler, DPMSolver) and sampler.solver_mode == "singlestep" and sampler.order >= 2:
                # DPM-Solver singlestep order >= 2: 追加の UNet フォワードが必要
                output = sampler.step_singlestep(
                    model_output=model_output,
                    timestep=t_tensor,
                    sample=latents,
                    unet_forward_fn=unet_forward_fn,
                    encoder_hidden_states=encoder_hidden_states,
                    guidance_scale=guidance_scale,
                )
            else:
                # それ以外 (DDPM, DDIM, Euler, LMS-2, DPM-Solver-1, DPM-Solver multistep)
                output = sampler.step(
                    model_output=model_output,
                    timestep=t_tensor,
                    sample=latents,
                )

            latents = output.prev_sample

        elapsed_time = time.perf_counter() - start_time

        # ----------------------------------------------------------
        # [5] latent → 画像に変換
        # ----------------------------------------------------------
        images = self.decode_latents(latents)

        return images, elapsed_time

    # ----------------------------------------------------------------
    # リファレンス生成 (DDIM 250step, 固定シード)
    # ----------------------------------------------------------------

    def generate_reference(
        self,
        prompt: str,
        seed: int = 42,
        num_inference_steps: int = 250,
        guidance_scale: float = 7.5,
        height: int = 512,
        width: int = 512,
    ) -> torch.Tensor:
        """
        品質比較の基準となるリファレンス画像を生成する。

        DDIM (η=0, 250step) を使って決定論的に生成する。
        PSNR / SSIM 計算の "Ground Truth" として使用。

        Args:
          prompt              : プロンプト文字列
          seed                : 固定シード (デフォルト 42)
          num_inference_steps : リファレンス用ステップ数 (デフォルト 250)
          guidance_scale      : CFG スケール

        Returns:
          image: shape [1, 3, H, W], float32, [0, 1]
        """
        from samplers import DDIMSampler

        sampler = DDIMSampler(eta=0.0)
        image, _ = self.generate(
            prompt=prompt,
            sampler=sampler,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            seed=seed,
            height=height,
            width=width,
        )
        return image
