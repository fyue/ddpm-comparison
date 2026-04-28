"""
experiment.py — 全サンプラーの比較実験スクリプト
=============================================================================

実験内容:
  - 10 種類のサンプラー × 3 ステップ (10 / 20 / 50) = 30 通りの組み合わせ
  - 各組み合わせで 1 枚の画像を生成し PNG として保存
  - リファレンス画像 (DDIM 250step, seed=42) も生成
  - 生成時間・設定を results.json に記録

使用方法:
  python experiment.py
  python experiment.py --dry-run          # モデルロードのみ確認
  python experiment.py --output-dir out/  # 出力ディレクトリを指定

出力:
  {output_dir}/reference.png
  {output_dir}/ddpm_steps10.png
  {output_dir}/dpm_solver_2_single_steps50.png
  ...
  {output_dir}/results.json

results.json の構造:
  [
    {
      "sampler"    : "ddim",
      "steps"      : 20,
      "nfe"        : 20,        // 実際の UNet フォワード回数
      "time_sec"   : 3.14,
      "image_path" : "ddim_steps20.png",
      "is_reference": false
    },
    ...
  ]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image

from pipeline import SDPipeline, get_device
from samplers import build_sampler, SAMPLER_REGISTRY, NFE_PER_STEP


# =============================================================================
# 実験設定
# =============================================================================

PROMPT = "a photo of an astronaut riding a horse on mars"
SEED   = 42
GUIDANCE_SCALE = 7.5
STEP_LIST = [10, 20, 50]  # 比較するステップ数
MODEL_ID  = "CompVis/stable-diffusion-v1-4"

# 実験するサンプラー名のリスト (SAMPLER_REGISTRY のキー)
SAMPLER_NAMES = [
    "ddpm",
    "ddim",
    "euler",
    "heun",
    "lms2",
    "dpm_solver_1",
    "dpm_solver_2_single",
    "dpm_solver_2_multi",
    "dpm_solver_3_single",
    "dpm_solver_3_multi",
]


# =============================================================================
# 画像保存ユーティリティ
# =============================================================================

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """
    [1, 3, H, W] の float32 テンソル (値 [0,1]) を PIL Image に変換する。
    """
    img = image_tensor.squeeze(0)          # [3, H, W]
    img = img.float().cpu()
    img = (img * 255).round().clamp(0, 255).to(torch.uint8)
    img = img.permute(1, 2, 0).numpy()     # [H, W, 3] numpy
    return Image.fromarray(img, mode="RGB")


def save_image(image_tensor: torch.Tensor, path: Path) -> None:
    """
    テンソルを PNG として保存する。
    """
    pil_image = tensor_to_pil(image_tensor)
    pil_image.save(str(path))
    print(f"  [保存] {path}")


# =============================================================================
# 実験ランナー
# =============================================================================

def run_experiments(
    output_dir: Path,
    dry_run: bool = False,
) -> List[Dict[str, Any]]:
    """
    全サンプラー × 全ステップ数の実験を実行し、結果を返す。

    Args:
      output_dir : 画像と results.json を保存するディレクトリ
      dry_run    : True の場合はモデルロードのみ実行し生成はスキップ

    Returns:
      results: List of dict (results.json の内容)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"[実験] デバイス: {device}")

    # モデルロード (全実験で共有)
    pipe = SDPipeline.from_pretrained(MODEL_ID, device=device)

    if dry_run:
        print("[dry-run] モデルロード成功。生成はスキップします。")
        return []

    results: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # リファレンス画像 (DDIM 250step)
    # ------------------------------------------------------------------
    print("\n[リファレンス] DDIM 250step を生成中...")
    ref_path = output_dir / "reference.png"

    t0 = time.perf_counter()
    ref_image = pipe.generate_reference(
        prompt=PROMPT,
        seed=SEED,
        num_inference_steps=250,
        guidance_scale=GUIDANCE_SCALE,
    )
    ref_time = time.perf_counter() - t0

    save_image(ref_image, ref_path)

    results.append({
        "sampler"      : "ddim",
        "order"        : None,
        "solver_mode"  : None,
        "steps"        : 250,
        "nfe"          : 250,
        "time_sec"     : round(ref_time, 3),
        "image_path"   : str(ref_path.name),
        "is_reference" : True,
    })

    # ------------------------------------------------------------------
    # 各サンプラー × 各ステップ数
    # ------------------------------------------------------------------
    total = len(SAMPLER_NAMES) * len(STEP_LIST)
    count = 0

    for sampler_name in SAMPLER_NAMES:
        for steps in STEP_LIST:
            count += 1
            print(f"\n[{count}/{total}] sampler={sampler_name}, steps={steps}")

            # サンプラーを毎回新規インスタンス化 (バッファリセット保証)
            sampler = build_sampler(sampler_name)

            # 実際の NFE = steps × NFE_per_step
            nfe_per_step = NFE_PER_STEP[sampler_name]
            nfe = steps * nfe_per_step

            # 画像生成
            image_filename = f"{sampler_name}_steps{steps:02d}.png"
            image_path = output_dir / image_filename

            try:
                image, elapsed = pipe.generate(
                    prompt=PROMPT,
                    sampler=sampler,
                    num_inference_steps=steps,
                    guidance_scale=GUIDANCE_SCALE,
                    seed=SEED,
                )
            except Exception as e:
                print(f"  [エラー] {sampler_name} steps={steps}: {e}")
                results.append({
                    "sampler"      : sampler_name,
                    "order"        : getattr(sampler, "order", None),
                    "solver_mode"  : getattr(sampler, "solver_mode", None),
                    "steps"        : steps,
                    "nfe"          : nfe,
                    "time_sec"     : None,
                    "image_path"   : None,
                    "is_reference" : False,
                    "error"        : str(e),
                })
                continue

            save_image(image, image_path)
            print(f"  NFE={nfe}, time={elapsed:.2f}s")

            results.append({
                "sampler"      : sampler_name,
                "order"        : getattr(sampler, "order", None),
                "solver_mode"  : getattr(sampler, "solver_mode", None),
                "steps"        : steps,
                "nfe"          : nfe,
                "time_sec"     : round(elapsed, 3),
                "image_path"   : image_filename,
                "is_reference" : False,
            })

    # ------------------------------------------------------------------
    # results.json に保存
    # ------------------------------------------------------------------
    results_path = output_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[完了] 結果を保存しました: {results_path}")

    return results


# =============================================================================
# CLI エントリーポイント
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diffusion Sampler 比較実験"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="画像と results.json を保存するディレクトリ (デフォルト: outputs)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="モデルロードのみ実行し、画像生成をスキップする",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_experiments(output_dir=output_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
