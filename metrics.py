"""
metrics.py — 画像品質メトリクスの計算スクリプト
=============================================================================

計算するメトリクス:
  1. PSNR (Peak Signal-to-Noise Ratio)
     - 参照画像 (DDIM 250step) との pixel-level な比較
     - 高いほど良い (単位: dB)

  2. SSIM (Structural Similarity Index Measure)
     - 輝度・コントラスト・構造の類似度
     - 範囲: [0, 1], 高いほど良い

  3. KID (Kernel Inception Distance) ← 主要メトリクス
     - FID の不偏推定量、少数サンプル (~50 枚) でも信頼性あり
     - 低いほど良い
     - 実装: clean-fid ライブラリ

  4. CLIP Score
     - プロンプトと生成画像の意味的整合性
     - 高いほど良い
     - 実装: open-clip-torch (ViT-B/32)

  5. FID (Fréchet Inception Distance) ← 参考値のみ
     - 分布の距離、N~50 では統計的に不安定
     - 低いほど良い

使用方法:
  # experiment.py 実行後に実行
  python metrics.py
  python metrics.py --output-dir outputs/ --results results.json

結果は results.json に各エントリの "metrics" キーとして追記される。
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np
from PIL import Image


# =============================================================================
# デバイス選択
# =============================================================================

def get_compute_device() -> torch.device:
    """
    メトリクス計算用のデバイスを選択する。

    Note:
      clean-fid の Inception ネットワークは MPS 非対応の場合があるため
      Inception 系メトリクス (KID/FID) は CPU にフォールバックする。
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")  # KID/FID: MPS 非対応のため CPU


# =============================================================================
# 画像ロードユーティリティ
# =============================================================================

def load_image_tensor(path: Path) -> torch.Tensor:
    """
    PNG 画像を float32 テンソル [1, 3, H, W], [0, 1] でロードする。
    """
    img = Image.open(str(path)).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3], [0,1]
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return tensor


def load_images_from_dir(dir_path: Path) -> Tuple[List[Path], List[torch.Tensor]]:
    """
    ディレクトリ内の全 PNG を読み込む。
    """
    paths = sorted(dir_path.glob("*.png"))
    tensors = [load_image_tensor(p) for p in paths]
    return paths, tensors


# =============================================================================
# PSNR / SSIM (torchmetrics)
# =============================================================================

def compute_psnr_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[float, float]:
    """
    PSNR と SSIM を計算する。

    ---------------------------------------------------------------
    [数式] PSNR:

      MSE = mean((pred − target)²)
      PSNR = 10 * log10(1.0² / MSE)     (pixel 値 [0,1] の場合)

    [数式] SSIM (Wang et al. 2004, Eq.13):

      SSIM(x, y) = (2μ_x μ_y + C1)(2σ_xy + C2)
                   / ((μ_x² + μ_y² + C1)(σ_x² + σ_y² + C2))

      C1 = (0.01 * L)²,  C2 = (0.03 * L)²,  L = 1.0 (最大輝度)

    引数:
      pred   : 生成画像 [1, 3, H, W], float32, [0, 1]
      target : 参照画像 [1, 3, H, W], float32, [0, 1]

    戻り値:
      (psnr, ssim)  どちらも float
    ---------------------------------------------------------------
    """
    try:
        from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
    except ImportError:
        warnings.warn("torchmetrics が見つかりません。PSNR/SSIM をスキップします。")
        return float("nan"), float("nan")

    device = torch.device("cpu")  # torchmetrics は CPU で十分
    pred_cpu   = pred.float().to(device)
    target_cpu = target.float().to(device)

    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    psnr_val = float(psnr_metric(pred_cpu, target_cpu).item())
    ssim_val = float(ssim_metric(pred_cpu, target_cpu).item())

    return psnr_val, ssim_val


# =============================================================================
# CLIP Score (open-clip-torch)
# =============================================================================

def compute_clip_score(
    image: torch.Tensor,
    prompt: str,
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    device: Optional[torch.device] = None,
) -> float:
    """
    CLIP Score を計算する。

    ---------------------------------------------------------------
    [数式] CLIP Score (Hessel et al. 2021):

      CLIP_Score(I, c) = max(cosine_similarity(f_I, f_c), 0) * 2.5

      f_I = CLIP Vision Encoder(I)
      f_c = CLIP Text Encoder(c)
      cosine_similarity(a, b) = (a · b) / (||a|| * ||b||)

    スケール 2.5 は [0, 2.5] 範囲に正規化するための定数。
    ---------------------------------------------------------------

    引数:
      image      : [1, 3, H, W], float32, [0, 1]
      prompt     : テキストプロンプト
      model_name : CLIP モデル名 (デフォルト: ViT-B-32)
      pretrained : 事前学習重み (デフォルト: openai)
      device     : 計算デバイス

    戻り値:
      clip_score: float
    """
    if device is None:
        device = get_compute_device()

    try:
        import open_clip
    except ImportError:
        warnings.warn("open-clip-torch が見つかりません。CLIP Score をスキップします。")
        return float("nan")

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    clip_model = clip_model.to(device).eval()
    tokenizer  = open_clip.get_tokenizer(model_name)

    # 画像の前処理 (CLIP 用の正規化込み)
    # image は [1, 3, H, W] float [0,1]
    pil_img = Image.fromarray(
        (image.squeeze(0).permute(1, 2, 0).float().numpy() * 255).astype(np.uint8)
    )
    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)  # [1, 3, 224, 224]

    text_tokens = tokenizer([prompt]).to(device)

    with torch.no_grad():
        img_feat  = clip_model.encode_image(img_tensor)
        txt_feat  = clip_model.encode_text(text_tokens)
        img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)
        txt_feat  = txt_feat  / txt_feat.norm(dim=-1, keepdim=True)
        cos_sim   = (img_feat * txt_feat).sum(dim=-1)

    clip_score = float(max(cos_sim.item(), 0.0) * 2.5)
    return clip_score


# =============================================================================
# KID / FID (clean-fid)
# =============================================================================

def compute_kid_fid(
    gen_dir: Path,
    ref_dir: Path,
    device: Optional[torch.device] = None,
) -> Tuple[float, float]:
    """
    KID と FID を計算する (clean-fid 使用)。

    ---------------------------------------------------------------
    [数式] FID (Heusel et al. 2017):

      FID = ||μ_r − μ_g||² + Tr(Σ_r + Σ_g − 2*(Σ_r Σ_g)^{1/2})

      μ, Σ: Inception v3 特徴量の平均・共分散
      ※ N~50 では Σ の推定が不安定 → 参考値のみ

    [数式] KID (Binkowski et al. 2018):

      KID = MMD²(p_r, p_g)
          = E[k(x,x')] − 2*E[k(x,y)] + E[k(y,y')]

      k: polynomial kernel  k(x,y) = (x^T y / d + 1)^3
      ここで d は Inception 特徴次元数

      FID と異なり不偏推定量のため小 N でも信頼性が高い。
    ---------------------------------------------------------------

    Args:
      gen_dir : 生成画像ディレクトリ (PNG)
      ref_dir : 参照画像ディレクトリ (PNG)
      device  : 計算デバイス (Inception は CPU 推奨)

    Returns:
      (kid, fid)
    """
    if device is None:
        device = torch.device("cpu")  # Inception は MPS 非対応の場合あり

    try:
        from cleanfid import fid as cleanfid
    except ImportError:
        warnings.warn("clean-fid が見つかりません。KID/FID をスキップします。")
        return float("nan"), float("nan")

    gen_dir_str = str(gen_dir)
    ref_dir_str = str(ref_dir)

    try:
        kid_val = cleanfid.compute_kid(gen_dir_str, ref_dir_str, mode="clean")
    except Exception as e:
        warnings.warn(f"KID 計算エラー: {e}")
        kid_val = float("nan")

    try:
        fid_val = cleanfid.compute_fid(gen_dir_str, ref_dir_str, mode="clean")
    except Exception as e:
        warnings.warn(f"FID 計算エラー: {e}")
        fid_val = float("nan")

    return kid_val, fid_val


# =============================================================================
# 全メトリクスの計算と results.json への書き込み
# =============================================================================

def compute_all_metrics(
    output_dir: Path,
    results_path: Optional[Path] = None,
    prompt: str = "a photo of an astronaut riding a horse on mars",
) -> List[Dict]:
    """
    results.json を読み込み、各エントリにメトリクスを追記して保存する。

    計算順序:
      1. リファレンス画像をロード (PSNR/SSIM の target)
      2. 各エントリに PSNR/SSIM を計算
      3. CLIP Score を計算
      4. KID/FID を全生成画像 vs リファレンスで計算 (ディレクトリ単位)

    Args:
      output_dir   : 画像が保存されているディレクトリ
      results_path : results.json のパス (None の場合 output_dir/results.json)
      prompt       : CLIP Score 計算用プロンプト

    Returns:
      results: メトリクス追記済みの list
    """
    if results_path is None:
        results_path = output_dir / "results.json"

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    # ------------------------------------------------------------------
    # リファレンス画像のロード
    # ------------------------------------------------------------------
    ref_entry = next((r for r in results if r.get("is_reference")), None)
    if ref_entry is None or ref_entry.get("image_path") is None:
        raise ValueError("results.json にリファレンスエントリが見つかりません。")

    ref_tensor = load_image_tensor(output_dir / ref_entry["image_path"])

    # ------------------------------------------------------------------
    # 各エントリに PSNR / SSIM / CLIP Score を計算
    # ------------------------------------------------------------------
    print("[メトリクス] PSNR / SSIM / CLIP Score を計算中...")

    for entry in results:
        if entry.get("image_path") is None:
            continue

        img_path = output_dir / entry["image_path"]
        if not img_path.exists():
            print(f"  [スキップ] {img_path} が存在しません")
            continue

        img_tensor = load_image_tensor(img_path)

        # PSNR / SSIM (vs reference)
        if not entry.get("is_reference"):
            psnr, ssim = compute_psnr_ssim(img_tensor, ref_tensor)
        else:
            psnr, ssim = float("inf"), 1.0  # リファレンス自身は完全一致

        # CLIP Score
        clip_score = compute_clip_score(img_tensor, prompt)

        entry["metrics"] = {
            "psnr"      : round(psnr, 4) if not (psnr != psnr) else None,  # NaN check
            "ssim"      : round(ssim, 4) if not (ssim != ssim) else None,
            "clip_score": round(clip_score, 4) if not (clip_score != clip_score) else None,
        }
        print(f"  {entry['image_path']}: PSNR={psnr:.2f}, SSIM={ssim:.4f}, CLIP={clip_score:.4f}")

    # ------------------------------------------------------------------
    # KID / FID (全生成画像 vs リファレンス、ディレクトリ単位)
    # ------------------------------------------------------------------
    print("\n[メトリクス] KID / FID を計算中 (clean-fid)...")

    # 参照画像ディレクトリを一時的に作成
    ref_dir = output_dir / "_ref_dir"
    ref_dir.mkdir(exist_ok=True)
    ref_img_path = ref_dir / "reference.png"

    # リファレンス PNG を ref_dir にコピー (既存なら OK)
    if not ref_img_path.exists():
        src = output_dir / ref_entry["image_path"]
        import shutil
        shutil.copy(str(src), str(ref_img_path))

    # 生成画像のみのディレクトリ (リファレンス除外)
    gen_dir = output_dir / "_gen_dir"
    gen_dir.mkdir(exist_ok=True)

    for entry in results:
        if entry.get("is_reference") or entry.get("image_path") is None:
            continue
        src = output_dir / entry["image_path"]
        if src.exists():
            import shutil
            dst = gen_dir / entry["image_path"]
            if not dst.exists():
                shutil.copy(str(src), str(dst))

    kid_val, fid_val = compute_kid_fid(gen_dir, ref_dir)
    print(f"  全体 KID: {kid_val:.6f}")
    print(f"  全体 FID: {fid_val:.4f} (参考値; N~50 では不安定)")

    # 全体 KID/FID を各エントリに一括記録 (ディレクトリ単位なので同一値)
    for entry in results:
        if not entry.get("is_reference") and "metrics" in entry:
            entry["metrics"]["kid_overall"] = round(kid_val, 6) if kid_val == kid_val else None
            entry["metrics"]["fid_overall"] = round(fid_val, 4) if fid_val == fid_val else None

    # ------------------------------------------------------------------
    # 更新した results.json を保存
    # ------------------------------------------------------------------
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[完了] メトリクスを {results_path} に保存しました。")

    return results


# =============================================================================
# CLI エントリーポイント
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="拡散サンプラー比較 — メトリクス計算")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="experiment.py の出力ディレクトリ (デフォルト: outputs)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="results.json のパス (省略時は output-dir/results.json)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a photo of an astronaut riding a horse on mars",
        help="CLIP Score 計算用プロンプト",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_path = Path(args.results) if args.results else None

    compute_all_metrics(output_dir, results_path, args.prompt)


if __name__ == "__main__":
    main()
