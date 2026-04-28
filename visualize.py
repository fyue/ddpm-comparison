"""
visualize.py — 比較結果の可視化スクリプト
=============================================================================

生成するプロット:
  1. grid.png          : 10 サンプラー × 4 列 (steps 10/20/50 + reference) グリッド
  2. heatmap_psnr.png  : PSNR ヒートマップ (sampler × steps)
  3. heatmap_ssim.png  : SSIM ヒートマップ
  4. timing_bar.png    : 生成時間の棒グラフ (sampler × steps)
  5. nfe_scatter.png   : 実 NFE vs 品質 (PSNR) の散布図
  6. dpm_compare.png   : DPM-Solver-2/3 の singlestep vs multistep 比較

使用方法:
  python visualize.py
  python visualize.py --output-dir outputs/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")  # ヘッドレス環境用
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image


# =============================================================================
# データロードユーティリティ
# =============================================================================

SAMPLER_DISPLAY_NAMES = {
    "ddpm"                : "DDPM",
    "ddim"                : "DDIM",
    "euler"               : "Euler",
    "heun"                : "Heun",
    "lms2"                : "LMS-2",
    "dpm_solver_1"        : "DPM-1",
    "dpm_solver_2_single" : "DPM-2S",
    "dpm_solver_2_multi"  : "DPM-2M",
    "dpm_solver_3_single" : "DPM-3S",
    "dpm_solver_3_multi"  : "DPM-3M",
}

STEPS = [10, 20, 50]
SAMPLER_ORDER = list(SAMPLER_DISPLAY_NAMES.keys())


def load_results(results_path: Path) -> List[Dict]:
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def get_entry(results: List[Dict], sampler: str, steps: int) -> Optional[Dict]:
    for r in results:
        if r.get("sampler") == sampler and r.get("steps") == steps and not r.get("is_reference"):
            return r
    return None


def get_reference(results: List[Dict]) -> Optional[Dict]:
    for r in results:
        if r.get("is_reference"):
            return r
    return None


def load_pil(output_dir: Path, image_path: Optional[str]) -> Optional[Image.Image]:
    if image_path is None:
        return None
    p = output_dir / image_path
    if not p.exists():
        return None
    return Image.open(str(p)).convert("RGB")


# =============================================================================
# 1. グリッド画像
# =============================================================================

def plot_grid(results: List[Dict], output_dir: Path, save_path: Path) -> None:
    """
    サンプラー × ステップ数のグリッド画像を生成する。

    レイアウト:
      列: steps=10, steps=20, steps=50, reference
      行: 各サンプラー (SAMPLER_ORDER の順)
    """
    cols = STEPS + ["ref"]
    n_rows = len(SAMPLER_ORDER)
    n_cols = len(cols)

    ref_entry = get_reference(results)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.5, n_rows * 2.5))
    plt.subplots_adjust(wspace=0.02, hspace=0.1)

    col_labels = [f"steps={s}" for s in STEPS] + ["Reference\n(DDIM 250)"]

    for col_idx, col in enumerate(cols):
        for row_idx, sampler_name in enumerate(SAMPLER_ORDER):
            ax = axes[row_idx][col_idx]
            ax.axis("off")

            if col == "ref":
                img = load_pil(output_dir, ref_entry["image_path"] if ref_entry else None)
            else:
                entry = get_entry(results, sampler_name, col)
                img = load_pil(output_dir, entry["image_path"] if entry else None)

            if img is not None:
                ax.imshow(img)

            # 行ラベル (最左列のみ)
            if col_idx == 0:
                ax.set_ylabel(
                    SAMPLER_DISPLAY_NAMES.get(sampler_name, sampler_name),
                    rotation=0, labelpad=55, va="center", fontsize=9, fontweight="bold"
                )

            # 列ラベル (最上行のみ)
            if row_idx == 0:
                ax.set_title(col_labels[col_idx], fontsize=9)

    fig.suptitle("Diffusion Sampler Comparison — Stable Diffusion 2.1", fontsize=13, y=1.01)
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# =============================================================================
# 2. PSNR / SSIM ヒートマップ
# =============================================================================

def plot_heatmap(
    results: List[Dict],
    metric_key: str,
    title: str,
    save_path: Path,
    cmap: str = "YlOrRd",
) -> None:
    """
    指定メトリクスのヒートマップ (sampler × steps) を生成する。

    Args:
      metric_key: "psnr" または "ssim"
      title      : プロット タイトル
      save_path  : 保存先 PNG パス
      cmap       : matplotlib カラーマップ名
    """
    data = np.full((len(SAMPLER_ORDER), len(STEPS)), np.nan)

    for row_idx, sampler in enumerate(SAMPLER_ORDER):
        for col_idx, steps in enumerate(STEPS):
            entry = get_entry(results, sampler, steps)
            if entry and "metrics" in entry:
                val = entry["metrics"].get(metric_key)
                if val is not None:
                    data[row_idx, col_idx] = val

    fig, ax = plt.subplots(figsize=(6, 7))
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=np.nanmin(data), vmax=np.nanmax(data))
    plt.colorbar(im, ax=ax, label=metric_key.upper())

    ax.set_xticks(range(len(STEPS)))
    ax.set_xticklabels([str(s) for s in STEPS])
    ax.set_xlabel("Steps")

    ax.set_yticks(range(len(SAMPLER_ORDER)))
    ax.set_yticklabels([SAMPLER_DISPLAY_NAMES.get(s, s) for s in SAMPLER_ORDER])

    # セル内に値を表示
    for r in range(len(SAMPLER_ORDER)):
        for c in range(len(STEPS)):
            val = data[r, c]
            if not np.isnan(val):
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=8, color="black")

    ax.set_title(title, fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# =============================================================================
# 3. 生成時間棒グラフ
# =============================================================================

def plot_timing_bar(results: List[Dict], save_path: Path) -> None:
    """
    生成時間の棒グラフを描画する。

    グループ: サンプラー名
    棒: steps=10, 20, 50 の 3 本
    """
    x = np.arange(len(SAMPLER_ORDER))
    width = 0.25
    colors = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(14, 5))

    for i, steps in enumerate(STEPS):
        times = []
        for sampler in SAMPLER_ORDER:
            entry = get_entry(results, sampler, steps)
            t = entry["time_sec"] if (entry and entry.get("time_sec") is not None) else 0.0
            times.append(t)

        bars = ax.bar(x + i * width, times, width, label=f"steps={steps}", color=colors[i], alpha=0.85)
        # 棒上に数値を表示
        for bar, t in zip(bars, times):
            if t > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02,
                    f"{t:.1f}s",
                    ha="center", va="bottom", fontsize=6.5, rotation=90,
                )

    ax.set_xlabel("Sampler")
    ax.set_ylabel("Time (sec)")
    ax.set_title("Generation Time per Sampler × Steps")
    ax.set_xticks(x + width)
    ax.set_xticklabels([SAMPLER_DISPLAY_NAMES.get(s, s) for s in SAMPLER_ORDER], rotation=20, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# =============================================================================
# 4. NFE vs 品質 散布図
# =============================================================================

def plot_nfe_scatter(results: List[Dict], save_path: Path) -> None:
    """
    実際の NFE (UNet フォワード回数) vs PSNR の散布図。

    サンプラーごとに色分けし、ステップ数に応じてマーカーサイズを変える。
    """
    from samplers import NFE_PER_STEP  # import here to avoid circular

    fig, ax = plt.subplots(figsize=(9, 6))

    # カラーマップ
    cmap = plt.cm.tab10
    colors = {name: cmap(i / len(SAMPLER_ORDER)) for i, name in enumerate(SAMPLER_ORDER)}
    step_sizes = {10: 60, 20: 120, 50: 240}

    for sampler in SAMPLER_ORDER:
        nfe_vals, psnr_vals = [], []
        for steps in STEPS:
            entry = get_entry(results, sampler, steps)
            if entry and "metrics" in entry:
                psnr = entry["metrics"].get("psnr")
                if psnr is not None:
                    nfe = entry.get("nfe", steps)
                    nfe_vals.append(nfe)
                    psnr_vals.append(psnr)
                    ax.scatter(
                        nfe, psnr,
                        c=[colors[sampler]],
                        s=step_sizes[steps],
                        marker="o",
                        alpha=0.85,
                        label=None,
                    )
        if nfe_vals:
            # 折れ線でサンプラーごとに繋ぐ
            sorted_pairs = sorted(zip(nfe_vals, psnr_vals))
            ax.plot(
                [p[0] for p in sorted_pairs],
                [p[1] for p in sorted_pairs],
                color=colors[sampler],
                linewidth=1.2,
                alpha=0.6,
                label=SAMPLER_DISPLAY_NAMES.get(sampler, sampler),
            )

    ax.set_xlabel("NFE (UNet forward passes)")
    ax.set_ylabel("PSNR (dB) vs Reference (DDIM 250step)")
    ax.set_title("Quality (PSNR) vs Actual NFE — lower NFE preferred")
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# =============================================================================
# 5. DPM-Solver singlestep vs multistep 比較
# =============================================================================

def plot_dpm_compare(results: List[Dict], save_path: Path) -> None:
    """
    DPM-Solver order 2/3 の singlestep vs multistep を PSNR で比較する。
    """
    dpm_pairs = [
        ("dpm_solver_2_single", "dpm_solver_2_multi",  "DPM-Solver-2"),
        ("dpm_solver_3_single", "dpm_solver_3_multi",  "DPM-Solver-3"),
    ]
    x = np.arange(len(STEPS))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)

    for ax_idx, (single_key, multi_key, title_prefix) in enumerate(dpm_pairs):
        ax = axes[ax_idx]
        single_psnr = []
        multi_psnr  = []

        for steps in STEPS:
            single_entry = get_entry(results, single_key, steps)
            multi_entry  = get_entry(results, multi_key,  steps)

            s_val = (single_entry["metrics"].get("psnr") if single_entry and "metrics" in single_entry else None) or 0.0
            m_val = (multi_entry["metrics"].get("psnr")  if multi_entry  and "metrics" in multi_entry  else None) or 0.0

            single_psnr.append(s_val)
            multi_psnr.append(m_val)

        b1 = ax.bar(x - width / 2, single_psnr, width, label="Singlestep", color="#e15759", alpha=0.85)
        b2 = ax.bar(x + width / 2, multi_psnr,  width, label="Multistep",  color="#4e79a7", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([str(s) for s in STEPS])
        ax.set_xlabel("Steps")
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(f"{title_prefix}: Singlestep vs Multistep")
        ax.legend()

        # 注記: NFE は singlestep がより大きい
        nfe_note = f"NFE: Single={STEPS[0]*({'2':2,'3':3}[title_prefix[-1]])}x, Multi={STEPS[0]}x"
        ax.annotate(
            "※Singlestep: NFE/step は Multi より大きい",
            xy=(0.02, 0.02), xycoords="axes fraction", fontsize=8, color="gray"
        )

    plt.suptitle("DPM-Solver: Singlestep vs Multistep PSNR Comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[保存] {save_path}")


# =============================================================================
# メイン: 全プロット生成
# =============================================================================

def generate_all_plots(output_dir: Path) -> None:
    results_path = output_dir / "results.json"
    results = load_results(results_path)

    print("[可視化] グリッド画像を生成中...")
    plot_grid(results, output_dir, output_dir / "grid.png")

    print("[可視化] PSNR ヒートマップを生成中...")
    plot_heatmap(results, "psnr", "PSNR (dB) vs Reference (DDIM 250step)", output_dir / "heatmap_psnr.png", cmap="YlGn")

    print("[可視化] SSIM ヒートマップを生成中...")
    plot_heatmap(results, "ssim", "SSIM vs Reference (DDIM 250step)", output_dir / "heatmap_ssim.png", cmap="Blues")

    print("[可視化] 生成時間棒グラフを生成中...")
    plot_timing_bar(results, output_dir / "timing_bar.png")

    print("[可視化] NFE vs PSNR 散布図を生成中...")
    plot_nfe_scatter(results, output_dir / "nfe_scatter.png")

    print("[可視化] DPM-Solver singlestep vs multistep 比較を生成中...")
    plot_dpm_compare(results, output_dir / "dpm_compare.png")

    print("\n[完了] 全プロット生成が終了しました。")


# =============================================================================
# CLI エントリーポイント
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="拡散サンプラー比較 — 可視化")
    parser.add_argument("--output-dir", type=str, default="outputs")
    args = parser.parse_args()
    generate_all_plots(Path(args.output_dir))


if __name__ == "__main__":
    main()
