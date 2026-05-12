# ddpm-comparison

Stable Diffusion の推論サンプラーを比較するための実験用プロジェクトです。`diffusers` のモデル部品を使いながら、DDPM / DDIM / Euler / Heun / LMS / DPM-Solver 系のサンプラーをこのリポジトリ内で実装し、同一プロンプト・同一 seed・同一モデル条件で生成画像、生成時間、NFE、品質指標を比較します。

デフォルトの生成プロンプトは `a photo of an astronaut riding a horse on mars`、モデルは `CompVis/stable-diffusion-v1-4`、画像サイズは 512 x 512 です。比較基準として DDIM 250 step のリファレンス画像も生成します。

## 目的

このプロジェクトは、拡散モデルのサンプリング手法ごとの違いを実験的に観察するためのものです。単に画像を生成するだけでなく、以下の観点で比較できるようになっています。

- ステップ数を変えたときの画質の変化
- サンプラーごとの生成時間
- NFE (Network Function Evaluations: UNet forward 回数) と品質の関係
- DDIM 250 step の参照画像に対する PSNR / SSIM
- プロンプトとの意味的整合性を測る CLIP Score
- 参考指標としての KID / FID

## 主な構成

- `experiment.py`: 比較実験の入口です。モデルをロードし、指定サンプラーとステップ数で画像を生成して `results.json` に実行結果を保存します。
- `pipeline.py`: Stable Diffusion の推論パイプラインです。UNet、VAE、CLIP tokenizer / text encoder をロードし、Classifier-Free Guidance とカスタムサンプラーを組み合わせて画像を生成します。
- `samplers/`: 各サンプラーの実装です。`BaseSampler` と `NoiseSchedule` を共通基盤として、DDPM、DDIM、Euler、Heun、LMS-2、DPM-Solver 系を実装しています。
- `metrics.py`: 生成画像に対して PSNR、SSIM、CLIP Score、KID、FID を計算し、`results.json` に追記します。
- `visualize.py`: 生成画像グリッド、品質ヒートマップ、生成時間グラフ、NFE と品質の散布図などを作成します。
- `analysis.ipynb`: 実験結果を対話的に確認するための Jupyter Notebook です。
- `environment.yml`: conda 環境と Python 依存関係の定義です。
- `setup.sh`: conda 環境作成を補助するセットアップスクリプトです。
- `outputs/`: 実験結果の出力先です。
- `outputs_bak/`: 過去の生成画像や `results.json` のバックアップです。

## 対応サンプラー

`samplers/__init__.py` の `SAMPLER_REGISTRY` で、次のサンプラーを名前から生成できます。

- `ddpm`
- `ddim`
- `euler`
- `heun`
- `lms2`
- `dpm_solver_1`
- `dpm_solver_2_single`
- `dpm_solver_2_multi`

`experiment.py` の `SAMPLER_NAMES` と `STEP_LIST_OVERRIDE` を編集すると、実験対象のサンプラーやステップ数を変更できます。現在の設定では `dpm_solver_1` を対象にし、10 / 20 / 50 step を実行する構成です。

## セットアップ

conda を使う場合は、次のコマンドで環境を作成します。

```bash
conda env create -f environment.yml
conda activate diffusion-sampler
```

または、付属スクリプトを使えます。

```bash
chmod +x setup.sh
./setup.sh
conda activate diffusion-sampler
```

`environment.yml` は CUDA 12.4 向けの PyTorch wheel を指定しています。Apple Silicon の MPS 環境や CPU 実行では、必要に応じて PyTorch のインストール方法を環境に合わせて調整してください。

## 実行方法

まず画像生成実験を実行します。

```bash
python experiment.py
```

出力先を変える場合は `--output-dir` を指定します。

```bash
python experiment.py --output-dir outputs
```

モデルロードだけを確認したい場合は dry run を使います。

```bash
python experiment.py --dry-run
```

画像生成後、メトリクスを計算します。

```bash
python metrics.py
```

最後に可視化画像を生成します。

```bash
python visualize.py
```

Notebook で分析する場合は、次のように起動します。

```bash
jupyter lab analysis.ipynb
```

## 出力

標準では `outputs/` に以下のようなファイルが保存されます。

- `reference.png`: DDIM 250 step で生成した比較基準画像
- `{sampler}_steps{steps}.png`: 各サンプラーとステップ数で生成した画像
- `results.json`: サンプラー名、ステップ数、NFE、生成時間、画像パス、メトリクスなどの実験結果
- `grid.png`: サンプラー別・ステップ数別の生成画像グリッド
- `heatmap_psnr.png`, `heatmap_ssim.png`: 品質指標のヒートマップ
- `timing_bar.png`: 生成時間の比較グラフ
- `nfe_scatter.png`: NFE と品質の関係を示す散布図
- `dpm_compare.png`: DPM-Solver 系の比較図

## 注意点

初回実行時は Hugging Face Hub から Stable Diffusion の重みをダウンロードします。モデルによっては利用規約への同意や認証が必要になる場合があります。

画像生成は GPU 環境を推奨します。CPU でも動作しますが、実験全体にはかなり時間がかかります。`experiment.py` のコメントでは、全サンプラーを広く実行する場合の目安として約 30 分程度が想定されています。

`pipeline.py` のコメントには Stable Diffusion 2.1 向けの説明も残っていますが、現在の実験設定では `experiment.py` の `MODEL_ID` により Stable Diffusion v1.4 (`CompVis/stable-diffusion-v1-4`) を使用します。モデルを切り替える場合は、prediction type やノイズスケジュールの前提も確認してください。
