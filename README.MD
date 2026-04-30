conda env create -f environment.yml && conda activate diffusion-sampler
python experiment.py          # 画像生成 (~30 分, outputs/ に保存)
python metrics.py             # メトリクス計算・results.json 更新
python visualize.py           # プロット生成 (grid.png 等)
jupyter lab analysis.ipynb    # 対話的分析