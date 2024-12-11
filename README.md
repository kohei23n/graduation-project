# 全体の流れ

## データの準備

- `combine_data.py`：10年分の試合データがそれぞれ別のCSVに保存されているため、`./csv/match_data.csv` に統合
- `scraper.py`：FIFA Ratings をページからスクレイピングし、`./csv/ratings_data.csv` に保存

## 特徴量エンジニアリング

- `feature_engineering.py`：特徴量エンジニアリングの関数を定義

### ランダムフォレスト

- `rf_opt_k_gamma.py`：`feature_engineering.py` を基に特徴量エンジニアリングを実行しながら k と gamma を最適化し、最終的な特徴量を生成
- `rf.py`：`rf_opt_k_gamma.py` で生成した特長量を基にハイパーパラメータチューニングし、その後モデルを訓練・テスト
  - ハイパーパラメータチューニングに使うのは `rf_hyperparameter_tuning.py`
  - 評価に使うのは `model_evaluation.py` で定義した RPS
