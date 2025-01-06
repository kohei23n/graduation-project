import pandas as pd

# データを読み込む
match_data_df = pd.read_csv("./csv/match_data.csv")

# クラス分布を確認
def check_class_imbalance(data, target_column):
    # ターゲット列のクラス分布をカウント
    class_counts = data[target_column].value_counts()
    total = len(data)
    
    print("クラス分布:")
    print(class_counts)
    print("\n各クラスの割合 (%):")
    print((class_counts / total * 100).round(2))


# 使用例
check_class_imbalance(match_data_df, "FTR")


# クラス分布:
# FTR
# H    1708
# A    1206
# D     886

# 各クラスの割合 (%):
# FTR
# H    44.95
# A    31.74
# D    23.32
