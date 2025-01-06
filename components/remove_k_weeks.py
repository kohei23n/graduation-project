# 各シーズンの最初の k 試合は IsPrediction=False を設定
def mark_prediction_flag(df, k):
    df = df.sort_values(by=["Season", "Date"]).copy()
    df["IsPrediction"] = df.groupby("Season").cumcount() >= k
    return df