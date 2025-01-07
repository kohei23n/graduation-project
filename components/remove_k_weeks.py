# 各シーズンの最初の k*10 試合は IsPrediction=False を設定
def mark_prediction_flag(df, k):
    df = df.sort_values(by=["Season", "Date"]).copy()
    df["IsPrediction"] = df.groupby("Season").cumcount() >= (k * 10) # cumcount は0から始まるから、>= でOK
    return df