import logging
import pandas as pd

# 進捗状況を表示するための設定
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

# データの読み込みと準備
logging.info("Loading and preparing data...")
match_data_df = pd.read_csv("./csv/match_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)
ratings_df = pd.read_csv("./csv/ratings_data.csv")
xg_df = pd.read_csv("./csv/xg_data.csv")
logging.info("Data loaded successfully.")

# -------------------------
# 1. Add Ratings
# -------------------------


# Ratings を試合データに結合する関数
def add_ratings(df, ratings_df):
    df = (
        df.merge(
            ratings_df, left_on=["HomeTeam", "Season"], right_on=["Team", "Season"]
        )
        .rename(
            columns={
                "ATT": "HT_AttackR",
                "MID": "HT_MidfieldR",
                "DEF": "HT_DefenceR",
                "OVR": "HT_OverallR",
            }
        )
        .drop(columns=["Team"])
    )
    df = (
        df.merge(
            ratings_df, left_on=["AwayTeam", "Season"], right_on=["Team", "Season"]
        )
        .rename(
            columns={
                "ATT": "AT_AttackR",
                "MID": "AT_MidfieldR",
                "DEF": "AT_DefenceR",
                "OVR": "AT_OverallR",
            }
        )
        .drop(columns=["Team"])
    )
    return df


logging.info("Adding ratings to match data...")
match_data_df = add_ratings(match_data_df, ratings_df)
logging.info("Ratings added successfully.")

# -------------------------
# 2. Add xG
# -------------------------


# XGを足してみる（テスト！）
def add_xg(df, xg_df):
    # HomeTeam に対するマージ
    df = df.merge(
        xg_df[["HomeTeam", "HomeXG", "Season"]], on=["HomeTeam", "Season"], how="left"
    )

    # AwayTeam に対するマージ
    df = df.merge(
        xg_df[["AwayTeam", "AwayXG", "Season"]], on=["AwayTeam", "Season"], how="left"
    )
    return df


logging.info("Adding xG to match data...")
match_data_df = add_xg(match_data_df, xg_df)
logging.info("xG added successfully.")

# 不要なカラムを削除
logging.info("Removing unnecessary columns...")
unnecessary_columns = ["HomeXG", "AwayXG"]

required_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "FTR",
    "FTHG",
    "FTAG",
    "HS",
    "AS",
    "HST",
    "AST",
    "HT_AttackR",
    "HT_MidfieldR",
    "HT_DefenceR",
    "HT_OverallR",
    "AT_AttackR",
    "AT_MidfieldR",
    "AT_DefenceR",
    "AT_OverallR",
    "B365H",
    "B365D",
    "B365A",
]
match_data_df = match_data_df[required_columns]
logging.info("Columns removed successfully.")

# 最終データを保存（CSV, HTML）
logging.info("Saving data to CSV Files...")
match_data_df.to_csv("./csv/final_match_data.csv", index=False)
logging.info("Data saved successfully!")
