import pandas as pd


# データの読み込みと準備
def load_and_format_date(file_path):
    match_data_df = pd.read_csv(file_path)
    match_data_df["Date"] = pd.to_datetime(match_data_df["Date"], format="%d/%m/%Y", dayfirst=True)

    return match_data_df


# 訓練データとテストデータに分割
def split_data(match_data_df):
    train_data = match_data_df[
        match_data_df["Season"].isin(match_data_df["Season"].unique()[:-2])
    ].copy()
    test_data = match_data_df[
        match_data_df["Season"].isin(match_data_df["Season"].unique()[-2:])
    ].copy()
    
    return train_data, test_data


# 各シーズンの最初の k*10 試合は除外
def remove_first_k_gameweeks(df, k):
    df = df.sort_values(by=["Season", "Date"]).copy()
    # シーズンごとの最初の k 試合を除外
    df = df[df.groupby("Season").cumcount() >= (k * 10)]

    return df


features = [
    # Ability
    "HT_AttackR",
    "HT_MidfieldR",
    "HT_DefenceR",
    "HT_OverallR",
    "AT_AttackR",
    "AT_MidfieldR",
    "AT_DefenceR",
    "AT_OverallR",
    "HT_AvgPPG",
    "AT_AvgPPG",
    "HT_AvgShots",
    "HT_AvgSOT",
    "HT_AvgShotsConceded",
    "HT_AvgSOTConceded",
    "AT_AvgShots",
    "AT_AvgSOT",
    "AT_AvgShotsConceded",
    "AT_AvgSOTConceded",
    # Home Advantage
    "HT_HomeWinRate",
    "HT_HomeDrawRate",
    "HT_HomeLossRate",
    "AT_AwayWinRate",
    "AT_AwayDrawRate",
    "AT_AwayLossRate",
    # Recent Performance
    "HT_RecentShots",
    "HT_RecentSOT",
    "HT_RecentShotsConceded",
    "HT_RecentSOTConceded",
    "AT_RecentShots",
    "AT_RecentSOT",
    "AT_RecentShotsConceded",
    "AT_RecentSOTConceded",
    "HT_RecentPPG",
    "AT_RecentPPG",
    # Elo
    "HT_Elo",
    "AT_Elo",
    # Betting Odds
    "B365H",
    "B365D",
    "B365A",
    # xG
    "HT_RecentXG",
    "AT_RecentXG",
    "HT_RecentXGConceded",
    "AT_RecentXGConceded",
    "HT_AvgXG",
    "AT_AvgXG",
    "HT_AvgXGConceded",
    "AT_AvgXGConceded",
]
