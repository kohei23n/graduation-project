import numpy as np
import pandas as pd
from components.common import load_and_prepare_data

# データの読み込みと準備
match_data_df, ratings_df = load_and_prepare_data(
    "./csv/match_data.csv", "./csv/ratings_data.csv"
)

# -------------------------
# 1. the different abilities of both teams (FIFA Ratings, Average Points)
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


# Average Points


## チームの過去の試合データを取得する関数
def get_past_matches(team, date, df, k=None):
    # 試合の日付に基づいて、シーズンを取得
    season = df.loc[df["Date"] == date, "Season"].values[0]

    # 該当するシーズン内の、指定した試合より前の試合データを取得
    past_matches = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
        & (df["Date"] < date)
        & (df["Season"] == season)
    ].sort_values(by="Date", ascending=False)

    # k が指定されていない場合、全試合を返す
    if k is None:
        return past_matches

    # past matches のうち、直近 k 試合分のデータのみ取得
    return past_matches.head(k)


## 試合結果をポイントに変換する関数
### c("H", "home") or c("A", "away") → 3 / c("D", "away" or "home") → 1 / c("H", "away") or c("A", "home") → 0
def calc_points(result, team_type):
    if result == "H" and team_type == "home":
        return 3
    if result == "A" and team_type == "away":
        return 3
    return 1 if result == "D" else 0


def calc_points_stats(team, date, df, k=None):
    past_matches = get_past_matches(team, date, df, k=k)

    if past_matches.empty or (k is not None and len(past_matches) < k):
        return np.nan

    points = []

    for _, match in past_matches.iterrows():
        if match["HomeTeam"] == team:
            points.append(calc_points(match["FTR"], "home"))
        else:
            points.append(calc_points(match["FTR"], "away"))

    return np.mean(points)


## 過去k試合の平均ポイントを計算する関数
def add_recent_ppg_stats(df, k):
    df["HT_RecentPPG"] = 0.0
    df["AT_RecentPPG"] = 0.0

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームの直近 k 試合の平均ポイント
        df.at[_, "HT_RecentPPG"] = calc_points_stats(home_team, date, df, k)

        # アウェイチームの直近 k 試合の平均ポイント
        df.at[_, "AT_RecentPPG"] = calc_points_stats(away_team, date, df, k)

    return df


## シーズンごとの累積平均ポイントを計算する関数
def add_avg_ppg_stats(df):
    df["HT_AvgPPG"] = 0.0
    df["AT_AvgPPG"] = 0.0

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームの累積平均ポイント
        df.at[_, "HT_AvgPPG"] = calc_points_stats(home_team, date, df, k=None)

        # アウェイチームの累積平均ポイント
        df.at[_, "AT_AvgPPG"] = calc_points_stats(away_team, date, df, k=None)

    return df


# -------------------------
# 2. home advantage (W, D, L %)
# -------------------------


## シーズンごとの累積平均ポイントを計算する関数
def add_wdl_rates(df):
    df["HT_HomeWinRate"] = 0.0
    df["HT_HomeDrawRate"] = 0.0
    df["HT_HomeLossRate"] = 0.0
    df["AT_AwayWinRate"] = 0.0
    df["AT_AwayDrawRate"] = 0.0
    df["AT_AwayLossRate"] = 0.0

    # シーズンごとにデータを処理
    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])

        # チームごとの累積試合結果（ホーム、アウェイ別）を初期化
        team_home_stats = {
            team: {"W": 0, "D": 0, "L": 0, "Matches": 0} for team in teams
        }
        team_away_stats = {
            team: {"W": 0, "D": 0, "L": 0, "Matches": 0} for team in teams
        }

        # 各試合を順に処理
        for idx, row in season_data.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            result = row["FTR"]

            # ホームチームの勝率、引き分け率、負け率を計算（試合前の値を使用）
            matches = team_home_stats[home_team]["Matches"]
            if matches > 0:
                df.at[idx, "HT_HomeWinRate"] = team_home_stats[home_team]["W"] / matches
                df.at[idx, "HT_HomeDrawRate"] = (
                    team_home_stats[home_team]["D"] / matches
                )
                df.at[idx, "HT_HomeLossRate"] = (
                    team_home_stats[home_team]["L"] / matches
                )

            # アウェイチームの勝率、引き分け率、負け率を計算（試合前の値を使用）
            matches = team_away_stats[away_team]["Matches"]
            if matches > 0:
                df.at[idx, "AT_AwayWinRate"] = team_away_stats[away_team]["W"] / matches
                df.at[idx, "AT_AwayDrawRate"] = (
                    team_away_stats[away_team]["D"] / matches
                )
                df.at[idx, "AT_AwayLossRate"] = (
                    team_away_stats[away_team]["L"] / matches
                )

            # 試合後の結果を累積に更新
            if result == "H":  # ホーム勝ち
                team_home_stats[home_team]["W"] += 1
                team_away_stats[away_team]["L"] += 1
            elif result == "A":  # アウェイ勝ち
                team_home_stats[home_team]["L"] += 1
                team_away_stats[away_team]["W"] += 1
            else:  # 引き分け
                team_home_stats[home_team]["D"] += 1
                team_away_stats[away_team]["D"] += 1

            # 試合数を更新
            team_home_stats[home_team]["Matches"] += 1
            team_away_stats[away_team]["Matches"] += 1

    return df


# -------------------------
# 3. recent performance (直近 k 試合の Shots, SOT, Shots Conceded, SOT Conceded)
# -------------------------


## 指定されたチームの直近 k 試合の Shots, SOT, Shots Conceded, SOT Conceded を計算する関数
def calc_shots_stats(team, date, df, k=None):
    past_matches = get_past_matches(team, date, df, k)

    if past_matches.empty or (k is not None and len(past_matches) < k):
        return np.nan, np.nan, np.nan, np.nan

    shots = []
    sot = []
    shots_conceded = []
    sot_conceded = []

    for _, match in past_matches.iterrows():
        if match["HomeTeam"] == team:
            shots.append(match["HS"])
            sot.append(match["HST"])
            shots_conceded.append(match["AS"])
            sot_conceded.append(match["AST"])
        else:
            shots.append(match["AS"])
            sot.append(match["AST"])
            shots_conceded.append(match["HS"])
            sot_conceded.append(match["HST"])

    return np.mean(shots), np.mean(sot), np.mean(shots_conceded), np.mean(sot_conceded)


## データフレームにShots, SOT, Shots Conceded, SOT Concededに関する統計を追加。
def add_recent_shots_stats(df, k):
    team_stats = {
        "HT_RecentShots": [],
        "HT_RecentSOT": [],
        "HT_RecentShotsConceded": [],
        "HT_RecentSOTConceded": [],
        "AT_RecentShots": [],
        "AT_RecentSOT": [],
        "AT_RecentShotsConceded": [],
        "AT_RecentSOTConceded": [],
    }

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームの統計
        ht_shots, ht_sot, ht_shots_conceded, ht_sot_conceded = calc_shots_stats(
            home_team, date, df, k
        )
        team_stats["HT_RecentShots"].append(ht_shots)
        team_stats["HT_RecentSOT"].append(ht_sot)
        team_stats["HT_RecentShotsConceded"].append(ht_shots_conceded)
        team_stats["HT_RecentSOTConceded"].append(ht_sot_conceded)

        # アウェイチームの統計
        at_shots, at_sot, at_shots_conceded, at_sot_conceded = calc_shots_stats(
            away_team, date, df, k
        )
        team_stats["AT_RecentShots"].append(at_shots)
        team_stats["AT_RecentSOT"].append(at_sot)
        team_stats["AT_RecentShotsConceded"].append(at_shots_conceded)
        team_stats["AT_RecentSOTConceded"].append(at_sot_conceded)

    for col, values in team_stats.items():
        df[col] = values

    return df


# AVG Shots, SOT, Shots Conceded, SOT Conceded in Season
def add_avg_shots_stats(df):
    team_stats = {
        "HT_AvgShots": [],
        "HT_AvgSOT": [],
        "HT_AvgShotsConceded": [],
        "HT_AvgSOTConceded": [],
        "AT_AvgShots": [],
        "AT_AvgSOT": [],
        "AT_AvgShotsConceded": [],
        "AT_AvgSOTConceded": [],
    }

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームの全試合平均
        ht_avg_shots, ht_avg_sot, ht_avg_shots_conceded, ht_avg_sot_conceded = (
            calc_shots_stats(home_team, date, df, k=None)
        )
        team_stats["HT_AvgShots"].append(ht_avg_shots)
        team_stats["HT_AvgSOT"].append(ht_avg_sot)
        team_stats["HT_AvgShotsConceded"].append(ht_avg_shots_conceded)
        team_stats["HT_AvgSOTConceded"].append(ht_avg_sot_conceded)

        # アウェイチームの全試合平均
        at_avg_shots, at_avg_sot, at_avg_shots_conceded, at_avg_sot_conceded = (
            calc_shots_stats(away_team, date, df, k=None)
        )
        team_stats["AT_AvgShots"].append(at_avg_shots)
        team_stats["AT_AvgSOT"].append(at_avg_sot)
        team_stats["AT_AvgShotsConceded"].append(at_avg_shots_conceded)
        team_stats["AT_AvgSOTConceded"].append(at_avg_sot_conceded)

    for col, values in team_stats.items():
        df[col] = values

    return df


# -------------------------
# 4. ability of opposition (Elo Ratings)
# -------------------------


def add_elo_rating(df, initial_rating=1000, k=20, c=10, d=400):

    df["HT_Elo"] = 0.0
    df["AT_Elo"] = 0.0

    ## シーズンごとにループ
    for season in df["Season"].unique():
        # シーズンごとのデータを取得
        season_data = df[df["Season"] == season]

        # シーズン内のチームとその初期レーティングを設定
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])
        team_elo = {team: initial_rating for team in teams}

        home_elo_ratings, away_elo_ratings = [], []

        # 試合ごとにEloを計算
        for idx, row in season_data.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            result = row["FTR"]

            home_elo_ratings.append(float(team_elo[home_team]))
            away_elo_ratings.append(float(team_elo[away_team]))

            if result == "H":
                result_home = 1
            elif result == "D":
                result_home = 0.5
            else:
                result_home = 0

            expected_home = 1 / (
                1 + c ** ((team_elo[away_team] - team_elo[home_team]) / d)
            )
            expected_away = 1 - expected_home

            new_home_elo = team_elo[home_team] + k * (result_home - expected_home)
            new_away_elo = team_elo[away_team] + k * ((1 - result_home) - expected_away)

            team_elo[home_team] = new_home_elo
            team_elo[away_team] = new_away_elo

        df.loc[season_data.index, "HT_Elo"] = home_elo_ratings
        df.loc[season_data.index, "AT_Elo"] = away_elo_ratings

    return df


# -------------------------
# 5. combine all features into a single function
# -------------------------


def add_team_stats(df, ratings_df, k):
    # ability
    df = add_ratings(df, ratings_df)
    df = add_avg_ppg_stats(df)
    df = add_avg_shots_stats(df)
    # home advantage
    df = add_wdl_rates(df)
    # recent performance
    df = add_recent_shots_stats(df, k)
    df = add_recent_ppg_stats(df, k)
    # ability of opposition
    df = add_elo_rating(df)
    return df
