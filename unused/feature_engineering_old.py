import numpy as np
import pandas as pd

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# -------------------------
# Ratings
# -------------------------


## Ratings データを 試合データーに結合する関数
def add_ratings(df, ratings_df):
    df = (
        df.merge(
            ratings_df, left_on=["HomeTeam", "Season"], right_on=["Team", "Season"]
        )
        .rename(
            columns={
                "ATT": "HomeAttackR",
                "MID": "HomeMidfieldR",
                "DEF": "HomeDefenceR",
                "OVR": "HomeOverallR",
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
                "ATT": "AwayAttackR",
                "MID": "AwayMidfieldR",
                "DEF": "AwayDefenceR",
                "OVR": "AwayOverallR",
            }
        )
        .drop(columns=["Team"])
    )
    return df


# -------------------------
# Elo Rating
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

            expected_home = 1 / (1 + c ** ((team_elo[away_team] - team_elo[home_team]) / d))
            expected_away = 1 - expected_home

            new_home_elo = team_elo[home_team] + k * (result_home - expected_home)
            new_away_elo = team_elo[away_team] + k * ((1 - result_home) - expected_away)

            team_elo[home_team] = new_home_elo
            team_elo[away_team] = new_away_elo

        df.loc[season_data.index, "HT_Elo"] = home_elo_ratings
        df.loc[season_data.index, "AT_Elo"] = away_elo_ratings

    return df


# -------------------------
# 直近 k 試合のゴール数、シュート数、枠内シュート数、得失点差、勝ち点を総合、ホーム、アウェイごとに計算
# -------------------------


# チームの直近 k 試合のデータを取得する関数
### g("Arsenal", "2014-08-20", df, k=2)
### ↓
###   Date           HomeTeam    AwayTeam    FTHG  FTAG  FTR   Season
###   2014-08-15     Chelsea     Arsenal     1     2     A     2014-15
###   2014-08-10     Arsenal     Liverpool   0     0     D     2014-15
def get_past_matches(team, date, df, k, is_home=None):
    ## 試合の日付に基づいて、シーズンを取得
    season = df.loc[df["Date"] == date, "Season"].values[0]
    ## 該当するシーズン内の、指定した試合より前の試合データを取得
    past_matches = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
        & (df["Date"] < date)
        & (df["Season"] == season)
    ].sort_values(by="Date", ascending=False)

    ## ホーム試合のみ、アウェイ試合のみに絞り込める is_home を定義（is_home=True か is_home=False を呼び出さない限り、特に関係ない）
    if is_home is not None:
        if is_home:
            past_matches = past_matches[past_matches["HomeTeam"] == team]
        else:
            past_matches = past_matches[past_matches["AwayTeam"] == team]

    ## 過去の試合のうち、直近 k 試合分のデータのみ取得
    return past_matches.head(k)


# 試合ごとにポイントを計算
### c("H", "home") or c("A", "away") → 3 / c("D", "away" or "home") → 1 / c("H", "away") or c("A", "home") → 0
def calculate_points(result, team_type):
    if result == "H" and team_type == "home":
        return 3
    if result == "A" and team_type == "away":
        return 3
    return 1 if result == "D" else 0


# 指定されたチームの直近 k 試合の平均ポイントを計算
###   Date           HomeTeam    AwayTeam    FTHG  FTAG  FTR   Season
###   2014-08-15     Chelsea     Arsenal     1     2     A     2014-15
###   2014-08-10     Arsenal     Liverpool   0     0     D     2014-15
### r("Arsenal", "2014-08-20", df, k=2)
### ↓
### ((3+1)/2=) 2
def recent_points(team, date, df, k, is_home=None):
    ## 直近 k 試合のデータを取得
    past_matches = get_past_matches(team, date, df, k, is_home)

    ## 直近の試合数が k 未満の場合は、NaN を返す
    if len(past_matches) < k:
        return np.nan

    ## calculate_points 関数を使って各試合のポイントを計算し、平均を取る
    points = past_matches.apply(
        lambda row: calculate_points(
            row["FTR"], "home" if row["HomeTeam"] == team else "away"
        ),
        axis=1,
    )

    return points.mean()


# ポイントに関する統計を計算し、データフレームに追加
def add_points_stats(df, k):
    team_stats = {
        "HT_RecentPoints": [],  # ホームチームの直近 k 試合の平均ポイント
        "HT_HomeRecentPoints": [],  # ホームチームの直近 k 試合の平均ポイント（ホーム試合のみ）
        "HT_AwayRecentPoints": [],  # ホームチームの直近 k 試合の平均ポイント（アウェイ試合のみ）
        "HT_TotalPoints": [],  # ホームチームの累積ポイント
        "AT_RecentPoints": [],  # アウェイチームの直近 k 試合の平均ポイント
        "AT_HomeRecentPoints": [],  # アウェイチームの直近 k 試合の平均ポイント（ホーム試合のみ）
        "AT_AwayRecentPoints": [],  # アウェイチームの直近 k 試合の平均ポイント（アウェイ試合のみ）
        "AT_TotalPoints": [],  # アウェイチームの累積ポイント
    }
    team_total_points = {}

    ## 各シーズンごとにループ
    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])
        team_total_points.update({team: 0 for team in teams})

        ## 各試合ごとにループ
        for _, row in season_data.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            date = row["Date"]

            ## ポイントの計算
            team_stats["HT_RecentPoints"].append(recent_points(home_team, date, df, k))
            team_stats["HT_HomeRecentPoints"].append(
                recent_points(home_team, date, df, k, is_home=True)
            )
            team_stats["HT_AwayRecentPoints"].append(
                recent_points(home_team, date, df, k, is_home=False)
            )

            team_stats["AT_RecentPoints"].append(recent_points(away_team, date, df, k))
            team_stats["AT_HomeRecentPoints"].append(
                recent_points(away_team, date, df, k, is_home=True)
            )
            team_stats["AT_AwayRecentPoints"].append(
                recent_points(away_team, date, df, k, is_home=False)
            )

            ## 累積ポイントを計算
            team_stats["HT_TotalPoints"].append(team_total_points[home_team])
            team_stats["AT_TotalPoints"].append(team_total_points[away_team])
            team_total_points[home_team] += calculate_points(row["FTR"], "home")
            team_total_points[away_team] += calculate_points(row["FTR"], "away")

    for col, values in team_stats.items():
        df[col] = values
    return df


# 指定されたチームの直近 k 試合のゴール数と得失点差を計算
def recent_goals_stats(team, date, df, k, is_home=None):
    past_matches = get_past_matches(team, date, df, k, is_home)

    if len(past_matches) < k:
        return np.nan, np.nan

    if is_home is True:  # ホーム試合のゴール平均
        goals = past_matches["FTHG"].mean()
    elif is_home is False:  # アウェイ試合のゴール平均
        goals = past_matches["FTAG"].mean()
    else:  # 全試合
        goals = (past_matches["FTHG"].sum() + past_matches["FTAG"].sum()) / len(
            past_matches
        )

    if is_home is True:  # ホーム試合の得失点差平均
        gd = past_matches["FTHG"].sub(past_matches["FTAG"]).mean()
    elif is_home is False:  # アウェイ試合の得失点差平均
        gd = past_matches["FTAG"].sub(past_matches["FTHG"]).mean()
    else:  # 全試合
        gd = (past_matches["FTHG"].sum() - past_matches["FTAG"].sum()) / len(
            past_matches
        )

    return goals, gd


# ゴールと得失点差に関する統計を計算し、データフレームに追加
def add_goals_stats(df, k):
    team_stats = {
        "HT_RecentGoals": [],  # ホームチームの直近 k 試合の平均ゴール数
        "HT_HomeRecentGoals": [],  # ホームチームの直近 k 試合の平均ゴール数（ホーム試合のみ）
        "HT_AwayRecentGoals": [],  # ホームチームの直近 k 試合の平均ゴール数（アウェイ試合のみ）
        "HT_RecentGD": [],  # ホームチームの直近 k 試合の平均得失点差
        "HT_HomeRecentGD": [],  # ホームチームの直近 k 試合の平均得失点差（ホーム試合のみ）
        "HT_AwayRecentGD": [],  # ホームチームの直近 k 試合の平均得失点差（アウェイ試合のみ）
        "HT_TotalGoals": [],  # ホームチームの累積ゴール数
        "HT_TotalGD": [],  # ホームチームの累積得失点差
        "AT_RecentGoals": [],  # アウェイチームの直近 k 試合の平均ゴール数
        "AT_HomeRecentGoals": [],  # アウェイチームの直近 k 試合の平均ゴール数（ホーム試合のみ）
        "AT_AwayRecentGoals": [],  # アウェイチームの直近 k 試合の平均ゴール数（アウェイ試合のみ）
        "AT_RecentGD": [],  # アウェイチームの直近 k 試合の平均得失点差
        "AT_HomeRecentGD": [],  # アウェイチームの直近 k 試合の平均得失点差（ホーム試合のみ）
        "AT_AwayRecentGD": [],  # アウェイチームの直近 k 試合の平均得失点差（アウェイ試合のみ）
        "AT_TotalGoals": [],  # アウェイチームの累積ゴール数
        "AT_TotalGD": [],  # アウェイチームの累積得失点差
    }

    team_total_goals = {}  # 累積ゴール数
    team_total_gd = {}  # 累積得失点差

    ## 各シーズンごとにループ
    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])
        team_total_goals.update({team: 0 for team in teams})
        team_total_gd.update({team: 0 for team in teams})

        ## 各試合ごとにループ
        for _, row in season_data.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            date = row["Date"]

            ## ホームチームの直近 k 試合の統計を計算（ホームの成績、アウェイの成績、全体の成績）
            home_goals_home, home_gd_home = recent_goals_stats(
                home_team, date, df, k, is_home=True
            )
            home_goals_away, home_gd_away = recent_goals_stats(
                home_team, date, df, k, is_home=False
            )
            all_home_goals, all_home_gd = recent_goals_stats(home_team, date, df, k)

            team_stats["HT_HomeRecentGoals"].append(home_goals_home)
            team_stats["HT_AwayRecentGoals"].append(home_goals_away)
            team_stats["HT_RecentGoals"].append(all_home_goals)
            team_stats["HT_HomeRecentGD"].append(home_gd_home)
            team_stats["HT_AwayRecentGD"].append(home_gd_away)
            team_stats["HT_RecentGD"].append(all_home_gd)

            ## アウェイチームの直近 k 試合の統計を計算（ホームの成績、アウェイの成績、全体の成績）
            away_goals_home, away_gd_home = recent_goals_stats(
                away_team, date, df, k, is_home=True
            )
            away_goals_away, away_gd_away = recent_goals_stats(
                away_team, date, df, k, is_home=False
            )
            all_away_goals, all_away_gd = recent_goals_stats(away_team, date, df, k)

            team_stats["AT_HomeRecentGoals"].append(away_goals_home)
            team_stats["AT_AwayRecentGoals"].append(away_goals_away)
            team_stats["AT_RecentGoals"].append(all_away_goals)
            team_stats["AT_HomeRecentGD"].append(away_gd_home)
            team_stats["AT_AwayRecentGD"].append(away_gd_away)
            team_stats["AT_RecentGD"].append(all_away_gd)

            ## ホームチームの累積ゴール数と累積得失点差を計算
            team_stats["HT_TotalGoals"].append(team_total_goals[home_team])
            team_stats["HT_TotalGD"].append(team_total_gd[home_team])
            team_total_goals[home_team] += row["FTHG"]
            team_total_gd[home_team] += row["FTHG"] - row["FTAG"]

            ## アウェイチームの累積ゴール数と累積得失点差を計算
            team_stats["AT_TotalGoals"].append(team_total_goals[away_team])
            team_stats["AT_TotalGD"].append(team_total_gd[away_team])
            team_total_goals[away_team] += row["FTAG"]
            team_total_gd[away_team] += row["FTAG"] - row["FTHG"]

    # データフレームに統計を追加
    for col, values in team_stats.items():
        df[col] = values

    return df


# 指定されたチームの直近 k 試合のシュート数と枠内シュート数を計算
def recent_shots_stats(team, date, df, k, is_home=None):
    """直近k試合のデータを取得し、シュート数と枠内シュート数を計算"""
    past_matches = get_past_matches(team, date, df, k, is_home)

    if len(past_matches) < k:
        return np.nan, np.nan

    if is_home is True:  # ホーム試合限定
        shots = past_matches["HS"].mean()
        sot = past_matches["HST"].mean()
    elif is_home is False:  # アウェイ試合限定
        shots = past_matches["AS"].mean()
        sot = past_matches["AST"].mean()
    else:  # 全試合
        shots = (past_matches["HS"].sum() + past_matches["AS"].sum()) / len(
            past_matches
        )
        sot = (past_matches["HST"].sum() + past_matches["AST"].sum()) / len(
            past_matches
        )

    return shots, sot


# シュート数と枠内シュート数に関する統計を計算
def add_shots_stats(df, k):
    team_stats = {
        "HT_RecentShots": [],  # ホームチームの直近 k 試合の平均シュート数
        "HT_HomeRecentShots": [],  # ホームチームの直近 k 試合の平均シュート数（ホーム試合のみ）
        "HT_AwayRecentShots": [],  # ホームチームの直近 k 試合の平均シュート数（アウェイ試合のみ）
        "HT_RecentSOT": [],  # ホームチームの直近 k 試合の平均枠内シュート数
        "HT_HomeRecentSOT": [],  # ホームチームの直近 k 試合の平均枠内シュート数（ホーム試合のみ）
        "HT_AwayRecentSOT": [],  # ホームチームの直近 k 試合の平均枠内シュート数（アウェイ試合のみ）
        "HT_TotalShots": [],  # ホームチームの累積シュート数
        "HT_TotalSOT": [],  # ホームチームの累積枠内シュート数
        "AT_RecentShots": [],  # アウェイチームの直近 k 試合の平均シュート数
        "AT_HomeRecentShots": [],  # アウェイチームの直近 k 試合の平均シュート数（ホーム試合のみ）
        "AT_AwayRecentShots": [],  # アウェイチームの直近 k 試合の平均シュート数（アウェイ試合のみ）
        "AT_RecentSOT": [],  # アウェイチームの直近 k 試合の平均枠内シュート数
        "AT_HomeRecentSOT": [],  # アウェイチームの直近 k 試合の平均枠内シュート数（ホーム試合のみ）
        "AT_AwayRecentSOT": [],  # アウェイチームの直近 k 試合の平均枠内シュート数（アウェイ試合のみ）
        "AT_TotalShots": [],  # アウェイチームの累積シュート数
        "AT_TotalSOT": [],  # アウェイチームの累積枠内シュート数
    }

    team_total_shots = {}  # 累積シュート数
    team_total_sot = {}  # 累積枠内シュート数

    ## シーズンごとのループ
    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])

        ## チームごとの累積データを初期化
        team_total_shots.update({team: 0 for team in teams})
        team_total_sot.update({team: 0 for team in teams})

        ## 試合ごとのループ
        for _, row in season_data.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            date = row["Date"]

            # ホームチームの直近 k 試合の統計を計算
            home_shots_home, home_sot_home = recent_shots_stats(
                home_team, date, df, k, is_home=True
            )
            home_shots_away, home_sot_away = recent_shots_stats(
                home_team, date, df, k, is_home=False
            )
            all_home_shots, all_home_sot = recent_shots_stats(home_team, date, df, k)

            team_stats["HT_HomeRecentShots"].append(home_shots_home)
            team_stats["HT_AwayRecentShots"].append(home_shots_away)
            team_stats["HT_RecentShots"].append(all_home_shots)
            team_stats["HT_HomeRecentSOT"].append(home_sot_home)
            team_stats["HT_AwayRecentSOT"].append(home_sot_away)
            team_stats["HT_RecentSOT"].append(all_home_sot)

            # アウェイチームの直近 k 試合の統計を計算
            away_shots_home, away_sot_home = recent_shots_stats(
                away_team, date, df, k, is_home=True
            )
            away_shots_away, away_sot_away = recent_shots_stats(
                away_team, date, df, k, is_home=False
            )
            all_away_shots, all_away_sot = recent_shots_stats(away_team, date, df, k)

            team_stats["AT_HomeRecentShots"].append(away_shots_home)
            team_stats["AT_AwayRecentShots"].append(away_shots_away)
            team_stats["AT_RecentShots"].append(all_away_shots)
            team_stats["AT_HomeRecentSOT"].append(away_sot_home)
            team_stats["AT_AwayRecentSOT"].append(away_sot_away)
            team_stats["AT_RecentSOT"].append(all_away_sot)

            ## ホームチームの累積データを計算
            team_stats["HT_TotalShots"].append(team_total_shots[home_team])
            team_stats["HT_TotalSOT"].append(team_total_sot[home_team])
            team_total_shots[home_team] += row["HS"]
            team_total_sot[home_team] += row["HST"]

            ## アウェイチームの累積データを計算
            team_stats["AT_TotalShots"].append(team_total_shots[away_team])
            team_stats["AT_TotalSOT"].append(team_total_sot[away_team])
            team_total_shots[away_team] += row["AS"]
            team_total_sot[away_team] += row["AST"]

    # 統計をデータフレームに追加
    for col, values in team_stats.items():
        df[col] = values

    return df


# それぞれの一括で統計量を計算し、まとめてデータフレームに追加
def add_team_stats(df, ratings_df, k):
    df = add_ratings(df, ratings_df)
    df = add_elo_rating(df)
    df = add_points_stats(df, k)
    df = add_goals_stats(df, k)
    df = add_shots_stats(df, k)
    return df
