import numpy as np
import pandas as pd

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# データ加工1：Form


## Formを 計算・更新する関数
def calculate_new_form(home_team, away_team, result, gamma, team_form, season):
    form_home = team_form[home_team]
    form_away = team_form[away_team]

    if result == "H":
        new_form_home = form_home + gamma * form_away
        new_form_away = form_away - gamma * form_away
    elif result == "A":
        new_form_home = form_home - gamma * form_home
        new_form_away = form_away + gamma * form_home
    else:
        form_diff = form_home - form_away
        new_form_home = form_home - gamma * form_diff
        new_form_away = form_away + gamma * form_diff

    return new_form_home, new_form_away


## Form の更新を試合ごとに適用
def add_form(df, gamma, teams):
    home_forms, away_forms = [], []
    current_season = None
    team_form = {team: 1.0 for team in teams}

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        result = row["FTR"]
        season = row["Season"]

        if season != current_season:
            current_season = season
            team_form = {team: 1.0 for team in teams}

        home_forms.append(team_form[home_team])
        away_forms.append(team_form[away_team])

        new_home_form, new_away_form = calculate_new_form(
            home_team, away_team, result, gamma, team_form, season
        )
        team_form[home_team] = new_home_form
        team_form[away_team] = new_away_form

    df["HomeForm"] = pd.Series(dtype="float64")
    df["AwayForm"] = pd.Series(dtype="float64")

    df.loc[:, "HomeForm"] = home_forms
    df.loc[:, "AwayForm"] = away_forms
    return df


# データ加工2： Streak, Weighted Streak


## 試合結果を数値に変換する関数
def calculate_points(result, team_type):
    if result == "H" and team_type == "home":
        return 3  # ホーム勝利
    elif result == "A" and team_type == "away":
        return 3  # アウェイ勝利
    elif result == "D":
        return 1  # 引き分け
    else:
        return 0  # 敗北


# シーズン累積ポイントとRecent Points (Weighted含む) 計算
def add_points(df, k):
    df = df.copy()
    team_total_points = {}

    home_recent_points, away_recent_points = [], []
    home_weighted_recent_points, away_weighted_recent_points = [], []
    home_total_points, away_total_points = [], []

    for season in df["Season"].unique():
        season_matches = df[df["Season"] == season]
        teams = set(season_matches["HomeTeam"]).union(season_matches["AwayTeam"])

        # 初期化: 各チームの累積ポイント
        team_total_points = {team: 0 for team in teams}

        for idx, row in season_matches.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            date = row["Date"]

            # 直近k試合のポイント取得
            def recent_points(team, team_type):
                past_matches = (
                    season_matches[
                        (
                            (season_matches["HomeTeam"] == team)
                            | (season_matches["AwayTeam"] == team)
                        )
                        & (season_matches["Date"] < date)
                    ]
                    .sort_values(by="Date", ascending=False)
                    .head(k)
                )

                if len(past_matches) < k:
                    return np.nan, np.nan

                points = [
                    calculate_points(
                        match["FTR"], "home" if match["HomeTeam"] == team else "away"
                    )
                    for _, match in past_matches.iterrows()
                ]

                recent = sum(points) / (3 * k)
                weighted_recent = sum((i + 1) * points[i] for i in range(k)) / (
                    3 * k * (k + 1)
                )
                return recent, weighted_recent

            # 直近ポイント
            home_recent, home_weighted = recent_points(home_team, "home")
            away_recent, away_weighted = recent_points(away_team, "away")

            home_recent_points.append(home_recent)
            home_weighted_recent_points.append(home_weighted)
            away_recent_points.append(away_recent)
            away_weighted_recent_points.append(away_weighted)

            # 累積ポイント
            home_points = calculate_points(row["FTR"], "home")
            away_points = calculate_points(row["FTR"], "away")

            home_total_points.append(team_total_points[home_team])
            away_total_points.append(team_total_points[away_team])

            team_total_points[home_team] += home_points
            team_total_points[away_team] += away_points

    # データフレームに追加
    df["HomeRecentPoints"] = home_recent_points
    df["AwayRecentPoints"] = away_recent_points
    df["HomeWeightedRecentPoints"] = home_weighted_recent_points
    df["AwayWeightedRecentPoints"] = away_weighted_recent_points
    df["HomeTotalPoints"] = home_total_points
    df["AwayTotalPoints"] = away_total_points

    return df


# データ加工3: "Past k..." データの追加


## 過去のパフォーマンス指標を取得する関数
def get_past_performance(team_name, specified_date, df, k):
    season = df.loc[df["Date"] == specified_date, "Season"].values[0]
    past_matches = df[
        ((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name))
        & (df["Date"] < specified_date)
        & (df["Season"] == season)
    ]
    past_matches = past_matches.sort_values(by="Date", ascending=False).head(k)

    if len(past_matches) < k:
        return np.nan, np.nan, np.nan

    total_goals = np.where(
        past_matches["HomeTeam"] == team_name,
        past_matches["FTHG"],
        past_matches["FTAG"],
    ).sum()
    avg_goals = total_goals / k

    total_shots = np.where(
        past_matches["HomeTeam"] == team_name, past_matches["HS"], past_matches["AS"]
    ).sum()
    avg_shots = total_shots / k

    total_sot = np.where(
        past_matches["HomeTeam"] == team_name, past_matches["HST"], past_matches["AST"]
    ).sum()
    avg_sot = total_sot / k

    return avg_goals, avg_shots, avg_sot


## 各試合にチームパフォーマンスを追加する関数
def add_team_performance_to_matches(df, k):
    home_goals, home_shots, home_sot = [], [], []
    away_goals, away_shots, away_sot = [], [], []

    for idx, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]
        season = row["Season"]

        if idx < k * 10:
            home_goals.append(np.nan)
            home_sot.append(np.nan)
            home_shots.append(np.nan)
            away_goals.append(np.nan)
            away_sot.append(np.nan)
            away_shots.append(np.nan)
        else:
            home_performance = get_past_performance(home_team, date, df, k)
            home_goals.append(home_performance[0])
            home_sot.append(home_performance[1])
            home_shots.append(home_performance[2])

            away_performance = get_past_performance(away_team, date, df, k)
            away_goals.append(away_performance[0])
            away_sot.append(away_performance[1])
            away_shots.append(away_performance[2])

    df["HomeGoals"] = pd.Series(dtype="float64")
    df["HomeShots"] = pd.Series(dtype="float64")
    df["HomeSOT"] = pd.Series(dtype="float64")
    df["AwayGoals"] = pd.Series(dtype="float64")
    df["AwayShots"] = pd.Series(dtype="float64")
    df["AwaySOT"] = pd.Series(dtype="float64")

    df.loc[:, "HomeGoals"] = home_goals
    df.loc[:, "HomeShots"] = home_shots
    df.loc[:, "HomeSOT"] = home_sot
    df.loc[:, "AwayGoals"] = away_goals
    df.loc[:, "AwayShots"] = away_shots
    df.loc[:, "AwaySOT"] = away_sot

    return df


# データ加工4: Ratings


## Ratingsデータを結合する関数
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


# データ加工5: Goal Difference

## 各チームのGoal Differenceを初期化
team_gd = {}


## Goal Differenceを計算する関数
def add_goal_difference(df):
    home_gd_list, away_gd_list = [], []

    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        team_gd = {team: 0 for team in set(df["HomeTeam"]).union(df["AwayTeam"])}

        for _, row in season_data.iterrows():
            home_team = row["HomeTeam"]
            away_team = row["AwayTeam"]
            home_goals = row["FTHG"]
            away_goals = row["FTAG"]

            home_gd_list.append(team_gd[home_team])
            away_gd_list.append(team_gd[away_team])

            team_gd[home_team] += home_goals - away_goals
            team_gd[away_team] += away_goals - home_goals

    df.loc[:, "HomeGD"] = home_gd_list
    df.loc[:, "AwayGD"] = away_gd_list

    return df


# データ加工6: Diff Data


## Diff Dataを追加する関数
def add_diffs(df):
    df = df.loc[:, ~df.columns.duplicated()]

    df.loc[:, "FormDiff"] = df["HomeForm"] - df["AwayForm"]
    df.loc[:, "PointsDiff"] = df["HomeTotalPoints"] - df["AwayTotalPoints"]
    df.loc[:, "RecentPointsDiff"] = df["HomeRecentPoints"] - df["AwayRecentPoints"]
    df.loc[:, "WeightedRecentPointsDiff"] = (
        df["HomeWeightedRecentPoints"] - df["AwayWeightedRecentPoints"]
    )
    df.loc[:, "GoalsDiff"] = df["HomeGoals"] - df["AwayGoals"]
    df.loc[:, "SOTDiff"] = df["HomeSOT"] - df["AwaySOT"]
    df.loc[:, "ShotsDiff"] = df["HomeShots"] - df["AwayShots"]
    df.loc[:, "ARDiff"] = df["HomeAttackR"] - df["AwayAttackR"]
    df.loc[:, "MRDiff"] = df["HomeMidfieldR"] - df["AwayMidfieldR"]
    df.loc[:, "DRDiff"] = df["HomeDefenceR"] - df["AwayDefenceR"]
    df.loc[:, "ORDiff"] = df["HomeOverallR"] - df["AwayOverallR"]
    df.loc[:, "GDDiff"] = df["HomeGD"] - df["AwayGD"]
    return df


# データ加工7: ホーム・アウェイのフラグ追加
def add_home_factor(df):
    # ホームチームに対して1
    df["HomeIsHome"] = 1

    # アウェイチームに対して0
    df["AwayIsHome"] = 0

    return df
