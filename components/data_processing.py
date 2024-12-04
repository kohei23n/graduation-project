import numpy as np
import pandas as pd

# データの読み込みと準備
match_data_df = pd.read_csv("./csv/match_data_10yr.csv")
ratings_df = pd.read_csv("./csv/ratings_data.csv")
match_data_df["Date"] = pd.to_datetime(
    match_data_df["Date"], format="%d/%m/%Y", dayfirst=True
)

# 訓練データとテストデータに分割
latest_season = match_data_df["Season"].max()

teams = set(match_data_df["HomeTeam"]).union(set(match_data_df["AwayTeam"]))

default_k = 6
default_gamma = 0.33

# データ加工1：Form


## Formを 計算・更新する関数
def update_form(home_team, away_team, result, gamma, team_form, season):
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
def calculate_form(df, gamma, teams):
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

        new_home_form, new_away_form = update_form(
            home_team, away_team, result, gamma, team_form, season
        )
        team_form[home_team] = new_home_form
        team_form[away_team] = new_away_form

    df.loc[:, "HomeForm"] = home_forms
    df.loc[:, "AwayForm"] = away_forms
    return df


# データ加工2： Streak, Weighted Streak


## 試合結果を数値に変換する関数
def get_result_points(result, team_type):
    if result == "H" and team_type == "home":
        return 3  # ホーム勝利
    elif result == "A" and team_type == "away":
        return 3  # アウェイ勝利
    elif result == "D":
        return 1  # 引き分け
    else:
        return 0  # 敗北


## 直近k試合のStreakとWeighted Streakを計算する関数
def calculate_streaks(team_name, current_date, df, k):
    season = df.loc[df["Date"] == current_date, "Season"].values[0]
    pAwayStreak_matches = df[
        ((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name))
        & (df["Date"] < current_date)
        & (df["Season"] == season)
    ]
    pAwayStreak_matches = pAwayStreak_matches.sort_values(by="Date", ascending=False).head(k)

    if len(pAwayStreak_matches) < k:
        return None, None

    results = []
    for _, game in pAwayStreak_matches.iterrows():
        if game["HomeTeam"] == team_name:
            points = get_result_points(game["FTR"], "home")
        else:
            points = get_result_points(game["FTR"], "away")
        results.append(points)

    streak = sum(results) / (3 * k)
    weighted_streak = sum(2 * (i + 1) * results[i] for i in range(k)) / (
        3 * k * (k + 1)
    )

    return streak, weighted_streak


## 各試合にStreakとWeighted Streakを追加する関数
def add_streaks(df, k):
    home_streaks, home_weighted_streaks = [], []
    away_streaks, away_weighted_streaks = [], []

    for _, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]

        home_streak, home_weighted_streak = calculate_streaks(home_team, date, df, k)
        away_streak, away_weighted_streak = calculate_streaks(away_team, date, df, k)

        home_streaks.append(home_streak)
        home_weighted_streaks.append(home_weighted_streak)
        away_streaks.append(away_streak)
        away_weighted_streaks.append(away_weighted_streak)

    df.loc[:, "HomeStreak"] = home_streaks
    df.loc[:, "HomeStreakWeighted"] = home_weighted_streaks
    df.loc[:, "AwayStreak"] = away_streaks
    df.loc[:, "AwayStreakWeighted"] = away_weighted_streaks

    return df


# データ加工3: "PAwayStreak k..." データの追加


## 過去のパフォーマンス指標を取得する関数
def get_pAwayStreak_performance(team_name, specified_date, df, k):
    season = df.loc[df["Date"] == specified_date, "Season"].values[0]
    pAwayStreak_matches = df[
        ((df["HomeTeam"] == team_name) | (df["AwayTeam"] == team_name))
        & (df["Date"] < specified_date)
        & (df["Season"] == season)
    ]
    pAwayStreak_matches = pAwayStreak_matches.sort_values(by="Date", ascending=False).head(k)

    if len(pAwayStreak_matches) < k:
        return None, None, None

    total_goals = np.where(
        pAwayStreak_matches["HomeTeam"] == team_name,
        pAwayStreak_matches["FTHG"],
        pAwayStreak_matches["FTAG"],
    ).sum()
    avg_goals = total_goals / k

    total_sot = np.where(
        pAwayStreak_matches["HomeTeam"] == team_name, pAwayStreak_matches["HomeStreak"], pAwayStreak_matches["AwayStreak"]
    ).sum()
    avg_sot = total_sot / k

    total_corners = np.where(
        pAwayStreak_matches["HomeTeam"] == team_name, pAwayStreak_matches["HC"], pAwayStreak_matches["AC"]
    ).sum()
    avg_corners = total_corners / k

    return avg_goals, avg_corners, avg_sot


## 各試合にチームパフォーマンスを追加する関数
def add_team_performance_to_matches(df, k):
    home_goals, home_sot, home_corners = [], [], []
    away_goals, away_sot, away_corners = [], [], []

    for idx, row in df.iterrows():
        home_team = row["HomeTeam"]
        away_team = row["AwayTeam"]
        date = row["Date"]
        season = row["Season"]

        if idx < k * 10:
            home_goals.append(None)
            home_sot.append(None)
            home_corners.append(None)
            away_goals.append(None)
            away_sot.append(None)
            away_corners.append(None)
        else:
            home_performance = get_pAwayStreak_performance(home_team, date, df, k)
            home_goals.append(home_performance[0])
            home_sot.append(home_performance[1])
            home_corners.append(home_performance[2])

            away_performance = get_pAwayStreak_performance(away_team, date, df, k)
            away_goals.append(away_performance[0])
            away_sot.append(away_performance[1])
            away_corners.append(away_performance[2])

    df.loc[:, "HomeGoals"] = home_goals
    df.loc[:, "HomeSOT"] = home_sot
    df.loc[:, "HomeCorners"] = home_corners
    df.loc[:, "AwayGoals"] = away_goals
    df.loc[:, "AwaySOT"] = away_sot
    df.loc[:, "AwayCorners"] = away_corners

    return df


# データ加工4: Ratings


## Ratingsデータを結合する関数
def merge_ratings(df, ratings_df):
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
def add_Diffs(df):
    df["FormDiff"] = df["HomeForm"] - df["AwayForm"]
    df["StreakDiff"] = df["HomeStreak"] - df["AwayStreak"]
    df["SOTDiff"] = df["HomeSOT"] - df["AwaySOT"]
    df["GoalsDiff"] = df["HomeGoals"] - df["AwayGoals"]
    df["CornersDiff"] = df["HomeCorners"] - df["AwayCorners"]
    df["ARDiff"] = df["HomeAttackR"] - df["AwayAttackR"]
    df["MRDiff"] = df["HomeMidfieldR"] - df["AwayMidfieldR"]
    df["DRDiff"] = df["HomeDefenceR"] - df["AwayDefenceR"]
    df["ORDiff"] = df["HomeOverallR"] - df["AwayOverallR"]
    df["GDDiff"] = df["HomeGD"] - df["AwayGD"]
    df["StreakWeightedDiff"] = df["HomeStreakWeighted"] - df["AwayStreakWeighted"]
    return df


# 特徴量生成の実行
match_data_df = calculate_form(match_data_df, default_gamma, teams)
match_data_df = add_streaks(match_data_df, default_k)
match_data_df = add_team_performance_to_matches(match_data_df, default_k)
match_data_df = merge_ratings(match_data_df, ratings_df)
match_data_df = add_goal_difference(match_data_df)
match_data_df = add_Diffs(match_data_df)

# 加工済みデータを CSV に出力
match_data_df.to_csv("./csv/engineered_data.csv", index=False)
