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


# -------------------------
# Form
# -------------------------


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


# -------------------------
# 直近 k 試合のゴール数、シュート数、枠内シュート数、得失点差、勝ち点を総合、ホーム、アウェイごとに計算
# -------------------------


def calculate_points(result, team_type):
    """試合結果からポイントを計算"""
    if result == "H" and team_type == "home":
        return 3
    if result == "A" and team_type == "away":
        return 3
    return 1 if result == "D" else 0


def get_past_matches(team, date, df, k, is_home=None):
    """
    チームの直近 k 試合を取得する
    - team: チーム名
    - date: 現在の日付
    - df: 試合データフレーム
    - k: 直近試合数
    - is_home: ホーム試合のみに限定する場合 (True: ホーム, False: アウェイ, None: 全て)
    """
    season = df.loc[df["Date"] == date, "Season"].values[0]
    past_matches = df[
        ((df["HomeTeam"] == team) | (df["AwayTeam"] == team))
        & (df["Date"] < date)
        & (df["Season"] == season)
    ].sort_values(by="Date", ascending=False)

    if is_home is not None:
        past_matches = past_matches[
            (
                past_matches["HomeTeam"] == team
                if is_home
                else past_matches["AwayTeam"] == team
            )
        ]

    return past_matches.head(k)


def recent_points(team, date, df, k, is_home=None):
    """
    指定されたチームの直近k試合の平均ポイントを計算
    - team: チーム名
    - date: 現在の日付
    - df: データフレーム
    - k: 直近試合数
    - is_home: ホーム試合に限定するか (True: ホームのみ, False: アウェイのみ, None: 全試合)
    """
    past_matches = get_past_matches(team, date, df, k, is_home)

    if len(past_matches) < k:
        return np.nan

    points = past_matches.apply(
        lambda row: calculate_points(
            row["FTR"], "home" if row["HomeTeam"] == team else "away"
        ),
        axis=1,
    )

    return points.mean()


def add_points_stats(df, k):
    """直近k試合と累積のポイントを計算"""
    team_stats = {
        "HT_RecentPoints": [],
        "HT_HomeRecentPoints": [],
        "HT_AwayRecentPoints": [],
        "HT_TotalPoints": [],
        "AT_RecentPoints": [],
        "AT_HomeRecentPoints": [],
        "AT_AwayRecentPoints": [],
        "AT_TotalPoints": [],
    }
    team_total_points = {}

    for season in df["Season"].unique():
        season_data = df[df["Season"] == season]
        teams = set(season_data["HomeTeam"]).union(season_data["AwayTeam"])
        team_total_points.update({team: 0 for team in teams})

        for _, row in season_data.iterrows():
            home_team, away_team = row["HomeTeam"], row["AwayTeam"]
            date = row["Date"]

            # ホームチーム
            team_stats["HT_RecentPoints"].append(recent_points(home_team, date, df, k))
            team_stats["HT_HomeRecentPoints"].append(
                recent_points(home_team, date, df, k, is_home=True)
            )
            team_stats["HT_AwayRecentPoints"].append(
                recent_points(home_team, date, df, k, is_home=False)
            )
            team_stats["HT_TotalPoints"].append(team_total_points[home_team])

            # アウェイチーム
            team_stats["AT_RecentPoints"].append(recent_points(away_team, date, df, k))
            team_stats["AT_HomeRecentPoints"].append(
                recent_points(away_team, date, df, k, is_home=True)
            )
            team_stats["AT_AwayRecentPoints"].append(
                recent_points(away_team, date, df, k, is_home=False)
            )
            team_stats["AT_TotalPoints"].append(team_total_points[away_team])

            # 累積ポイント更新
            if row["FTR"] == "H":
                team_total_points[home_team] += 3
            elif row["FTR"] == "A":
                team_total_points[away_team] += 3
            elif row["FTR"] == "D":
                team_total_points[home_team] += 1
                team_total_points[away_team] += 1

    for col, values in team_stats.items():
        if len(values) != len(df):
            raise ValueError(
                f"Length mismatch for column {col}: {len(values)} vs {len(df)}"
            )
        df[col] = values
    return df


def add_goals_stats(df, k):
    """
    ゴールと得失点差に関する統計を計算
    - HT_RecentGoals, HT_HomeRecentGoals, HT_AwayRecentGoals, HT_RecentGD, ...
    """
    team_stats = {
        "HT_RecentGoals": [],
        "HT_HomeRecentGoals": [],
        "HT_AwayRecentGoals": [],
        "HT_RecentGD": [],
        "HT_HomeRecentGD": [],
        "HT_AwayRecentGD": [],
        "AT_RecentGoals": [],
        "AT_HomeRecentGoals": [],
        "AT_AwayRecentGoals": [],
        "AT_RecentGD": [],
        "AT_HomeRecentGD": [],
        "AT_AwayRecentGD": [],
    }

    def recent_goals_stats(team, date, df, k, is_home=None):
        """直近k試合のデータを取得し、ゴールと得失点差を計算"""
        past_matches = get_past_matches(team, date, df, k, is_home)

        if len(past_matches) < k:
            return np.nan, np.nan

        goals = past_matches["FTHG"].mean() if is_home else past_matches["FTAG"].mean()
        gd = (
            past_matches["FTHG"].sub(past_matches["FTAG"]).mean()
            if is_home
            else past_matches["FTAG"].sub(past_matches["FTHG"]).mean()
        )
        return goals, gd

    for _, row in df.iterrows():
        home_team, away_team = row["HomeTeam"], row["AwayTeam"]
        date = row["Date"]

        # ホームチーム
        home_goals_home, home_gd_home = recent_goals_stats(home_team, date, df, k, is_home=True)
        home_goals_away, home_gd_away = recent_goals_stats(home_team, date, df, k, is_home=False)
        all_home_goals, all_home_gd = recent_goals_stats(home_team, date, df, k)

        team_stats["HT_HomeRecentGoals"].append(home_goals_home)
        team_stats["HT_AwayRecentGoals"].append(home_goals_away)
        team_stats["HT_RecentGoals"].append(all_home_goals)
        team_stats["HT_HomeRecentGD"].append(home_gd_home)
        team_stats["HT_AwayRecentGD"].append(home_gd_away)
        team_stats["HT_RecentGD"].append(all_home_gd)

        # アウェイチーム
        away_goals_home, away_gd_home = recent_goals_stats(away_team, date, df, k, is_home=True)
        away_goals_away, away_gd_away = recent_goals_stats(away_team, date, df, k, is_home=False)
        all_away_goals, all_away_gd = recent_goals_stats(away_team, date, df, k)

        team_stats["AT_HomeRecentGoals"].append(away_goals_home)
        team_stats["AT_AwayRecentGoals"].append(away_goals_away)
        team_stats["AT_RecentGoals"].append(all_away_goals)
        team_stats["AT_HomeRecentGD"].append(away_gd_home)
        team_stats["AT_AwayRecentGD"].append(away_gd_away)
        team_stats["AT_RecentGD"].append(all_away_gd)


    for col, values in team_stats.items():
        if len(values) != len(df):
            print(f"[DEBUG] Column {col} length mismatch: {len(values)} vs {len(df)}")
            raise ValueError(f"Length mismatch for column {col}")
        df[col] = values
    return df


def add_shots_stats(df, k):
    """
    シュート数と枠内シュート数に関する統計を計算
    - HT_RecentShots, HT_HomeRecentShots, HT_AwayRecentShots, HT_RecentSOT, ...
    """
    team_stats = {
        "HT_RecentShots": [],
        "HT_HomeRecentShots": [],
        "HT_AwayRecentShots": [],
        "HT_RecentSOT": [],
        "HT_HomeRecentSOT": [],
        "HT_AwayRecentSOT": [],
        "AT_RecentShots": [],
        "AT_HomeRecentShots": [],
        "AT_AwayRecentShots": [],
        "AT_RecentSOT": [],
        "AT_HomeRecentSOT": [],
        "AT_AwayRecentSOT": [],
    }

    def recent_shots_stats(team, date, df, k, is_home=None):
        """直近k試合のデータを取得し、シュート数と枠内シュート数を計算"""
        past_matches = get_past_matches(team, date, df, k, is_home)

        if len(past_matches) < k:
            return np.nan, np.nan

        shots = past_matches["HS"].mean() if is_home else past_matches["AS"].mean()
        sot = past_matches["HST"].mean() if is_home else past_matches["AST"].mean()
        return shots, sot

    for _, row in df.iterrows():
        home_team, away_team = row["HomeTeam"], row["AwayTeam"]
        date = row["Date"]

        # ホームチーム
        home_shots_home, home_sot_home = recent_shots_stats(home_team, date, df, k, is_home=True)
        home_shots_away, home_sot_away = recent_shots_stats(home_team, date, df, k, is_home=False)
        all_home_shots, all_home_sot = recent_shots_stats(home_team, date, df, k)

        team_stats["HT_HomeRecentShots"].append(home_shots_home)
        team_stats["HT_AwayRecentShots"].append(home_shots_away)
        team_stats["HT_RecentShots"].append(all_home_shots)
        team_stats["HT_HomeRecentSOT"].append(home_sot_home)
        team_stats["HT_AwayRecentSOT"].append(home_sot_away)
        team_stats["HT_RecentSOT"].append(all_home_sot)

        # アウェイチーム
        away_shots_home, away_sot_home = recent_shots_stats(away_team, date, df, k, is_home=True)
        away_shots_away, away_sot_away = recent_shots_stats(away_team, date, df, k, is_home=False)
        all_away_shots, all_away_sot = recent_shots_stats(away_team, date, df, k)

        team_stats["AT_HomeRecentShots"].append(away_shots_home)
        team_stats["AT_AwayRecentShots"].append(away_shots_away)
        team_stats["AT_RecentShots"].append(all_away_shots)
        team_stats["AT_HomeRecentSOT"].append(away_sot_home)
        team_stats["AT_AwayRecentSOT"].append(away_sot_away)
        team_stats["AT_RecentSOT"].append(all_away_sot)


    for col, values in team_stats.items():
        if len(values) != len(df):
            raise ValueError(
                f"Length mismatch for column {col}: {len(values)} vs {len(df)}"
            )
        df[col] = values
    return df


def add_team_stats(df, k):
    """
    チーム統計量を計算し、データフレームに追加
    """
    df = add_points_stats(df, k)
    df = add_goals_stats(df, k)
    df = add_shots_stats(df, k)
    return df


# -------------------------
# Diff Data
# -------------------------


## Diff Dataを追加する関数
def add_diffs(df):
    df = df.loc[:, ~df.columns.duplicated()]

    # 基本的な差分
    df["FormDiff"] = df["HomeForm"] - df["AwayForm"]
    df["PointsDiff"] = df["HT_TotalPoints"] - df["AT_TotalPoints"]
    df["RecentPointsDiff"] = df["HT_RecentPoints"] - df["AT_RecentPoints"]
    df["HomeRecentPointsDiff"] = df["HT_HomeRecentPoints"] - df["AT_AwayRecentPoints"]
    df["AwayRecentPointsDiff"] = df["HT_AwayRecentPoints"] - df["AT_HomeRecentPoints"]

    # ゴール数の差分
    df["GoalsDiff"] = df["HT_RecentGoals"] - df["AT_RecentGoals"]
    df["HomeGoalsDiff"] = df["HT_HomeRecentGoals"] - df["AT_AwayRecentGoals"]
    df["AwayGoalsDiff"] = df["HT_AwayRecentGoals"] - df["AT_HomeRecentGoals"]

    # シュート数の差分
    df["ShotsDiff"] = df["HT_RecentShots"] - df["AT_RecentShots"]
    df["HomeShotsDiff"] = df["HT_HomeRecentShots"] - df["AT_AwayRecentShots"]
    df["AwayShotsDiff"] = df["HT_AwayRecentShots"] - df["AT_HomeRecentShots"]

    # 枠内シュート数の差分
    df["SOTDiff"] = df["HT_RecentSOT"] - df["AT_RecentSOT"]
    df["HomeSOTDiff"] = df["HT_HomeRecentSOT"] - df["AT_AwayRecentSOT"]
    df["AwaySOTDiff"] = df["HT_AwayRecentSOT"] - df["AT_HomeRecentSOT"]

    # 得失点差（GD）の差分
    df["GDDiff"] = df["HT_RecentGD"] - df["AT_RecentGD"]
    df["HomeGDDiff"] = df["HT_HomeRecentGD"] - df["AT_AwayRecentGD"]
    df["AwayGDDiff"] = df["HT_AwayRecentGD"] - df["AT_HomeRecentGD"]

    # Ratingsの差分
    df["ARDiff"] = df["HomeAttackR"] - df["AwayAttackR"]
    df["MRDiff"] = df["HomeMidfieldR"] - df["AwayMidfieldR"]
    df["DRDiff"] = df["HomeDefenceR"] - df["AwayDefenceR"]
    df["ORDiff"] = df["HomeOverallR"] - df["AwayOverallR"]

    return df
