# -------------------------
# データ加工: 累積・直近k試合の統計量追加
# -------------------------


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