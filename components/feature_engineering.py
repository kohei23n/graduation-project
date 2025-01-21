import numpy as np
import logging

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

    # past matches のうち、直近 k 試合分のデータのみ取得
    return past_matches.head(k) if k else past_matches


# -------------------------
# 1. the different abilities of both teams
# -------------------------

# Average Points

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
    for idx, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームとアウェイチームの直近 k 試合の平均ポイントを計算
        df.at[idx, "HT_RecentPPG"] = calc_points_stats(home_team, date, df, k)
        df.at[idx, "AT_RecentPPG"] = calc_points_stats(away_team, date, df, k)

    return df


## シーズンごとの累積平均ポイントを計算する関数
def add_avg_ppg_stats(df):
    for idx, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        # ホームチームとアウェイチームの直近 k 試合の平均ポイントを計算
        df.at[idx, "HT_AvgPPG"] = calc_points_stats(home_team, date, df)
        df.at[idx, "AT_AvgPPG"] = calc_points_stats(away_team, date, df)

    return df


# -------------------------
# 2. home advantage
# -------------------------

def calc_wdl_rates(team, date, df, k=None):
    past_matches = get_past_matches(team, date, df, k=k)

    if past_matches.empty or (k is not None and len(past_matches) < k):
        return np.nan, np.nan, np.nan
    
    wins, draws, losses = 0, 0, 0
    
    for _, match in past_matches.iterrows():
        if match["HomeTeam"] == team:
            if match["FTR"] == "H":
                wins += 1
            elif match["FTR"] == "D":
                draws += 1
            else:
                losses += 1
        else:
            if match["FTR"] == "A":
                wins += 1
            elif match["FTR"] == "D":
                draws += 1
            else:
                losses += 1
                
    total_matches = len(past_matches)
    
    return wins / total_matches, draws / total_matches, losses / total_matches


## シーズンごとの累積平均ポイントを計算する関数
def add_wdl_rates(df):
    wdl_stats = {
        "HT_HomeWinRate": [],
        "HT_HomeDrawRate": [],
        "HT_HomeLossRate": [],
        "AT_AwayWinRate": [],
        "AT_AwayDrawRate": [],
        "AT_AwayLossRate": [],
    }
    
    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]
        
        home_win_rate, home_draw_rate, home_loss_rate = calc_wdl_rates(home_team, date, df)
        wdl_stats["HT_HomeWinRate"].append(home_win_rate)
        wdl_stats["HT_HomeDrawRate"].append(home_draw_rate)
        wdl_stats["HT_HomeLossRate"].append(home_loss_rate)
        
        away_win_rate, away_draw_rate, away_loss_rate = calc_wdl_rates(away_team, date, df)
        wdl_stats["AT_AwayWinRate"].append(away_win_rate)
        wdl_stats["AT_AwayDrawRate"].append(away_draw_rate)
        wdl_stats["AT_AwayLossRate"].append(away_loss_rate)

    for col, values in wdl_stats.items():
        df[col] = values

    return df


# -------------------------
# 3. recent performance
# -------------------------


## 指定されたチームの直近 k 試合の Shots, SOT, Shots Conceded, SOT Conceded を計算する関数
def calc_shots_stats(team, date, df, k=None):
    past_matches = get_past_matches(team, date, df, k)

    if past_matches.empty or (k is not None and len(past_matches) < k):
        return np.nan, np.nan, np.nan, np.nan

    shots, sot, shots_conceded, sot_conceded = [], [], [], []

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
    recent_shots_stats = {
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
        recent_shots_stats["HT_RecentShots"].append(ht_shots)
        recent_shots_stats["HT_RecentSOT"].append(ht_sot)
        recent_shots_stats["HT_RecentShotsConceded"].append(ht_shots_conceded)
        recent_shots_stats["HT_RecentSOTConceded"].append(ht_sot_conceded)

        # アウェイチームの統計
        at_shots, at_sot, at_shots_conceded, at_sot_conceded = calc_shots_stats(
            away_team, date, df, k
        )
        recent_shots_stats["AT_RecentShots"].append(at_shots)
        recent_shots_stats["AT_RecentSOT"].append(at_sot)
        recent_shots_stats["AT_RecentShotsConceded"].append(at_shots_conceded)
        recent_shots_stats["AT_RecentSOTConceded"].append(at_sot_conceded)

    for col, values in recent_shots_stats.items():
        df[col] = values

    return df


# AVG Shots, SOT, Shots Conceded, SOT Conceded in Season
def add_avg_shots_stats(df):
    avg_shots_stats = {
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
        avg_shots_stats["HT_AvgShots"].append(ht_avg_shots)
        avg_shots_stats["HT_AvgSOT"].append(ht_avg_sot)
        avg_shots_stats["HT_AvgShotsConceded"].append(ht_avg_shots_conceded)
        avg_shots_stats["HT_AvgSOTConceded"].append(ht_avg_sot_conceded)

        # アウェイチームの全試合平均
        at_avg_shots, at_avg_sot, at_avg_shots_conceded, at_avg_sot_conceded = (
            calc_shots_stats(away_team, date, df, k=None)
        )
        avg_shots_stats["AT_AvgShots"].append(at_avg_shots)
        avg_shots_stats["AT_AvgSOT"].append(at_avg_sot)
        avg_shots_stats["AT_AvgShotsConceded"].append(at_avg_shots_conceded)
        avg_shots_stats["AT_AvgSOTConceded"].append(at_avg_sot_conceded)

    for col, values in avg_shots_stats.items():
        df[col] = values

    return df


# -------------------------
# 4. ability of opposition
# -------------------------

def calc_elo_rating(team, date, df, initial_rating=1000, k=20, c=10, d=400):
    past_matches = get_past_matches(team, date, df)

    if past_matches.empty:
        return initial_rating

    elo = initial_rating

    for _, match in past_matches.iterrows():
        if match["HomeTeam"] == team:
            opponent = match["AwayTeam"]
            opponent_elo = match["AT_Elo"]
            result = (
                1 if match["FTR"] == "H" 
                else 0.5 if match["FTR"] == "D" 
                else 0
            )
        else:
            opponent = match["HomeTeam"]
            opponent_elo = match["HT_Elo"]
            result = (
                1 if match["FTR"] == "A"
                else 0.5 if match["FTR"] == "D"
                else 0
            )

        expected_score = 1 / (1 + c ** ((opponent_elo - elo) / d))
        elo += k * (result - expected_score)

    return elo


def add_elo_stats(df):
    for idx, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        df.at[idx, "HT_Elo"] = calc_elo_rating(home_team, date, df)
        df.at[idx, "AT_Elo"] = calc_elo_rating(away_team, date, df)

    return df


# -------------------------
# X. Bonus!
# -------------------------


def calc_xg_stats(team, date, df, k=None):
    past_matches = get_past_matches(team, date, df, k=k)

    if past_matches.empty or (k is not None and len(past_matches) < k):
        return np.nan, np.nan

    xg, xg_conceded = [], []

    for _, match in past_matches.iterrows():
        if match["HomeTeam"] == team:
            xg.append(match["HomeXG"])
            xg_conceded.append(match["AwayXG"])
        else:
            xg.append(match["AwayXG"])
            xg_conceded.append(match["HomeXG"])

    return np.mean(xg), np.mean(xg_conceded)


## 過去k試合の平均ポイントを計算する関数
def add_recent_xg_stats(df, k):
    recent_xg_stats = {
        "HT_RecentXG": [],
        "HT_RecentXGConceded": [],
        "AT_RecentXG": [],
        "AT_RecentXGConceded": [],
    }

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        ht_xg, ht_xg_conceded = calc_xg_stats(home_team, date, df, k)
        recent_xg_stats["HT_RecentXG"].append(ht_xg)
        recent_xg_stats["HT_RecentXGConceded"].append(ht_xg_conceded)
        
        at_xg, at_xg_conceded = calc_xg_stats(away_team, date, df, k)
        recent_xg_stats["AT_RecentXG"].append(at_xg)
        recent_xg_stats["AT_RecentXGConceded"].append(at_xg_conceded)

    for col, values in recent_xg_stats.items():
        df[col] = values

    return df


## シーズンごとの累積平均ポイントを計算する関数
def add_avg_xg_stats(df):
    avg_xg_stats = {
        "HT_AvgXG": [],
        "HT_AvgXGConceded": [],
        "AT_AvgXG": [],
        "AT_AvgXGConceded": [],
    }

    for _, row in df.iterrows():
        home_team, away_team, date = row["HomeTeam"], row["AwayTeam"], row["Date"]

        ht_xg, ht_xg_conceded = calc_xg_stats(home_team, date, df, k=None)
        avg_xg_stats["HT_AvgXG"].append(ht_xg)
        avg_xg_stats["HT_AvgXGConceded"].append(ht_xg_conceded)
        
        at_xg, at_xg_conceded = calc_xg_stats(away_team, date, df, k=None)
        avg_xg_stats["AT_AvgXG"].append(at_xg)
        avg_xg_stats["AT_AvgXGConceded"].append(at_xg_conceded)

    for col, values in avg_xg_stats.items():
        df[col] = values

    return df


# -------------------------
# 5. combine all features into a single function
# -------------------------


def add_team_stats(df, k):
    logging.info("Adding ability stats...")
    df = add_avg_ppg_stats(df)
    df = add_avg_shots_stats(df)
    df = add_avg_xg_stats(df)
    logging.info("Adding home advantage stats...")
    df = add_wdl_rates(df)
    logging.info("Adding recent stats...")
    df = add_recent_shots_stats(df, k)
    df = add_recent_ppg_stats(df, k)
    df = add_recent_xg_stats(df, k)
    logging.info("Adding elo ratings...")
    df = add_elo_stats(df)

    return df
