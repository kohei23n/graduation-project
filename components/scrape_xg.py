import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

# URL の定義
urls = [
    "https://fbref.com/en/comps/9/2023-2024/schedule/2023-2024-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2022-2023/schedule/2022-2023-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2021-2022/schedule/2021-2022-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2020-2021/schedule/2020-2021-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2019-2020/schedule/2019-2020-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2018-2019/schedule/2018-2019-Premier-League-Scores-and-Fixtures",
    "https://fbref.com/en/comps/9/2017-2018/schedule/2017-2018-Premier-League-Scores-and-Fixtures",
]

# "2023-24", "2022-23" の形式で作成
seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2023, 2013, -1)]

scraper = cloudscraper.CloudScraper()
scraper.headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/"
}

# データのタイトルを定義
data_titles = ["home_team", "home_xg", "away_team", "away_xg"]

all_data = []

# すべての URL に対して繰り返し処理
for url, season in zip(urls, seasons):
    print(f"Scraping URL for season {season}: {url}")
    page = scraper.get(url, timeout=100)

    # HTML を解析
    soup = BeautifulSoup(page.text, "html.parser")

    # すべての <tr> 要素を取得
    for row in soup.find_all("tr"):
        # <td> 要素の data-title 属性を取得
        row_data = {}
        for data_title in data_titles:
            td = row.find("td", {"data-stat": data_title})
            if td and td.get_text(strip=True):
                row_data[data_title] = td.get_text(strip=True)

        # データがある場合は、Season キーを追加
        if row_data:
            row_data["Season"] = season
            all_data.append(row_data)

# Convert all data to a DataFrame
df = pd.DataFrame(all_data)

# Rename the 'Name' column to 'Team'
df.rename(
    columns={
        "home_team": "HomeTeam",
        "away_team": "AwayTeam",
        "home_xg": "HomeXG",
        "away_xg": "AwayXG",
    },
    inplace=True,
)

# Define the replacements as a dictionary
replacements = {
    "Cardiff City": "Cardiff",
    "Hull City": "Hull",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Luton Town": "Luton",
    "Manchester City": "Man City",
    "Manchester Utd": "Man United",
    "Newcastle Utd": "Newcastle",
    "Norwich City": "Norwich",
    "Nott'ham Forest": "Nott'm Forest",
    "Sheffield Utd": "Sheffield United",
    "Stoke City": "Stoke",
    "Swansea City": "Swansea",
    "Queens Park Rangers": "QPR",
}

# Replace the values in the 'HomeTeam' and 'AwayTeam' columns
df["HomeTeam"] = df["HomeTeam"].replace(replacements)
df["AwayTeam"] = df["AwayTeam"].replace(replacements)

# Save to CSV
df.to_csv("./csv/xg_data.csv", index=False)
