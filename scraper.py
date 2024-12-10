import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

# URL の定義
urls = [
    "https://www.fifaindex.com/teams/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa23_589/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa22_555/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa21_486/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa20_419/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa19_353/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa18_278/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa17_173/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa16_73/?league=13&order=desc",
    "https://www.fifaindex.com/teams/fifa15_14/?league=13&order=desc",
]

# "2023-24", "2022-23" の形式で作成
seasons = [f"{year}-{str(year+1)[-2:]}" for year in range(2023, 2013, -1)]

scraper = cloudscraper.CloudScraper()
scraper.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'}

# データのタイトルを定義
data_titles = ["Name", "ATT", "MID", "DEF", "OVR"]

all_data = []

# すべての URL に対して繰り返し処理
for url, season in zip(urls, seasons):
    print(f"Scraping URL for season {season}: {url}")
    page = scraper.get(url, timeout=100)

    # HTML を解析
    soup = BeautifulSoup(page.text, 'html.parser')

    # すべての <tr> 要素を取得
    for row in soup.find_all('tr'):
        # <td> 要素の data-title 属性を取得
        row_data = {}
        for data_title in data_titles:
            td = row.find('td', {'data-title': data_title})
            if td:
                row_data[data_title] = td.get_text(strip=True)
        
        # データがある場合は、Season キーを追加
        if row_data:
            row_data['Season'] = season
            all_data.append(row_data)

# Convert all data to a DataFrame
df = pd.DataFrame(all_data)

# Rename the 'Name' column to 'Team'
df.rename(columns={'Name': 'Team'}, inplace=True)

# Define the replacements as a dictionary
replacements = {
    'AFC Bournemouth': 'Bournemouth',
    'Brighton & Hove Albion': 'Brighton',
    'Cardiff City': 'Cardiff',
    'Huddersfield Town': 'Huddersfield',
    'Hull City': 'Hull',
    'Leicester City': 'Leicester',
    'Luton Town': 'Luton',
    'Leeds United': 'Leeds',
    'Manchester City': 'Man City',
    'Manchester United': 'Man United',
    'Manchester Utd': 'Man United',
    'NewcAwayStreakle United': 'NewcAwayStreakle',
    'NewcAwayStreakle Utd': 'NewcAwayStreakle',
    'Norwich City': 'Norwich',
    'Nottingham Forest': "Nott'm Forest",
    'Queens Park Rangers': 'QPR',
    'Spurs': 'Tottenham',
    "Stoke City": "Stoke",
    'Swansea City': 'Swansea',
    'Tottenham Hotspur': 'Tottenham',
    'West Bromwich Albion': 'West Brom',
    'West Ham United': 'West Ham',
    'Wolverhampton Wanderers': 'Wolves',
}

# Replace the values in the 'Team' column based on the dictionary
df['Team'] = df['Team'].replace(replacements)

# Save to CSV
df.to_csv('./csv/ratings_data.csv', index=False)
