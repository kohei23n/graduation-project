import cloudscraper
from bs4 import BeautifulSoup
import pandas as pd

# Define URLS
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
seasons = list(range(2023, 2013, -1))

scraper = cloudscraper.CloudScraper()
scraper.headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/'}

# Define the data-title attributes
data_titles = ["Name", "ATT", "MID", "DEF", "OVR"]

# Initialize a list to store all data
all_data = []

# Loop through each URL and season
for url, season in zip(urls, seasons):
    print(f"Scraping URL for season {season}: {url}")
    page = scraper.get(url, timeout=100)

    # Parse the HTML using BeautifulSoup
    soup = BeautifulSoup(page.text, 'html.parser')

    # Iterate through each row (<tr>)
    for row in soup.find_all('tr'):
        # Find all <td> elements in the row with the specified data-title attributes
        row_data = {}
        for data_title in data_titles:
            td = row.find('td', {'data-title': data_title})
            if td:
                row_data[data_title] = td.get_text(strip=True)
        
        # Add season column if data is found
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
