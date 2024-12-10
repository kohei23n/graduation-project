import pandas as pd
import os

# フォルダとアウトプットファイルのパスを指定
data_folder = "./csv/"
output_file = "./csv/match_data_10yr.csv"

# 取得する season ファイルのリストを作成（"seasonYYYY-YY.csv"）
seasons = [f"season{year}-{str(year+1)[-2:]}.csv" for year in range(2014, 2024)]

dataframes = []

# シーズンごとにデータを読み込む
for season_file in seasons:
    file_path = os.path.join(data_folder, season_file)
    if os.path.exists(file_path):
        # Load the CSV file into a DataFrame
        season_df = pd.read_csv(file_path)
        # Add a 'Season' column for clarity
        season_df["Season"] = season_file.split("season")[1].split(".csv")[0]
        dataframes.append(season_df)
        print(f"Loaded {season_file} successfully.")
    else:
        print(f"File {season_file} not found. Skipping...")

# Combine all DataFrames into one
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Save the combined DataFrame to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"All data successfully combined and saved to {output_file}.")
else:
    print("No data loaded. Check your file paths and names.")

# Verify the first few rows of the combined DataFrame
print(combined_df.head())
