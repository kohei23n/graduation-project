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
        # CSV ファイルを読み込む
        season_df = pd.read_csv(file_path)
        # シーズン情報を追加（"YYYY-YY"）
        season_df["Season"] = season_file.split("season")[1].split(".csv")[0]
        dataframes.append(season_df)
        print(f"Loaded {season_file} successfully.")
    else:
        print(f"File {season_file} not found. Skipping...")

# データを結合
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    # 新しい CSV ファイルに保存
    combined_df.to_csv(output_file, index=False)
    print(f"All data successfully combined and saved to {output_file}.")
else:
    print("No data loaded. Check your file paths and names.")

# 確認用
print(combined_df.head())
