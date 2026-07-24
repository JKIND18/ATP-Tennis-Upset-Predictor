import glob
import pandas as pd
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Combine all csv files in Data folder
print("Scanning Data/ directory for tennis match logs...")
path = 'Data/'
files = glob.glob(path + '*.csv')

df_list = []
for f in files:
    # Skips the combined file
    if "combined" in f.lower() or "final" in f.lower():
        continue
    
    print(f"   Reading: {f}")
    df = pd.read_csv(f)
    df_list.append(df)

if df_list:
    df_final = pd.concat(df_list, ignore_index=True)

    df_final['draw_size'] = pd.to_numeric(df_final['draw_size'], errors='coerce')
    df_final['match_num'] = pd.to_numeric(df_final['match_num'], errors='coerce')
    df_final['winner_rank'] = pd.to_numeric(df_final['winner_rank'], errors='coerce')
    df_final['loser_rank'] = pd.to_numeric(df_final['loser_rank'], errors='coerce')
    df_final['tourney_date'] = pd.to_numeric(df_final['tourney_date'], errors='coerce')
    
    # Overwrite the combined file with the complete, updated master list
    df_final.to_csv('Data/combined_data.csv', index=False)
    print(f"Combined {len(df_final):,} total matches into 'Data/combined_data.csv'.")
else:
    print("No raw match CSV files found in the Data/ directory.")


df_final.info()

# Refresh the PostgreSQL tables with the latest data
print("\nMirroring raw datasets to PostgreSQL tables...")
engine = create_engine(os.environ.get('TENNIS_DB_URL', 'fallback-value'))

with engine.connect() as conn:
    conn.execute(text("DROP VIEW IF EXISTS vw_tennis_with_weather CASCADE;"))
    conn.commit()

# Upload the combined match records
df_final.to_sql('raw_tennis_matches', engine, if_exists='replace', index=False)

# Upload the master weather CSV pulled from Meteostat
df_weather = pd.read_csv('modern_tennis_weather.csv')
df_weather.to_sql('raw_meteostat_weather', engine, if_exists='replace', index=False)
print("Both raw relational tables successfully refreshed.")


# Execute the SQL script to merge weather data with tennis match records
print("\nReading and running structural rules from merge_weather.sql...")

# Read the pure SQL file contents
with open('merge_weather.sql', 'r') as sql_file:
    sql_script = sql_file.read()

# Fire the SQL logic script into Postgres to recreate the calculation view
with engine.connect() as connection:
    connection.execute(text(sql_script))
    connection.commit()

print("\n" + "="*40)
print("SYSTEM FULLY SYNCHRONISED!")
print("Your View 'vw_tennis_with_weather' is updated and live!")
print("="*40)