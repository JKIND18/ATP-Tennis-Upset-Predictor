from datetime import datetime
import pandas as pd
import meteostat as ms
from meteostat import daily, hourly

ms.config.block_large_requests = False

print("Running complete weather pipeline engine...")


df_master = pd.read_csv('Data/combined_data.csv')

latest_date_int = df_master['tourney_date'].max()
earliest_date_int = df_master['tourney_date'].min()

latest_date_str = datetime.strptime(str(int(latest_date_int)), '%Y%m%d').strftime('%Y-%m-%d')
earliest_date_str = datetime.strptime(str(int(earliest_date_int)), '%Y%m%d').strftime('%Y-%m-%d')

start_date = datetime.strptime(earliest_date_str, '%Y-%m-%d')
end_date = datetime.strptime(latest_date_str, '%Y-%m-%d')


# Global WMO airport weather stations
station_lookup = {
    'Brisbane': '94578', 'Doha': 'OTHH0', 'Chennai': '43279', 'Sydney': '94767', 
    'Auckland': '93110', 'Quito': '84071', 'Delray Beach': '72203', 'Rio de Janeiro': '83755', 
    'Dubai': '41194', 'Buenos Aires': '87585', 'Acapulco': '76805', 'Indian Wells': '72286', 'Miami': '72202', 
    'Casablanca': '60155', 'Houston': '72243', 'Monte Carlo': '07650', 
    'Barcelona': '08181', 'Bucharest': '15420', 'Munich': '10865', 'Istanbul': '17060', 
    'Estoril': '08535', 'Madrid': '08221', 'Rome': '16239', 'Geneva': '06700', 
    'Nice': '07690', 'Paris': '07149', 'Stuttgart': '10738', 's-Hertogenbosch': '06260', 
    'London': '03772', 'Halle': '10348', 'Nottingham': '03772', 'Newport': '72506', 
    'Bastad': '02616', 'Umag': '16110', 'Bogota': '80222', 'Gstaad': '06700', 'Hamburg': '10147', 'Atlanta': '72219', 
    'Kitzbuhel': '11120', 'Washington': '72405', 'Toronto': '71624', 'Cincinnati': '72421', 'Winston-Salem': '72434', 
    'New York': '74486', 'Shenzhen': '59493', 'Tokyo': '47662', 'Beijing': '54511', 
    'Shanghai': '58367', 'Sao Paulo': '83755', 'Marrakech': '60230', 'Los Cabos': '76741', 
    'Chengdu': '56294', 'Budapest': '12843', 'Lyon': '07481', 'Eastbourne': '03772', 
    'Antalya': '17300', 'Pune': '43279', 'Cabo San Lucas': '76741', 'Cordoba': '87585', 
    'Zhuhai': '59287', 'Adelaide': '94672', 'Santiago': '85574', 'Marbella': '08482', 'Cagliari': '16560', 
    'Belgrade': '13274', 'Parma': '16022', 'Mallorca': '08306', 'San Diego': '72290', 
    'Melbourne': '94866', 'Seoul': '47108', 'Naples': '16289', 'Banja Luka': '14542', 
    'Hong Kong': '45005', 'Hangzhou': '58457'
}

compiled_frames = []

for city_name, station_id in station_lookup.items():
    try:
        if city_name == 'Doha':
            data_hourly = ms.hourly(station_id, start_date, end_date).fetch()

            if not data_hourly.empty:
                df_clean = data_hourly.groupby(data_hourly.index.date).agg(
                    temp=('temp', 'mean'),
                    tmax=('temp', 'max'),
                    wspd=('wspd', 'mean')
                ).reset_index()
                df_clean.rename(columns={'index': 'time'}, inplace=True)
                df_clean['target_city'] = 'Doha'
                compiled_frames.append(df_clean)
                print(f"    Successfully retrieved: {city_name} (Station {station_id})")
            else:
                print(f"    Station {station_id} returned empty results for {city_name}")
        else:
            query = ms.daily(station_id, start_date, end_date)
            df_raw = query.fetch()
            
            if df_raw is not None and not df_raw.empty:
                df_flat = df_raw.reset_index()
                df_clean = df_flat[['time', 'temp', 'tmax', 'wspd']].copy()
                df_clean['target_city'] = city_name
                compiled_frames.append(df_clean)
                print(f"    Successfully retrieved: {city_name} (Station {station_id})")
            else:
                print(f"    Station {station_id} returned empty results for {city_name}")
                
    except Exception as e:
        print(f"    Failed connection step for {city_name}: {str(e)}")

# 4. Concatenate and overwrite output
if compiled_frames:
    final_climate_df = pd.concat(compiled_frames, ignore_index=True)
    
    final_climate_df.rename(columns={
        'time': 'weather_date',
        'temp': 'average_temp',
        'tmax': 'max_temp',
        'wspd': 'wind_speed_kmh'
    }, inplace=True)
    
    output_filename = "modern_tennis_weather.csv"
    final_climate_df.to_csv(output_filename, index=False)
    
    print("\n" + "="*40)
    print(f"SUMMARY")
    print(f"Saved total rows: {len(final_climate_df):,}")
    print(f"Output location: {output_filename}")
    print("="*40)
else:
    print("\nProcess complete, but no data entries were compiled successfully.")