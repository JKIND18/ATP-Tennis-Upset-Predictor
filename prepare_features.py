import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

engine = create_engine(os.environ.get('TENNIS_DB_URL', 'fallback-value'))

print("Loading the tennis with weather dataset...")

df = pd.read_sql("SELECT * FROM vw_tennis_with_weather;", engine)
df.to_csv("Data/final_dataset_with_weather.csv", index=False)

print(df.info())
print(df.head())

# Filling blanks with a rank of 2000
df['winner_rank'] = df['winner_rank'].fillna(2000)
df['loser_rank'] = df['loser_rank'].fillna(2000)

print("Checking for remaining missing ranks:")
print(f"Missing winner ranks: {df['winner_rank'].isnull().sum()}")
print(f"Missing loser ranks:  {df['loser_rank'].isnull().sum()}")

# Cleaning the dates
df['estimated_match_date'] = pd.to_datetime(df['estimated_match_date'])

# Removing any blank estimated dates
df = df.dropna(subset=['estimated_match_date']).copy()

# Minutes to numbers rather than a string
df['minutes'] = pd.to_numeric(df['minutes'], errors='coerce').fillna(0).astype(int)

# Remove Davis Cup rows
is_davis_cup = df['tourney_name'].str.contains('Davis Cup', case=False, na=False)
df = df[~is_davis_cup].copy()

# Sort chronologically
df = df.sort_values('estimated_match_date').reset_index(drop=True)

# Remapping the rounds to numbers for later
round_mapping = {
    'Q1': 1,
    'Q2': 2,
    'Q3': 3,
    'R128': 4,
    'R64': 5,
    'R32': 6,
    'R16': 7,
    'QF': 8,
    'SF': 9,
    'F': 10
}
df['round_num'] = df['round'].map(round_mapping).fillna(1)


# Create a retirement or walkover indicator
df['retirement_or_wo'] = df['score'].astype(str).str.contains('RET|W/O', regex=True)


# Number of sets dropped this tournament
# Count dashes for total sets played
df['best_of_num'] = pd.to_numeric(df['best_of'], errors='coerce').fillna(3).astype(int)
df['total_sets_played'] = df['score'].astype(str).str.count('-')
df['sets_needed_to_win'] = ((df['best_of_num'] + 1) / 2).astype(int)
df['winner_sets_dropped'] = (df['total_sets_played'] - df['sets_needed_to_win']).clip(lower=0)
df['loser_sets_dropped'] = df['sets_needed_to_win']
# 0 sets dropped for a walkover
is_wo = df['score'].astype(str).str.contains('W/O', na=False)
df.loc[is_wo, ['winner_sets_dropped', 'loser_sets_dropped']] = 0

# Weather percentages
df['is_extreme_heat'] = (df['max_temp'] >= 30).astype(int)
df['is_hot_weather'] = (df['average_temp'] >= 25).astype(int)
df['is_windy_weather'] = (df['wind_speed_kmh'] >= 20).astype(int)



# Engineering features for the model


# Stacking every player's rank, combining winner and loser name
winners = df[['estimated_match_date', 'winner_name', 'winner_age', 'winner_rank', 'minutes', 'tourney_id', 'surface', 'loser_name', 'loser_age', 'loser_rank', 'loser_hand', 'round_num', 'winner_ioc', 'target_city', 'winner_sets_dropped', 'is_extreme_heat', 'is_hot_weather', 'is_windy_weather']].rename(
    columns={
        'winner_name': 'player', 
        'winner_rank': 'rank',
        'winner_age': 'age',
        'loser_name': 'opponent',
        'loser_age': 'opponent_age',
        'loser_rank': 'opponent_rank',
        'loser_hand': 'opponent_hand',
        'winner_ioc': 'player_ioc',
        'winner_sets_dropped': 'sets_dropped_this_match'
    }
)
winners['match_won'] = 1

# Ony want retirement or walkovers to affect the loser
winners['injury_loss'] = 0

# Change round number to an 8 if they won the final
winners.loc[winners['round_num'] == 10, 'round_num'] = 11


losers = df[['estimated_match_date', 'loser_name', 'loser_age', 'loser_rank', 'minutes', 'tourney_id', 'surface', 'winner_name', 'winner_age', 'winner_rank', 'winner_hand', 'round_num', 'loser_ioc', 'target_city', 'retirement_or_wo', 'loser_sets_dropped', 'is_extreme_heat', 'is_hot_weather', 'is_windy_weather']].rename(
    columns={
        'loser_name': 'player',
        'loser_rank': 'rank',
        'loser_age': 'age',
        'winner_name': 'opponent',
        'winner_age': 'opponent_age',
        'winner_rank': 'opponent_rank',
        'winner_hand': 'opponent_hand',
        'loser_ioc': 'player_ioc',
        'loser_sets_dropped': 'sets_dropped_this_match'
    }
)
losers['match_won'] = 0

timeline = pd.concat([winners, losers]).sort_values('estimated_match_date')

timeline = timeline.reset_index(drop=False).rename(columns={'index': 'match_row_id'})

# Change rank 2000 back to NA to avoid skew
timeline.loc[timeline['rank'] == 2000, 'rank'] = pd.NA
# Blank minutes to 0
timeline['minutes'] = timeline['minutes'].fillna(0)
timeline['match_year'] = timeline['estimated_match_date'].dt.year

# Lookup table so we can merge back to the dataset
lookup = timeline.drop_duplicates(subset=['estimated_match_date', 'player'])


# FEATURES:

# 1. Running average rank
timeline['avg_career_rank'] = timeline.groupby('player')['rank'].transform(lambda x: x.expanding().mean()).fillna(2000)


# 2. Career High Ranking
timeline['career_high_rank'] = timeline.groupby('player')['rank'].transform(lambda x: x.expanding().min()).fillna(2000)


# 3. Days since career high
is_new_high = (timeline['rank'] == timeline['career_high_rank'])
# Record date if new high, otherwise NA
timeline['career_high_date'] = timeline['estimated_match_date'].where(is_new_high)
# Forward fill last career high date
timeline['career_high_date'] = timeline.groupby('player')['career_high_date'].ffill()
# Calculate days since career high
timeline['days_since_career_high'] = (timeline['estimated_match_date'] - timeline['career_high_date']).dt.days.fillna(0)


# 4. Days since last match
timeline['previous_match_date'] = timeline.groupby('player')['estimated_match_date'].shift(1)
timeline['days_since_last_match'] = (timeline['estimated_match_date'] - timeline['previous_match_date']).dt.days.fillna(30)


# 5. Time on court last match
timeline['time_on_court_last_match'] = timeline.groupby('player')['minutes'].shift(1).fillna(0)


# 6. Time on court this tournament
timeline['time_on_court_this_tournament'] = timeline.groupby(['player', 'tourney_id'])['minutes'].transform(lambda x: x.cumsum().shift(1)).fillna(0)


# 7. Number of matches on surface this season
timeline['matches_on_surface_this_season'] = timeline.groupby(['player', 'surface', 'match_year']).cumcount().fillna(0)


# 8. Surface win percentage
timeline['surface_win_pct'] = timeline.groupby(['player', 'surface'])['match_won'].transform(
    lambda x: x.cumsum().shift(1) / np.arange(len(x))
).fillna(0.5)
# Fill NA with 50%
# We don't shift len(x) as this starts at 0


# 9. Surface win streak
def calculate_streak(series):
    shifted = series.shift(1).fillna(0)
    streak = []
    current_streak = 0
    for val in shifted:
        if val == 1:
            current_streak += 1
        else:
            current_streak = 0
        streak.append(current_streak)
    return pd.Series(streak, index=series.index)

timeline['surface_win_streak'] = timeline.groupby(['player', 'surface'])['match_won'].transform(calculate_streak).fillna(0)


# 10. Win percentage against opponent dominant hand
timeline['win_pct_against_opponent_hand'] = timeline.groupby(['player', 'opponent_hand'])['match_won'].transform(
    lambda x: x.cumsum().shift(1) / np.arange(len(x))
).fillna(0.5)


# 11. Wins against opponent
timeline['h2h_wins'] = timeline.groupby(['player', 'opponent'])['match_won'].transform(
    lambda x: x.cumsum().shift(1)
).fillna(0)


# 12. Head to Head Total Matches
timeline['h2h_total_matches'] = timeline.groupby(['player', 'opponent'])['match_won'].transform(
    lambda x: np.arange(len(x))
).fillna(0)


# 13. Head to Head Win Percentage
timeline['h2h_win_pct'] = (timeline['h2h_wins'] / timeline['h2h_total_matches']).fillna(0.5)


# 14. Number of main draw wins in the last tournament
wins_this_tourney = timeline.groupby(['player', 'tourney_id'])['match_won'].transform('sum').fillna(0)
tourney_changed = timeline['tourney_id'] != timeline.groupby('player')['tourney_id'].shift(1)
timeline['main_draw_wins_last_tournament'] = wins_this_tourney.shift(1).where(tourney_changed).ffill().fillna(0)


# 15. Home Ground Advantage
city_to_ioc = {
    # Australia
    'Melbourne': 'AUS', 'Sydney': 'AUS', 'Brisbane': 'AUS', 'Adelaide': 'AUS', 'Auckland': 'NZL',
    
    # Europe (West & Central)
    'London': 'GBR', 'Eastbourne': 'GBR', 'Nottingham': 'GBR',
    'Paris': 'FRA', 'Marseille': 'FRA', 'Montpellier': 'FRA', 'Lyon': 'FRA', 'Nice': 'FRA',
    'Madrid': 'ESP', 'Barcelona': 'ESP', 'Mallorca': 'ESP', 'Marbella': 'ESP', 'Gijon': 'ESP',
    'Rome': 'ITA', 'Florence': 'ITA', 'Naples': 'ITA', 'Parma': 'ITA',
    'Rotterdam': 'NED', 's-Hertogenbosch': 'NED',
    'Antwerp': 'BEL', 'Brussels': 'BEL',
    'Geneva': 'SUI', 'Gstaad': 'SUI', 'Basel': 'SUI',
    'Halle': 'GER', 'Hamburg': 'GER', 'Munich': 'GER', 'Stuttgart': 'GER', 'Cologne 1': 'GER', 'Cologne 2': 'GER',
    'Kitzbuhel': 'AUT', 'Vienna': 'AUT',
    
    # Europe (East, North & Balkans)
    'Estoril': 'POR',
    'Monte Carlo': 'MON',
    'Banja Luka': 'BIH',
    'Belgrade': 'SRB',
    'Zagreb': 'CRO', 'Umag': 'CRO',
    'Sofia': 'BUL',
    'Bucharest': 'ROU',
    'Budapest': 'HUN',
    'Athens': 'GRE',
    'Stockholm': 'SWE', 'Bastad': 'SWE',
    'Antalya': 'TUR',
    'St. Petersburg': 'RUS', 'Moscow': 'RUS',
    
    # North America
    'Indian Wells': 'USA', 'Miami': 'USA', 'New York': 'USA', 'Cincinnati': 'USA', 
    'Washington': 'USA', 'Winston-Salem': 'USA', 'Atlanta': 'USA', 'Dallas': 'USA', 
    'Houston': 'USA', 'Delray Beach': 'USA', 'New York Open': 'USA', 'Newport': 'USA', 'San Diego': 'USA',
    'Toronto': 'CAN', 'Montreal': 'CAN',
    'Acapulco': 'MEX', 'Los Cabos': 'MEX', 'Cabo San Lucas': 'MEX',
    
    # South America
    'Rio de Janeiro': 'BRA', 'Sao Paulo': 'BRA',
    'Buenos Aires': 'ARG', 'Cordoba': 'ARG',
    'Santiago': 'CHI',
    'Bogota': 'COL',
    'Quito': 'ECU',
    
    # Asia & Middle East
    'Tokyo': 'JPN',
    'Beijing': 'CHN', 'Shanghai': 'CHN', 'Shenzhen': 'CHN', 'Zhuhai': 'CHN', 'Chengdu': 'CHN', 'Hangzhou': 'CHN',
    'Hong Kong': 'HKG',
    'Seoul': 'KOR',
    'Singapore': 'SGP',
    'Pune': 'IND', 'Chennai': 'IND',
    'Almaty': 'KAZ', 'Astana': 'KAZ', 'Nur-Sultan': 'KAZ',
    'Doha': 'QAT',
    'Dubai': 'ARE',
    'Kuala Lumpur': 'MYS',
    
    # Year-End Finals Locations (Mapping to their physical host cities)
    'ATP Finals': 'ITA',          # Held in Turin, Italy
    'ATP Tour Finals': 'ITA',     # Held in Turin / London (using ITA as modern default)
    'Tour Finals': 'ITA',
    'Next Gen ATP Finals': 'SAU', # Held in Jeddah, Saudi Arabia
    'Next Gen Finals': 'SAU',
    'NextGen Finals': 'SAU',
    'Laver Cup': 'EUR'            # Neutral designation
}

# Map the tournament city to its country's IOC code
timeline['tourney_ioc'] = timeline['target_city'].map(city_to_ioc)

timeline['is_home_ground'] = (timeline['player_ioc'] == timeline['tourney_ioc']).astype(int)


# 16. Days Since Last Retirement or Walkover
injury_dates = timeline['estimated_match_date'].where(timeline['injury_loss'] == 1)
timeline['last_injury_date'] = injury_dates.groupby(timeline['player']).transform(lambda x: x.ffill())
timeline['days_since_last_injury'] = (timeline['estimated_match_date'] - timeline['last_injury_date']).dt.days.fillna(9999)  # Fill with a large number if no prior injury


# 17. Number of sets dropped that tournament
timeline['sets_dropped_this_tournament'] = timeline.groupby(['player', 'tourney_id'])['sets_dropped_this_match'].transform(
    lambda x: x.cumsum().shift(1)
).fillna(0)


# 18. Highest ranked player they beat this tournament
timeline['beaten_opponent_rank'] = timeline['opponent_rank'].where(timeline['match_won'] == 1)
timeline['best_win_this_tournament'] = timeline.groupby(['player', 'tourney_id'])['beaten_opponent_rank'].transform(
    lambda x: x.expanding().min().shift(1)
).fillna(2000)


# 19. Highest ranked player they beat in the past 30 days
timeline = timeline.sort_values(by=['player', 'estimated_match_date'])

rolling_series = (
    timeline.groupby('player')
    .rolling('30D', on='estimated_match_date', closed='left')['beaten_opponent_rank']
    .min()
)
timeline['best_win_last_30_days'] = rolling_series.to_numpy()

timeline['best_win_last_30_days'] = timeline['best_win_last_30_days'].fillna(2000)
timeline = timeline.sort_values(by=['estimated_match_date', 'round_num'])


# 20. Weather win percentages
# Extreme heat win percentage
timeline['extreme_heat_win_pct'] = timeline.groupby(['player', 'is_extreme_heat'])['match_won'].transform(
    lambda x: x.cumsum().shift(1) / np.arange(len(x))
).fillna(0.5)

# Hot weather win percentage
timeline['hot_weather_win_pct'] = timeline.groupby(['player', 'is_hot_weather'])['match_won'].transform(
    lambda x: x.cumsum().shift(1) / np.arange(len(x))
).fillna(0.5)

# Windy weather win percentage
timeline['windy_weather_win_pct'] = timeline.groupby(['player', 'is_windy_weather'])['match_won'].transform(
    lambda x: x.cumsum().shift(1) / np.arange(len(x))
).fillna(0.5)



# Checking the features worked:
test_player = 'Jannik Sinner'
player_slice = timeline[timeline['player'] == test_player].tail(10)
timeline[timeline['player'] == 'Jannik Sinner'].to_csv('sinner_check.csv', index=False)

# Saving the features dataset
print("Saving the features dataset...")
timeline.to_csv('tennis_features.csv', index=False)
print("Saved in 'tennis_features.csv'.")