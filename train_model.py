import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, log_loss


print("Loading the features dataset...")
df = pd.read_csv('tennis_features.csv', low_memory=False)
print(f"Loaded {len(df)} rows and {len(df.columns)} columns from 'tennis_features.csv'.")


# Only main draw matches
df = df[df['round_num'] >= 4].copy()

winners = df[df['match_won'] == 1].copy()
losers = df[df['match_won'] == 0].copy()


# Side join together
merged = pd.merge(
    winners, losers, 
    left_on=['estimated_match_date', 'tourney_id', 'round_num', 'player', 'opponent'],
    right_on=['estimated_match_date', 'tourney_id', 'round_num', 'opponent', 'player'],
    suffixes=('_win', '_loss')
)


# Flip half so winners aren't always "winning"
np.random.seed(42)
mask = np.random.rand(len(merged)) < 0.5

train_df = pd.DataFrame()

# 1 if winner stays p1, 0 is loser flips to p1
train_df['p1_won'] = np.where(mask, 1, 0)


# Features to calculate the difference with
features_to_compare = [
    'rank', 
    'age',
    'avg_career_rank',
    'career_high_rank',
    'days_since_career_high',
    'days_since_last_match',
    'main_draw_wins_last_tournament',
    'time_on_court_last_match',
    'time_on_court_this_tournament',
    'sets_dropped_this_tournament',
    'best_win_this_tournament',
    'best_win_last_30_days',
    'days_since_last_injury',
    'matches_on_surface_this_season',
    'surface_win_pct',
    'surface_win_streak',
    'win_pct_against_opponent_hand',
    'h2h_wins',
    'h2h_total_matches',
    'h2h_win_pct',
    'is_home_ground',
    'extreme_heat_win_pct',
    'hot_weather_win_pct',
    'windy_weather_win_pct'
]


# Assign player 1 and 2
for feat in features_to_compare:
    col_win  = f"{feat}_win"
    col_loss = f"{feat}_loss"
    
    # Fill missing values cleanly
    merged[col_win]  = pd.to_numeric(merged[col_win], errors='coerce').fillna(0)
    merged[col_loss] = pd.to_numeric(merged[col_loss], errors='coerce').fillna(0)
    
    # Assign p1 and p2 based on the coin flip mask
    p1_val = np.where(mask, merged[col_win], merged[col_loss])
    p2_val = np.where(mask, merged[col_loss], merged[col_win])
    
    # Calculate the subtraction feature (p1 minus p2)
    train_df[f"{feat}_diff"] = p1_val - p2_val


p1_won = np.where(mask, 1, 0)
print(f"Total Collapsed Match Rows: {len(merged)}")
print(f"Player 1 Wins (1s): {np.sum(p1_won == 1)} ({np.mean(p1_won):.1%})")
print(f"Player 1 Losses (0s): {np.sum(p1_won == 0)} ({1 - np.mean(p1_won):.1%})")


# Track who was the underdog
p1_rank = np.where(mask, merged['rank_win'], merged['rank_loss'])
p2_rank = np.where(mask, merged['rank_loss'], merged['rank_win'])
train_df['p1_is_underdog'] = (p1_rank > p2_rank).astype(int)


# Dates in order
train_df['estimated_match_date'] = merged['estimated_match_date']
train_df = train_df.sort_values('estimated_match_date').reset_index(drop=True)

y = train_df['p1_won']
X = train_df.drop(columns=['p1_won', 'estimated_match_date', 'p1_is_underdog'])


# Split 80% Train, 20% Test
split_idx = int(len(train_df) * 0.8)

X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

underdog_test = train_df['p1_is_underdog'].iloc[split_idx:]

print(f"\nChronological Split Complete:")
print(f"Training Matches: {len(X_train)} (Older matches)")
print(f"Testing Matches:  {len(X_test)} (Most recent matches)")


# Train the model
model = xgb.XGBClassifier(
    n_estimators=150,      # Number of decision trees
    max_depth=4,           # Max depth per tree (prevents overfitting)
    learning_rate=0.03,    # Step size shrinkage
    subsample=0.8,         # Row sampling per tree
    colsample_bytree=0.8,  # Feature sampling per tree
    random_state=42,
    eval_metric='logloss'
)

print("\nTraining XGBoost model...")
model.fit(X_train, y_train)


# Evaluate performance on future matches
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1] # Probabilities for Player 1 winning

acc = accuracy_score(y_test, y_pred)
loss = log_loss(y_test, y_proba)

print(f"\n==========================================")
print(f"  MODEL ACCURACY: {acc:.2%}")
print(f"  LOG LOSS:       {loss:.4f}")
print(f"==========================================")


# Top 10 most important features
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 10 Most Influential Predictors:")
print(importances.head(10))




actual_upsets = ((underdog_test == 1) & (y_test == 1)) | ((underdog_test == 0) & (y_test == 0))
predicted_upsets = ((underdog_test == 1) & (y_pred == 1)) | ((underdog_test == 0) & (y_pred == 0))

hits = actual_upsets & predicted_upsets
misses = actual_upsets & (~predicted_upsets)
false_alarms = (~actual_upsets) & predicted_upsets

print("\n==========================================")
print("        UPSET PERFORMANCE BREAKDOWN       ")
print("==========================================")
print(f"Total Test Matches Evaluated:        {len(X_test)}")
print(f"Actual Upsets That Occurred:        {actual_upsets.sum()} ({actual_upsets.mean():.1%})")
print(f"Total Upsets Predicted by Model:     {predicted_upsets.sum()}")
print(f"------------------------------------------")
print(f"Correctly Predicted Upsets (Hits): {hits.sum()}")
print(f"Missed Upsets (Pick Favorite):     {misses.sum()}")
print(f"False Upset Calls (Pick Underdog): {false_alarms.sum()}")
print(f"------------------------------------------")
if predicted_upsets.sum() > 0:
    precision = hits.sum() / predicted_upsets.sum()
    print(f"Upset Prediction Precision:        {precision:.2%}")
if actual_upsets.sum() > 0:
    catch = hits.sum() / actual_upsets.sum()
    print(f"Upset Prediction Catch Rate:        {catch:.2%}")
print("==========================================")