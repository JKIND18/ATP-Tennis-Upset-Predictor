# ATP Tennis Match Upset Predictor & Weather Analytics Pipeline

A end-to-end machine learning, feature engineering, and relational database pipeline built to predict match outcomes and upset probabilities on the ATP Tennis Tour. 

By pairing historical match logs with high-resolution weather data from global WMO airport stations (Meteostat API), this pipeline quantifies the trade-off between court time benefit (match sharpness) and cumulative fatigue, as well as weather tolerance, surface adaptability, and player momentum.

---

## Project Motivation & Research Goals

In grand slam tennis—such as Jannik Sinner's grueling multi-set battles at Roland Garros—there is a subtle trade-off between **court time advantage** (gaining tournament rhythm and match sharpness) and **cumulative fatigue** (physical wear and tear).

This project was built to research and model these dynamics:
1. **The Court Time / Rest Dynamics:** Does extra time on court help a player adjust to conditions, or does it leave them vulnerable to physical fatigue against a rested opponent?
2. **Weather Sensitivity:** How do extreme temperatures ($\ge 30^\circ\text{C}$ heat, $\ge 25^\circ\text{C}$ warm, $\ge 20\text{ km/h}$ winds) impact player performance and upset rates?
3. **Upset Mechanics:** Can custom feature engineering identify when an underdog (lower-ranked player) is statistically favored to beat a top-ranked opponent?

---

## 📐 System Architecture

```
┌────────────────────────┐      ┌───────────────────────────┐
│ ATP Match CSVs (Data/) │      │ Meteostat Weather API     │
└───────────┬────────────┘      └─────────────┬─────────────┘
            │                                 │
            ▼                                 ▼
   combined_data.csv              modern_tennis_weather.csv
            │                                 │
            └────────────────┬────────────────┘
                             │ (SQL View Synchronization)
                             ▼
              ┌─────────────────────────────┐
              │ PostgreSQL Database         │
              │  - raw_tennis_matches       │
              │  - raw_meteostat_weather    │
              │  - vw_tennis_with_weather   │
              └──────────────┬──────────────┘
                             │
                             ▼
                     prepare_features.py
             (Time-series feature engineering)
                             │
                             ▼
                    tennis_features.csv
                             │
                             ▼
                      train_model.py
         (Chronological Split & XGBoost Classifier)
```

---

## Data Engineering & Pipeline Journey

### 1. Weather Integration & SQL Normalization (`merge_weather.sql`)
* **Station Lookup & Hourly Aggregations:** Mapped ATP tournament locations to WMO airport weather station IDs. For cities with incomplete daily summaries (e.g., Doha returning hourly logs only), built custom Python aggregation scripts to convert hourly weather into daily metrics (`average_temp`, `max_temp`, `wind_speed_kmh`).
* **Match Date Estimation:** ATP records only provided tournament start dates (`tourney_date`). Implemented SQL logic to estimate exact match dates based on draw size ($128, 96, 64$) and match sequence numbers, shifting qualifying rounds ($Q1, Q2, Q3$) back by 1–2 days.
* **Mean Imputation Fallback:** Unrecognized venues or missing dates were filled using historical monthly averages for that target city to ensure zero missing weather attributes for outdoor matches.

### 2. Time-Series Feature Engineering (`prepare_features.py`)
To prevent data leakage, all features were computed using strict `.shift(1)` and chronological rolling windows:
* **Fatigue & Rest Tracing:** `days_since_last_match`, `time_on_court_last_match`, `time_on_court_this_tournament`, and `sets_dropped_this_tournament`.
* **Momentum & Form:** `surface_win_streak`, `main_draw_wins_last_tournament`, and 30-day rolling `best_win_last_30_days` (using time-indexed rolling minimums of opponent ranks beaten).
* **Head-to-Head & Handedness:** Opponent dominant hand win percentages and `h2h_win_pct`.
* **Environmental Win Rates:** Individual win percentages calculated under extreme heat ($\ge 30^\circ\text{C}$), warm weather ($\ge 25^\circ\text{C}$), and high winds ($\ge 20\text{ km/h}$).
* **Career Milestones:** Expanding mean career rank, `career_high_rank`, and `days_since_career_high`.
* **Home Ground Advantage:** Mapped tournament target cities to IOC country codes to detect when a player was competing on home turf.

### 3. Bidirectional Side-Join & Cartesian Bug Resolution
* **The Problem:** Initial side-joining on `[estimated_match_date, tourney_id, round_num]` caused a $16 \times 16$ cross-join explosion on day 1 of tournaments (expanding 29,231 matches into 165,490 duplicated rows).
* **The Fix:** Applied a bidirectional `left_on` and `right_on` merge on player names (`left_on=['player', 'opponent']`, `right_on=['opponent', 'player']`), collapsing the dataset down to **28,496 clean physical matches**.

### 4. Target Balancing & Differential Transformations (`train_model.py`)
* Applied a 50/50 `np.random.rand` coin-flip mask to assign Player 1 and Player 2 dynamically, balancing the target outcome (`p1_won`) to **49.9% / 50.1%**.
* Formatted input vectors as differential pairs ($\text{Feature}_{\text{Diff}} = \text{P1} - \text{P2}$) so the XGBoost classifier learns relative athletic and form edges rather than absolute values.


### Feature Dataset Sample (`sinner_check.csv`)
To inspect the engineered time-series format without downloading the multi-gigabyte dataset, a 450-match sample tracking Jannik Sinner's career progression from 2019 onward is included in `sinner_check.csv`.

| Date | Player | Opponent | Surface | Rank | Opp. Rank | Avg Career Rank | Days Rest | Court Time This Tourney (m) | Surface Win % | H2H Win % |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2019-04-20** | Jannik Sinner | Lukas Rosol | Clay | 314 | 139 | 314.0 | 30.0 | 0.0 | 50.0% | 50.0% |
| **2019-04-21** | Jannik Sinner | Yannick Maden | Clay | 314 | 109 | 314.0 | 1.0 | 43.0 | 100.0% | 50.0% |
| **2019-04-23** | Jannik Sinner | Mate Valkusz | Clay | 314 | 323 | 314.0 | 2.0 | 139.0 | 50.0% | 50.0% |

---

## Model Performance & Results

Model trained on **22,796 historical matches** and evaluated on **5,700 unseen future test matches** using a strict chronological split.

### Overall Performance Indicators

| Indicator | Result | Insight |
| :--- | :--- | :--- |
| **Model Accuracy** | **71.49%** | Correctly predicts match winners on unseen future tour matches |
| **Log Loss** | **0.5616** | Significant calibration improvement over a 0.6931 coin-flip baseline |
| **Upset Precision** | **67.13%** | Hit rate when predicting an underdog win (overall test set) |
| **Clean Upset Precision** | **69.56%** | Hit rate on upsets when excluding non-completed retirement/WO matches |
| **Upset Catch Rate (Recall)**| **42.13%** | Percentage of all actual tour upsets successfully identified |

---

### Top 10 Most Influential Predictors

```text
1. rank_diff                             19.96%  (Current ATP Ranking Gap)
2. main_draw_wins_last_tournament_diff   17.37%  (Previous Tournament Form & Momentum)
3. surface_win_pct_diff                   6.49%  (Surface Specialization)
4. career_high_rank_diff                  5.57%  (Peak Proven Quality)
5. win_pct_against_opponent_hand_diff     5.54%  (Left-handed / Right-handed Matchup)
6. windy_weather_win_pct_diff             4.98%  (Wind Adaptability Edge)
7. avg_career_rank_diff                   4.27%  (Long-Term Career Baseline)
8. hot_weather_win_pct_diff               3.81%  (Heat & Temperature Tolerance)
9. surface_win_streak_diff                3.36%  (Active Surface Form)
10. best_win_last_30_days_diff            3.32%  (30-Day Peak Win Quality)
```

---

## Repository Structure

```text
├── Data/                          # Directory for raw match CSVs (git-ignored)
├── weather_data.py                # Meteostat API fetcher & hourly-to-daily station converter
├── all_data.py                    # Concatenates raw CSVs and mirrors to PostgreSQL tables
├── merge_weather.sql              # SQL view script for match date estimation & weather join
├── prepare_features.py            # Master feature engineering pipeline
├── train_model.py                 # Chronological XGBoost training & evaluation script
├── .env.example                   # Environment variable template
├── .gitignore                     # Git tracking exclusions
└── README.md                      # Project documentation
```

---

## How to Run Locally

### 1. Installation
```bash
git clone [https://github.com/JKIND18/ATP-Tennis-Upset-Predictor.git](https://github.com/JKIND18/ATP-Tennis-Upset-Predictor.git)
cd ATP-Tennis-Upset-Predictor
pip install pandas numpy xgboost scikit-learn sqlalchemy psycopg2-binary python-dotenv meteostat
```

### 2. Configure Environment
Create a `.env` file from the provided template:
```bash
cp .env.example .env
```
Update `.env` with your PostgreSQL connection parameters:
```text
TENNIS_DB_URL=postgresql://postgres:your_password@localhost:5432/tennis_db
```

### 3. Pipeline Execution Sequence
```bash
# 1. Fetch weather station data from Meteostat API
python weather_data.py

# 2. Synchronize PostgreSQL database tables and views
python all_data.py

# 3. Process time-series features
python prepare_features.py

# 4. Train XGBoost classifier & output performance analytics
python train_model.py
```