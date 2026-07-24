DROP VIEW IF EXISTS vw_tennis_with_weather CASCADE;

CREATE VIEW vw_tennis_with_weather AS 
WITH cleaned_matches AS (
    SELECT 
        *,
        TO_DATE(tourney_date::text, 'YYYYMMDD') as clean_tourney_date,
        
        CASE -- Cleaning tourney_name in a new column so it matches with weather cities
            WHEN tourney_name = 'Wimbledon' THEN 'London'
            WHEN tourney_name = 'Roland Garros' THEN 'Paris'
            WHEN tourney_name = 'US Open' THEN 'New York'
            WHEN tourney_name = 'Australian Open' THEN 'Melbourne'
            WHEN tourney_name LIKE '%Madrid%' THEN 'Madrid'
            WHEN tourney_name LIKE '%Rome%' THEN 'Rome'
            WHEN tourney_name LIKE '%Miami%' THEN 'Miami'
            WHEN tourney_name LIKE '%Indian Wells%' THEN 'Indian Wells'
            WHEN tourney_name LIKE '%Monte Carlo%' THEN 'Monte Carlo'
            WHEN tourney_name LIKE '%Shanghai%' THEN 'Shanghai'
            WHEN tourney_name LIKE '%Cincinnati%' THEN 'Cincinnati'
            WHEN tourney_name LIKE '%Canada%' THEN 'Toronto'
            WHEN tourney_name LIKE '%Adelaide%' THEN 'Adelaide'
            WHEN tourney_name LIKE '%Belgrade%' THEN 'Belgrade'
            WHEN tourney_name IN ('Winston Salem', 'Winston-Salem') THEN 'Winston-Salem'
            WHEN tourney_name IN ('ATP Cup', 'Atp Cup', 'United Cup') THEN 'Sydney'
            WHEN tourney_name IN ('Great Ocean Road Open', 'Murray River Open') THEN 'Melbourne'
            WHEN tourney_name IN ('Rio Olympics', 'Rio De Janeiro', 'Rio de Jainero') THEN 'Rio de Janeiro'
            WHEN tourney_name = 'Tokyo Olympics' THEN 'Tokyo'
            WHEN tourney_name = 'Paris Olympics' THEN 'Paris'
            WHEN tourney_name = 'Queen''s Club' THEN 'London'
            WHEN tourney_name = 'Sardinia' THEN 'Cagliari'
            WHEN tourney_name LIKE '%Serbia%' THEN 'Belgrade'
            WHEN tourney_name LIKE '%Hertogenbosch%' THEN 's-Hertogenbosch'
            WHEN tourney_name = 'Buenoa Aires' THEN 'Buenos Aires'
            WHEN tourney_name = 'Marrakesh' THEN 'Marrakech'
            WHEN tourney_name LIKE '%Doha%' THEN 'Doha' 
            
            ELSE tourney_name 
        END as target_city,

        CASE -- Calculating the estimated match date based on the tournament start date and match number as we are only given the tournament start date in the dataset
            -- 2026 has the proper dates, so we don't have to change them
            WHEN LEFT(tourney_date::text, 4) = '2026' THEN TO_DATE(tourney_date::text, 'YYYYMMDD')

            WHEN round IN ('Q1', 'Q2', 'Q3') THEN 
                TO_DATE(tourney_date::text, 'YYYYMMDD') + (
                    CASE 
                        WHEN round = 'Q1' THEN -2 -- Saturday before the tournament starts
                        WHEN round = 'Q2' THEN -1 -- Sunday before the tournament starts
                        ELSE 0                    -- Q3 is Monday morning
                    END
                ) * INTERVAL '1 day'
            
            ELSE TO_DATE(tourney_date::text, 'YYYYMMDD') + (
                CASE 
                    WHEN draw_size <= 1 OR match_num IS NULL THEN 0
                    WHEN draw_size = 128 THEN LEAST(13, FLOOR((match_num::float / (draw_size - 1)) * 14)::int)
                    WHEN draw_size = 96  THEN LEAST(11, FLOOR((match_num::float / (draw_size - 1)) * 12)::int)
                    ELSE LEAST(6, FLOOR((match_num::float / (draw_size - 1)) * 7)::int)
                END
            ) * INTERVAL '1 day'
        END::date as estimated_match_date
    FROM raw_tennis_matches
),

-- Dealing with missing weather data with monthly averages
monthly_baselines AS (
    SELECT 
        target_city,
        EXTRACT(MONTH FROM weather_date::date)::int as weather_month,
        AVG(average_temp) as fallback_avg_temp,
        AVG(max_temp) as fallback_max_temp,
        AVG(wind_speed_kmh) as fallback_wspd
    FROM raw_meteostat_weather
    GROUP BY target_city, EXTRACT(MONTH FROM weather_date::date)
)

-- Match up the tennis matches with the weather, turning to monthly averages if the specific date is missing
SELECT 
    m.*,
    CASE 
        WHEN m.indoor = 'O' THEN ROUND(COALESCE(w.average_temp, b.fallback_avg_temp)::numeric, 1)
        ELSE NULL 
    END as average_temp,
    
    CASE 
        WHEN m.indoor = 'O' THEN ROUND(COALESCE(w.max_temp, b.fallback_max_temp)::numeric, 1)
        ELSE NULL 
    END as max_temp,
    
    CASE 
        WHEN m.indoor = 'O' THEN ROUND(COALESCE(w.wind_speed_kmh, b.fallback_wspd)::numeric, 1)
        ELSE NULL 
    END as wind_speed_kmh
FROM cleaned_matches m

LEFT JOIN raw_meteostat_weather w 
    ON m.target_city = w.target_city 
    AND m.estimated_match_date = w.weather_date::date

LEFT JOIN monthly_baselines b
    ON m.target_city = b.target_city
    AND EXTRACT(MONTH FROM m.estimated_match_date) = b.weather_month;


-- Testing if it has worked...
SELECT * FROM vw_tennis_with_weather
LIMIT 5;
-- All 5 outdoor matches have weather data attached


-- Testing if the tournament name successfully links with the target_city I set
SELECT DISTINCT tourney_name, target_city
FROM vw_tennis_with_weather
WHERE indoor = 'O'
ORDER BY tourney_name;
-- All tournaments are linked to their respective city


-- Check if any weather data is empty
SELECT DISTINCT tourney_name, target_city, estimated_match_date, average_temp, max_temp, wind_speed_kmh
FROM vw_tennis_with_weather
WHERE indoor = 'O'
AND (
      average_temp IS NULL 
      OR max_temp IS NULL 
      OR wind_speed_kmh IS NULL
  )
ORDER BY tourney_name;