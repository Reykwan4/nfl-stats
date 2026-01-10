# Data Directory

Place your NFL historical data CSV here.

## Expected CSV Format

The backend expects a file named `historical_games.csv` with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| date | Date (YYYY-MM-DD) | Game date |
| home_team | String | Home team name |
| away_team | String | Away team name |
| home_score | Integer | Points scored by home team |
| away_score | Integer | Points scored by away team |
| home_turnovers | Integer | Turnovers committed by home team |
| away_turnovers | Integer | Turnovers committed by away team |

## Example CSV

```csv
date,home_team,away_team,home_score,away_score,home_turnovers,away_turnovers
2023-09-07,Bills,Jets,31,10,0,2
2023-09-10,Chiefs,Lions,21,20,1,1
2023-09-10,Eagles,Patriots,25,20,0,3
```

## Data Sources

You can obtain historical NFL data from:

- [NFL.com](https://www.nfl.com) - Official stats
- [Pro Football Reference](https://www.pro-football-reference.com) - Comprehensive historical data
- [Kaggle NFL datasets](https://www.kaggle.com/search?q=nfl) - Community datasets

## Note

If no CSV is provided, the backend will generate sample data for demonstration purposes. This allows you to test the app immediately without data preparation.
