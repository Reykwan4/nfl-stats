# NFL Game Predictor - MVP v0.1

A mobile-first, single-page web app that displays weekly NFL game predictions with natural language explanations. This is an analytical tool, not a betting app, focused on transparency and interpretability.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Frontend                             │
│                    (React Single Page)                       │
│                                                              │
│  - Fetches predictions from backend API                     │
│  - Displays game cards (mobile-first)                       │
│  - Shows win probabilities and explanations                 │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │ HTTP GET /predictions/weekly
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                         Backend                              │
│                    (Python + FastAPI)                        │
│                                                              │
│  1. Data Loading: Reads historical games from CSV           │
│  2. Feature Engineering:                                     │
│     - Rolling averages (offense, defense, turnovers)        │
│     - Elo rating system                                      │
│  3. Model Training: Logistic regression                      │
│     Features: Elo diff, rolling stats, home field           │
│  4. Prediction: Win probabilities for upcoming games         │
│  5. Explainability: Natural language reasoning               │
│                                                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Reads CSV
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                      Data Layer                              │
│                                                              │
│  historical_games.csv                                        │
│  - Team-level game results                                  │
│  - Scores, turnovers, dates                                 │
│  (No database for MVP - just CSV files)                     │
└─────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

**Why this architecture?**

1. **Single file backend** (`backend/main.py`): Everything in one place for easy understanding and iteration. No premature abstraction.

2. **Logistic regression**: Simple, interpretable, fast. Coefficients directly show feature importance.

3. **Elo ratings**: Battle-tested system for relative team strength. Updates after each game.

4. **Rolling averages**: Recent performance (last 4 games) better predicts future than season-long stats.

5. **CSV data layer**: No database complexity for MVP. Easy to inspect, version control, and update.

6. **React SPA**: Simple state management, no routing needed. Mobile-first CSS without framework overhead.

7. **Explainability first**: Every prediction includes natural language explanation based on model coefficients.

## Project Structure

```
NFL_Predictor/
├── backend/
│   ├── main.py              # Complete backend (FastAPI + ML model)
│   └── requirements.txt     # Python dependencies
├── frontend/
│   ├── package.json         # Node dependencies
│   ├── public/
│   │   └── index.html       # HTML shell
│   └── src/
│       ├── index.js         # React entry point
│       ├── App.jsx          # Main app component
│       └── App.css          # Mobile-first styles
├── data/
│   ├── README.md            # Data format documentation
│   └── historical_games.csv # Your historical NFL data (add this)
└── README.md                # This file
```

## How to Run Locally

### Prerequisites

- Python 3.8+
- Node.js 16+ and npm
- (Optional) Historical NFL data CSV

### Step 1: Set Up Backend

```bash
# Navigate to backend directory
cd backend

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the backend server
python main.py
```

The backend will start on `http://localhost:8000`

**Note**: If you don't have a `data/historical_games.csv` file, the backend will automatically generate sample data for testing.

### Step 2: Set Up Frontend

Open a new terminal window:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start the development server
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

### Step 3: View Predictions

Once both servers are running, visit `http://localhost:3000` in your browser (mobile or desktop).

You should see:
- Weekly game predictions
- Win probabilities for each team
- Predicted winner highlighted
- Natural language explanations

## API Endpoints

The backend exposes these endpoints:

### `GET /`
Health check

**Response:**
```json
{
  "message": "NFL Prediction API v0.1",
  "status": "running",
  "model_trained": true
}
```

### `GET /predictions/weekly`
Get predictions for upcoming week

**Query params:**
- `week` (optional): Week identifier (default: "upcoming")

**Response:**
```json
{
  "week": "upcoming",
  "total_games": 5,
  "games": [
    {
      "home_team": "Bills",
      "away_team": "Chiefs",
      "home_win_probability": 62.3,
      "away_win_probability": 37.7,
      "predicted_winner": "Bills",
      "home_elo": 1650,
      "away_elo": 1580,
      "explanation": "The model favors the Bills (62% confidence) primarily due to higher Elo rating, stronger recent offense, and home-field advantage."
    }
  ]
}
```

### `GET /predict`
Predict a single game

**Query params:**
- `home_team`: Home team name
- `away_team`: Away team name

**Example:**
```
GET /predict?home_team=Bills&away_team=Chiefs
```

### `GET /teams`
Get all teams with current Elo ratings

**Response:**
```json
{
  "teams": [
    {"team": "Chiefs", "elo_rating": 1650},
    {"team": "Bills", "elo_rating": 1620}
  ]
}
```

## Model Details

### Features

The model uses 8 features to predict game outcomes:

1. **Elo difference**: Home team Elo - Away team Elo + 65 (home advantage)
2. **Home offense**: Home team's rolling average points scored (last 4 games)
3. **Away offense**: Away team's rolling average points scored
4. **Home defense**: Home team's rolling average points allowed
5. **Away defense**: Away team's rolling average points allowed
6. **Home turnover margin**: Home team's turnover differential
7. **Away turnover margin**: Away team's turnover differential
8. **Home field**: Binary indicator (always 1 for home team)

### Training Process

1. Load historical games from CSV
2. Calculate rolling averages for each team over time
3. Initialize Elo ratings (1500 for all teams)
4. For each historical game:
   - Extract features at game time
   - Record actual outcome
   - Update Elo ratings after game
5. Train logistic regression on all historical games
6. Use current Elo/rolling stats to predict upcoming games

### Explainability

For each prediction, the system:
1. Calculates each feature's contribution (coefficient × feature_value)
2. Ranks features by absolute contribution
3. Selects top 3 most important factors
4. Generates natural language explanation

Example:
> "The model favors the Bills (62% confidence) primarily due to higher Elo rating, stronger recent offense (28.5 ppg), and home-field advantage."

## Adding Your Own Data

Replace the sample data with real NFL data:

1. Create `data/historical_games.csv` with these columns:
   - `date`: YYYY-MM-DD format
   - `home_team`: Team name
   - `away_team`: Team name
   - `home_score`: Integer
   - `away_score`: Integer
   - `home_turnovers`: Integer
   - `away_turnovers`: Integer

2. Restart the backend server

The model will automatically retrain on your data.

## Testing the App

### Quick Smoke Test

1. Start both backend and frontend
2. Visit `http://localhost:3000`
3. Verify you see game predictions with:
   - Team names
   - Win probabilities (0-100%)
   - Predicted winner badge
   - Explanation text

### Test Backend Directly

```bash
# Health check
curl http://localhost:8000/

# Get predictions
curl http://localhost:8000/predictions/weekly

# Predict specific game
curl "http://localhost:8000/predict?home_team=Bills&away_team=Chiefs"

# View team Elo ratings
curl http://localhost:8000/teams
```

## Next 3 Incremental Improvements

Once you have the MVP running, consider these enhancements (one at a time):

### 1. Improve Model Accuracy
**What to add:**
- Weather data (temperature, wind, precipitation)
- Injury reports (key player availability)
- Rest days (short week, bye week effects)
- Head-to-head history between specific teams

**Why:** More features = more nuanced predictions. These factors significantly impact game outcomes.

**How:** Add columns to CSV, create new feature engineering functions, retrain model.

### 2. Historical Accuracy Tracking
**What to add:**
- Store predictions in a simple JSON file
- After games complete, compare predictions vs actual results
- Display model accuracy metrics (overall %, by team, by week)
- Show calibration curve (predicted 60% should win ~60% of time)

**Why:** Builds trust in predictions and helps identify model weaknesses.

**How:** Add a `/results` endpoint, create a results page in frontend, log predictions with timestamps.

### 3. Confidence Intervals and Uncertainty
**What to add:**
- Display prediction confidence bands (e.g., "55-65% win probability")
- Flag "toss-up games" (45-55% probability)
- Show historical accuracy for similar probability predictions
- Add uncertainty based on data quality (team with few games = higher uncertainty)

**Why:** Predictions are never certain. Showing confidence helps users interpret responsibly.

**How:** Use sklearn's decision function, calculate standard errors, add uncertainty to UI.

## Troubleshooting

### Backend won't start
- Check Python version: `python --version` (need 3.8+)
- Ensure virtual environment is activated
- Install dependencies: `pip install -r requirements.txt`

### Frontend won't start
- Check Node version: `node --version` (need 16+)
- Delete `node_modules` and `package-lock.json`, reinstall: `npm install`
- Check port 3000 is available

### No predictions showing
- Open browser console (F12) and check for errors
- Verify backend is running: visit `http://localhost:8000/`
- Check CORS is enabled in backend (it is by default)

### Predictions look wrong
- Verify your CSV data format matches expected schema
- Check backend logs for data loading errors
- Ensure historical games have reasonable scores and dates

## Philosophy: Why This MVP Works

This MVP prioritizes:

**Simplicity over scalability**
- No database, no authentication, no caching
- Everything you need to understand and modify in ~500 lines of code
- Can run on your laptop without any infrastructure

**Explainability over accuracy**
- Model coefficients are interpretable
- Explanations link directly to features
- Users understand *why* a prediction was made

**Iteration over perfection**
- Version 0.1 is meant to be replaced
- Easy to add features one at a time
- No architectural decisions that lock you in

**Mobile-first, not mobile-only**
- Works great on phone (primary use case)
- Progressively enhances on larger screens
- No separate mobile/desktop codebases

## Contributing Ideas

Once you've built the MVP, here are directions to explore:

**Model improvements:**
- Try gradient boosting (XGBoost, LightGBM)
- Ensemble multiple models
- Add deep learning for complex interactions

**Feature engineering:**
- Player-level statistics (QB rating, injuries)
- Situational factors (weather, rivalry games)
- Advanced metrics (DVOA, EPA)

**UI enhancements:**
- Dark mode toggle
- Filter by team or conference
- Historical predictions browser
- Comparison view (model vs Vegas lines)

**Data pipeline:**
- Automated data scraping
- Real-time score updates
- Push notifications for game day

## License

This is a personal project. Use and modify as you see fit.

## Questions?

This is Version 0.1 - a starting point, not a finished product. Start here, then iterate based on what you learn.

**Remember:** One change per iteration. Don't rebuild from scratch.
