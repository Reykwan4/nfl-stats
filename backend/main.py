"""
NFL Game Prediction Backend - MVP Version 0.1

This single-file backend implements:
1. CSV data loading (historical NFL team stats)
2. Feature engineering: rolling averages for offense, defense, turnover margin
3. Elo rating system for team strength
4. Logistic regression model for win probability
5. Natural language explanations based on model coefficients
6. FastAPI endpoint to serve weekly predictions

Design decisions:
- Everything in one file for simplicity
- Model trains on historical data, predicts upcoming games
- Features: Elo diff, rolling averages (offense/defense/turnovers), home field
- Explainability: rank features by coefficient * feature_value contribution
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Any
import os
from pathlib import Path

app = FastAPI(title="NFL Prediction API")

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to data directory (relative to this file)
DATA_DIR = Path(__file__).parent.parent / "data"

# Elo rating parameters
INITIAL_ELO = 1500
K_FACTOR = 20  # How much ratings change per game
HOME_ADVANTAGE = 65  # Elo points advantage for home team

# Rolling average window (games)
ROLLING_WINDOW = 4  # Look at last 4 games for rolling averages

# Feature names used in the model
FEATURE_NAMES = [
    "elo_diff",           # Difference in Elo ratings (home - away)
    "home_off_avg",       # Home team's rolling average points scored
    "away_off_avg",       # Away team's rolling average points scored
    "home_def_avg",       # Home team's rolling average points allowed
    "away_def_avg",       # Away team's rolling average points allowed
    "home_to_margin",     # Home team's turnover margin (turnovers forced - lost)
    "away_to_margin",     # Away team's turnover margin
    "is_home"             # Binary: 1 for home team advantage
]

# ============================================================================
# DATA LOADING
# ============================================================================

def load_historical_data() -> pd.DataFrame:
    """
    Load historical NFL game data from CSV.

    Expected CSV columns:
    - date: game date
    - home_team: home team name
    - away_team: away team name
    - home_score: points scored by home team
    - away_score: points scored by away team
    - home_turnovers: turnovers committed by home team
    - away_turnovers: turnovers committed by away team

    Returns DataFrame sorted by date.
    """
    csv_path = DATA_DIR / "historical_games.csv"

    if not csv_path.exists():
        # Return sample data for demonstration if CSV doesn't exist
        print(f"Warning: {csv_path} not found. Using sample data.")
        return create_sample_data()

    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)

    return df

def create_sample_data() -> pd.DataFrame:
    """
    Creates sample NFL game data for demonstration.
    In production, this would be replaced with real CSV data.

    Now includes offensive breakdowns:
    - Rush yards and pass yards
    - Rush TDs and pass TDs
    """
    np.random.seed(42)
    teams = ['Bills', 'Chiefs', 'Eagles', 'Cowboys', '49ers', 'Ravens',
             'Dolphins', 'Bengals', 'Jaguars', 'Chargers']

    data = []
    dates = pd.date_range(start='2023-09-01', periods=100, freq='W')

    for date in dates:
        home = np.random.choice(teams)
        away = np.random.choice([t for t in teams if t != home])

        # Simulate rushing stats (realistic NFL ranges)
        home_rush_yards = np.random.randint(80, 180)
        away_rush_yards = np.random.randint(80, 180)
        home_rush_tds = np.random.randint(0, 3)
        away_rush_tds = np.random.randint(0, 3)

        # Simulate passing stats (realistic NFL ranges)
        home_pass_yards = np.random.randint(180, 350)
        away_pass_yards = np.random.randint(180, 350)
        home_pass_tds = np.random.randint(0, 4)
        away_pass_tds = np.random.randint(0, 4)

        # Calculate total scores from TDs (simplified: TD = 7 points, add some FGs)
        home_score = (home_rush_tds + home_pass_tds) * 7 + np.random.randint(0, 10)
        away_score = (away_rush_tds + away_pass_tds) * 7 + np.random.randint(0, 10)

        data.append({
            'date': date,
            'home_team': home,
            'away_team': away,
            'home_score': home_score,
            'away_score': away_score,
            'home_turnovers': np.random.randint(0, 4),
            'away_turnovers': np.random.randint(0, 4),
            # Offensive stats
            'home_rush_yards': home_rush_yards,
            'away_rush_yards': away_rush_yards,
            'home_pass_yards': home_pass_yards,
            'away_pass_yards': away_pass_yards,
            'home_rush_tds': home_rush_tds,
            'away_rush_tds': away_rush_tds,
            'home_pass_tds': home_pass_tds,
            'away_pass_tds': away_pass_tds,
            # Defensive stats (yards allowed)
            'home_rush_yards_allowed': away_rush_yards,  # Home defense vs away rush
            'away_rush_yards_allowed': home_rush_yards,  # Away defense vs home rush
            'home_pass_yards_allowed': away_pass_yards,  # Home defense vs away pass
            'away_pass_yards_allowed': home_pass_yards   # Away defense vs home pass
        })

    return pd.DataFrame(data)

# ============================================================================
# ELO RATING SYSTEM
# ============================================================================

class EloRatingSystem:
    """
    Elo rating system for NFL teams.

    Higher Elo = stronger team. Elo adjusts after each game based on:
    - Expected win probability (based on rating difference)
    - Actual result (win/loss)
    - K-factor (magnitude of rating change)
    """

    def __init__(self, k_factor: float = K_FACTOR, initial_elo: float = INITIAL_ELO):
        self.k_factor = k_factor
        self.initial_elo = initial_elo
        self.ratings: Dict[str, float] = {}

    def get_rating(self, team: str) -> float:
        """Get current Elo rating for a team. Initialize if new."""
        if team not in self.ratings:
            self.ratings[team] = self.initial_elo
        return self.ratings[team]

    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Calculate expected win probability for team A vs team B.
        Uses logistic curve: 1 / (1 + 10^((rating_b - rating_a) / 400))
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update_ratings(self, team_a: str, team_b: str, score_a: float,
                       margin_of_victory: int = 0):
        """
        Update Elo ratings after a game.

        Args:
            team_a: First team
            team_b: Second team
            score_a: Actual score (1 for team_a win, 0 for loss, 0.5 for tie)
            margin_of_victory: Point differential (optional, not used in basic Elo)
        """
        rating_a = self.get_rating(team_a)
        rating_b = self.get_rating(team_b)

        expected_a = self.expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a

        # Update ratings
        self.ratings[team_a] = rating_a + self.k_factor * (score_a - expected_a)
        self.ratings[team_b] = rating_b + self.k_factor * ((1 - score_a) - expected_b)

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_rolling_averages(df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    """
    Calculate rolling averages for each team:
    - Points scored (offense)
    - Points allowed (defense)
    - Turnover margin (turnovers forced - turnovers committed)
    - Rush yards, pass yards (if available)
    - Rush TDs, pass TDs (if available)

    Returns DataFrame with team stats over time.
    """
    team_stats = []

    all_teams = set(df['home_team'].unique()) | set(df['away_team'].unique())

    # Check if detailed offensive and defensive stats are available
    has_detailed_stats = all(col in df.columns for col in [
        'home_rush_yards', 'away_rush_yards',
        'home_pass_yards', 'away_pass_yards',
        'home_rush_tds', 'away_rush_tds',
        'home_pass_tds', 'away_pass_tds',
        'home_rush_yards_allowed', 'away_rush_yards_allowed',
        'home_pass_yards_allowed', 'away_pass_yards_allowed'
    ])

    for team in all_teams:
        # Get all games for this team (as home or away)
        home_games = df[df['home_team'] == team].copy()
        away_games = df[df['away_team'] == team].copy()

        # Create unified view of team performance
        home_games['points_scored'] = home_games['home_score']
        home_games['points_allowed'] = home_games['away_score']
        home_games['turnovers_committed'] = home_games['home_turnovers']
        home_games['turnovers_forced'] = home_games['away_turnovers']

        away_games['points_scored'] = away_games['away_score']
        away_games['points_allowed'] = away_games['home_score']
        away_games['turnovers_committed'] = away_games['away_turnovers']
        away_games['turnovers_forced'] = away_games['home_turnovers']

        # Add detailed offensive and defensive stats if available
        if has_detailed_stats:
            # Offensive stats
            home_games['rush_yards'] = home_games['home_rush_yards']
            home_games['pass_yards'] = home_games['home_pass_yards']
            home_games['rush_tds'] = home_games['home_rush_tds']
            home_games['pass_tds'] = home_games['home_pass_tds']

            away_games['rush_yards'] = away_games['away_rush_yards']
            away_games['pass_yards'] = away_games['away_pass_yards']
            away_games['rush_tds'] = away_games['away_rush_tds']
            away_games['pass_tds'] = away_games['away_pass_tds']

            # Defensive stats (yards allowed)
            home_games['rush_yards_allowed'] = home_games['home_rush_yards_allowed']
            home_games['pass_yards_allowed'] = home_games['home_pass_yards_allowed']

            away_games['rush_yards_allowed'] = away_games['away_rush_yards_allowed']
            away_games['pass_yards_allowed'] = away_games['away_pass_yards_allowed']

        # Select columns for concatenation
        base_cols = ['date', 'points_scored', 'points_allowed',
                     'turnovers_committed', 'turnovers_forced']
        detail_cols = ['rush_yards', 'pass_yards', 'rush_tds', 'pass_tds',
                       'rush_yards_allowed', 'pass_yards_allowed'] if has_detailed_stats else []

        # Combine and sort by date
        team_games = pd.concat([
            home_games[base_cols + detail_cols],
            away_games[base_cols + detail_cols]
        ]).sort_values('date').reset_index(drop=True)

        # Calculate rolling averages
        team_games['off_avg'] = team_games['points_scored'].rolling(
            window=window, min_periods=1).mean()
        team_games['def_avg'] = team_games['points_allowed'].rolling(
            window=window, min_periods=1).mean()
        team_games['to_margin'] = (
            team_games['turnovers_forced'] - team_games['turnovers_committed']
        ).rolling(window=window, min_periods=1).mean()

        # Calculate detailed offensive and defensive rolling averages if available
        if has_detailed_stats:
            # Offensive rolling averages
            team_games['rush_yards_avg'] = team_games['rush_yards'].rolling(
                window=window, min_periods=1).mean()
            team_games['pass_yards_avg'] = team_games['pass_yards'].rolling(
                window=window, min_periods=1).mean()
            team_games['rush_tds_avg'] = team_games['rush_tds'].rolling(
                window=window, min_periods=1).mean()
            team_games['pass_tds_avg'] = team_games['pass_tds'].rolling(
                window=window, min_periods=1).mean()

            # Defensive rolling averages (yards allowed)
            team_games['rush_yards_allowed_avg'] = team_games['rush_yards_allowed'].rolling(
                window=window, min_periods=1).mean()
            team_games['pass_yards_allowed_avg'] = team_games['pass_yards_allowed'].rolling(
                window=window, min_periods=1).mean()

        team_games['team'] = team
        team_stats.append(team_games)

    return pd.concat(team_stats, ignore_index=True)

def build_training_data(df: pd.DataFrame, elo_system: EloRatingSystem) -> tuple:
    """
    Build feature matrix X and target vector y for model training.

    For each historical game:
    1. Get Elo ratings before the game
    2. Get rolling averages before the game
    3. Create feature vector
    4. Update Elo ratings after the game

    Returns (X, y, feature_scaler) where:
    - X: feature matrix (n_games, n_features)
    - y: binary outcomes (1 = home win, 0 = away win)
    - feature_scaler: StandardScaler fitted on X
    """
    # Calculate rolling stats for all teams
    team_stats = calculate_rolling_averages(df)

    X_list = []
    y_list = []

    for idx, game in df.iterrows():
        date = game['date']
        home_team = game['home_team']
        away_team = game['away_team']
        home_score = game['home_score']
        away_score = game['away_score']

        # Get Elo ratings BEFORE this game
        home_elo = elo_system.get_rating(home_team)
        away_elo = elo_system.get_rating(away_team)
        elo_diff = home_elo - away_elo + HOME_ADVANTAGE

        # Get rolling averages BEFORE this game
        # (Most recent game before current date)
        home_stats = team_stats[
            (team_stats['team'] == home_team) &
            (team_stats['date'] < date)
        ].tail(1)

        away_stats = team_stats[
            (team_stats['team'] == away_team) &
            (team_stats['date'] < date)
        ].tail(1)

        # Skip if insufficient history
        if home_stats.empty or away_stats.empty:
            continue

        # Build feature vector
        features = [
            elo_diff,
            home_stats['off_avg'].values[0],
            away_stats['off_avg'].values[0],
            home_stats['def_avg'].values[0],
            away_stats['def_avg'].values[0],
            home_stats['to_margin'].values[0],
            away_stats['to_margin'].values[0],
            1  # is_home indicator
        ]

        X_list.append(features)
        y_list.append(1 if home_score > away_score else 0)

        # Update Elo ratings AFTER this game
        home_won = 1 if home_score > away_score else 0
        elo_system.update_ratings(home_team, away_team, home_won)

    X = np.array(X_list)
    y = np.array(y_list)

    # Standardize features (important for logistic regression)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """
    Train logistic regression model to predict game outcomes.

    Logistic regression is chosen because:
    - Outputs probabilities (0-1) naturally
    - Coefficients are interpretable (feature importance)
    - Fast to train and predict
    - Works well with ~8 features

    Returns trained model.
    """
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'  # Good for small datasets
    )

    model.fit(X, y)

    print(f"Model trained on {len(X)} games")
    print(f"Model accuracy: {model.score(X, y):.3f}")

    return model

# ============================================================================
# PREDICTION & EXPLAINABILITY
# ============================================================================

def predict_game(home_team: str, away_team: str, model: LogisticRegression,
                 scaler: StandardScaler, elo_system: EloRatingSystem,
                 team_stats: pd.DataFrame) -> Dict[str, Any]:
    """
    Predict outcome of a single game and generate explanation.

    Args:
        home_team: Home team name
        away_team: Away team name
        model: Trained logistic regression model
        scaler: Fitted StandardScaler
        elo_system: Current Elo ratings
        team_stats: Rolling averages for all teams

    Returns dict with prediction and explanation.
    """
    # Get current Elo ratings
    home_elo = elo_system.get_rating(home_team)
    away_elo = elo_system.get_rating(away_team)
    elo_diff = home_elo - away_elo + HOME_ADVANTAGE

    # Get most recent rolling averages
    home_stats = team_stats[team_stats['team'] == home_team].tail(1)
    away_stats = team_stats[team_stats['team'] == away_team].tail(1)

    # Handle case where team has no history (use league averages)
    if home_stats.empty:
        home_off = 24.0
        home_def = 24.0
        home_to = 0.0
    else:
        home_off = home_stats['off_avg'].values[0]
        home_def = home_stats['def_avg'].values[0]
        home_to = home_stats['to_margin'].values[0]

    if away_stats.empty:
        away_off = 24.0
        away_def = 24.0
        away_to = 0.0
    else:
        away_off = away_stats['off_avg'].values[0]
        away_def = away_stats['def_avg'].values[0]
        away_to = away_stats['to_margin'].values[0]

    # Build feature vector
    features = np.array([[
        elo_diff,
        home_off,
        away_off,
        home_def,
        away_def,
        home_to,
        away_to,
        1  # is_home
    ]])

    # Scale and predict
    features_scaled = scaler.transform(features)
    home_win_prob = model.predict_proba(features_scaled)[0][1]
    away_win_prob = 1 - home_win_prob

    # Generate explanation
    explanation = generate_explanation(
        home_team, away_team, features[0], model.coef_[0],
        home_win_prob, away_win_prob
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_win_probability": round(home_win_prob * 100, 1),
        "away_win_probability": round(away_win_prob * 100, 1),
        "predicted_winner": home_team if home_win_prob > 0.5 else away_team,
        "explanation": explanation,
        "home_elo": round(home_elo, 0),
        "away_elo": round(away_elo, 0)
    }

def generate_explanation(home_team: str, away_team: str, features: np.ndarray,
                        coefficients: np.ndarray, home_prob: float,
                        away_prob: float) -> str:
    """
    Generate natural language explanation based on feature contributions.

    Explanation logic:
    1. Calculate contribution of each feature (coefficient * feature_value)
    2. Rank features by absolute contribution
    3. Take top 3 most important factors
    4. Convert to human-readable text

    Returns explanation string.
    """
    winner = home_team if home_prob > 0.5 else away_team
    confidence = max(home_prob, away_prob)

    # Calculate feature contributions (how much each feature pushes toward home win)
    contributions = coefficients * features

    # Map features to explanations
    feature_explanations = [
        ("higher Elo rating", contributions[0] > 0),
        (f"stronger recent offense ({features[1]:.1f} ppg)", contributions[1] > 0),
        (f"weaker opponent offense ({features[2]:.1f} ppg)", contributions[2] < 0),
        (f"stronger defense (allowing {features[3]:.1f} ppg)", contributions[3] < 0),
        (f"weaker opponent defense (allowing {features[4]:.1f} ppg)", contributions[4] > 0),
        (f"better turnover margin ({features[5]:.2f})", contributions[5] > 0),
        (f"worse opponent turnover margin ({features[6]:.2f})", contributions[6] < 0),
        ("home-field advantage", contributions[7] > 0)
    ]

    # Get indices of top 3 features by absolute contribution
    top_indices = np.argsort(np.abs(contributions))[-3:][::-1]

    # Build explanation focusing on winner's advantages
    factors = []
    for idx in top_indices:
        explanation, helps_home = feature_explanations[idx]

        # Include factor if it helps the predicted winner
        if (home_prob > 0.5 and helps_home) or (away_prob > 0.5 and not helps_home):
            factors.append(explanation)

    # Construct sentence
    if len(factors) >= 2:
        factors_text = ", ".join(factors[:-1]) + f", and {factors[-1]}"
    elif len(factors) == 1:
        factors_text = factors[0]
    else:
        factors_text = "overall team metrics"

    explanation = (
        f"The model favors the {winner} ({confidence*100:.0f}% confidence) "
        f"primarily due to {factors_text}."
    )

    return explanation

# ============================================================================
# GLOBAL STATE (Model, Elo, etc.)
# ============================================================================

# These will be initialized on startup
model: LogisticRegression = None
scaler: StandardScaler = None
elo_system: EloRatingSystem = None
team_stats: pd.DataFrame = None
historical_df: pd.DataFrame = None

def initialize_model():
    """
    Load data, train model, and initialize global state.
    Called once on startup.
    """
    global model, scaler, elo_system, team_stats, historical_df

    print("Loading historical data...")
    historical_df = load_historical_data()

    print("Initializing Elo system...")
    elo_system = EloRatingSystem()

    print("Building training data...")
    X, y, scaler = build_training_data(historical_df, elo_system)

    print("Training model...")
    model = train_model(X, y)

    print("Calculating team statistics...")
    team_stats = calculate_rolling_averages(historical_df)

    print("Model initialization complete!")

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model when server starts."""
    initialize_model()

@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "message": "NFL Prediction API v0.1",
        "status": "running",
        "model_trained": model is not None
    }

@app.get("/predictions/weekly")
async def get_weekly_predictions(week: str = "upcoming"):
    """
    Get predictions for upcoming week's games.

    For MVP, this returns predictions for a sample set of matchups.
    In production, you would:
    1. Have a CSV with upcoming week's schedule
    2. Load it based on 'week' parameter
    3. Return predictions for those specific games

    Returns list of game predictions with explanations.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    # Sample upcoming matchups (in production, load from CSV)
    upcoming_games = [
        ("Bills", "Chiefs"),
        ("Eagles", "Cowboys"),
        ("49ers", "Seahawks"),
        ("Ravens", "Bengals"),
        ("Dolphins", "Jets")
    ]

    predictions = []

    for home_team, away_team in upcoming_games:
        try:
            prediction = predict_game(
                home_team, away_team, model, scaler, elo_system, team_stats
            )
            predictions.append(prediction)
        except Exception as e:
            print(f"Error predicting {home_team} vs {away_team}: {e}")
            continue

    return {
        "week": week,
        "games": predictions,
        "total_games": len(predictions)
    }

@app.get("/teams")
async def get_teams():
    """
    Get list of all teams with current Elo ratings.
    Useful for debugging and understanding model state.
    """
    if elo_system is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    teams = [
        {
            "team": team,
            "elo_rating": round(rating, 0)
        }
        for team, rating in sorted(
            elo_system.ratings.items(),
            key=lambda x: x[1],
            reverse=True
        )
    ]

    return {"teams": teams}

@app.get("/predict")
async def predict_single_game(home_team: str, away_team: str):
    """
    Predict outcome of a single game.

    Query parameters:
    - home_team: Name of home team
    - away_team: Name of away team

    Example: /predict?home_team=Bills&away_team=Chiefs
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        prediction = predict_game(
            home_team, away_team, model, scaler, elo_system, team_stats
        )
        return prediction
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.get("/analytics")
async def get_game_analytics(home_team: str, away_team: str):
    """
    Get detailed analytics breakdown for a specific game.

    Returns:
    - Feature contributions (how much each feature influences the prediction)
    - Team statistics comparison
    - Model coefficients
    - Historical rolling averages

    Query parameters:
    - home_team: Name of home team
    - away_team: Name of away team

    Example: /analytics?home_team=Bills&away_team=Chiefs
    """
    if model is None or scaler is None or elo_system is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    try:
        # Get current Elo ratings
        home_elo = elo_system.get_rating(home_team)
        away_elo = elo_system.get_rating(away_team)
        elo_diff = home_elo - away_elo + HOME_ADVANTAGE

        # Get most recent rolling averages
        home_stats = team_stats[team_stats['team'] == home_team].tail(1)
        away_stats = team_stats[team_stats['team'] == away_team].tail(1)

        # Check if detailed offensive and defensive stats are available
        has_detailed_stats = not home_stats.empty and 'rush_yards_avg' in home_stats.columns

        # Handle case where team has no history
        if home_stats.empty:
            home_off = 24.0
            home_def = 24.0
            home_to = 0.0
            home_rush_yards = 120.0
            home_pass_yards = 250.0
            home_rush_tds = 1.0
            home_pass_tds = 2.0
            home_rush_yards_allowed = 120.0
            home_pass_yards_allowed = 250.0
        else:
            home_off = home_stats['off_avg'].values[0]
            home_def = home_stats['def_avg'].values[0]
            home_to = home_stats['to_margin'].values[0]
            if has_detailed_stats:
                home_rush_yards = home_stats['rush_yards_avg'].values[0]
                home_pass_yards = home_stats['pass_yards_avg'].values[0]
                home_rush_tds = home_stats['rush_tds_avg'].values[0]
                home_pass_tds = home_stats['pass_tds_avg'].values[0]
                home_rush_yards_allowed = home_stats['rush_yards_allowed_avg'].values[0]
                home_pass_yards_allowed = home_stats['pass_yards_allowed_avg'].values[0]
            else:
                home_rush_yards = 120.0
                home_pass_yards = 250.0
                home_rush_tds = 1.0
                home_pass_tds = 2.0
                home_rush_yards_allowed = 120.0
                home_pass_yards_allowed = 250.0

        if away_stats.empty:
            away_off = 24.0
            away_def = 24.0
            away_to = 0.0
            away_rush_yards = 120.0
            away_pass_yards = 250.0
            away_rush_tds = 1.0
            away_pass_tds = 2.0
            away_rush_yards_allowed = 120.0
            away_pass_yards_allowed = 250.0
        else:
            away_off = away_stats['off_avg'].values[0]
            away_def = away_stats['def_avg'].values[0]
            away_to = away_stats['to_margin'].values[0]
            if has_detailed_stats:
                away_rush_yards = away_stats['rush_yards_avg'].values[0]
                away_pass_yards = away_stats['pass_yards_avg'].values[0]
                away_rush_tds = away_stats['rush_tds_avg'].values[0]
                away_pass_tds = away_stats['pass_tds_avg'].values[0]
                away_rush_yards_allowed = away_stats['rush_yards_allowed_avg'].values[0]
                away_pass_yards_allowed = away_stats['pass_yards_allowed_avg'].values[0]
            else:
                away_rush_yards = 120.0
                away_pass_yards = 250.0
                away_rush_tds = 1.0
                away_pass_tds = 2.0
                away_rush_yards_allowed = 120.0
                away_pass_yards_allowed = 250.0

        # Build feature vector
        features = np.array([[
            elo_diff,
            home_off,
            away_off,
            home_def,
            away_def,
            home_to,
            away_to,
            1  # is_home
        ]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        home_win_prob = model.predict_proba(features_scaled)[0][1]
        away_win_prob = 1 - home_win_prob

        # Calculate feature contributions
        coefficients = model.coef_[0]
        contributions = coefficients * features[0]

        # Build detailed feature breakdown
        feature_details = [
            {
                "feature": "Elo Difference",
                "value": round(elo_diff, 1),
                "coefficient": round(coefficients[0], 4),
                "contribution": round(contributions[0], 4),
                "description": f"{home_team} Elo: {round(home_elo, 0)}, {away_team} Elo: {round(away_elo, 0)}, Home advantage: +{HOME_ADVANTAGE}"
            },
            {
                "feature": "Home Offense",
                "value": round(home_off, 1),
                "coefficient": round(coefficients[1], 4),
                "contribution": round(contributions[1], 4),
                "description": f"{home_team} averaging {round(home_off, 1)} points per game (last {ROLLING_WINDOW} games)"
            },
            {
                "feature": "Away Offense",
                "value": round(away_off, 1),
                "coefficient": round(coefficients[2], 4),
                "contribution": round(contributions[2], 4),
                "description": f"{away_team} averaging {round(away_off, 1)} points per game (last {ROLLING_WINDOW} games)"
            },
            {
                "feature": "Home Defense",
                "value": round(home_def, 1),
                "coefficient": round(coefficients[3], 4),
                "contribution": round(contributions[3], 4),
                "description": f"{home_team} allowing {round(home_def, 1)} points per game (last {ROLLING_WINDOW} games)"
            },
            {
                "feature": "Away Defense",
                "value": round(away_def, 1),
                "coefficient": round(coefficients[4], 4),
                "contribution": round(contributions[4], 4),
                "description": f"{away_team} allowing {round(away_def, 1)} points per game (last {ROLLING_WINDOW} games)"
            },
            {
                "feature": "Home Turnover Margin",
                "value": round(home_to, 2),
                "coefficient": round(coefficients[5], 4),
                "contribution": round(contributions[5], 4),
                "description": f"{home_team} turnover differential: {round(home_to, 2)} per game"
            },
            {
                "feature": "Away Turnover Margin",
                "value": round(away_to, 2),
                "coefficient": round(coefficients[6], 4),
                "contribution": round(contributions[6], 4),
                "description": f"{away_team} turnover differential: {round(away_to, 2)} per game"
            },
            {
                "feature": "Home Field Advantage",
                "value": 1,
                "coefficient": round(coefficients[7], 4),
                "contribution": round(contributions[7], 4),
                "description": "Built-in home field advantage factor"
            }
        ]

        # Sort by absolute contribution (most impactful first)
        feature_details_sorted = sorted(
            feature_details,
            key=lambda x: abs(x['contribution']),
            reverse=True
        )

        return {
            "home_team": home_team,
            "away_team": away_team,
            "prediction": {
                "home_win_probability": round(home_win_prob * 100, 1),
                "away_win_probability": round(away_win_prob * 100, 1),
                "predicted_winner": home_team if home_win_prob > 0.5 else away_team
            },
            "team_stats": {
                "home": {
                    "elo": round(home_elo, 0),
                    "offense_avg": round(home_off, 1),
                    "defense_avg": round(home_def, 1),
                    "turnover_margin": round(home_to, 2)
                },
                "away": {
                    "elo": round(away_elo, 0),
                    "offense_avg": round(away_off, 1),
                    "defense_avg": round(away_def, 1),
                    "turnover_margin": round(away_to, 2)
                }
            },
            "offensive_breakdown": {
                "home": {
                    "rush_yards_avg": round(home_rush_yards, 1),
                    "pass_yards_avg": round(home_pass_yards, 1),
                    "rush_tds_avg": round(home_rush_tds, 2),
                    "pass_tds_avg": round(home_pass_tds, 2),
                    "total_yards_avg": round(home_rush_yards + home_pass_yards, 1)
                },
                "away": {
                    "rush_yards_avg": round(away_rush_yards, 1),
                    "pass_yards_avg": round(away_pass_yards, 1),
                    "rush_tds_avg": round(away_rush_tds, 2),
                    "pass_tds_avg": round(away_pass_tds, 2),
                    "total_yards_avg": round(away_rush_yards + away_pass_yards, 1)
                }
            },
            "defensive_breakdown": {
                "home": {
                    "rush_yards_allowed_avg": round(home_rush_yards_allowed, 1),
                    "pass_yards_allowed_avg": round(home_pass_yards_allowed, 1),
                    "total_yards_allowed_avg": round(home_rush_yards_allowed + home_pass_yards_allowed, 1)
                },
                "away": {
                    "rush_yards_allowed_avg": round(away_rush_yards_allowed, 1),
                    "pass_yards_allowed_avg": round(away_pass_yards_allowed, 1),
                    "total_yards_allowed_avg": round(away_rush_yards_allowed + away_pass_yards_allowed, 1)
                }
            },
            "feature_breakdown": feature_details_sorted,
            "model_info": {
                "model_type": "Logistic Regression",
                "features_used": len(FEATURE_NAMES),
                "rolling_window": ROLLING_WINDOW,
                "home_advantage_points": HOME_ADVANTAGE
            }
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Analytics error: {str(e)}")

# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
