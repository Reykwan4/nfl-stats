/**
 * NFL Predictor Frontend - MVP Version 0.1
 *
 * Single-page mobile-first app that displays weekly game predictions.
 *
 * Design decisions:
 * - Mobile-first: Stacked cards, large touch targets, readable on phone
 * - Simple CSS: No dependencies on CSS frameworks for MVP
 * - Auto-refresh: Fetch predictions on mount
 * - Clear visual hierarchy: Winner highlighted, probability prominent
 * - Explanations: Natural language reasoning for each prediction
 */

import React, { useState, useEffect } from 'react';
import './App.css';

function App() {
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [week, setWeek] = useState('upcoming');

  // Analytics modal state
  const [showAnalytics, setShowAnalytics] = useState(false);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [analyticsLoading, setAnalyticsLoading] = useState(false);

  // Fetch predictions from backend on component mount
  useEffect(() => {
    fetchPredictions();
  }, []);

  const fetchPredictions = async () => {
    setLoading(true);
    setError(null);

    try {
      // Call backend API (proxied through React dev server)
      const response = await fetch('/predictions/weekly?week=upcoming');

      if (!response.ok) {
        throw new Error('Failed to fetch predictions');
      }

      const data = await response.json();
      setPredictions(data.games);
      setWeek(data.week);
    } catch (err) {
      setError(err.message);
      console.error('Error fetching predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  const fetchAnalytics = async (homeTeam, awayTeam) => {
    setAnalyticsLoading(true);
    setShowAnalytics(true);
    setAnalyticsData(null);

    try {
      const response = await fetch(
        `/analytics?home_team=${encodeURIComponent(homeTeam)}&away_team=${encodeURIComponent(awayTeam)}`
      );

      if (!response.ok) {
        throw new Error('Failed to fetch analytics');
      }

      const data = await response.json();
      setAnalyticsData(data);
    } catch (err) {
      console.error('Error fetching analytics:', err);
      setAnalyticsData({ error: err.message });
    } finally {
      setAnalyticsLoading(false);
    }
  };

  const closeAnalytics = () => {
    setShowAnalytics(false);
    setAnalyticsData(null);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1 className="title">NFL Predictor</h1>
        <p className="subtitle">Data-driven weekly game analysis</p>
      </header>

      {/* Main content */}
      <main className="main">
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <p>Loading predictions...</p>
          </div>
        )}

        {error && (
          <div className="error">
            <p>Error: {error}</p>
            <button onClick={fetchPredictions} className="retry-button">
              Retry
            </button>
          </div>
        )}

        {!loading && !error && predictions.length === 0 && (
          <div className="empty">
            <p>No predictions available</p>
          </div>
        )}

        {!loading && !error && predictions.length > 0 && (
          <>
            <div className="week-header">
              <h2>Week: {week}</h2>
              <button onClick={fetchPredictions} className="refresh-button">
                Refresh
              </button>
            </div>

            <div className="games-container">
              {predictions.map((game, index) => (
                <GameCard key={index} game={game} onViewAnalytics={fetchAnalytics} />
              ))}
            </div>
          </>
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <p>Predictions based on Elo ratings, rolling averages, and logistic regression</p>
      </footer>

      {/* Analytics Modal */}
      {showAnalytics && (
        <AnalyticsModal
          data={analyticsData}
          loading={analyticsLoading}
          onClose={closeAnalytics}
        />
      )}
    </div>
  );
}

/**
 * GameCard component - displays single game prediction
 *
 * Shows:
 * - Team names with winner highlighted
 * - Win probabilities with visual bar
 * - Elo ratings
 * - Natural language explanation
 */
function GameCard({ game, onViewAnalytics }) {
  const {
    home_team,
    away_team,
    home_win_probability,
    away_win_probability,
    predicted_winner,
    explanation,
    home_elo,
    away_elo
  } = game;

  const homeIsWinner = predicted_winner === home_team;

  const handleViewAnalytics = () => {
    onViewAnalytics(home_team, away_team);
  };

  return (
    <div className="game-card">
      {/* Teams and probabilities */}
      <div className="matchup">
        <div className={`team ${homeIsWinner ? 'winner' : 'loser'}`}>
          <div className="team-info">
            <span className="team-name">{home_team}</span>
            <span className="team-label">(Home)</span>
          </div>
          <div className="probability">
            <span className="prob-value">{home_win_probability}%</span>
            <div className="prob-bar-container">
              <div
                className="prob-bar home"
                style={{ width: `${home_win_probability}%` }}
              ></div>
            </div>
          </div>
          <div className="elo">Elo: {home_elo}</div>
        </div>

        <div className="vs">VS</div>

        <div className={`team ${!homeIsWinner ? 'winner' : 'loser'}`}>
          <div className="team-info">
            <span className="team-name">{away_team}</span>
            <span className="team-label">(Away)</span>
          </div>
          <div className="probability">
            <span className="prob-value">{away_win_probability}%</span>
            <div className="prob-bar-container">
              <div
                className="prob-bar away"
                style={{ width: `${away_win_probability}%` }}
              ></div>
            </div>
          </div>
          <div className="elo">Elo: {away_elo}</div>
        </div>
      </div>

      {/* Winner badge */}
      <div className="predicted-winner">
        <span className="winner-badge">Predicted Winner: {predicted_winner}</span>
      </div>

      {/* Explanation */}
      <div className="explanation">
        <h4>Analysis:</h4>
        <p>{explanation}</p>
      </div>

      {/* View Analytics Button */}
      <div className="analytics-button-container">
        <button onClick={handleViewAnalytics} className="analytics-button">
          View Detailed Analytics
        </button>
      </div>
    </div>
  );
}

/**
 * AnalyticsModal component - displays detailed game analytics
 *
 * Shows:
 * - Feature contribution breakdown (sorted by impact)
 * - Team statistics comparison
 * - Model coefficients and values
 * - Model information
 */
function AnalyticsModal({ data, loading, onClose }) {
  const [activeTab, setActiveTab] = useState('team'); // 'team', 'qb', 'coach'

  // Close modal when clicking outside
  const handleBackdropClick = (e) => {
    if (e.target.className === 'modal-backdrop') {
      onClose();
    }
  };

  return (
    <div className="modal-backdrop" onClick={handleBackdropClick}>
      <div className="modal-content">
        {/* Header */}
        <div className="modal-header">
          <h2>Detailed Analytics</h2>
          <button onClick={onClose} className="modal-close">
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="modal-body">
          {loading && (
            <div className="modal-loading">
              <div className="spinner"></div>
              <p>Loading analytics...</p>
            </div>
          )}

          {!loading && data && data.error && (
            <div className="modal-error">
              <p>Error loading analytics: {data.error}</p>
            </div>
          )}

          {!loading && data && !data.error && (
            <>
              {/* Game Info */}
              <div className="analytics-section">
                <h3>{data.home_team} vs {data.away_team}</h3>
                <div className="prediction-summary">
                  <div className="pred-item">
                    <span className="pred-label">Predicted Winner:</span>
                    <span className="pred-value winner-text">
                      {data.prediction.predicted_winner}
                    </span>
                  </div>
                  <div className="pred-item">
                    <span className="pred-label">{data.home_team}:</span>
                    <span className="pred-value">
                      {data.prediction.home_win_probability}%
                    </span>
                  </div>
                  <div className="pred-item">
                    <span className="pred-label">{data.away_team}:</span>
                    <span className="pred-value">
                      {data.prediction.away_win_probability}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Tab Navigation */}
              <div className="analytics-tabs">
                <button
                  className={`tab-button ${activeTab === 'team' ? 'active' : ''}`}
                  onClick={() => setActiveTab('team')}
                >
                  Team Stats
                </button>
                <button
                  className={`tab-button ${activeTab === 'qb' ? 'active' : ''}`}
                  onClick={() => setActiveTab('qb')}
                >
                  QB Stats
                </button>
                <button
                  className={`tab-button ${activeTab === 'coach' ? 'active' : ''}`}
                  onClick={() => setActiveTab('coach')}
                >
                  Coach Stats
                </button>
              </div>

              {/* Team Stats Tab */}
              {activeTab === 'team' && (
                <>
                  {/* Team Stats Comparison */}
                  <div className="analytics-section">
                <h3>Team Statistics</h3>
                <div className="stats-comparison">
                  <div className="stat-row">
                    <div className="stat-label">Elo Rating</div>
                    <div className="stat-values">
                      <span className="stat-home">{data.team_stats.home.elo}</span>
                      <span className="stat-away">{data.team_stats.away.elo}</span>
                    </div>
                  </div>
                  <div className="stat-row">
                    <div className="stat-label">Offensive Avg (PPG)</div>
                    <div className="stat-values">
                      <span className="stat-home">{data.team_stats.home.offense_avg}</span>
                      <span className="stat-away">{data.team_stats.away.offense_avg}</span>
                    </div>
                  </div>
                  <div className="stat-row">
                    <div className="stat-label">Defensive Avg (PPG)</div>
                    <div className="stat-values">
                      <span className="stat-home">{data.team_stats.home.defense_avg}</span>
                      <span className="stat-away">{data.team_stats.away.defense_avg}</span>
                    </div>
                  </div>
                  <div className="stat-row">
                    <div className="stat-label">Turnover Margin</div>
                    <div className="stat-values">
                      <span className="stat-home">{data.team_stats.home.turnover_margin}</span>
                      <span className="stat-away">{data.team_stats.away.turnover_margin}</span>
                    </div>
                  </div>
                </div>
                <div className="stats-legend">
                  <span className="legend-home">{data.home_team}</span>
                  <span className="legend-vs">vs</span>
                  <span className="legend-away">{data.away_team}</span>
                </div>
              </div>

              {/* Offensive Breakdown */}
              {data.offensive_breakdown && (
                <div className="analytics-section">
                  <h3>Offensive Breakdown</h3>
                  <p className="section-description">
                    Detailed breakdown of rushing and passing offense (last {data.model_info.rolling_window} games)
                  </p>
                  <div className="stats-comparison">
                    {/* Rushing Stats */}
                    <div className="stat-row">
                      <div className="stat-label">Rush Yards per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.rush_yards_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.rush_yards_avg}</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Rush TDs per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.rush_tds_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.rush_tds_avg}</span>
                      </div>
                    </div>
                    {/* Passing Stats */}
                    <div className="stat-row">
                      <div className="stat-label">Pass Yards per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.pass_yards_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.pass_yards_avg}</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Pass TDs per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.pass_tds_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.pass_tds_avg}</span>
                      </div>
                    </div>
                    {/* Field Goals */}
                    <div className="stat-row">
                      <div className="stat-label">Field Goals per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.field_goals_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.field_goals_avg}</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Field Goal Success Rate</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.offensive_breakdown.home.field_goals_made}/{data.offensive_breakdown.home.field_goals_attempted}
                        </span>
                        <span className="stat-away">
                          {data.offensive_breakdown.away.field_goals_made}/{data.offensive_breakdown.away.field_goals_attempted}
                        </span>
                      </div>
                    </div>
                    {/* Total Yards */}
                    <div className="stat-row">
                      <div className="stat-label">Total Yards per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.offensive_breakdown.home.total_yards_avg}</span>
                        <span className="stat-away">{data.offensive_breakdown.away.total_yards_avg}</span>
                      </div>
                    </div>
                  </div>
                  <div className="stats-legend">
                    <span className="legend-home">{data.home_team}</span>
                    <span className="legend-vs">vs</span>
                    <span className="legend-away">{data.away_team}</span>
                  </div>
                </div>
              )}

              {/* Defensive Breakdown */}
              {data.defensive_breakdown && (
                <div className="analytics-section">
                  <h3>Defensive Breakdown</h3>
                  <p className="section-description">
                    Detailed breakdown of rushing and passing defense - yards allowed (last {data.model_info.rolling_window} games)
                  </p>
                  <div className="stats-comparison">
                    <div className="stat-row">
                      <div className="stat-label">Rush Yards Allowed per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.defensive_breakdown.home.rush_yards_allowed_avg}</span>
                        <span className="stat-away">{data.defensive_breakdown.away.rush_yards_allowed_avg}</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Pass Yards Allowed per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.defensive_breakdown.home.pass_yards_allowed_avg}</span>
                        <span className="stat-away">{data.defensive_breakdown.away.pass_yards_allowed_avg}</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Total Yards Allowed per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">{data.defensive_breakdown.home.total_yards_allowed_avg}</span>
                        <span className="stat-away">{data.defensive_breakdown.away.total_yards_allowed_avg}</span>
                      </div>
                    </div>
                  </div>
                  <div className="stats-legend">
                    <span className="legend-home">{data.home_team}</span>
                    <span className="legend-vs">vs</span>
                    <span className="legend-away">{data.away_team}</span>
                  </div>
                  <p className="section-note">
                    Lower numbers are better for defense (fewer yards allowed)
                  </p>
                </div>
              )}
                </>
              )}

              {/* QB Stats Tab */}
              {activeTab === 'qb' && (
                <>
                  {/* QB Breakdown */}
              {data.qb_breakdown && (
                <div className="analytics-section">
                  <h3>Quarterback Stats</h3>
                  <p className="section-description">
                    Season quarterback performance metrics (real NFL data)
                  </p>
                  <div className="stats-comparison">
                    <div className="stat-row">
                      <div className="stat-label">Passer Rating</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.passer_rating : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.passer_rating : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Completion %</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.completion_pct + '%' : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.completion_pct + '%' : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Pass Attempts per Game</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.attempts_per_game : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.attempts_per_game : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Yards per Attempt</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.yards_per_attempt : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.yards_per_attempt : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">TD/INT</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available
                            ? `${data.qb_breakdown.home.passing_tds}/${data.qb_breakdown.home.interceptions}`
                            : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available
                            ? `${data.qb_breakdown.away.passing_tds}/${data.qb_breakdown.away.interceptions}`
                            : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">TD/INT Ratio</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.td_int_ratio : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.td_int_ratio : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Passing EPA</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.passing_epa : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.passing_epa : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Deep Throws per Game (20+ yards)</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? data.qb_breakdown.home.deep_attempts_per_game : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? data.qb_breakdown.away.deep_attempts_per_game : 'N/A'}
                        </span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Deep Pass Completion %</div>
                      <div className="stat-values">
                        <span className="stat-home">
                          {data.qb_breakdown.home.data_available ? `${data.qb_breakdown.home.deep_completion_pct}%` : 'N/A'}
                        </span>
                        <span className="stat-away">
                          {data.qb_breakdown.away.data_available ? `${data.qb_breakdown.away.deep_completion_pct}%` : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="stats-legend">
                    <span className="legend-home">{data.home_team}</span>
                    <span className="legend-vs">vs</span>
                    <span className="legend-away">{data.away_team}</span>
                  </div>
                  <p className="section-note">
                    QB stats are aggregated from nfl-data-py (season {data.qb_breakdown.home.season || '2024'})
                  </p>
                </div>
              )}
                </>
              )}

              {/* Coach Stats Tab */}
              {activeTab === 'coach' && (
                <div className="analytics-section">
                  <h3>Coach Statistics</h3>
                  <p className="section-description">
                    Head coach experience and performance metrics (coming soon)
                  </p>
                  <div className="stats-comparison">
                    <div className="stat-row">
                      <div className="stat-label">Head Coach</div>
                      <div className="stat-values">
                        <span className="stat-home">Data Coming Soon</span>
                        <span className="stat-away">Data Coming Soon</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Career Win %</div>
                      <div className="stat-values">
                        <span className="stat-home">-</span>
                        <span className="stat-away">-</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Years Experience</div>
                      <div className="stat-values">
                        <span className="stat-home">-</span>
                        <span className="stat-away">-</span>
                      </div>
                    </div>
                    <div className="stat-row">
                      <div className="stat-label">Playoff Appearances</div>
                      <div className="stat-values">
                        <span className="stat-home">-</span>
                        <span className="stat-away">-</span>
                      </div>
                    </div>
                  </div>
                  <div className="stats-legend">
                    <span className="legend-home">{data.home_team}</span>
                    <span className="legend-vs">vs</span>
                    <span className="legend-away">{data.away_team}</span>
                  </div>
                  <p className="section-note">
                    Coach statistics will be added in a future update
                  </p>
                </div>
              )}

              {/* Feature Breakdown - Visible on all tabs */}
              <div className="analytics-section">
                <h3>Feature Contributions</h3>
                <p className="section-description">
                  How each feature influences the prediction (sorted by impact)
                </p>
                <div className="feature-list">
                  {data.feature_breakdown.map((feature, index) => (
                    <div key={index} className="feature-item">
                      <div className="feature-header">
                        <span className="feature-name">{feature.feature}</span>
                        <span className={`feature-contribution ${feature.contribution > 0 ? 'positive' : 'negative'}`}>
                          {feature.contribution > 0 ? '+' : ''}{feature.contribution}
                        </span>
                      </div>
                      <div className="feature-details">
                        <span className="feature-detail">Value: {feature.value}</span>
                        <span className="feature-detail">Coefficient: {feature.coefficient}</span>
                      </div>
                      <div className="feature-description">
                        {feature.description}
                      </div>
                      {/* Contribution bar */}
                      <div className="contribution-bar-container">
                        <div
                          className={`contribution-bar ${feature.contribution > 0 ? 'positive' : 'negative'}`}
                          style={{
                            width: `${Math.min(Math.abs(feature.contribution) * 10, 100)}%`
                          }}
                        ></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Model Info */}
              <div className="analytics-section">
                <h3>Model Information</h3>
                <div className="model-info">
                  <div className="info-item">
                    <span className="info-label">Model Type:</span>
                    <span className="info-value">{data.model_info.model_type}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Features Used:</span>
                    <span className="info-value">{data.model_info.features_used}</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Rolling Window:</span>
                    <span className="info-value">{data.model_info.rolling_window} games</span>
                  </div>
                  <div className="info-item">
                    <span className="info-label">Home Advantage:</span>
                    <span className="info-value">+{data.model_info.home_advantage_points} Elo points</span>
                  </div>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;
