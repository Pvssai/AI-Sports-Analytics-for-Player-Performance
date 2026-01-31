from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import os
from models.advanced_analytics import AdvancedPlayerAnalytics

app = Flask(__name__)
CORS(app)

# Initialize analytics
analytics = AdvancedPlayerAnalytics()

@app.route('/')
def landing():
    """Render the landing page"""
    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/api/players')
def get_players():
    """Get list of all players"""
    try:
        players = analytics.data['Name'].unique().tolist()
        return jsonify(players)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>')
def get_player_stats(player_name):
    """Get statistics for a specific player"""
    try:
        stats = analytics.data[analytics.data['Name'] == player_name].to_dict('records')
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>/similar')
def get_similar_players(player_name):
    """Get similar players based on performance metrics"""
    try:
        similar_players = analytics.get_player_similarity(player_name)
        return jsonify(similar_players)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>/trends')
def get_player_trends(player_name):
    """Get performance trends for a player"""
    try:
        trends = analytics.analyze_player_trends(player_name)
        return jsonify(trends)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>/predict')
def predict_player_performance(player_name):
    """Predict future performance for a player"""
    try:
        prediction = analytics.predict_player_performance(player_name)
        # Ensure we return a properly structured response
        return jsonify({
            'predicted_points': float(prediction) if prediction is not None else 0.0,
            'confidence': 0.85,  # Example confidence score
            'last_updated': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clusters')
def get_player_clusters():
    """Get player clusters for visualization"""
    try:
        clusters = analytics.cluster_players()
        return jsonify(clusters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def compare_players():
    """Compare two players"""
    try:
        data = request.get_json()
        player1 = data.get('player1')
        player2 = data.get('player2')
        
        if not player1 or not player2:
            return jsonify({'error': 'Both players must be specified'}), 400
            
        player1_stats = analytics.data[analytics.data['Name'] == player1].to_dict('records')
        player2_stats = analytics.data[analytics.data['Name'] == player2].to_dict('records')
        
        return jsonify({
            'player1': {'stats': player1_stats},
            'player2': {'stats': player2_stats}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature-importance')
def get_feature_importance():
    """Get feature importance for the model"""
    try:
        if analytics.feature_importance is None:
            return jsonify({'error': 'Model not trained yet'}), 400
        return jsonify(analytics.feature_importance)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>/advanced-stats')
def get_advanced_stats(player_name):
    """Get advanced statistics for a player"""
    try:
        player_data = analytics.data[analytics.data['Name'] == player_name]
        if player_data.empty:
            return jsonify({'error': 'Player not found'}), 404
            
        # Calculate advanced metrics
        advanced_stats = {
            'efficiency': {
                'true_shooting': float(player_data['TrueShootingPercent'].mean()),
                'usage_rate': float(player_data['UsageRate'].mean()),
                'win_shares': float(player_data['WinShares'].mean())
            },
            'advanced_metrics': {
                'per': float(player_data['PER'].mean()),
                'box_plus_minus': float(player_data['BoxPlusMinus'].mean()),
                'value_over_replacement': float(player_data['VORP'].mean())
            },
            'shooting': {
                'effective_fg': float(player_data['EffectiveFieldGoalPercent'].mean()),
                'three_point_rate': float(player_data['ThreePointRate'].mean()),
                'free_throw_rate': float(player_data['FreeThrowRate'].mean())
            }
        }
        
        return jsonify(advanced_stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/player/<player_name>/season-stats')
def get_season_stats(player_name):
    """Get season-by-season statistics for a player"""
    try:
        player_data = analytics.data[analytics.data['Name'] == player_name]
        if player_data.empty:
            return jsonify({'error': 'Player not found'}), 404
            
        # Group by season and calculate averages
        season_stats = player_data.groupby('Season').agg({
            'PointsPerGame': 'mean',
            'Rebounds': 'mean',
            'Assists': 'mean',
            'Steals': 'mean',
            'Blocks': 'mean',
            'FieldGoalPercent': 'mean',
            'ThreePointPercent': 'mean',
            'FreeThrowPercent': 'mean'
        }).reset_index()
        
        return jsonify(season_stats.to_dict('records'))
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 