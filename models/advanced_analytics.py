import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from typing import Dict, Any

class AdvancedPlayerAnalytics:
    def __init__(self, data_path='summary.csv'):
        # Load and prepare data
        self.data = pd.read_csv(data_path)
        self.scaler = StandardScaler()
        
        # Initialize models
        self.cluster_model = None
        self.performance_model = None
        self.feature_importance = None
        
        # Train models immediately
        try:
            self.train_performance_model()
            self.cluster_players()
        except Exception as e:
            print(f"Warning: Error during model initialization: {str(e)}")
    
    def prepare_features(self, player_data):
        # Select relevant numerical features for analysis
        features = [
            'GamesPlayed', 'MinutesPlayed', 'PointsPerGame',
            'FieldGoalPercent', '3PointPercent', 'FreeThrowPercent',
            'Rebounds', 'Assists', 'Steals', 'Blocks', 'Turnovers'
        ]
        
        # Handle missing values and ensure all features exist
        for feature in features:
            if feature not in player_data.columns:
                print(f"Warning: Feature {feature} not found in dataset")
                player_data[feature] = 0
        
        X = player_data[features].fillna(0)
        return X
    
    def cluster_players(self, n_clusters=5):
        """Cluster players based on their performance metrics"""
        try:
            X = self.prepare_features(self.data)
            X_scaled = self.scaler.fit_transform(X)
            
            self.cluster_model = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = self.cluster_model.fit_predict(X_scaled)
            
            # Add cluster information to the data
            self.data['Cluster'] = clusters
            
            return {
                'clusters': clusters.tolist(),
                'centers': self.cluster_model.cluster_centers_.tolist()
            }
        except Exception as e:
            print(f"Error in cluster_players: {str(e)}")
            return {'clusters': [], 'centers': []}
    
    def train_performance_model(self, target='PointsPerGame'):
        """Train a model to predict player performance"""
        try:
            if target not in self.data.columns:
                print(f"Warning: Target {target} not found in dataset")
                return {'model_score': 0, 'feature_importance': {}}
            
            X = self.prepare_features(self.data)
            y = self.data[target]
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train multiple models
            models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
            }
            
            best_score = -np.inf
            best_model = None
            
            for name, model in models.items():
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    self.performance_model = model
            
            # Calculate feature importance
            if hasattr(best_model, 'feature_importances_'):
                self.feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            
            return {
                'model_score': best_score,
                'feature_importance': self.feature_importance
            }
        except Exception as e:
            print(f"Error in train_performance_model: {str(e)}")
            return {'model_score': 0, 'feature_importance': {}}
    
    def predict_player_performance(self, player_name: str) -> float:
        """Predict player's future performance"""
        try:
            # Check if player exists
            player_data = self.data[self.data['Name'] == player_name]
            if player_data.empty:
                print(f"Player {player_name} not found")
                return 0.0

            # Prepare features for prediction
            features = self.prepare_features(player_data)
            if features is None or features.empty:
                print(f"Could not prepare features for {player_name}")
                return 0.0

            # Ensure model is trained
            if self.performance_model is None:
                print("Training performance model...")
                self.train_performance_model()

            # Make prediction
            prediction = self.performance_model.predict(features)
            
            # Ensure prediction is a valid number
            if prediction is None or np.isnan(prediction).any():
                print(f"Invalid prediction for {player_name}")
                return 0.0

            # Return the mean prediction if multiple predictions
            return float(np.mean(prediction))

        except Exception as e:
            print(f"Error predicting performance for {player_name}: {str(e)}")
            return 0.0
    
    def get_player_similarity(self, player_name, n_similar=5):
        """Find similar players based on performance metrics"""
        try:
            X = self.prepare_features(self.data)
            X_scaled = self.scaler.fit_transform(X)
            
            player_idx = self.data[self.data['Name'] == player_name].index[0]
            player_vector = X_scaled[player_idx]
            
            # Calculate Euclidean distance to all other players
            distances = np.linalg.norm(X_scaled - player_vector, axis=1)
            
            # Get indices of most similar players (excluding the player themselves)
            similar_indices = np.argsort(distances)[1:n_similar+1]
            
            similar_players = self.data.iloc[similar_indices][['Name', 'PointsPerGame', 'Rebounds', 'Assists']]
            return similar_players.to_dict('records')
        except Exception as e:
            print(f"Error in get_player_similarity: {str(e)}")
            return []
    
    def analyze_player_trends(self, player_name: str) -> Dict[str, Any]:
        """Analyze player performance trends"""
        try:
            player_data = self.data[self.data['Name'] == player_name]
            if player_data.empty:
                return {
                    'points': {'mean': 0, 'std': 0, 'trend': 'stable'},
                    'rebounds': {'mean': 0, 'std': 0, 'trend': 'stable'},
                    'assists': {'mean': 0, 'std': 0, 'trend': 'stable'}
                }
            
            # Calculate trends for key metrics
            points_trend = self._calculate_trend(player_data['PointsPerGame'])
            rebounds_trend = self._calculate_trend(player_data['Rebounds'])
            assists_trend = self._calculate_trend(player_data['Assists'])
            
            # Convert NaN values to None for JSON serialization
            def convert_nan_to_none(value):
                return None if pd.isna(value) else float(value)
            
            return {
                'points': {
                    'mean': convert_nan_to_none(player_data['PointsPerGame'].mean()),
                    'std': convert_nan_to_none(player_data['PointsPerGame'].std()),
                    'trend': points_trend
                },
                'rebounds': {
                    'mean': convert_nan_to_none(player_data['Rebounds'].mean()),
                    'std': convert_nan_to_none(player_data['Rebounds'].std()),
                    'trend': rebounds_trend
                },
                'assists': {
                    'mean': convert_nan_to_none(player_data['Assists'].mean()),
                    'std': convert_nan_to_none(player_data['Assists'].std()),
                    'trend': assists_trend
                }
            }
        except Exception as e:
            print(f"Error analyzing player trends: {str(e)}")
            return {
                'points': {'mean': 0, 'std': 0, 'trend': 'stable'},
                'rebounds': {'mean': 0, 'std': 0, 'trend': 'stable'},
                'assists': {'mean': 0, 'std': 0, 'trend': 'stable'}
            }
    
    def _calculate_trend(self, series):
        """Calculate trend direction and magnitude"""
        try:
            if len(series) < 2:
                return 'insufficient_data'
            
            # Simple linear regression
            x = np.arange(len(series))
            slope = np.polyfit(x, series, 1)[0]
            
            if abs(slope) < 0.1:
                return 'stable'
            elif slope > 0:
                return 'improving'
            else:
                return 'declining'
        except Exception as e:
            print(f"Error in _calculate_trend: {str(e)}")
            return 'unknown'
    
    def save_models(self, base_path='models'):
        """Save trained models"""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            if self.cluster_model:
                joblib.dump(self.cluster_model, f'{base_path}/cluster_model.joblib')
            
            if self.performance_model:
                joblib.dump(self.performance_model, f'{base_path}/performance_model.joblib')
            
            if self.feature_importance:
                joblib.dump(self.feature_importance, f'{base_path}/feature_importance.joblib')
        except Exception as e:
            print(f"Error in save_models: {str(e)}")
    
    def load_models(self, base_path='models'):
        """Load trained models"""
        try:
            if os.path.exists(f'{base_path}/cluster_model.joblib'):
                self.cluster_model = joblib.load(f'{base_path}/cluster_model.joblib')
            
            if os.path.exists(f'{base_path}/performance_model.joblib'):
                self.performance_model = joblib.load(f'{base_path}/performance_model.joblib')
            
            if os.path.exists(f'{base_path}/feature_importance.joblib'):
                self.feature_importance = joblib.load(f'{base_path}/feature_importance.joblib')
        except Exception as e:
            print(f"Error in load_models: {str(e)}") 