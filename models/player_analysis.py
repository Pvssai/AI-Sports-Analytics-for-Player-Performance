import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os

class PlayerAnalyzer:
    def __init__(self, data_path='summary.csv'):
        self.data = pd.read_csv(data_path)
        self.model = None
        self.scaler = StandardScaler()
        
    def preprocess_data(self):
        # Select numerical columns for analysis
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        X = self.data[numerical_cols].fillna(0)
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled
    
    def train_model(self, target_column):
        X = self.preprocess_data()
        y = self.data[target_column]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train the model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        return self.model.score(X_test, y_test)
    
    def predict_performance(self, player_data):
        if self.model is None:
            raise ValueError("Model not trained. Call train_model first.")
        
        # Preprocess the input data
        X = self.scaler.transform(player_data)
        
        # Make prediction
        prediction = self.model.predict(X)
        return prediction
    
    def get_player_trends(self, player_name):
        player_data = self.data[self.data['Name'] == player_name]
        return player_data
    
    def compare_players(self, player1_name, player2_name):
        p1_data = self.data[self.data['Name'] == player1_name]
        p2_data = self.data[self.data['Name'] == player2_name]
        
        return {
            'player1': p1_data.to_dict('records'),
            'player2': p2_data.to_dict('records')
        }
    
    def save_model(self, path='models/player_model.joblib'):
        if self.model is None:
            raise ValueError("No model to save")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
    
    def load_model(self, path='models/player_model.joblib'):
        if os.path.exists(path):
            self.model = joblib.load(path)
            return True
        return False 