import json
import requests
import pandas as pd
import logging
import os
import numpy as np
import math 
from .milestone2_tidy_data import *
from .milestone2_feature_engineering import *
import time


class GameClient:
    def __init__(self):
        self.tracker = 0
        self.game = None
        self.home_team = None
        self.away_team = None
        self.dashboard_time = float('inf')
        self.dashboard_period = 0
        
    def get_game(self,game_id):
        self.game_id = game_id
        file_path = './' + str(self.game_id) + '.json'
        data = requests.get(f'https://statsapi.web.nhl.com/api/v1/game/{self.game_id}/feed/live/')
        if (data.status_code == 404):
            return None
        with open(file_path, 'w') as f:
            json.dump(data.json(), f)
        
        return file_path
    
    def update_model_df_length(self):
        self.model_df_length = self.game.shape[0]
   
    def ping_game(self,file_path):
        df_game_tidied = tidy_data(file_path)
        df_game_features = feature_engineering(df_game_tidied)
        last_event = df_game_features.iloc[-1]
        self.game = df_game_features
        self.update_model_df_length()
        tracker = self.model_df_length
       
        return df_game_features, last_event, tracker
