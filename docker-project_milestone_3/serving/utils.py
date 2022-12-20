import pandas as pd
import os
import numpy as np
# One hot encoding
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer


# import sys
# sys.path.insert(0, os.path.join("..",'..','src','models'))
# import xgboostcometfinal

def dataload(test_link):
    """
    This function is for loading the dataset.
    params: train_link
    return train_df
    """
    if not isinstance(test_link, pd.DataFrame):
        test_df = pd.read_csv(test_link)
    else:
        test_df = test_link
    test_df.rename(columns={'game date': 'game_date', 'period time': 'period_time',
                             'game id': 'game_id', 'home team': 'home_team',
                             'away team': 'away_team', 'is goal': 'is_goal',
                             'team shot': 'team_shot', 'x shot': 'x_shot',
                             'y shot': 'y_shot', 'shot type': 'shot_type',
                             'empty net': 'empty_net', 'home goal': 'home_goal',
                             'away goal': 'away_goal', 'is rebound': 'is_rebound',
                             'game seconds': 'game_seconds',
                             'last event type': 'last_event_type',
                             'x last event': 'x_last_event',
                             'y last event': 'y_last_event',
                             'time from last event': 'time_from_last_event',
                             'num player home': 'num_player_home',
                             'num player away': 'num_player_away',
                             'time power play': 'time_power_play',
                             'shot distance': 'shot_distance',
                             'shot angle': 'shot_angle',
                             'change in shot angle': 'change_in_shot_angle',
                             'distance from last event': 'distance_from_last_event'
                             }, inplace=True)

    test_df = test_df[['game_seconds', 'shot_distance', 'shot_angle', 'period',
                         'x_shot', 'y_shot', 'shot_type', 'last_event_type',
                         'x_last_event', 'y_last_event', 'time_from_last_event',
                         'distance_from_last_event', 'is_rebound',
                         'change_in_shot_angle', 'speed']]
    test_df['is_rebound'] = np.where(test_df['is_rebound'] == False, 0, 1)
    # One hot-encoding for categorical variables
    transformer = make_column_transformer(
        (OneHotEncoder(), ['shot_type', 'last_event_type']),
        remainder='passthrough')
    transformed = transformer.fit_transform(test_df)
    transformed_X = pd.DataFrame(transformed,
                                 columns=transformer.get_feature_names_out())
    transformed_X.dropna(inplace=True)
    return transformed_X


def load_data(data=None, features=None):
    X_test = data[features]
    return X_test


# current_dir_path = os.getcwd()
# dir_path = os.path.join("\\".join(current_dir_path.split('\\')[:-2]), "data", "processed",
#                         "df_feature_engineering.csv")
# X_test = dataload(dir_path)

