import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


def read_file_json(file_path) -> dict:
    """
    Read a json file for a particular game based on the file path
    :param file_path: Path to the file containing the game data
    :return: A dictionary containing data of the json file
    """
    if os.path.exists(file_path):
        with open(file_path) as f:
            return json.load(f)
    else:
        print("Cannot find the game in: "+file_path)
        return


def game_events_to_np(file_path):
    """
    Get all shot and goal events in a particular game and convert into a numpy array
    :param file_path: Path to the file containing the game data
    :return: A numpy with information about all shot and goal events
    """
    # read json data into a dictionary
    data_dict = read_file_json(file_path)

    # define desirable features we want to cover
    event_features = ['season', 'game time', 'period', 'period time', 'game id', 'home team', 'away team',
                      'goal', 'team shot', 'goal home coordinates', 'coordinates', 'shooter', 'goalie',
                      'shot type', 'empty net', 'strength', 'home goal', 'away goal', 'rebound', 'shot distance',
                      'rink side']

    event_features.append('shot distance inv')
    np_game = np.array(event_features)

    # get the corresponding features from json
    all_plays = data_dict['liveData']['plays']['allPlays']

    # If there is no data in this file, return an empty list
    if (len(all_plays) == 0):
        return []

    game_time = data_dict['gameData']['datetime']['dateTime']
    game_id = data_dict['gamePk']
    season = str(game_id)[0:4]
    team_away = data_dict['gameData']['teams']['away']['triCode']
    team_home = data_dict['gameData']['teams']['home']['triCode']

    # coordinates of two teams' goal
    goal_left_coordinates = np.array([-89, 0])
    goal_right_coordinates = np.array([89, 0])
    try:
        first_period_home_side = data_dict['liveData']['linescore']['periods'][0]['home']['rinkSide']
    except:
        return []
    #         first_period_home_side=""
    #         print(f"Game ID {game_id} does not have rinkSide")

    # initialize
    goal_home_coordinates = goal_right_coordinates
    goal_away_coordinates = goal_left_coordinates

    # loop through all events
    for play_indx, play in enumerate(all_plays):
        event_type = play['result']['event']
        is_goal = event_type == 'Goal'
        period_time = play['about']['periodTime']
        period = play['about']['period']
        home_goal = play['about']['goals']['home']
        away_goal = play['about']['goals']['away']

        # Switch sides
        if ((all_plays[play_indx - 1]['about']['period'] != period and period != 1)):
            goal_home_coordinates_current = goal_home_coordinates
            goal_home_coordinates = goal_away_coordinates
            goal_away_coordinates = goal_home_coordinates_current

        # only get shots and goals
        if (event_type not in ['Goal', 'Shot']):
            continue

        team_shot = play['team']['triCode']

        if (is_goal):
            is_empty_net = play['result']['emptyNet'] if 'emptyNet' in play['result'] else ""
            strength = play['result']['strength']['name'] if 'strength' in play['result'] else ""
        else:
            is_empty_net = ""
            strength = ""

        # is a shot/goal rebound
        is_rebound = False
        # a shot/goal is rebound if it is from a blocked shot of the same team
        if ((all_plays[play_indx - 1]['result']['event'] == "Blocked Shot") and (
                team_shot == all_plays[play_indx - 1]['team']['name'])):
            is_rebound = True

        # whether the information of corrdinates missing
        is_corr_available = all(cor in play['coordinates'] for cor in ['x', 'y'])
        coordinates = [play['coordinates']['x'], play['coordinates']['y']] if is_corr_available else ""

        # does not keep event without coordinates
        if (len(coordinates) == 0):
            continue

        # whether team home takes this shot
        is_team_home_shot = play['team']['triCode'] == team_home
        # coordinates of the goal being shot
        goal_shot_coordinates = goal_away_coordinates if is_team_home_shot else goal_home_coordinates
        goal_shot_coordinates_inv = goal_home_coordinates if is_team_home_shot else goal_away_coordinates
        # distance of the shot to the goal
        shot_distance = np.linalg.norm(np.array(goal_shot_coordinates) - np.array(coordinates))
        shot_distance_inv = np.linalg.norm(np.array(goal_shot_coordinates_inv) - np.array(coordinates))

        shot_type = play['result']['secondaryType'] if 'secondaryType' in play['result'] else ""

        goalie = ""
        for player in play['players']:
            if (player['playerType'] == "Goalie"):
                goalie = player['player']['fullName']
                continue
            if (player['playerType'] in ["Scorer", "Shooter"]):
                shooter = player['player']['fullName']

        # Adding rink side for the complex_visuals code
        type = 'home' if team_home == team_shot else 'away'
        if len(data_dict['liveData']['linescore']['periods']) > 0 and period <= 4:
            rink_side = data_dict['liveData']['linescore']['periods'][period-1][type]['rinkSide']

        # a particular event
        event_data = [season, game_time, period, period_time, game_id, team_home, team_away,
                      is_goal, team_shot, goal_home_coordinates,
                      coordinates, shooter, goalie, shot_type, is_empty_net,
                      strength, home_goal, away_goal, is_rebound, shot_distance, rink_side, shot_distance_inv]

        np_game = np.vstack((np_game, event_data))

    df_game = pd.DataFrame(data=np_game[1:], columns=event_features)
    if df_game['shot distance inv'].sum() < df_game['shot distance'].sum():
        df_game['shot distance'] = df_game['shot distance inv']
        df_game['goal home coordinates'] = df_game['goal home coordinates'] * (-1)

    # The first row of np_game is just the header
    np_game = df_game.values

    np_game = np_game[:, :-1]

    return np_game


def get_list_of_files(dir_path):
    """
    Get the list of all files in a directory and its sub-directories
    :param dir_path: Path to the directory
    :return: A list with all files in the directory and its sub-directories
    """
    # Get all files and directories of the given top directory
    list_entries_top_dir = os.listdir(dir_path)

    # list of all files
    list_all_files = list()

    # loop through all sub-directories
    for entry in list_entries_top_dir:
        entry_path = os.path.join(dir_path, entry)
        # check whether the entry is a file or a directory
        if (os.path.isdir(entry_path)):
            list_all_files = list_all_files + get_list_of_files(entry_path)
        else:
            list_all_files.append(entry_path)
    return list_all_files


def all_games_events_to_df(dir_path):
    """
    Get all shots and goals of all games inside a directory and its sub-directories
    :param dir_path: Path to the directory containing hockey games as json files
    :return: A single dataframe containing information of all shots and goals in the directory
    """
    list_all_files = get_list_of_files(dir_path)

    # define desirable features we want to cover
    event_features = ['season', 'game time', 'period', 'period time', 'game id', 'home team', 'away team',
                      'goal', 'team shot', 'goal home coordinates', 'coordinates', 'shooter', 'goalie',
                      'shot type', 'empty net', 'strength', 'home goal', 'away goal', 'rebound', 'shot distance',
                      'rinkSide']

    np_all_game_events = np.array(event_features)
    #     list_all_game_events = [[0 for columns in range(len(event_features))] for rows in range(len(list_all_files))]
    #     np_all_game_events = np.empty((len(list_all_files), len(event_features)))

    for file_path in tqdm(list_all_files):
        np_game_events = game_events_to_np(file_path)
        if (len(np_game_events) == 0):
            continue
        #         list_all_game_events[file_idx] = np_game_events
        #         np_all_game_events[file_idx] = np_game_events
        np_all_game_events = np.vstack((np_all_game_events, np_game_events))

    df_all_game_events = pd.DataFrame(data=np_all_game_events[1:], columns=np_all_game_events[0])
    #     df_all_game_events = pd.DataFrame(data=list_all_game_events, columns=event_features)
    return df_all_game_events


# Test for all dataset
current_dir_path = os.getcwd()
dir_path = os.path.join("\\".join(current_dir_path.split('\\')[:-3]), "milestone1", "data", "raw", "regular_season")
df_game_events = all_games_events_to_df(dir_path)
save_dir_path = os.path.join("\\".join(current_dir_path.split('\\')[:-2]), "data", "processed", "complex_visuals.csv")
df_game_events.to_csv(save_dir_path, index=False)
df_game_events
