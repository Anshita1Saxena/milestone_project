import json
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

class penalty_object:
    def __init__(self, start_time_seconds, penalty_minutes, is_team_home):
        self.start_time_seconds = start_time_seconds
        self.end_time_seconds = start_time_seconds + penalty_minutes*60
        self.end_time_first_minor = start_time_seconds + 2*60
        self.is_major = penalty_minutes==5
        self.is_double_minor = penalty_minutes==4
        self.is_team_home = is_team_home
        self.available = True
        
    def update_infor(self, current_time_seconds, is_goal):
        # if the penalty time expire, set the state to unavailable
        if(current_time_seconds >= self.end_time_seconds):
            self.available = False
            return
        
        # if the penalty time has not expired, check is_goal
        if(is_goal):
            if(self.is_major): # if this is a major penalty, the penalty still continue
                return
            elif(self.is_double_minor): # this is a double minor penalty
                if(current_time_seconds < self.end_time_first_minor): # the first minor time has not expired yet
                    self.end_time_seconds = current_time_seconds + 2*60
                else: # the first minor time has expired already, so the second one will be finished immediately
                    self.available = False
            else: # this is a minor penalty, so it expires after a shot
                self.available = False
        

def read_file_json(file_path) -> dict:
    """
    Read a json file for a particular game based on the file path
    :param file_path: Path to the file containing the game data
    :return: A dictionary containing data of the json file
    """
    if(os.path.exists(file_path)):
        with open(file_path) as f:
            return json.load(f)
    else:
        print("Cannot find the game in: "+file_path)
        return
    
    
def game_events_to_np(file_path, features_set):
    """
    Get all shot and goal events in a particular game and convert into a numpy array
    :param file_path: Path to the file containing the game data
    :param features_set: The set of all desirable features
    :return: A numpy with information about all shot and goal events
    """
    # read json data into a dictionary
    data_dict = read_file_json(file_path)
    
    # initial the numpy which will contain all events information
    np_game = np.array(features_set)
    
    # get the corresponding features from json
    all_plays = data_dict['liveData']['plays']['allPlays']
    
    # If there is no data in this file, return an empty list
    if(len(all_plays)==0):
        return []
    
    try:
        first_period_home_side = data_dict['liveData']['linescore']['periods'][0]['home']['rinkSide']
    except:
        return []
    
    game_date = data_dict['gameData']['datetime']['dateTime'][0:10]
    
    game_id = data_dict['gamePk']
    season = str(game_id)[0:4]
    team_away = data_dict['gameData']['teams']['away']['triCode']
    team_home = data_dict['gameData']['teams']['home']['triCode']
    
    # coordinates of two teams' goal
    goal_left_coordinates = np.array([-89,0])
    goal_right_coordinates = np.array([89,0])
        
    # initialize coordinates (if this is not correct, we will correct it later in feature engineering)
    goal_home_coordinates = goal_right_coordinates
    goal_away_coordinates = goal_left_coordinates
    
    penalty_object_list = []
    time_start_penalty = 0
    is_currently_penalty = False
    time_power_play = 0
    
    # loop through all events
    for play_indx, play in enumerate(all_plays):
        event_type = play['result']['event']
        is_goal = event_type == 'Goal'
        home_goal = play['about']['goals']['home']
        away_goal = play['about']['goals']['away']
        
        # Game time for the current event
        period_time = play['about']['periodTime']
        period = play['about']['period']
        period_minutes = int(period_time.split(':')[0])
        period_seconds = int(period_time.split(':')[1])
        game_seconds = (period-1)*20*60 + period_minutes*60 + period_seconds
        
        # handle penalty event
        if(event_type == "Penalty"):
            # start the penalty counting time
            if(len(penalty_object_list)==0):
                time_start_penalty = game_seconds
                is_currently_penalty = True
            
            # append the penalty to the penalty list
            penalty_minutes = play['result']['penaltyMinutes']
            is_team_home = play['team']['triCode'] == team_home
            penalty_object_list.append(penalty_object(game_seconds, penalty_minutes, is_team_home))
        
        num_player_home = 5
        num_player_away = 5
        
        # update penalty-related information
        for penalty_instance in penalty_object_list:
            if(penalty_instance.available):
                if(penalty_instance.is_team_home):
                    num_player_home -= 1
                else:
                    num_player_away -= 1
            else: 
                penalty_object_list.remove(penalty_instance) # the penalty expires
            penalty_instance.update_infor(game_seconds, is_goal)
        
        if(is_currently_penalty):
            time_power_play = game_seconds - time_start_penalty
        
        time_power_play_store = time_power_play
        
        # if it's currently penalty and the number of players in both teams comback to 5 --> the end of penalty
        if(is_currently_penalty and num_player_home==5 and num_player_away==5):
            is_currently_penalty = False
            time_power_play = 0
        
        # Game time for the previous event
        period_time_last_event = all_plays[play_indx-1]['about']['periodTime']
        period_last_event = all_plays[play_indx-1]['about']['period']
        period_minutes_last_event = int(period_time_last_event.split(':')[0])
        period_seconds_last_event = int(period_time_last_event.split(':')[1])
        game_seconds_last_event = (period_last_event-1)*20*60 + period_minutes_last_event*60 + period_seconds_last_event
        
        # Time distance from the last event
        time_from_last_event = game_seconds - game_seconds_last_event
        
        # Switch sides
        if((all_plays[play_indx-1]['about']['period'] != period and period!=1)):
            goal_home_coordinates_current = goal_home_coordinates
            goal_home_coordinates = goal_away_coordinates
            goal_away_coordinates = goal_home_coordinates_current
        
        # only get shots and goals
        if(event_type not in ['Goal', 'Shot']):
            continue
        
        team_shot = play['team']['triCode']
        
        if(is_goal):
            is_empty_net = play['result']['emptyNet'] if 'emptyNet' in play['result'] else False
            strength = play['result']['strength']['name'] if 'strength' in play['result'] else ""
        else:
            is_empty_net = False
            strength = ""
        
        # is a shot/goal rebound
        is_rebound = False
        # a shot/goal is rebound if it is from a blocked shot of the same team
        if((all_plays[play_indx-1]['result']['event'] in ["Blocked Shot","Shot"])
           and(team_shot==all_plays[play_indx-1]['team']['triCode'])
           and time_from_last_event<5):
            is_rebound = True
        
        # whether the information of corrdinates missing
        is_corr_available = all(cor in play['coordinates'] for cor in ['x', 'y'])
        coordinates = [play['coordinates']['x'], play['coordinates']['y']] if is_corr_available else ""
        
        # does not keep event without coordinates
        if(len(coordinates)==0):
            continue
            
            
        shot_type = play['result']['secondaryType'] if 'secondaryType' in play['result'] else ""
        
        goalie = ""
        for player in play['players']:
            if(player['playerType'] == "Goalie"):
                goalie =  player['player']['fullName']
                continue
            if(player['playerType'] in ["Scorer", "Shooter"]):
                shooter = player['player']['fullName']
                
        # Adding rink side for the complex_visuals code
        type = 'home' if team_home == team_shot else 'away'
        if len(data_dict['liveData']['linescore']['periods']) > 0 and period <= 4:
            rink_side = data_dict['liveData']['linescore']['periods'][period-1][type]['rinkSide']
            
        x_shot = coordinates[0]
        y_shot = coordinates[1]
        
        
        # Add information of the previous event
        last_event_type = all_plays[play_indx-1]['result']['event']
        
        is_corr_available_last_event = all(cor in all_plays[play_indx-1]['coordinates'] for cor in ['x', 'y'])
        last_event_coordinates = [all_plays[play_indx-1]['coordinates']['x'], all_plays[play_indx-1]['coordinates']['y']] if is_corr_available_last_event else ""
        if(last_event_coordinates!=''):
            x_last_event = last_event_coordinates[0]
            y_last_event = last_event_coordinates[1]
        else:
            x_last_event = ""
            y_last_event = ""
        
        # a particular event
        event_data = [season, game_date, period, period_time, game_id, team_home, team_away,
                      is_goal, team_shot, x_shot, y_shot, shooter, goalie, shot_type,
                      is_empty_net, strength, home_goal, away_goal, is_rebound, rink_side,
                      game_seconds, last_event_type, x_last_event, y_last_event, time_from_last_event,
                      num_player_home, num_player_away, time_power_play_store]

        np_game = np.vstack((np_game, event_data))
    
    
    # The first row of np_game is just the header
    df_game = pd.DataFrame(data=np_game[1:], columns=np_game[0])
        
    np_game = df_game.values

    return np_game


def get_list_of_files(dir_path):
    """
    A Supporting function to get the list of all files in a directory and its sub-directories
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
        if(os.path.isdir(entry_path)):
            list_all_files = list_all_files + get_list_of_files(entry_path)
        else:
            list_all_files.append(entry_path)
    return list_all_files

def all_games_events_to_df(dir_path, features_set):
    """
    Get all shots and goals of all games inside a directory and its sub-directories
    :param dir_path: Path to the directory containing hockey games as json files
    :param features_set: The set of all desirable features
    :return: A single dataframe containing information of all shots and goals in the directory
    """
    list_all_files = get_list_of_files(dir_path)
    
    np_all_game_events = np.array(features_set)

    for file_path in tqdm(list_all_files):
        np_game_events = game_events_to_np(file_path, features_set)
        if(len(np_game_events)==0):
            continue
        np_all_game_events = np.vstack((np_all_game_events, np_game_events))
        
    df_all_game_events = pd.DataFrame(data=np_all_game_events[1:], columns=np_all_game_events[0])
    return df_all_game_events

def tidy_data(seasons_list, raw_data_dir_path):
    # define desirable features we want to cover
    features_set = ['season','game date','period','period time','game id','home team','away team',
                      'is goal','team shot','x shot', 'y shot','shooter','goalie',
                      'shot type','empty net','strength','home goal','away goal','is rebound', 'rinkSide',
                      'game seconds','last event type', 'x last event', 'y last event', 'time from last event',
                      'num player home', 'num player away', 'time power play']
    df_game = pd.DataFrame(columns=features_set)
    for season in seasons_list:
        print(f"Getting data of the season {season}")
        dir_path = os.path.join(raw_data_dir_path, season)
        df_game_events = all_games_events_to_df(dir_path, features_set)
        df_game = pd.concat([df_game, df_game_events])
    return df_game
    
# Test for all dataset
# raw_data_dir_path = os.path.join("..","data","raw")
# seasons_list = ['2016', '2017', '2018', '2019']
current_dir_path = os.getcwd()
dir_path = os.path.join("\\".join(current_dir_path.split('\\')[:-3]), "milestone1", "data", "raw", "playoffs")
seasons_list = ['2020']
df_game_tidied = tidy_data(seasons_list, dir_path)
df_game_tidied.to_csv('df_tidy_data.csv',index=False)

