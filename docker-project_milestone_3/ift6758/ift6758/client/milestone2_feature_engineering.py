import pandas as pd
import numpy as np

def calc_shot_distance(x_shot, y_shot, rink_side):
    """
    A supporting function to calculate distance from the shot to the net
    :param x_shot: x coordinate of the shot
    :param y_shot: y coordinate of the shot
    :param rink_side: side of the net, only accepts "left" or "right"
    """
    if(rink_side not in ['left','right']):
        return ""
    coor_net = np.array([89,0]) if rink_side=="right" else np.array([-89,0])
    coor_shot = np.array([x_shot, y_shot]).astype(np.float)
    return round(np.linalg.norm(coor_shot - coor_net))
    

def add_shot_distance_and_correct_coordinates(df_game, is_fix_wrong_coor=False):
    """
    Calculate the shot distance and detect incorrect coordinates to fix them
    :param df_game: The tidied data-frame
    :return: A dataframe after correcting the coordinates with the added "shot distance" column
    """
    df_game_added_distance = df_game.copy()
    df_game_added_distance = df_game_added_distance.astype({'x shot':'float','y shot':'float'})
    np_shot_distance = df_game_added_distance.apply(lambda event: calc_shot_distance(
                                                    event['x shot'],
                                                    event['y shot'],
                                                    event['rinkSide']),
                                                    axis=1)
    
    np_shot_distance_inv = df_game_added_distance.apply(lambda event: calc_shot_distance(
                                                    -(event['x shot']),
                                                    event['y shot'],
                                                    event['rinkSide']),
                                                    axis=1)
    
    if(np.mean(np_shot_distance_inv) < np.mean(np_shot_distance)):
        df_game_added_distance['shot distance'] = np_shot_distance_inv
        # invert the rinkSide because this information is not correct
        df_game_added_distance['rinkSide'] = df_game_added_distance['rinkSide'].apply(lambda side: "right" if side=="left" else "left")
        # df_game_added_distance['x shot'] = df_game_added_distance['x shot']*(-1)
        # df_game_added_distance['x last event'] = df_game_added_distance['x last event']*(-1)
    else:
        df_game_added_distance['shot distance'] = np_shot_distance
    
    return df_game_added_distance

def calc_shot_angle(x_shot, y_shot, rink_side):
    """
    Calculate the angle of the shot from the net
    :param shot_coordinates: A numpy array indicating coordinates of the shot (e.g., [46, 25])
    :param net_coordinates: A numpy array indicating coordinates of the net (e.g., [-89, 0])
    :return: angle of the shot from net in degree
    """
    x_shot = float(x_shot)
    y_shot = float(y_shot)
    if(rink_side not in ['left','right'] or np.isnan(float(x_shot)) or np.isnan(float(y_shot))):
        return ""
    
    x_net = 89 if rink_side=="right" else (-89)
    
    x_dist_abs = np.abs(x_net - float(x_shot))
    y_dist_abs = np.abs(float(y_shot))
    
    is_shot_behind_net = (x_net==89 and x_shot>89) or (x_net==-89 and x_shot<-89)
    is_shot_perpendicular_net = (x_net==x_shot)
    
    if(y_shot == 0):
        angle = 0
    else:
        if(is_shot_perpendicular_net):
            angle = np.pi/2
        else:
            angle = np.arctan(y_dist_abs/x_dist_abs)
            if(is_shot_behind_net):
                angle += np.pi/2

    return round(np.rad2deg(angle))


def add_shot_angle(df_game):
    """
    Add a "shot angle" column into the dataframe, which indicates the angle of the shot
    :param df_game: The tidied data-frame
    :return: A dataframe with the added "shot angle" column
    """
    
    df_game_added_angle = df_game.copy()
    df_game_added_angle['shot angle'] = df_game_added_angle.apply(lambda event: calc_shot_angle(
                                                                event['x shot'],
                                                                event['y shot'],
                                                                event['rinkSide']),
                                                                axis=1)
    return df_game_added_angle

def calc_distance_two_events(x_shot, y_shot, x_last, y_last):
    """
    A supporting function for calculating the distance between two events, automatically ignore inappropriate values
    :param x_shot: x coordinate of the shot
    :param y_shot: y coordinate of the shot
    :param x_last: x coordinate of the previous event
    :param y_last: y coordinate of the previous event
    :return: The distance between the shot and the previous event
    """
    if(np.isnan(x_last) or np.isnan(y_last)):
        return ""
    else:
        shot_coordinates = np.array([x_shot, y_shot])
        last_event_coordinates = np.array([x_last, y_last])
        return round(np.linalg.norm(shot_coordinates - last_event_coordinates))

def calc_speed(distance_from_last_event, time_from_last_event):
    """
    A supporting function for calculating the speed of the shot, automatically ignore inappropriate values
    :param distance_from_last_event: Distance of the shot from the previous event
    :param time_from_last_event: Time from the previous event to the current shot
    :return: The speed of the shot
    """
    if(distance_from_last_event!="" and time_from_last_event!=0):
        return round(float(distance_from_last_event)/float(time_from_last_event))
    else:
        return ""

def add_distance_from_last_event_and_speed(df_game):
    """
    Add two columns which are "distance from last event" and "speed" to the dataframe
    :param df_game: The dataframe which has been added "time from last event" column
    :return: The dataframe after adding the two columns
    """
    df_game_added_features = df_game.copy()
    df_game_added_features = df_game_added_features.astype({'x shot':'float','y shot':'float','x last event':'float','y last event':'float'})
    df_game_added_features['distance from last event'] = df_game_added_features.apply(lambda event: calc_distance_two_events(
                                                                event['x shot'],
                                                                event['y shot'],
                                                                event['x last event'],
                                                                event['y last event']),
                                                                axis=1)
    df_game_added_features['speed'] = df_game_added_features.apply(lambda event:calc_speed(
                                                                event['distance from last event'],
                                                                event['time from last event']),
                                                                axis=1)
    return df_game_added_features


def calc_change_angle(is_rebound, current_shot_angle, y_shot, x_last, y_last, rinkSide):
    """
    A supporting function to calculate the change in angle between two consecutive shot of a rebound
    :param is_rebound: Whether the shot is rebound or not
    :param current_shot_angle: Shot angle of the current shot
    :param y_shot: y coordinate of the shot
    :param x_last: x coordinate of the previous shot
    :param y_last: x coordinate of the previous shot
    :param rinkSide: Side of the net. To calculate the angle of the previous shot
    :return: The change in angle of the rebound
    """
    y_shot = float(y_shot)
    y_last = float(y_last)
    x_last = float(x_last)
    if(not is_rebound):
        return 0

    last_shot_angle = calc_shot_angle(x_last, y_last, rinkSide)
    if(last_shot_angle==""):
        return 0

    if(np.sign(y_shot)==np.sign(y_last)): # two shots at the same vertical side
        return np.abs(last_shot_angle-current_shot_angle)
    else:
        return (last_shot_angle+current_shot_angle)

def add_change_in_shot_angle(df_game):
    """
    Add the column "change in shot angle" indicating the change in angle of a rebound
    :param df_game: The dataframe which has been added "shot angle" column
    """
    df_game_change_angle = df_game.copy()
    df_game_change_angle['change in shot angle'] = df_game_change_angle.apply(lambda event: calc_change_angle(
                                                                            event['is rebound'],
                                                                            event['shot angle'],
                                                                            event['y shot'],
                                                                            event['x last event'],
                                                                            event['y last event'],
                                                                            event['rinkSide']), axis=1)
    return df_game_change_angle


def feature_engineering(df_game_tidied):
    """
    The main function of the task feature engineering
    :param df_game_tidied: The tidied version of nhl data, stored in a dataframe
    :return: A new dataframe added all neccesary features.
    """
    df_game = df_game_tidied.copy()
    
    print("Start feature engineering for the tidied dataframe...")
    
    print("Start correcting incorrect coordinates and adding shot distance...")
    #This function must always be called at first of all, because it help correcting wrong coordinates
    df_game = add_shot_distance_and_correct_coordinates(df_game)

    print("Start adding shot angle...")
    df_game = add_shot_angle(df_game)

    print("Start adding change in shot angle...")
    df_game = add_change_in_shot_angle(df_game)

    print("Start adding distance from last event and the shot speed...")
    df_game = add_distance_from_last_event_and_speed(df_game)

    print("Finish feature engineering!")
    return df_game
    