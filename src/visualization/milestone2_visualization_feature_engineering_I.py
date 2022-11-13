import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

df_game = pd.read_csv("df_feature_engineering.csv")

def milestone2_q2_1_shot_by_distance(df_game):
    figure = plt.figure(figsize=(10, 10))
    ax = sns.histplot(df_game, x="shot distance", hue="is goal", multiple ="stack", bins=50)
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Shot count')
    plt.title('Histogram of shot counts, binned by distance')
    ax.legend(['Goal','No goal'], title = "Result")
    
def milestone2_q2_1_shot_by_angle(df_game):
    figure = plt.figure(figsize=(10, 10))
    ax = sns.histplot(df_game, x="shot angle", hue="is goal", multiple ="stack", bins=50)
    plt.xlabel('Shot angle (degree)')
    plt.ylabel('Shot count')
    plt.title('Histogram of shot counts, binned by shot angle')
    ax.legend(['Goal','No goal'], title = "Result")
    
def milestone2_q2_1_shot_by_distance_and_angle(df_game):
    figure = plt.figure(figsize=(20, 20))
    ax = sns.jointplot(data=df_game, x="shot distance", y="shot angle", kind="hist")
    ax.ax_joint.set_xlabel("Shot distance (ft)")
    ax.ax_joint.set_ylabel("Shot angle (degree)")
    ax.fig.tight_layout()

def milestone2_q2_2_goal_rate_by_distance(df_game):
    df_groupby_distance = df_game.groupby(["shot distance"])["is goal"].mean().to_frame().reset_index()
    ax = sns.lineplot(data=df_groupby_distance , x='shot distance', y='is goal')
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal rate')
    plt.xticks(np.arange(0, 220, 20))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.title('Relation between goal rate and shot distance')
    
def milestone2_q2_2_goal_rate_by_angle(df_game):
    df_groupby_distance = df_game.groupby(["shot angle"])["is goal"].mean().to_frame().reset_index()
    ax = sns.lineplot(data=df_groupby_distance , x='shot angle', y='is goal')
    plt.xlabel('Shot angle (degree)')
    plt.xticks(np.arange(0, 220, 20))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.ylabel('Goal rate')
    plt.title('Relation between goal rate and shot angle')
    
def milestone2_q2_3_goal_non_empty_net_by_distance(df_game):
    df_goal = df_game[df_game['is goal']==True]
    df_goal_non_empty_net = df_goal[df_goal['empty net']==False]
    df_goal_empty_net = df_goal[df_goal['empty net']==True]
    
    fig = plt.figure(figsize=(10, 15))
    
    plt.subplot(211)
    ax = sns.histplot(data=df_goal_non_empty_net, x='shot distance', bins=50)
    plt.xticks(np.arange(0, 220, 20))
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal count')
    plt.title('Histogram of goal counts by distance - Non-empty net')
    
    plt.subplot(212)
    ax = sns.histplot(data=df_goal_empty_net, x='shot distance', bins=50)
    plt.xticks(np.arange(0, 220, 20))
    plt.xlabel('Shot distance (ft)')
    plt.ylabel('Goal count')
    plt.title('Histogram of goal counts by distance - Empty net')
    
    df_goal_non_empty_net_defensive_zone = df_goal_non_empty_net[df_goal_non_empty_net['shot distance']>100]
    list_columns_keep = ['game date','period','period time','game id','is goal','empty net',
                         'home team','away team', 'team shot','x shot','y shot','shooter',
                         'rinkSide','shot distance']
    
    return df_goal_non_empty_net_defensive_zone[list_columns_keep].rename(columns={'rinkSide':'net side'})
    
milestone2_q2_1_shot_by_distance_and_angle(df_game)