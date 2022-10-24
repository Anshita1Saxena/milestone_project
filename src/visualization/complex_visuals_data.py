import pandas as pd


def get_games_num(df):
    """
    This function takes a dataframe with play_by_play events of the whole season
    computes the number of games a specific team played in that season.
    """
    list_teams = list(df['team shot'].unique())
    games_per_team = {}
    for team in list_teams:
        new_df = df[df['team shot'] == team]
        games_per_team[team] = new_df['game id'].unique().shape[0]
    return games_per_team


def aggregate_team_location(df, games_per_team):
    """
    Computes the number of shots per location for each team in each for each season
    And the shot rate per hour is equal to the total number of shots in a location
    divided by total number of games (1 game = 60 mins = 1h = 3 periods of 20 minutes, sometimes overtimes)
    """
    # fit the plot in the rink
    df['y'] = df['y_transformed'] * (-1)
    # bins creation, choosen the coordinates based on the x-axis and y-axis of an image
    y_bins, goal_dist_bins = list(range(-40, 40, 2)), list(range(0, 93, 2))
    df['y_bins'], df['goal_dist_bins'] = pd.cut(df['y'], y_bins), pd.cut(df['goal_dist'], goal_dist_bins)
    # data will be grouped by season, team_shot, y_bins, goal_dist_bins - Team wise shot group
    new_df = df.groupby(['season', 'team shot', 'y_bins', 'goal_dist_bins'])['goal'].size().to_frame('total').reset_index()
    new_df['games_per_team'] = new_df['team shot'].apply(lambda x: games_per_team.get(x))
    # total be the number of games of team
    new_df['average_per_hour'] = new_df['total'] / new_df['games_per_team']
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right) / 2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right) / 2)
    # file for the tracing of data
    new_df.to_csv("complex_team.csv", index=False, encoding='utf-8-sig')
    return new_df


def aggregate_shot_location(df):
    """
    Computes the number of shots at each location for the whole league
    And the shot rate per hour = total # of shots in a location/total number of games (since 1game = 1h)
    """
    # y_transformed coordinate into the rink, actual shot location coordinate
    df['y'] = df['y_transformed'] * (-1)
    # total number of games
    total_games = df['game id'].unique().shape[0]
    # bins creation for displaying the densities of shot goal per locations by team in each season
    y_bins, goal_dist_bins = list(range(-40, 40, 2)), list(range(0, 93, 2))
    df['y_bins'], df['goal_dist_bins'] = pd.cut(df['y'], y_bins), pd.cut(df['goal_dist'], goal_dist_bins)
    # data will be grouped by season, y_bins, goal_dist_bins
    new_df = df.groupby(['season', 'y_bins', 'goal_dist_bins'])['goal'].size().to_frame('total').reset_index()
    # two teams are playing at same time for a match
    new_df['average_per_hour'] = new_df['total'] / (2*total_games)
    new_df['y_mid'] = new_df['y_bins'].apply(lambda x: (x.left + x.right) / 2)
    new_df['goal_mid'] = new_df['goal_dist_bins'].apply(lambda x: (x.left + x.right) / 2)
    # recorded the data for tracing
    new_df.to_csv("complex_league.csv", index=False, encoding='utf-8-sig')
    return new_df


def transform_coordinates(rinkSide, coordinate):
    """
    This function transforms a coordinate from the left offensive side to the right one.
    """
    # Left offensive zone transformation
    if rinkSide == "right":
        return (-1) * coordinate
    else:
        return coordinate


# function to normalize coordinates and compute distances
def transformed_col(df):
    """
    This function adds columns to the data frame:
     - x_transformed and y_transformed: the coordinates transposed to the right side of the rink
     - euclidean distance: the euclidean distance from a shot location to the center of the rink
     - goal_dist: the distance from a shot location to the goal line: 89-x_transformed
    """
    # str.split() with expand=True option results in a data frame and without
    # that we will get Pandas Series object as output.
    df[['coordinates_x', 'coordinates_y']] = df.coordinates.str.split(", ", expand=True,)
    df['coordinates_x'] = df['coordinates_x'].str.replace('[', '', regex=True)
    df['coordinates_y'] = df['coordinates_y'].str.replace(']', '', regex=True)
    # convert the coordinates into numeric columns
    df[['coordinates_x', 'coordinates_y']] = df[['coordinates_x', 'coordinates_y']].apply(pd.to_numeric)
    # transformed coordinates have to be adjusted
    df['x_transformed'] = df.apply(lambda x: transform_coordinates(x['rinkSide'], x['coordinates_x']), axis=1)
    df['y_transformed'] = df.apply(lambda x: transform_coordinates(x['rinkSide'], x['coordinates_y']), axis=1)
    # field less than 25 is not the part of offensive zone
    df = df.drop(df[df.x_transformed < 25].index)
    # goal distance
    df['goal_dist'] = df.apply(lambda x: (89 - x['x_transformed']), axis=1)
    return df


def get_season_agg(y_mid, goal_mid, league_df):
    """"
    This function returns the shot rate per hour of the whole league for a specific location.
    """
    league = league_df.loc[(league_df["y_mid"] == y_mid) & (league_df["goal_mid"] == goal_mid), 'average_per_hour']
    return league.iloc[0]


def main():
    # Read the tidy data generated in Question 4
    df = pd.read_csv("complex_visuals.csv")
    # drop the events where rinkSide is not present.
    df = df.dropna(subset=['rinkSide'])
    # add columns: x,y transposed to the right side of the rink, goal dist, and on which side of the y_axis
    df = transformed_col(df)
    # Dictionary consists of the number of games for each team
    games_per_team = get_games_num(df)
    # Taking season in group by to keep the data season wise
    # df with shot rate per hour group by location (across all teams)
    df_league = league_df = aggregate_shot_location(df)
    # df with shot rate per hour grouped by team and location
    df_team = aggregate_team_location(df, games_per_team)
    # Add the corresponding shot rate per hour of the league in each row of df_team
    df_team['league_average'] = df_team.apply(lambda x: get_season_agg(x['y_mid'], x['goal_mid'], df_league), axis=1)
    # Excess shots calculation
    df_team['raw_diff'] = df_team['average_per_hour'] - df_team['league_average']
    # Save the data to easily process it via visualization Dash app.
    df_team.to_csv("complex_diff.csv", index=False, encoding='utf-8-sig')


if __name__ == "__main__":
    main()
