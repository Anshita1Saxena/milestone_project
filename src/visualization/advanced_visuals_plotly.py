# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter
import scipy.stats as st
from scipy.interpolate import griddata
import numpy as np

import pandas as pd
from PIL import Image

# Below CSS is for aligning the radio buttons vertically.
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)

# Initializing Dash Application
app = Dash(__name__)
# Make sure that the web app should be hosted on Heroku
server = app.server

# Read the dataframe
hockey_df = pd.read_csv('complex_diff.csv')

# Store the seasons along with their team list.
all_options = {}
for season in hockey_df.season.unique().tolist():
    season_df = hockey_df.loc[hockey_df['season'] == season]
    shot_team_list = list(set(season_df['team shot'].unique().tolist()))
    all_options[season] = shot_team_list

# Prepare the app layout consists of interactive radio buttons and Shot Map.
app.layout = html.Div([

    # Season Radio Button list
    html.Div([
        dcc.RadioItems(
            list(all_options.keys()),
            list(all_options.keys())[0],
            id='season-radio',
        ),

        html.Hr(),

        # Team Radio Button List
        dcc.RadioItems(id='team-radio'),

        html.Hr(),

        # Simple callback to check the selected values
        html.Div(id='display-selected-values')
    ]),

    # The shot map graph
    dcc.Graph(id='indicator-graphic')
])


# Display all the seasons on the dash app
@app.callback(
    Output('team-radio', 'options'),
    Input('season-radio', 'value'))
def set_seasons_options(selected_season):
    return [{'label': i, 'value': i} for i in all_options[selected_season]]


# Display all the Teams on the dash app
@app.callback(
    Output('team-radio', 'value'),
    Input('team-radio', 'options'))
def set_teams_value(available_options):
    return available_options[0]['value']


# Display the selection of season and team dynamically on dash app
@app.callback(
    Output('display-selected-values', 'children'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def set_display_children(selected_season, selected_team):
    return u'{} is a team played in {} season'.format(
        selected_team, selected_season,
    )


# Display the updated graph based on the user input values
@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def update_graph(selected_season, selected_team):

    # Add the background rink image
    im = Image.open("nhl_rink.jpg")
    # Transpose the image to rotate the x-axis and y-axis
    rot_img = im.transpose(Image.Transpose.ROTATE_90)
    img_width, img_height = rot_img.size

    # Selection of data based on user's selected team and season
    team_df = hockey_df.loc[(hockey_df['team shot'] == selected_team) & (hockey_df['season'] == selected_season)]
    # Image Contour width and height setting
    x_rink = np.sort(team_df['y_mid'].unique())
    y_rink = np.sort(team_df['goal_mid'].unique())
    [x, y] = np.round(np.meshgrid(x_rink, y_rink))
    # Return the value determined from a piecewise cubic, continuously differentiable (C1),
    # & approximately curvature-minimizing polynomial surface.
    diff = griddata((team_df['y_mid'], team_df['goal_mid']), team_df['raw_diff'], (x, y),
                    method='cubic', fill_value=0)

    # Add go figure contour plot showing densities
    fig = go.Figure(data=
    go.Contour(
        z=gaussian_filter(diff, sigma=2),
        x=x_rink,
        y=y_rink,
        opacity=0.7,
        # with _r, the colorbar scale will be inverted
        colorscale='RdBu_r'
    ))
    # To put the contour plot upside down
    fig.update_yaxes(autorange="reversed")

    # The background image setting based on the axis and the data
    fig.add_layout_image(
        dict(
            source=rot_img,
            xref="x",
            yref="y",
            # Set the image starting and ending
            x=-40,
            y=-10,
            # Set the image width and image height based on data and x-axis and y-axis
            sizex=img_width/6,
            sizey=img_height/5.5,
            sizing="stretch",
            opacity=0.5,
            layer="above")
    )
    # Add the empty scatter plot to adjust the figure on desired axis
    fig.add_trace(
        go.Scatter(
            x=[-40, 40],
            y=[-4, None, 100],
            showlegend=False)
    )
    # Set the range of x-axis and y-axis
    fig.update_xaxes(range=[-40, 40], title_text='Distance from the center of the rink(ft)')
    fig.update_yaxes(range=[-10, 100], title_text='Distance from the goal line(ft)')
    # Show the exact axis on the line required by the question
    fig.update_yaxes(tickvals=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90])

    # Remove the blue grid effect from image using the below code "plotly_white"
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        template="plotly_white",
        title="5v5 Offence"
    )

    return fig


if __name__ == '__main__':
    # Run the application on the specific port. This saves the conflicts for acquiring the ports from the machine.
    app.run_server(debug=True, port=8051)
