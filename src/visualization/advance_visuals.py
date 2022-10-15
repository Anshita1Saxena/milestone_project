# -*- coding: utf-8 -*-
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import plotly.express as px

import pandas as pd
from PIL import Image

# Below CSS is for aligning the radio buttons vertically.
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)

app = Dash(__name__)

hockey_df = pd.read_csv('all_game_events.csv')

all_options = {}
for season in hockey_df.season.unique().tolist():
    away_home_team_list = list(set(
        hockey_df.loc[hockey_df.season == season]['home team'].unique().tolist() +
        hockey_df.loc[hockey_df.season == season]['away team'].unique().tolist()))
    all_options[season] = away_home_team_list

app.layout = html.Div([

    html.Div([
        dcc.RadioItems(
            list(all_options.keys()),
            list(all_options.keys())[0],
            id='season-radio',
        ),

        html.Hr(),

        dcc.RadioItems(id='team-radio'),

        html.Hr(),

        html.Div(id='display-selected-values')
    ]),
    dcc.Graph(id='indicator-graphic', figure={
        'layout': go.Layout(xaxis={'visible': False, 'showgrid': False},
                            yaxis={'visible': False, 'showgrid': False})
    }, )
])


@app.callback(
    Output('team-radio', 'options'),
    Input('season-radio', 'value'))
def set_cities_options(selected_season):
    return [{'label': i, 'value': i} for i in all_options[selected_season]]


@app.callback(
    Output('team-radio', 'value'),
    Input('team-radio', 'options'))
def set_cities_value(available_options):
    return available_options[0]['value']


@app.callback(
    Output('display-selected-values', 'children'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def set_display_children(selected_season, selected_team):
    return u'{} is a city in {}'.format(
        selected_team, selected_season,
    )


@app.callback(
    Output('indicator-graphic', 'figure'),
    Input('season-radio', 'value'),
    Input('team-radio', 'value'))
def update_graph(selected_season, selected_team):
    # Initialize figure
    fig = go.Figure()

    # Add the behind image
    im = Image.open("nhl_rink.jpg")
    rot_img = im.transpose(Image.Transpose.ROTATE_90)
    img_width, img_height = rot_img.size
    scale_factor = 0.5

    desired_coordinates = hockey_df.loc[(hockey_df.season == selected_season) &
                                        ((hockey_df['home team'] == selected_team) |
                                         (hockey_df['away team'] == selected_team))]['coordinates'].values.tolist()

    coor_x_list = []
    coor_y_list = []
    for coordinates in desired_coordinates:
        coor_x, coor_y = coordinates.replace('[', '').replace(']', '').split(',')
        coor_x_list.append(coor_x)
        coor_y_list.append(float(coor_y))

    print(coor_x_list)
    print(coor_y_list)

    # Add surface trace
    fig.add_trace(go.Heatmap(z=coor_x_list, colorscale="Viridis"))
    # print(desired_coordinates)
    print(img_height, img_width)

    fig.add_layout_image(
        dict(
            source=rot_img,
            xref="x",
            yref="y",
            x=-40,
            sizex=img_width/5.83,
            y=-9,
            sizey=img_height/5.6,
            layer="above",
            # "stretch" in the sizing will make the figure compressed
            sizing="stretch",
            opacity=0.5
        )
    )

    # Configure axes
    fig.update_yaxes(
        ticks="outside",
        visible=True,
        range=[90, -9],
        showgrid=False,
        dtick=10
    )

    fig.update_xaxes(
        ticks="outside",
        visible=True,
        range=[-40, 40],
        # scaleanchor attribute ensures that the aspect ratio stays constant
        # scaleanchor attribute makes the x-axis unaligned
        # scaleanchor="x",
        showgrid=False
    )
    # fig.add_trace(go.Scatter(x=coor_y_list, y=coor_x_list))
    # Not working in Plotly. Rotation of figure should be at image level
    # fig.update_polars(angularaxis_rotation=180)

    # # Disable the autosize on double click because it adds unwanted margins around the image
    # # More detail: https://plotly.com/python/configuration-options/
    # # fig.show(config={'doubleClick': 'reset'})
    # fig.update_yaxes(range=[90, -9], dtick=10)
    # fig.update_layout(yaxis=dict(range=[90, -9]))
    # Set templates
    fig.update_layout(template="plotly_white")

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
