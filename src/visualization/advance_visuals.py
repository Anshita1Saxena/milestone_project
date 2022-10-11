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
    img_width, img_height = im.size
    scale_factor = 0.5
    print(img_width)

    fig.add_layout_image(
        dict(
            source=Image.open("nhl_rink.jpg"),
            xref="x",
            yref="y",
            x=-400,
            sizex=img_width/2 + 2.5,
            y=img_height,
            sizey=img_height,
            layer="above",
            # "stretch" in the sizing will make the figure compressed
            sizing="fill",
            opacity=0.5
        )
    )

    # Configure axes
    fig.update_xaxes(
        ticks="outside",
        visible=True,
        range=[0, img_width],
        showgrid=False
    )

    fig.update_yaxes(
        ticks="outside",
        visible=True,
        range=[0, img_height],
        # scaleanchor attribute ensures that the aspect ratio stays constant
        # scaleanchor attribute makes the x-axis unaligned
        scaleanchor="x",
        showgrid=False
    )

    # print(hockey_df.loc[(hockey_df.season == selected_season) &
    #                     (hockey_df['home team'] == selected_team) &
    #                     (hockey_df['away team'] == selected_team)
    #                     ]['coordinates'].values.tolist())
    desired_coordinates = hockey_df.loc[(hockey_df.season == selected_season) &
                                        ((hockey_df['home team'] == selected_team) |
                                        (hockey_df['away team'] == selected_team))]['coordinates'].values.tolist()

    coor_x_list = []
    coor_y_list = []
    for coordinates in desired_coordinates:
        coor_x, coor_y = coordinates.replace('[', '').replace(']', '').split(',')
        coor_x_list.append(coor_x)
        coor_y_list.append(coor_y)
        print(coor_x, coor_y)

    fig.add_trace(go.Scatter(x=coor_x_list,
                     y=coor_y_list))
    # x = [489.99378204345703, 424.4607162475586, 665.4505157470703, 665.1176452636719]
    # y = [709.4012403488159, 253.38330745697021, 519.5582628250122, 519.5164632797241]
    # for x, y in zip(x, y):
    #     print(x, y)

    # # Add surface trace
    # fig.add_trace(
    #     go.Heatmap(z=hockey_df.loc[(hockey_df.season == selected_season) &
    #                                (hockey_df['home team'] == selected_team)]['coordinates'].values.tolist(),
    #                colorscale="Viridis"))

    # # fig.update_yaxes(autorange="reversed")
    # # Disable the autosize on double click because it adds unwanted margins around the image
    # # More detail: https://plotly.com/python/configuration-options/
    # # fig.show(config={'doubleClick': 'reset'})

    # Set templates
    fig.update_layout(template="plotly_white")

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
