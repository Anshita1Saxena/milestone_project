import streamlit as st
import pandas as pd
from serving import ServingClient
from serving import GameClient
import json
import traceback

# Setting the app title
st.title("Hockey Visualization App")
sc = ServingClient.ServingClient(ip="127.0.0.1", port=5000)
gc = GameClient.GameClient()

with st.sidebar:
    # This sidebar uses Serving Client for the model Download
    workspace = st.text_input('Workspace', 'Workspace x')
    model = st.text_input('Model', 'Model y')
    version = st.text_input('Version', 'Version z')
    if st.button('Get Model'):
        print(workspace)
        print(model)
        print(version)
        sc.download_registry_model(
            workspace=workspace,
            model=model,
            version=version
        )
        st.write('Model Downloaded')


def ping_game_id(game_id):
    with st.container():
        # Initialization of session variables to track the dataframe length
        # st.session_state preserves the state of the variables between different reruns
        # session_state is the key functionality in streamlit app
        if 'session_tracker' not in st.session_state and 'previous_session_tracker' not in st.session_state:
            st.session_state['session_tracker'] = 0
            st.session_state['previous_session_tracker'] = 0
            # st.write(st.session_state.session_tracker)
            # st.write(st.session_state.previous_session_tracker)
        try:
            # Get the filepath of the recent game_id downloaded json
            filepath = gc.get_game(game_id=game_id)
            # Get the dataframe, last event, and dataframe_length
            model_df, last_event_df, new_dataframe_length = gc.ping_game(filepath)
            # If the session ran already, then the session_tracker should update with
            # new length that is newly loaded dataframe length - old evaluated dataframe length
            if 'session_tracker' in st.session_state and st.session_state.session_tracker > 0:
                # st.write(st.session_state.session_tracker)
                st.session_state['session_tracker'] = new_dataframe_length - st.session_state.session_tracker
                # st.write(st.session_state.session_tracker)
            # If there is zero event for this session the session_tracker will become 0
            # but previous session tracker holds the result of last dataframe length value
            if st.session_state.session_tracker == 0 and st.session_state.previous_session_tracker > 0:
                st.write("Displaying same dashboard as previous because there are " + str(st.session_state.session_tracker) + " new events!")
                st.session_state['previous_session_tracker'] = new_dataframe_length - st.session_state.previous_session_tracker
            # If current session and previous session have dataframe length then it has new events
            if st.session_state.session_tracker > 0 and st.session_state.previous_session_tracker > 0:
                st.write("There are total " + str(st.session_state.session_tracker) + " new events!")
                st.session_state['previous_session_tracker'] = new_dataframe_length - st.session_state.previous_session_tracker
            # If both session and previous session do not have the events then it load is first time for game id
            if st.session_state.session_tracker == 0 and st.session_state.previous_session_tracker == 0:
                st.session_state['previous_session_tracker'] = new_dataframe_length
                st.session_state['session_tracker'] = new_dataframe_length
            # If both current session and previous session have values
            # then we should get the dataframe with only new values
            if st.session_state.session_tracker > 0 and st.session_state.previous_session_tracker > 0:
                model_df = model_df.copy()
                model_df = model_df.iloc[-st.session_state.session_tracker:]
                st.session_state.session_tracker = new_dataframe_length
                st.session_state.previous_session_tracker = new_dataframe_length
            # Current and Previous session values will be equal once we will reach at the state of predicting the model
            # because we set both session variables as same values
            # so that in next run previous session will give us the value of this run
            if st.session_state.session_tracker == st.session_state.previous_session_tracker:
                preds = sc.predict(model_df)
                preds = json.loads(preds)
                preds = [pred for pred in preds['0'].values()]
                model_df["Model Output"] = preds
                model_df.is_goal = model_df.is_goal.astype(str)
                # Calculating the predict and actual goals probability
                grouped_prob_df = model_df.where(model_df.is_goal == "True").groupby("team_shot")["Model Output"].agg("sum").round(decimals=2)
                grouped_goal_df = model_df.where(model_df.is_goal == "True").groupby("team_shot")["is_goal"].count()
                grouped_prob_df = pd.DataFrame(grouped_prob_df).transpose()
                grouped_goal_df = pd.DataFrame(grouped_goal_df).transpose()
                last_event_df = pd.DataFrame(last_event_df).transpose()
                last_event_df = last_event_df.reset_index()
                # Calculating the time left in period
                time_remain_df = str(pd.DataFrame(pd.to_datetime("20:00", format="%M:%S")
                             - pd.to_datetime(last_event_df["period time"], format="%M:%S"))['period time'][0])
                print(time_remain_df.split("days")[-1])
                st.subheader("Game " + str(game_id) + ": " + str(model_df.home_team[0]) + " vs " + str(model_df.away_team[0]))
                st.text("Period " + str(last_event_df.period[0]) + " - "
                        + str(":".join(time_remain_df.split("days")[-1].split(":")[1:])) + " left")
                # Arranging the values in two columns
                col1, col2 = st.columns(2)
                col1.metric(label=str(model_df.home_team[0]) + " xG(actual)", value=str(grouped_prob_df[model_df.home_team[0]][0])+" ("
                                                                                  + str(grouped_goal_df[model_df.home_team[0]][0])
                                                                                  + ")",
                          delta=str(float(float(grouped_goal_df[model_df.home_team[0]][0])-float(
                              grouped_prob_df[model_df.home_team[0]][0])).__round__(2)))
                col2.metric(label=str(model_df.away_team[0]) + " xG(actual)",
                          value=str(grouped_prob_df[model_df.away_team[0]][0]) + " ("
                                + str(grouped_goal_df[model_df.away_team[0]][0])
                                + ")",
                          delta=str(float(float(grouped_goal_df[model_df.away_team[0]][0]) - float(
                              grouped_prob_df[model_df.away_team[0]][0])).__round__(2)))

                with st.container():
                    # Fetching the entire dataframe
                    st.subheader("Data used for predictions (and predictions)")
                    st.table(model_df)
        except Exception as e:
            st.write("Please turn on your prediction service.")
            print(e)
            print(traceback.format_exc())
            st.session_state['session_tracker'] = 0
            st.session_state['previous_session_tracker'] = 0


with st.container():
    # This is the ping game container consists of the Game ID and button
    game_id = st.text_input('Game ID', '2021020329')
    if st.button('Ping game'):
        ping_game_id(game_id)
