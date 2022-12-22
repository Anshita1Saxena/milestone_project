# milestone_project
Hockey NHL Project (Data Science Project)

Advanced Plot Visualization Shot Map Dash Application: https://advancedvisuals.herokuapp.com/

## MILESTONE 1
#### Question Codes and Figures:

Question 1 Data Acquisition (25%):-
The builded the code `download_raw_data.ipynb` which is placed in the directory `milestone_project/src/data/`. As per suggested by TA, we didn't place the downloaded data in the data directory folder.

Question 2 Interactive Debugging Tool (10%):-
The notebook of ipywidget intractive tool is placed in `milestone_project/notebooks/ipywidgets.ipynb`

Question 3 Tidy Data (10%):-
The code is placed in `tidy_data_add.py` which is placed in `milestone_project/src/features`. The data `complex_visuals.csv` generated by this file is placed in the `milestone_project/data/processed/`.

Question 4 Simple Visualizations (25%):-
The code is placed in `simple visualizations.ipynb` along with the plots.

Question 5 Advanced Visualizations: Shot Maps (30%):-
There are two codes for this question. The data aggregation code `complex_visuals_data.py` is placed in `milestone_project/src/visualization/` and the data visualization code `advanced_visuals_plotly.py` is placed in `milestone_project/src/visualization/`. The aggregated data `complex_diff.csv` generated is placed in `milestone_project/data/processed`. The interactive plot `Milestone1_Q_6_1.html` is placed in `milestone_project/reports/figures`. Additionally, we have launched it as a website: `https://advancedvisuals.herokuapp.com/`.

#### BlogPost Image, File and figures used in the blogpost:

We have also placed an image of our blogpost: `milestone_project/reports/Blogpost Image/Blogpost Image.png`
Our images attached in blogpost are placed in `milestone_project/blogpost_milestone_1/assets/images/`
Our blogpost .md file `2022-10-16-Milestone1.md` is placed in `milestone_project/blogpost_milestone_1/_posts/`


## MILESTONE 2
#### Question Codes and Figures:

Question 1:
Comet-ml was setup and the link for this comet-ml is available here: `https://www.comet.com/anshitasaxena/milestone-project-2/view/new/experiments`

Question 2 (10%):-
The code is present under `src/features` directory, namely `milestone2_feature_engineering.py` and `milestone2_tidy_data.py`. Images are placed under the blogpost `Assets` directory. And visualization kept in `src/visualization` in the code `milestone2_visualization_feature_engineering_I.py`.

Question 3 (15%):-
The code is present under the `src/model` directory, namely `basemodel_logreg.py` and it's notebook is present under `notebook` directory namely `basemodel_logreg.ipynb`.

Question 4 (15% + bonus 5%):-
The code is present under `src/features` directory, namely `milestone2_feature_engineering.py` and `milestone2_tidy_data.py`. Images are placed under the blogpost `Assets` directory. 

Question 5 (20%):-
The code is present under `src/model` directory, namely `xgboostcometfinal.py` and it's notebook is present under `notebook` directory namely `xgboostcometfinal.ipynb`.

Question 6 (20%):-
The code is present under `src/model` directory, namely `final_best_shot_try.py` and it's notebook is present under `notebook` directory namely `final_best_shot_try.ipynb`.

Question 7 (10%):-
Plots are placed under `asset` directory.

Figures and plots are kept under `asset` directory of the blogpost. Under comet, we kept are the experiments, models, plots, and confusion matrix along with other matrices.

#### BlogPost Image, File and figures used in the blogpost:

Our images attached in blogpost are placed in `milestone_project/blogpost_milestone_1_2/assets/images/`
Our blogpost .md file `2022-11-09-Milestone2.md` is placed in `milestone_project/blogpost_milestone_1_2/_posts/`

## MILESTONE 2
#### Question Codes and Figures:

Question 1: (30%) Modeling Serving Flask App:-
The code is present inside `milestone_project/docker-project_milestone_3` with filename `app.py`. It writes the logs about model state into `flask.log`. We have used waitress for this flask application.
This will run and test for models- `base-xgboost`, `feature-selection-xgboost`, and `tuned-xgboost` as written in question 1. It can also support other models too.


Question 2: (10%) Clients:-
The code is kept inside `milestone_project/docker-project_milestone_3/ift6758/ift6758/client` with filenames as `ServingClient.py`, `GameClient.py`, with other supporting files.


Question 3: (15%) Docker part 1 - Serving:-
We have `.env` file inside which we kept COMET_API_KEY. We have used Docker Desktop in windows to setup the docker containers. The files included in this tasks are `docker-compose.yaml`, `Dockerfile.serving`, `runj.sh`, `run.sh`, and `build.sh`. 
Images are displayed in last question 5.


Question 4: (30%) Streamlit App:-
We have implemented the number of records/newly updated records or batch in this part using streamlit `session_state`. 
It works in a way that for each game id it will every time take newly updated records, however, if there are no new records, it will show us the previous computed results. For a new different game id, we have to download the model again and then generate the prediction for this new game id because the model is using One hot encoding which will create categorical columns based on the values of the game id. Hence, each game id can vary in terms of categorical columns based on the values they have in the column.
The code is kept in `milestone_project/docker-project_milestone_3/streamlit_app.py`.
Below image shows the stream app.
![Streamlit App](https://github.com/Anshita1Saxena/milestone_project/blob/main/docker-project_milestone_3/figures/streamlit_app.png)


Question 5: (15%) Docker part 2 - Streamlit:-
The Streamlit app is containerized into docker containers. Some of the information of this question is written in Question 3. The files included in this tasks are `docker-compose.yaml`, `Dockerfile.streamlit`, and `.env` file in which we kept COMET_API_KEY.
Below are the images showing docker containers, images, network, docker desktop, and docker compose up command:
Docker Containers, Images, Network
![Docker Containers Status](https://github.com/Anshita1Saxena/milestone_project/blob/main/docker-project_milestone_3/figures/docker_containers_status.png)

Docker Compose Command
![Docker Compose UP](https://github.com/Anshita1Saxena/milestone_project/blob/main/docker-project_milestone_3/figures/docker_compose_up_1.png)

![Docker Compose UP](https://github.com/Anshita1Saxena/milestone_project/blob/main/docker-project_milestone_3/figures/docker_compose_up_2.png)

Docker Desktop
![Docker Desktop Image](https://github.com/Anshita1Saxena/milestone_project/blob/main/docker-project_milestone_3/figures/docker_desktop_images.png)
