import json
import os
import urllib.request
import sys

from urllib.error import HTTPError


def download_raw_data(year) -> None:
    print('Starting download of Regular Season Data:')
    regular_path_dir = os.path.join('\\'.join(str(os.getcwd()).split('\\')[:-2]), 'data', 'raw', 'regular_season')
    doc_limit = 0
    if year <= '2016':
        # 2016-17 season game with only 1230 games
        doc_limit = 1231
    elif year >= '2017':
        # 2017 onwards season game with 1270 games
        doc_limit = 1271
    if year:
        for i in range(1, doc_limit):
            i = appending_0(i)
            game_id = year + '02' + str(i)
            regular_path = os.path.join(regular_path_dir, year, game_id + '.json')
            if not os.path.exists(regular_path):
                regular_dir = os.path.join(regular_path_dir, year)
                if not os.path.exists(regular_dir):
                    os.makedirs(regular_dir)
                download_data(regular_path, game_id)

    print('Starting download of Playoffs Data:')
    playoff_path_dir = os.path.join('\\'.join(str(os.getcwd()).split('\\')[:-2]), 'data', 'raw', 'playoffs')
    game_id = year + '030'
    for digit in range(0, 777):
        if len(str(digit)) == 1:
            digit = '00' + str(digit)
        elif len(str(digit)) == 2:
            digit = '0' + str(digit)
        exact_game_id = game_id + str(digit)
        playoff_path = os.path.join(playoff_path_dir, year, exact_game_id + '.json')
        if not os.path.exists(playoff_path):
            playoff_dir = os.path.join(playoff_path_dir, year)
            if not os.path.exists(playoff_dir):
                os.makedirs(playoff_dir)
            download_data(playoff_path, exact_game_id)


def appending_0(i) -> str:
    if len(str(i)) == 1:
        i = '000' + str(i)
    if len(str(i)) == 2:
        i = '00' + str(i)
    if len(str(i)) == 3:
        i = '0' + str(i)
    return i


def download_data(path, game_id) -> None:
    try:
        with urllib.request.urlopen("https://statsapi.web.nhl.com/api/v1/game/" + game_id + "/feed/live/") as url:
            data = json.load(url)
            if "messageNumber" in data and "message" in data \
                and data["messageNumber"] == 2 and data["message"] == "Game data couldn't be found":
                pass
            else:
                with open(path, 'w') as outfile:
                    json.dump(data, outfile)
    except HTTPError as he:
        print(game_id)
        print(he.reason)
    except Exception:
        print(game_id)
        e_type, e_value, e_traceback = sys.exc_info()
        print(e_value)


if __name__ == "__main__":
    print('I am working')
    years = ['2016', '2017', '2018', '2019', '2020']
    for year in years:
        download_raw_data(year)
