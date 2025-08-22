import asyncio
import time
from fastapi import APIRouter, WebSocket
from fastapi.responses import JSONResponse
import pandas as pd
from pandas import DataFrame
import json
import os
import torch
import numpy as np
from random import Random
from fastapi import Request
from Users.project.src.base.logger import ConsoleLogger, SaveLogger
from Users.project.src.data_container.data_container import AzureStorageAccess
from Users.project.src.deep_learning_pipeline.context import Context
from Users.project.src.deep_learning_pipeline.model_creator import ModelManager
from Users.project.src.deep_learning_pipeline.scaler_manager import ScalerManager
from Users.project.src.predict_lab_time_module.all_track_model import SimpleModel_64_Hidden_2_Creator
from Users.project.src.predict_lab_time_module.lap_data_collector import LapDataCollector, arrange_feature_and_label_has_nan
from Users.project.src.data_container.data_container import AzureStorageAccess


router = APIRouter()
data_access = AzureStorageAccess()

best_model_path = os.path.join("Users", "project", "model", "lap_time_predict", "best_checkpoint.pth")
scaler_save_path = os.path.join("Users", "project", "model", "lap_time_predict", "best_scaler.joblib")
model_manager = ModelManager(best_model_path)
scaler_manager = ScalerManager(scaler_save_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
context = Context(device, model_manager, scaler_manager)

lap_data_collector = LapDataCollector()
model_creator = SimpleModel_64_Hidden_2_Creator()
logger = SaveLogger()

@router.post("/predict_lap_time")
async def predict_lap_time(request: Request):
    predict_data = await request.json()

    car_data = pd.DataFrame([predict_data["CarData"]])
    lap_data_collector.add_car_data(car_data)
    if predict_data["IsLapChange"] == True:
        try:
            laps_data = pd.DataFrame([predict_data["LapData"]]).iloc[0]
            weather_data = pd.DataFrame([predict_data["WeatherData"]])
            feature = lap_data_collector.get_feature_by_data_frame(laps_data, weather_data)
            label = pd.DataFrame([predict_data["LapTime"]])
            feature = feature.fillna(0)

            context.test_one_lap(feature, label, model_creator, logger)

            log = logger.get_str()
            logger.clear_log()
        except Exception as e:
            print(e)
            return JSONResponse({"success": False, "error": str(e)})
        return JSONResponse({"success": False, "result": log})
    else:
        return JSONResponse({"success": True})
    
@router.post("/choice_random_lap_time_predict")
async def choice_random_lap_time_predict():
    correct_game_file_path = os.path.join("Users", "project", "correct_file", "correct_game")
    game_list = []
    with open(correct_game_file_path, "r", encoding="utf-8") as file:
        for f in file.readlines():
            game_list.append(f)
    rand = Random()
    game = rand.choice(game_list)
    game = game.replace("\n", "")
    data_access = AzureStorageAccess()

    car_data_file = game + "car_data_all.csv"
    laps_file = game + "laps.csv"
    weather_file = game[0 : -4] + "weather_data.csv"

    car_data = data_access.get_file_by_data_frame(car_data_file)
    laps_data = data_access.get_file_by_data_frame(laps_file)
    weather_data = data_access.get_file_by_data_frame(weather_file)

    feature, label = arrange_feature_and_label_has_nan(car_data, laps_data, weather_data)

    context.test_df(feature, label, model_creator, logger)
    log = logger.get_str()
    logger.clear_log()
    return JSONResponse({"success": True, "result": log})

















def sanitize_json(obj):
    return {
        k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in obj.items()
    }

@router.websocket("/check_file")
async def check_file(web_socket: WebSocket):
    await web_socket.accept()
    data = await web_socket.receive_json()
    Year = data.get("Year")
    Track = data.get("Track")
    Session = data.get("Session")
    Team = data.get("Team")
    FileName = data.get("FileName")

    if Year == 2022:
        file_pre = f"{Year}/{Year}_{Track}_{Session}/{Team}/"
    else:
        file_pre = f"{Year}/{Year}_{Track}_Grand_Prix_{Session}/{Team}/"

    file_pre_token = file_pre.split("/")

    car_data_all = data_access.read_csv_by_data_frame(file_pre + "car_data_all.csv")
    laps = data_access.read_csv_by_data_frame(file_pre + "laps.csv")
    position_data = data_access.read_csv_by_data_frame(file_pre + "position_data.csv")
    weather_data = data_access.read_csv_by_data_frame(
        f"{file_pre_token[0]}/{file_pre_token[1]}/" + "weather_data.csv"
    )

    if (
        len(car_data_all) == 0
        or len(laps) == 0
        or len(position_data) == 0
        or len(weather_data) == 0
    ):
        await web_socket.send_json({"Result": False})
    else:
        await web_socket.send_json(
            {
                "Result": True,
                "Year": Year,
                "Track": Track,
                "Session": Session,
                "Team": Team,
            }
        )
    await web_socket.close()


@router.websocket("/request_data_sequential")
async def request_car_data(web_socket: WebSocket):
    await web_socket.accept()
    try:
        data = await web_socket.receive_json()
        Year = data.get("Year")
        Track = data.get("Track")
        Session = data.get("Session")
        Team = data.get("Team")
        FileName = data.get("FileName")
        ReactionTime = data.get("ReactionTime")

        if Year == 2022:
            file_pre = f"{Year}/{Year}_{Track}_{Session}/{Team}/"
        else:
            file_pre = f"{Year}/{Year}_{Track}_Grand_Prix_{Session}/{Team}/"

        if FileName == "weather_data.csv":
            file_token = file_pre.split("/")
            file_pre = f"{file_token[0]}/{file_token[1]}/"

        file_name = file_pre + FileName

        data_frame = data_access.read_csv_by_data_frame(file_name)

        for _, row in data_frame.iterrows():
            sanitized = sanitize_json(row.to_dict())
            await web_socket.send_json(sanitized)
            await asyncio.sleep(ReactionTime)

    except Exception as e:
        print({"error": str(e)})

    finally:
        await web_socket.close()

@router.websocket("/request_data_once")
async def request_car_data(web_socket: WebSocket):
    await web_socket.accept()
    try:
        data = await web_socket.receive_json()
        Year = data.get("Year")
        Track = data.get("Track")
        Session = data.get("Session")
        Team = data.get("Team")
        FileName = data.get("FileName")
        if Year == 2022:
            file_pre = f"{Year}/{Year}_{Track}_{Session}/{Team}/"
        else:
            file_pre = f"{Year}/{Year}_{Track}_Grand_Prix_{Session}/{Team}/"

        if FileName == "weather_data.csv":
            file_token = file_pre.split("/")
            file_pre = f"{file_token[0]}/{file_token[1]}/"

        file_name = file_pre + FileName

        data_frame = data_access.read_csv_by_data_frame(file_name)
        all_data = data_frame.to_dict(orient='records')
        sanitized = [sanitize_json(row) for row in all_data]
        await web_socket.send_json(sanitized)
    except Exception as e:
        print(e)
    finally:
        await web_socket.close()