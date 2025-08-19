import asyncio
import random
import time
from fastapi import APIRouter, WebSocket
import pandas as pd
from pandas import DataFrame
import json
import os
import numpy as np

from Users.project.src.data_container.data_container import AzureStorageAccess

router = APIRouter()
data_access = AzureStorageAccess()

@router.post("/predict_lap_time")
async def predict_lap_time(predict_data: dict):
    print(predict_data)
    return {"message": predict_data}

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


@router.websocket("/get_random_format")
async def response(web_socket: WebSocket):
    await web_socket.accept()
    await web_socket.receive_text()
    base_dir = os.path.dirname(__file__)
    json_path = os.path.join(base_dir, "..", "file_format", "format.json")
    # JSON 파일 열기
    with open(json_path, "r", encoding="utf-8") as file:
        json_data = json.load(file)

        while True:
            Year = random.choice(json_data["Year"])
            Track = random.choice(json_data["Track"])
            Session = random.choice(json_data["Session"])
            Team = random.choice(json_data["Team"])

            if Year == 2022:
                file_pre = f"{Year}/{Year}_{Track}_{Session}/{Team}/"
            else:
                file_pre = f"{Year}/{Year}_{Track}_Grand_Prix_{Session}/{Team}/"

            file_pre_token = file_pre.split("/")

            car_data_all = data_access.read_csv_by_data_frame(
                file_pre + "car_data_all.csv"
            )
            laps = data_access.read_csv_by_data_frame(file_pre + "laps.csv")
            position_data = data_access.read_csv_by_data_frame(
                file_pre + "position_data.csv"
            )
            weather_data = data_access.read_csv_by_data_frame(
                f"{file_pre_token[0]}/{file_pre_token[1]}/" + "weather_data.csv"
            )
            if (
                0 < len(car_data_all)
                and 0 < len(laps)
                and 0 < len(position_data)
                and 0 < len(weather_data)
            ):
                break

        await web_socket.send_json(
            {"Year": Year, "Track": Track, "Session": Session, "Team": Team}
        )

    await web_socket.close()
