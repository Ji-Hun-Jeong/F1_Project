import pandas as pd


class LapDataCollector:
    def __init__(self) -> None:
        self.data_list = []
        
    def add_car_data(self, data: pd.DataFrame):
        self.data_list.append(data)