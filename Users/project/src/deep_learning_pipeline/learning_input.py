import pandas as pd

class LearningInput:
    def __init__(self, features: list, labels: list):
        self.features: list = features
        self.labels: list = labels
        
    def get_data_as_data_frame_do_not_has_none(self):
        return pd.DataFrame(self.features), pd.DataFrame(self.labels)

    

    