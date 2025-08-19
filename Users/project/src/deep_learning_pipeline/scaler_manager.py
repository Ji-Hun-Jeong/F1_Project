from sklearn.base import BaseEstimator, TransformerMixin # TransformerMixin도 함께 알아두시면 좋습니다.
import joblib

class ScalerManager:
    def __init__(self, save_path: str):
        self.save_path = save_path
        
    def save_scaler(self, scaler: TransformerMixin):
        joblib.dump(scaler, self.save_path)
        
    def load_scaler(self) -> TransformerMixin:
        return joblib.load(self.save_path)