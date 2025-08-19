import numpy as np
import json
import joblib

def convert_numpy_types(obj):
    """
    딕셔너리, 리스트를 순회하며 NumPy 타입을 Python 기본 타입으로 변환하는 함수
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def save_json(data, file_path):
    """JSON 파일을 저장하는 헬퍼 함수"""
    serializable_data = convert_numpy_types(data)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    print(f"✅ JSON 파일이 '{file_path}'에 저장되었습니다.")

def save_model(model, file_path):
    """Joblib 또는 XGBoost 모델을 저장하는 헬퍼 함수"""
    if hasattr(model, 'save_model'): # XGBoost 모델인 경우
        model.save_model(file_path)
    else: # Scikit-learn 객체인 경우
        joblib.dump(model, file_path)
    print(f"✅ 모델/객체가 '{file_path}'에 저장되었습니다.")