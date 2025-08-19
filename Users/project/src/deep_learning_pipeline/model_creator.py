import torch
from torch import nn

class IModelCreator:
    def create_model(self, in_dim: int) -> nn.Module:
        pass
    
    def get_save_name(self) -> str:
        pass
    
class ModelManager:
    def __init__(self, model_save_path: str):
        self.model_save_path = model_save_path
    
    def save_model(self, model_state_info_dict: dict):
        # 딕셔너리의 포맷을 구조체로 정해주고 그걸 받아야 할것같은데 그냥 일단 이렇게 하자
        torch.save(model_state_info_dict, self.model_save_path)
        
    def set_model_state(self, model: nn.Module, device: torch.device):
        model_data = torch.load(self.model_save_path, map_location=device)
        model.load_state_dict(model_data['model_state_dict'])