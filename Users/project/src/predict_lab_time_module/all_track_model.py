from Users.project.src.deep_learning_pipeline.model_creator import IModelCreator
import torch
from torch import nn

class BatchNormalModel(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        
        # 과적합 완화를 위해 레이어 수를 줄이거나
        # 드롭아웃 레이어를 추가하는 것을 고려할 수 있습니다.
        # 여러 패턴을 학습하기 위해 레이어를 쌓는것
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.BatchNorm1d(64),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.BatchNorm1d(16),  # 배치 정규화 추가
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class BatchNormalModelCreator(IModelCreator):
    def create_model(self, in_dim: int) -> nn.Module:
        return BatchNormalModel(in_dim)
    
    def get_save_name(self) -> str:
        return "batch_normal_model_64_"
    
class SimpleModel_128_Hidden_1(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        
        # 과적합 완화를 위해 레이어 수를 줄이거나
        # 드롭아웃 레이어를 추가하는 것을 고려할 수 있습니다.
        # 여러 패턴을 학습하기 위해 레이어를 쌓는것
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleModel_128_Hidden_1_Creator(IModelCreator):
    def create_model(self, in_dim: int) -> nn.Module:
        return SimpleModel_128_Hidden_1(in_dim)
    
    def get_save_name(self) -> str:
        return "simple_model_128_hidden_1_"
    
class SimpleModel_128_Hidden_2(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        
        # 과적합 완화를 위해 레이어 수를 줄이거나
        # 드롭아웃 레이어를 추가하는 것을 고려할 수 있습니다.
        # 여러 패턴을 학습하기 위해 레이어를 쌓는것
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),
            
            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleModel_128_Hidden_2_Creator(IModelCreator):
    def create_model(self, in_dim: int) -> nn.Module:
        return SimpleModel_128_Hidden_2(in_dim)
    
    def get_save_name(self) -> str:
        return "simple_model_128_hidden_2_"
    
class SimpleModel_64_Hidden_2(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        
        # 과적합 완화를 위해 레이어 수를 줄이거나
        # 드롭아웃 레이어를 추가하는 것을 고려할 수 있습니다.
        # 여러 패턴을 학습하기 위해 레이어를 쌓는것
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleModel_64_Hidden_2_Creator(IModelCreator):
    def create_model(self, in_dim: int) -> nn.Module:
        return SimpleModel_64_Hidden_2(in_dim)
    
    def get_save_name(self) -> str:
        return "simple_model_64_hidden_2_"
    
class SimpleModel_64_Hidden_1(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        
        # 과적합 완화를 위해 레이어 수를 줄이거나
        # 드롭아웃 레이어를 추가하는 것을 고려할 수 있습니다.
        # 여러 패턴을 학습하기 위해 레이어를 쌓는것
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),

            nn.Linear(64, 16),
            nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class SimpleModel_64_Hidden_1_Creator(IModelCreator):
    def create_model(self, in_dim: int) -> nn.Module:
        return SimpleModel_64_Hidden_1(in_dim)
    
    def get_save_name(self) -> str:
        return "simple_model_64_hidden_1_"