from Users.project.src.base.logger import ILogger
from Users.project.src.deep_learning_pipeline.learning_input import LearningInput
from Users.project.src.deep_learning_pipeline.loss_func import ILossFunc, ILossFuncCreator
from Users.project.src.deep_learning_pipeline.model_creator import IModelCreator, ModelManager
from Users.project.src.deep_learning_pipeline.optimizer import IOptimizer, IOptimizerCreator
from sklearn.base import BaseEstimator, TransformerMixin # TransformerMixin도 함께 알아두시면 좋습니다.
from sklearn.model_selection import train_test_split
from Users.project.src.deep_learning_pipeline.scaler_manager import ScalerManager
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import pandas as pd
import numpy as np
import os

class Context:
    def __init__(self, device: torch.device
                , model_manager: ModelManager
                , scaler_manager: ScalerManager):
        self.device = device
        self.model_manager = model_manager
        self.scaler_manager = scaler_manager
                
    def train(self, learning_input: LearningInput
                  , data_scaler: TransformerMixin
                , model_creator: IModelCreator
                , optimizer_creator: IOptimizerCreator
                , loss_func_creator: ILossFuncCreator
                , logger: ILogger
                , test_size: float, random_state: int
                , learning_rate: float, num_epochs: int):
        features, labels = learning_input.get_data_as_data_frame_do_not_has_none()
        
        # y = ax + b <- 이런 느낌 x가 입력 y가 출력
        x_train, x_test, y_train, y_test = train_test_split(
            features.values, labels.values, test_size=test_size, random_state=random_state)
        
        # 학습된 데이터로 스케일링 기준 정의
        data_scaler.fit(x_train);
        x_train, x_test = data_scaler.transform(x_train), data_scaler.transform(x_test)
        
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).to(self.device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1).to(self.device)
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # --- 1. TensorDataset과 DataLoader 생성 ---
        # 먼저 학습 데이터와 검증(테스트) 데이터를 TensorDataset으로 묶습니다.
        train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
        
        # 배치 크기 설정
        batch_size = 64
        
        # DataLoader를 생성합니다.
        # 학습용 로더는 매 에폭마다 데이터를 섞어주는 것이 좋습니다 (shuffle=True)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        # 검증용 로더는 섞을 필요가 없습니다.
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

        in_dim = features.shape[1]
        
        model: nn.Module = model_creator.create_model(in_dim).to(self.device)
        optimizer: IOptimizer = optimizer_creator.create_optimizer(model, learning_rate)
        loss_func: ILossFunc = loss_func_creator.create_loss_func()
        
        # 이 변수에 가장 좋았던 모델의 상태를 저장할 것입니다.
        best_val_loss = float('inf')
        best_model_state_dict = None
        best_optimizer_state_dict = None
        best_epoch = 0
        patience = 50  # 50번의 에폭 동안 성능 개선이 없으면 중단
        patience_counter = 0
        for epoch in range(num_epochs):
            # --- 학습 모드 ---
            model.train()
            train_loss = 0.0

            # DataLoader를 사용하여 미니배치를 하나씩 꺼내옵니다.
            for x_batch, y_batch in train_loader:
                # 데이터를 현재 device로 이동
                # x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)

                # 순전파
                outputs = model(x_batch)
                loss_func.calculate_loss(outputs, y_batch)

                # 역전파 및 최적화
                optimizer.zero_grad()
                loss_func.backward()
                optimizer.step()

                train_loss += loss_func.get_loss_value() # 배치 손실을 누적

            # --- 검증 모드 ---
            model.eval()
            validation_loss = 0.0
            with torch.no_grad():
                # 검증 데이터도 배치 단위로 처리
                for x_test_batch, y_test_batch in test_loader:
                    # x_test_batch, y_test_batch = x_test_batch.to(self.device), y_test_batch.to(self.device)

                    test_outputs = model(x_test_batch)
                    loss_func.calculate_loss(test_outputs, y_test_batch)
                    validation_loss += loss_func.get_loss_value() # 배치 손실을 누적

            # 에폭의 평균 손실 계산
            avg_train_loss = train_loss / len(train_loader)
            avg_validation_loss = validation_loss / len(test_loader)

            # 에폭 결과 출력
            if (epoch + 1) % 10 == 0:
                logger.println(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_validation_loss:.4f}')

            # 현재 에포크의 검증 손실이 기존의 최저 손실보다 낮으면
            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                best_epoch = epoch + 1
                # 현재 모델의 상태를 딕셔너리로 저장합니다.
                best_model_state_dict = model.state_dict().copy()
                best_optimizer_state_dict = optimizer.state_dict().copy()
                # 인내심 카운터 리셋
                patience_counter = 0
            else:
                # 성능 개선이 없었음. 인내심 카운터 증가
                patience_counter += 1
                
            if patience_counter >= patience:
                logger.println(f"\nEarly stopping triggered at epoch {epoch + 1} as validation loss did not improve for {patience} epochs.")
                break
        
        # 훈련이 모두 끝난 후, 가장 좋았던 모델을 별도로 저장합니다.
        if best_model_state_dict != None:
            logger.println(f"\nTraining finished. Best model from Epoch {best_epoch} with Validation Loss {best_val_loss:.4f} is being saved.")
            self.model_manager.save_model({
                'epoch': best_epoch,
                'model_state_dict': best_model_state_dict,  # 나중에 모델 불러올 때
                'optimizer_state_dict': best_optimizer_state_dict,  # 나중에 추가적인 학습을 진행할 경우
                'val_loss': best_val_loss,
                'in_dim': in_dim
            })
        else:
            logger.println("\nNo best model found. This should not happen.")
            
        self.scaler_manager.save_scaler(data_scaler)
        

    def test(self, learning_input: LearningInput
                  , model_creator: IModelCreator
                  , logger: ILogger
                  , test_size: float, random_state: int):
        features, labels = learning_input.get_data_as_data_frame_do_not_has_none()
        
        x_train, x_test, y_train, y_test = train_test_split(
            features.values, labels.values, test_size=test_size, random_state=random_state, shuffle=False)
        
        data_scaler = self.scaler_manager.load_scaler()
        x_test = data_scaler.transform(x_test)
                
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
        
        in_dim = features.shape[1]
        
        model = model_creator.create_model(in_dim).to(self.device)
        self.model_manager.set_model_state(model, self.device)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test_tensor)
            
        predictions_array = val_outputs.cpu().numpy()
        y_predict_array = y_test_tensor.cpu().numpy()

        # 예측값과 실제값의 차이 계산
        # 절대값 차이를 계산하여 음수 값이 없도록 합니다.
        difference_array = np.abs(predictions_array - y_predict_array)
        # 예측값, 실제값, 차이를 하나의 배열로 합쳐서 반환
        max = 0
        idx = 0
        count_05 = 0
        count_025 = 0
        count_01 = 0
        for i in range(predictions_array.shape[0]):
            pred = predictions_array[i][0]
            actual = y_predict_array[i][0]
            diff = difference_array[i][0]
            logger.println(f"행 {i+1}: 예측값={pred:.4f}, 실제값={actual:.4f}, 차이={diff:.4f}\n")
            if max < diff:
                max = diff
                idx = i
            if 0.5 < diff:
                count_05 += 1
            if 0.25 < diff:
                count_025 += 1
            if 0.1 < diff:
                count_01 += 1
        logger.println(f"행: {idx+1}, 최댓값: {max}, 0.5이상: {count_05}, 0.25이상: {count_025}, 0.1이상: {count_01}")
        logger.println(str(model))
        
        
    def test_df(self, features: pd.DataFrame, labels: pd.DataFrame
                  , model_creator: IModelCreator
                  , logger: ILogger):
        x_test = features.values
        y_test = labels.values
        data_scaler = self.scaler_manager.load_scaler()
        x_test = data_scaler.transform(x_test)
                
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(self.device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1).to(self.device)
        
        in_dim = features.shape[1]
        
        model = model_creator.create_model(in_dim).to(self.device)
        self.model_manager.set_model_state(model, self.device)
        
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test_tensor)
            
        predictions_array = val_outputs.cpu().numpy()
        y_predict_array = y_test_tensor.cpu().numpy()

        # 예측값과 실제값의 차이 계산
        # 절대값 차이를 계산하여 음수 값이 없도록 합니다.
        difference_array = np.abs(predictions_array - y_predict_array)
        # 예측값, 실제값, 차이를 하나의 배열로 합쳐서 반환
        max = 0
        idx = 0
        count_05 = 0
        count_025 = 0
        count_01 = 0
        for i in range(predictions_array.shape[0]):
            pred = predictions_array[i][0]
            actual = y_predict_array[i][0]
            diff = difference_array[i][0]
            logger.println(f"행 {i+1}: 예측값={pred:.4f}, 실제값={actual:.4f}, 차이={diff:.4f}\n")
            if max < diff:
                max = diff
                idx = i
            if 0.5 < diff:
                count_05 += 1
            if 0.25 < diff:
                count_025 += 1
            if 0.1 < diff:
                count_01 += 1
        logger.println(f"행: {idx+1}, 최댓값: {max}, 0.5이상: {count_05}, 0.25이상: {count_025}, 0.1이상: {count_01}")