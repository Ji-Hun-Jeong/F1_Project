from torch import nn

class ILossFunc:
    def __init__(self):
        pass

    def calculate_loss(self, y_predict, y_data):
        pass

    def backward(self):
        pass

    def get_loss_value(self):
        pass

class MSELoss(ILossFunc):
    def __init__(self):
        super().__init__()
        self.loss = None
        self.loss_func = nn.MSELoss()

    def calculate_loss(self, y_predict, y_data):
        self.loss = self.loss_func(y_predict, y_data)

    def backward(self):
        self.loss.backward()

    def get_loss_value(self):
        return self.loss.item()


class ILossFuncCreator:
    def create_loss_func(self):
        pass
    
class MSELossFuncCreator(ILossFuncCreator):
    def create_loss_func(self):
        return MSELoss()