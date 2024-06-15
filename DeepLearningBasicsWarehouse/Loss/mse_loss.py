import torch
import torch.nn as nn


# 底层实现的均方误差损失函数
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        
    def forward(self, predicts, targets):
        # 计算均方误差损失
        loss = torch.mean((predicts - targets) ** 2)
        return loss


# 测试
if __name__ == '__main__':
    # 创建模型
    loss_func = MSELoss()
    
    # 创建输入数据
    predicts = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 2.0, 2.0], dtype=torch.float32)

    loss = loss_func(predicts, targets)
    print(loss)