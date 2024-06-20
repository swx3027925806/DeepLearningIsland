import torch
import torch.nn as nn

# 定义多层感知机模型
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二个全连接层

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# 实例化模型
input_size = 784  # 假设输入是28x28的图像，共784个像素
hidden_size = 128
output_size = 10  # 假设有10个类别
model = MLP(input_size, hidden_size, output_size)

# 创建随机输入数据
x = torch.randn(1, input_size)

# 前向传播
output = model(x)