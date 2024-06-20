import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchsummary import summary


# 定义多层感知机模型
# Acc: 0.9698
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 第一个全连接层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 第二个全连接层

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 定义一个简单的CNN模型
# 0.9797
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一个卷积层，输入通道数为1（灰度图像），输出通道数为10，卷积核大小为3x3
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=3)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 最大池化层，窗口大小为2x2
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 全连接层，输入特征数为10*13*13（假设经过卷积和池化后的尺寸），输出特征数为10（MNIST有10个类别）
        self.fc = nn.Linear(10 * 13 * 13, 10)

    def forward(self, x):
        # 输入通过第一个卷积层
        x = self.conv1(x)
        # 应用ReLU激活函数
        x = self.relu(x)
        # 应用最大池化层
        x = self.pool(x)
        # 将输出扁平化，以便输入到全连接层
        x = x.view(-1, 10 * 13 * 13)
        # 输入到全连接层
        x = self.fc(x)
        return x


# Acc: 0.9578
class SimpleRNN(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.cnn = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.rnn = nn.RNN(49, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN提取特征
        x = self.cnn(x)        # [b, 1, 28, 28] -> [b, 16, 7, 7]
        # 压平特征
        b, c, h, w = x.size()
        x = torch.reshape(x, (b, c, -1)).transpose(1, 0)
        # RNN处理特征
        out, _ = self.rnn(x)
        out = self.fc(out[-1, :, :])
        return out
    

# Acc: 0.9578
class SimpleLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.cnn = nn.Conv2d(1, 32, kernel_size=4, stride=4)
        self.lstm = nn.LSTM(49, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # CNN提取特征
        x = self.cnn(x)        # [b, 1, 28, 28] -> [b, 16, 7, 7]
        # 压平特征
        b, c, h, w = x.size()
        x = torch.reshape(x, (b, c, -1)).transpose(1, 0)
        # RNN处理特征
        out, _ = self.lstm(x)
        out = self.fc(out[-1, :, :])
        return out


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=10):
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader), 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = test_model(model, test_loader)
        print('Epoch {}, Loss: {:.4f}, Acc: {:.4f}'.format(epoch + 1, running_loss / len(train_loader), acc))
    return model


def test_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total


if __name__ == '__main__':
    # 定义模型、损失函数和优化器
    # model = MLP(input_size=28*28, hidden_size=500, output_size=10)
    # model = SimpleCNN()
    # model = SimpleRNN(hidden_size=128, output_size=10, num_layers=1)
    model = SimpleLSTM(hidden_size=128, output_size=10, num_layers=1)
    # summary(model, (1, 28, 28))
    # 设置损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 加载MNIST数据集
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)

    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False)

    model = train_model(model, criterion, optimizer, trainloader, testloader, num_epochs=24)
    test_model(model, testloader)