import torch
import torch.nn as nn


# 底层实现的交叉损失函数
class CrossEntropyLoss(nn.Module):
    def __init__(self, num_classes):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = num_classes                         # 分类的数量
        self.log_softmax = nn.LogSoftmax(dim=1)                # 计算log(softmax(x))

    def forward(self, predicts, targets):
        """
        损失函数的计算
        需要注意的是，通常我们的predicts应当为(batch_size, num_classes)的一个张量，
        而targets应当为(batch_size,)的一个张量。
        """
        # 计算log(softmax(x))
        log_probs = self.log_softmax(predicts)
        # 对targets做one-hot编码
        one_hot_targets = torch.zeros_like(log_probs)
        one_hot_targets.scatter_(1, targets.unsqueeze(1), 1)   # 沿着维度1进行scatter操作，即将targets中的元素复制到one_hot_targets中

        # 计算损失
        loss = (-one_hot_targets * log_probs).sum(dim=1).mean()
        return loss


# 测试
if __name__ == '__main__':
    # 创建模型
    loss_func = CrossEntropyLoss(num_classes=10)

    predicts = torch.rand((32, 10), dtype=torch.float32)
    targets = torch.randint(low=0, high=10, size=(32,), dtype=torch.int64)

    # 计算损失
    loss = loss_func(predicts, targets)
    print(loss)