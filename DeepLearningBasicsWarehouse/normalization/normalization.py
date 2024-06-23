import torch
from torch import nn


class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))
        
    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True)
            exponential_average_factor = self.momentum
            if self.num_batches_tracked is not None:
                exponential_average_factor = self.momentum * (
                    1 - self.momentum ** self.num_batches_tracked) ** 0.5
            
                self.running_mean.mul_(1 - exponential_average_factor).add_(
                    batch_mean.data, alpha=exponential_average_factor
                )
                self.running_var.mul_(1 - exponential_average_factor).add_(
                    batch_var.data, alpha=exponential_average_factor
                )
                
                self.num_batches_tracked += 1
                
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
            
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        
    def forward(self, x):
        batch_mean = x.mean(dim=(1, 2, 3), keepdim=True)
        batch_var = x.var(dim=(1, 2, 3), keepdim=True)
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat * self.weight + self.bias


class InstanceNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super(InstanceNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_mean', torch.zeros(1, num_features, 1, 1))
        self.register_buffer('running_var', torch.ones(1, num_features, 1, 1))

    def forward(self, x):
        if self.training:
            batch_mean = x.mean(dim=(0, 2, 3), keepdim=True)
            batch_var = x.var(dim=(0, 2, 3), keepdim=True)
            exponential_average_factor = self.momentum
            if self.num_batches_tracked is not None:
                exponential_average_factor = self.momentum * (
                    1 - self.momentum ** self.num_batches_tracked) ** 0.5
                
                self.running_mean.mul_(1 - exponential_average_factor).add_(
                    batch_mean.data, alpha=exponential_average_factor
                )
                self.running_var.mul_(1 - exponential_average_factor).add_(
                    batch_var.data, alpha=exponential_average_factor
                    )
                
            self.num_batches_tracked += 1
            
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var
            
        x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
        return x_hat * self.weight + self.bias



class RMSNorm(nn.Module):
  def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
    super().__init__()
    self.eps = eps
    self.weight = nn.Parameter(torch.ones(hidden_size))
  
  def _norm(self, hidden_states):
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    return hidden_states * torch.rsqrt(variance + self.eps)
  
  def forward(self, hidden_states):
    return self.weight * self._norm(hidden_states.float()).type_as(hidden_states)
