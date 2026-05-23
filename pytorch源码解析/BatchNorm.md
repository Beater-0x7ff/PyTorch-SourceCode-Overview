## BN & SyncBN

### BatchNorm原理

BatchNorm 最早在全连接网络中被提出，对每个神经元的输入做归一化。拓展到 CNN 中，就是对每个卷积核的输入做归一化。

BN有以下好处：

- 防止过拟合
- 加快收敛
- 防止梯度弥散

BN 的数学表达式为：
$$
y = \frac{x-E[x]}{\sqrt{Var[x]+\epsilon}} * \gamma + \beta
$$
其中 $\gamma, \beta$ 是可学习的参数



### BatchNorm 的 PyTorch 实现

PyTorch 中与 BN 相关的几个类放在 torch.nn.modules.batchnorm 中，包含以下几个类：

- ```_NormBase``` ：```nn.Module``` 的子类，定义了 BN 中一系列属性与初始化，读数据的方法
- ```_BatchNorm``` ：```_NormBase``` 的子类，定义了 ```forward``` 方法
- ```BatchNorm1d``` & ```BatchNorm2d``` & ```BatchNorm3d``` ：```_BatchNorm``` 的子类，定义了不同的 ```_check_input_dim``` 方法

#### _NormBase类

```
Module
  ↓
_NormBase
  ↓
_BatchNorm
  ↓
BatchNorm1d / BatchNorm2d / BatchNorm3d
```

`_NormBase` 是公共父类，负责**定义和初始化 BatchNorm/InstanceNorm 共用的参数和状态**；`_BatchNorm` 继承 `_NormBase`，负责实现 **BatchNorm 的 forward 逻辑**。

[源码](https://github.com/pytorch/pytorch/blob/090b49d99bb9c26144cedea3b9c566fbe6ede275/torch/nn/modules/batchnorm.py#L152)



```_NormBase``` 定义了以下参数

```python
def __init__(
    self,
    num_features,
    eps=1e-5,
    momentum=0.1,
    affine=True,
    track_running_stats=True,
    device=None,
    dtype=None,
    *,
    bias=True,
)
```

| 参数                  | 作用                                    |
| --------------------- | --------------------------------------- |
| `num_features`        | 通道数，例如 `BatchNorm2d(64)` 中的 64  |
| `eps`                 | 防止除零的小常数                        |
| `momentum`            | 更新 running mean/var 的动量            |
| `affine`              | 是否使用可学习的 `weight` 和 `bias`     |
| `track_running_stats` | 是否维护 `running_mean` / `running_var` |
| `bias`                | 当 `affine=True` 时，是否创建 bias      |



##### running_mean, running_var 的更新

BatchNorm 默认打开 `track_running_stats`，因此每次 forward 时都会依据当前 minibatch 的统计量来更新 `running_mean` 和 `running_var`。

`momentum` 默认值为 0.1，更新时有
$$
running\_mean = running\_mean * (1-momentum) + E[x] * momentum \\
running\_var = running\_var * (1-momentum) + Var[x] * momentum
$$


#####  $\gamma, \beta$ 的更新

BatchNorm 的 `weight(γ)` 和 `bias(β)` 通过 `loss.backward()` + `optimizer.step()` 被梯度下降更新。

```python
import torch
import torch.nn as nn

# 一个 BatchNorm1d 层，3 个 feature
bn = nn.BatchNorm1d(3, affine=True)

# 手动设置初始 gamma 和 beta，方便观察
bn.weight.data = torch.tensor([1.0, 1.0, 1.0])  # gamma
bn.bias.data = torch.tensor([0.0, 0.0, 0.0])    # beta

optimizer = torch.optim.SGD(bn.parameters(), lr=0.1)

x = torch.tensor([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0],
    [3.0, 6.0, 9.0],
    [4.0, 8.0, 12.0],
])

target = torch.zeros_like(x)

print("Before update:")
print("gamma =", bn.weight.data)
print("beta  =", bn.bias.data)

# forward
y = bn(x)

# 假设希望输出接近 0
loss = ((y - target) ** 2).mean()

# backward
optimizer.zero_grad()
loss.backward()

print("\nGradient:")
print("gamma.grad =", bn.weight.grad)
print("beta.grad  =", bn.bias.grad)

# update
optimizer.step()

print("\nAfter update:")
print("gamma =", bn.weight.data)
print("beta  =", bn.bias.data)

# Before update:
# gamma = tensor([1., 1., 1.])
# beta  = tensor([0., 0., 0.])

# Gradient:
# gamma.grad = tensor([0.6667, 0.6667, 0.6667])
# beta.grad  = tensor([5.2154e-08, 5.2154e-08, 5.2154e-08])

# After update:
# gamma = tensor([0.9333, 0.9333, 0.9333])
# beta  = tensor([-5.2154e-09, -5.2154e-09, -5.2154e-09])
```



##### eval 模式

eval 模式有几个重要参数

- ```track_running_stats``` 默认为 True， train 模式下统计 ```running_mean``` 和```running_var```，eval 模式下用统计数据作为 mean 和 var。设置为 False 时， eval模式直接计算输入的均值和方差
- ```running_mean``` 和 ```running_var``` train 模式下的统计量



#### BatchNormNd 类

```BatchNorm1d```

- 2D Input	[N, C] 例如 batch_size = 32, feature_dim = 128
- 3D Input        [N, C, L] 例如 NLP/时序 batch, channel, sequence_length

```BatchNorm2d```

- 4D Input        [N, C, H, W] 例如图像 batch，channel， height，width

```BatchNorm3d```

- 5D Input        [N, C, D, H, W] 例如视频 新增了一个 Depth



#### SyncBatchNorm 的 PyTorch 实现

`SyncBatchNorm`（Synchronized BatchNorm）是 GPU 训练时，跨所有 GPU 同步 batch 统计量的 BatchNorm。

例如总 batch size 为 32、使用 8 张 GPU 时，普通 BatchNorm 每张卡实际上只会基于 batch size=4 统计均值方差；当单卡 batch 很小时，统计量会非常不稳定，容易影响训练效果。`SyncBatchNorm` 通过 GPU 间通信（通常是 all_reduce）汇总所有卡的统计信息，使每张卡使用相同的全局统计量，从而让多卡训练结果更加稳定，也更接近单卡大 batch 的行为。

在 PyTorch 中通常这样使用

```python
model = nn.SyncBatchNorm.convert_batchnorm(model)
model = DistributedDataParallel(model)
```

遍历整个模型，把所有 `BatchNorm` 层替换成 `SyncBatchNorm` 层，并复制原有参数和状态。