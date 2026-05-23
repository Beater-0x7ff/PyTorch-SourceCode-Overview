# Torch 源码解析

## torch.autograd

### torch.autograd.Function

```Function``` 的作用是允许用户 **自定义一个可参与自动求导的算子**我们在自己构建 ```nn.Module``` 的时候通常只需要 实现 ```forward``` 只要其中使用的 PyTorch支持的 autograd 的 Tensor操作， PyTorch 就会自动构建 **计算图**，反向传播时自动求梯度但是在以下情况中 PyTorch 不知道怎么自动求导时，需要自己定义 autograd 的逻辑，例如：

- 自定义 CUDA/C++ operator
- 在 forward 里调用了不可微的外部库
- 实现更高效的 backward
- 实现特殊的梯度规则

就需要以 ```Function``` 定义的 forward，backward 作为基类实现自定义的前向和反向传播例如：

```python
import torch
from torch.autograd.function import Function

class Exp(Function):                   

    @staticmethod
    def forward(ctx, i):                
        result = i.exp()
        # torch.exp(i)
        ctx.save_for_backward(result)   
        return result

    @staticmethod
    def backward(ctx, grad_output):     
        result, = ctx.saved_tensors     
        return grad_output * result     


x = torch.tensor([1.], requires_grad=True) #梯度反传开启
ret = Exp.apply(x)                         
print(ret)                                 
ret.backward()                              
print(x.grad)
```



### torch.autograd.functional

普通的 autograd 只需要最终得到每个参数应该更新多少，即只对梯度值感兴趣而 ```functional``` 是一套计算梯度特征的工具箱，例如：

- 梯度值受哪些输入值影响
- 梯度值分别如何依赖每个输入
- 二阶导是多少
- 某个方向上变化率是多少

PyTorch 在前向传播时会自动记录 Tensor 是如何一步步计算出来的，形成一个 DAG，之后 backward 时， PyTorch 沿着这个图反向传播梯度，并调用每个 operation 对应的 backward 函数，用链式法则自动计算所有参数的梯度。



关注到 Tensor的 ```.backward()``` ，实际上这只是个入口，真正实现反向传播的是 ```torch.autograd.backward()``` 

```python
class Tensor(torch._C._TensorBase)

    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        relevant_args = (self,)
        ...
        torch.autograd.backward(self, gradient, retain_graph, create_graph)
```

关键变量解释：

- ```gradient```

  loss 只有为标量的时候才能直接 ```backward()```，如果 loss 是向量， PyTorch 不知道你想计算哪个方向上的导数，所以需要指定 **沿哪个方向反向传播**，例如：

  ```python
  import torch
  
  x = torch.tensor([2., 3.], requires_grad=True)
  y = x ** 2        # y = [4, 9]
  
  # False
  # y.backward()
  # RuntimeError: grad can be implicitly created only for scalar outputs
  
  y.backward(torch.tensor([1., 1.]))
  print(x.grad)
  # tensor([4., 6.])
  ```

  

- ```retain_graph``` 

  默认情况下 ```loss.backward()``` 反向传播后 PyTorch 会释放中间计算图缓存

  因此，无法对同一个图再次 backward：

  ```python
  import torch
  
  x = torch.tensor(2.0, requires_grad=True)
  
  y = x ** 3
  loss = y
  
  # loss.backward()
  # loss.backward()
  # RuntimeError: Trying to backward through the graph a second time (or directlyaccess saved tensors after they have already been freed). Saved intermediate vaccess saved tensors after they have already been freed). Saved intermediate vaccess saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
  
  loss.backward(retain_graph=True)
  print(x.grad)
  # tensor(4.)
  
  loss.backward()
  print(x.grad)
  # tensor(8.)
  ```

  这里需要注意两次梯度发生了变化，是因为 PyTorch 默认累加 ```.grad``` 

  如果不想累加需要在两个 ```backward()``` 之间 ```x.grad.zero_```

- ```create_graph```

  正如名字一样，这个变量决定了是否在计算一阶梯度的时候，也为 **一阶梯度计算过程建立计算图**，从而允许继续求二阶导，或者更高阶导，例如：

  ```python
  import torch
  
  x = torch.tensor(2.0, requires_grad=True)
  
  y = x ** 3
  
  grad = torch.autograd.grad(y, x, create_graph=True)[0]
  print(grad)  # tensor(12., grad_fn=<MulBackward0>)
  
  grad.backward()
  print(x.grad)  # tensor(12.)
  ```

  ```retain_graph``` 和 ```create_graph``` 的区别：

  前者保留 forward 图，用来重复反传

  后者创建 backward 图，用来求高阶导



#### jacobian()

```jacobian(func, inputs)``` 用来计算 **函数每一个输出元素，对每一个输入元素的偏导数**

```python
from torch.autograd.functional import jacobian
import torch

def f(x):
    return x ** 2

x = torch.tensor([1., 2., 3.], requires_grad=True)

J = jacobian(f, x)
print(J)
# tensor([[2., 0., 0.],
#        [0., 4., 0.],
#        [0., 0., 6.]])
```

**jacobian.shape = output.shape + input.shape**

换句话说 **jacobain 本质上是输入空间到输出空间的局部线性映射**，所有 output 维度和所有 input 维度的偏导数组合。



#### hessian()

```hessian(func, input)``` 用来计算 **标量函数对输入的二阶偏导数**，例如：

```python
import torch
from torch.autograd.functional import hessian

def f(x):
    return torch.sum(x ** 2)

x = torch.tensor([1., 2., 3.], requires_grad=True)

H = hessian(f, x)
print(H)
```

```hessian.shape = input.shape + input.shape```



#### hook

之前我们提到 ```autograd.backward()``` 为了节约空间，仅会保存叶节点的梯度。为了获取某一中间结果的梯度，我们可以使用 ```autograd.grad()``` 接口或者 ```hook``` 机制

```python
import torch

A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
C = A * B
D = C.exp()
torch.autograd.grad(D, (C, A))  
# (dD/dC, dD/dA)

def variable_hook(grad):                       
    print('the gradient of C is：', grad)

A = torch.tensor(2., requires_grad=True)
B = torch.tensor(.5, requires_grad=True)
C = A * B
hook_handle = C.register_hook(variable_hook)   

D = C.exp()                 
D.backward()                                   

hook_handle.remove() 
```

```grad()``` 和 ```backward()``` 的区别：

- 前者计算然后只返回
- 后者计算然后累加到 ```.grad```



在 PyTorch 里，hook 就是 **在反向传播或前向传播过程中插入一个回调函数**，可以让你在模型运行到某个 Tensor 或 Module时，**查看，记录，甚至修改中间的结果或者梯度**。

例如：

```python
import torch

A = torch.tensor(2., requires_grad=True)
B = torch.tensor(0.5, requires_grad=True)

C = A * B
D = C.exp()

def hook_fn1(grad):
    print("C received grad:", grad)
    
def hook_fn2(grad):
    print("original grad:", grad)
    return grad * 0.1

hooks = [
    C.register_hook(hook_fn1),
    C.register_hook(hook_fn2),
    C.register_hook(hook_fn1),
]

# C received grad: tensor(2.7183)
# original grad: tensor(2.7183)
# C received grad: tensor(0.2718)

D.backward()

for h in hooks:
    h.remove()
```

对于 Module 上的 forward hook 和 backward hook：

```python
import torch
from torch import nn

layer = nn.Linear(3, 2)


# 1. forward 前触发
def pre_hook(module, inputs):
    print("\n[forward_pre_hook]")
    print("input:", inputs)


# 2. forward 后触发
def forward_hook(module, inputs, output):
    print("\n[forward_hook]")
    print("input:", inputs)
    print("output:", output)


# 3. backward 时触发
def backward_hook(module, grad_input, grad_output):
    print("\n[full_backward_hook]")
    print("grad_input:", grad_input)
    print("grad_output:", grad_output)

hooks = [
    layer.register_forward_pre_hook(pre_hook),
    layer.register_forward_hook(forward_hook),
    layer.register_full_backward_hook(backward_hook),
]

x = torch.ones(3, requires_grad=True)

y = layer(x)
loss = y.sum()

print("\nloss:", loss)

loss.backward()


for hook in hooks:
    hook.remove()
    
# [forward_pre_hook]
# input: (tensor([1., 1., 1.], requires_grad=True),)

# [forward_hook]
# input: (tensor([1., 1., 1.], grad_fn=<BackwardHookFunctionBackward>),)        
# output: tensor([0.1311, 0.3671], grad_fn=<ViewBackward0>)

# loss: tensor(0.4982, grad_fn=<SumBackward0>)

# [full_backward_hook]
# grad_input: (tensor([ 0.8775, -0.6117,  0.2577]),)
# grad_output: (tensor([1., 1.]),)
```









### torch.autograd.gradcheck

```gradcheck``` 是 PyTorch 用来 **验证自定义backward是否正确** 的工具。具体来说是比较 **解析梯度** 和 **数值梯度**，如果两者接近则正确，反之错误。

解析梯度就是自定义 backward 计算得出的梯度，数值梯度则是
$$
f'(x) \approx \frac{f(x+\epsilon)-f(x-\epsilon)}{2\epsilon}
$$
即 **有限差分**，用函数变化（$[-\epsilon,\epsilon]$​）均值近似导数，且误差小于一定限度

```python
import torch
from torch.autograd import Function

class Sigmoid(Function):
                                               
    @staticmethod
    def forward(ctx, x): 
        output = 1 / (1 + torch.exp(-x))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output): 
        output,  = ctx.saved_tensors
        grad_x = output * (1 - output) * grad_output
        return grad_x

test_input = torch.randn(4, dtype=torch.float32 ,requires_grad=True)     
torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-3)    # pass
torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-3)    # pass
torch.autograd.gradcheck(Sigmoid.apply, (test_input,), eps=1e-4)    # fail
torch.autograd.gradcheck(torch.sigmoid, (test_input,), eps=1e-4)    # fail

#     raise GradcheckError(
# torch.autograd.gradcheck.GradcheckError: Jacobian mismatch for output 0 with respect to input 0,        
# numerical:tensor([[0.2497, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.1010, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.2235, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.2453]])
# analytical:tensor([[0.2500, 0.0000, 0.0000, 0.0000],
#         [0.0000, 0.1011, 0.0000, 0.0000],
#         [0.0000, 0.0000, 0.2234, 0.0000],
#         [0.0000, 0.0000, 0.0000, 0.2451]])
```





### torch.autograd.anomaly_mode

```anomaly mode``` 是 PyTorch autograd 的 **梯度异常调试模式**，主要用来定位：

- backward 中出现 ```nan```
- backward 中出现 ```inf```
- 自定义 ```Function.backward()``` 报错
- 某个 op 的反向传播计算失败

例如：

```python
import torch
from torch.autograd import Function

class BadGrad(Function):
    @staticmethod
    def forward(ctx, x):
        return x ** 2

    @staticmethod
    def backward(ctx, grad_output):
        # 故意制造 NaN 梯度
        grad_x = grad_output * torch.tensor(float("nan"))
        return grad_x


x = torch.tensor(2.0, requires_grad=True)

with torch.autograd.detect_anomaly():
    y = BadGrad.apply(x)
    loss = y.sum()
    loss.backward()
    
#    return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
# RuntimeError: Function 'BadGradBackward' returned nan values in its 0th output.
```

> [!NOTE]
>
> ```anomaly_mode``` 会明显降低速度，增加内存开销，一般只在调试时开启，不建议正式训练长期使用



### torch.autograd.grad_mode

```grad_mode``` 是 PyTorch 里控制 **是否记录计算图，是否启用梯度计算**的一组上下文管理工具。我们在 inference 的时候 因为不需要训练模型，因此也不需要 backward 和梯度，所以没有必要让 autograd 去建立计算图、保存中间信息。关闭 autograd 可以减少大量额外开销。

可以利用 ```torch.no_grad()``` 来关闭自动求导

```python
from torchvision.models import resnet50
import torch

net = resnet50().cuda(0)
num = 128
inp = torch.ones([num, 3, 224, 224]).cuda(0)

with torch.no_grad():
    net(inp)
```



### model.eval()

```model.eval()``` 会将模型切换到推理模式:

- dropout 层不再随机丢弃
- BatchNorm 使用训练时累计的running mean/var
- 不会关闭 autograd，不使用 no_grad 节省显存



### torch.autograd.profiler

```torch.autograd.profiler``` 是 PyTorch 用来**分析模型运行耗时和显存/内存开销**的性能分析工具。

| 信息                  | 含义                                     |
| --------------------- | ---------------------------------------- |
| 每个 op 的耗时        | 例如 `conv2d`、`matmul`、`relu` 花了多久 |
| CPU/GPU 时间          | 分别统计 CPU 和 CUDA kernel 时间         |
| 调用次数              | 某个 op 执行了多少次                     |
| 内存开销              | 某些模式下可以记录显存/内存使用          |
| forward/backward 开销 | 可以分析训练时哪部分最慢                 |

```python
import torch
from torchvision.models import resnet18

x = torch.randn((1, 3, 224, 224), requires_grad=True)
model = resnet18()
with torch.autograd.profiler.profile() as prof:
    for _ in range(10):
        y = model(x)
        y = torch.sum(y)
        y.backward()

print(prof.key_averages().table(sort_by="self_cpu_time_total"))
```



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







## torch.utils.data

### 迭代器

在 ```Dataset, Sampler, DataLoader``` 三个类中都会用到 python 抽象类的魔法方法，```__len__(self), __getitem__(self), __iter(self)```

- `__len__(self)`: 定义当被 `len()` 函数调用时的行为，一般返回迭代器中元素的个数
- `__getitem__(self)`: 定义获取容器中指定元素时的行为，相当于 `self[key]` ，即允许类对象拥有索引操作
- `__iter__(self)`: 定义当迭代容器中的元素时的行为



迭代器一般满足以下几种特性：

- 迭代器是⼀个对象
- 迭代器可以被 next() 函数调⽤，并返回⼀个值
- 迭代器可以被 iter() 函数调⽤，并返回一个迭代器（可以是自身）
- 连续被 next() 调⽤时依次返回⼀系列的值
- 如果到了迭代的末尾，则抛出 StopIteration 异常
- 迭代器也可以没有末尾，只要被 next() 调⽤，就⼀定会返回⼀个值
- Python 中， next() 内置函数调⽤的是对象的 **next**() 方法
- Python 中， iter() 内置函数调⽤的是对象的 **iter**() 方法
- ⼀个实现了迭代器协议的的对象可以被 for 语句循环迭代直到终止



### Dataset

```Dataset``` 负责对 raw data source 封装，将其封装成 Python 可识别的数据结构，其必须提供提取数据个体的接口。

Dataset 共有 Map-style datasets 和 Iterable-style datasets 两种：

#### Map-style dataset

这是一种通过实现`__getitem__()` 和 `__len()__` 来获取数据的 Dataset。如名字所示，通过构建键值对形成的数据集。

这种类型的数据集居多，接口定义如下：

```python
class Dataset(Generic[T_co]):
    # Generic is an Abstract base class for generic types.

    def __getitem__(self, index) -> T_co:
        raise NotImplementedError

    def __add__(self, other: 'Dataset[T_co]') -> 'ConcatDataset[T_co]':
        return ConcatDataset([self, other])
```

PyTorch 中所有定义的 Dataset 都是其子类



#### Iterable-style dataset

它是一种实现 `__iter__()` 来获取数据的 Dataset，这种类型的数据集特别适用于以下情况：随机读取代价很大甚至不大可能，且 batch size 取决于获取的数据。其接口定义如下：

```python
class IterableDataset(Dataset[T_co]):

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError

    def __add__(self, other: Dataset[T_co]):
        return ChainDataset([self, other])
```

当 ```DataLoader``` 的 ```num_workers > 0``` 时，每个 worker 都将具有数据对象的不同样本，因此需要独立对每个副本进行配置，以防每个 worker 产生的数据不同。数据加载顺序**完全由用户定义**的可迭代样式控制。这允许更容易地实现块读取和动态批次大小。



#### 其它

- `torch.utils.data.ConcatDataset`: 用于连接多个 `ConcatDataset` 数据集
- `torch.utils.data.ChainDataset` : 用于连接多个 `IterableDataset` 数据集，在 `IterableDataset` 的 `__add__()` 方法中被调用
- `torch.utils.data.Subset`: 用于获取指定一个索引序列对应的子数据集

```python
class Subset(Dataset[T_co]):

    dataset: Dataset[T_co]
    indices: Sequence[int]

    def __init__(self, dataset: Dataset[T_co], indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)
```

- `torch.utils.data.TensorDataset`: 用于获取封装成 tensor 的数据集，每一个样本都通过索引张量来获得。

```python
class TensorDataset(Dataset):
    def __init__(self, *tensor):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in tensors

    def __len__(self):
        return self.tensors[0].size(0)
```



### Sampler

`torch.utils.data.Sampler` 负责提供一种遍历数据集所有元素索引的方式。可支持用户自定义，也可以用 PyTorch 提供的

```python
class Sampler(Generic[T_co]):

    def __init__(self, data_source: Optional[Sized]) -> None:
        pass

    def __iter__(self) -> Iterator[T_co]:
        raise NotImplementedError
```

PyTorch 也在此基础上提供了其他类型的 Sampler 子类

- `torch.utils.data.SequentialSampler` : 顺序采样样本，始终按照同一个顺序
- `torch.utils.data.RandomSampler`: 可指定有无放回地，进行随机采样样本元素
- `torch.utils.data.SubsetRandomSampler`: 无放回地按照给定的索引列表采样样本元素
- `torch.utils.data.WeightedRandomSampler`: 按照给定的概率来采样样本。样本元素来自 `[0,…,len(weights)-1]` ， 给定概率（权重）
- `torch.utils.data.BatchSampler`: 在一个batch中封装一个其他的采样器, 返回一个 batch 大小的 index 索引
- `torch.utils.data.DistributedSample`: 将数据加载限制为数据集子集的采样器。与 `torch.nn.parallel.DistributedDataParallel` 结合使用。 在这种情况下，每个进程都可以将 `DistributedSampler` 实例作为 `DataLoader` 采样器传递



### DataLoader

`torch.utils.data.DataLoader` 是 PyTorch 数据加载的核心，负责加载数据，同时支持 Map-style 和 Iterable-style Dataset，支持**单进程/多进程**，还可以设置 loading order, batch size, pin memory 等加载参数。

```python
DataLoader(dataset, 
           batch_size=1, 
           shuffle=False, 
           sampler=None,
           batch_sampler=None, 
           num_workers=0, 
           collate_fn=None,
           pin_memory=False, 
           drop_last=False, 
           timeout=0,
           worker_init_fn=None, 
           *, 
           prefetch_factor=2,
           persistent_workers=False
          )
```

| 参数                 | 默认值  | 数据类型                        | 作用                                                         |
| -------------------- | ------- | ------------------------------- | ------------------------------------------------------------ |
| `dataset`            | 必填    | `Dataset`                       | 数据集对象，提供数据来源，通常实现 `__len__` 和 `__getitem__` |
| `batch_size`         | `1`     | `int` / `None`                  | 每个 batch 包含多少个样本                                    |
| `shuffle`            | `False` | `bool`                          | 是否在每个 epoch 打乱数据顺序                                |
| `sampler`            | `None`  | `Sampler` / `Iterable` / `None` | 自定义样本索引生成方式；与 `shuffle` 通常不能同时使用        |
| `batch_sampler`      | `None`  | `Sampler` / `Iterable` / `None` | 自定义 batch 级别的索引生成器；与 `batch_size`、`shuffle`、`sampler`、`drop_last` 通常互斥 |
| `num_workers`        | `0`     | `int`                           | 用多少个子进程加载数据；`0` 表示在主进程加载                 |
| `collate_fn`         | `None`  | `callable` / `None`             | 定义如何把多个样本合并成一个 batch                           |
| `pin_memory`         | `False` | `bool`                          | 是否将数据放入 pinned memory，加速 CPU 到 GPU 的拷贝         |
| `drop_last`          | `False` | `bool`                          | 如果最后一个 batch 不满 `batch_size`，是否丢弃               |
| `timeout`            | `0`     | `float` / `int`                 | worker 加载 batch 的超时时间，单位是秒                       |
| `worker_init_fn`     | `None`  | `callable` / `None`             | 每个 worker 启动时调用的初始化函数                           |
| `prefetch_factor`    | `2`     | `int` / `None`                  | 每个 worker 预取多少个 batch                                 |
| `persistent_workers` | `False` | `bool`                          | 一个 epoch 结束后是否保留 worker 进程，避免反复创建销毁      |



### 批处理

`DataLoader` 支持通过参数`batch_size`, `drop_last`, `batch_sampler`，自动地把取出的数据整理 (collate) 成批次样本 (batch)

在使用 sampler 产生的 indices 获取采样到的数据时，DataLoader 使用 `collate_fn` 参数将样本列表整理成 batch。

当用户想用 dataset 代码手动处理 batch，或仅加载单个 sample data 时，可将 `batch_size` 和 `batch_sampler` 设为 `None`, 将关闭自动批处理。此时，由 `Dataset` 产生的 sample 将会直接被 `collate_fn` 处理。



#### ```collate_fn```

关闭 automatic batching 时 ```collate_fn``` 处理单个 sample

开启的时候 ```collate_fn``` 作用于数据样本列表，将输入样本整理成 batch，这个过程有以下几步

- 添加新的维度（batch 维）
- 自动 tensor 化
- 原始数据的数据结构不变

为了解决如数据的维度不总是相同的，类似的情况，那么用户还可以自定义排序规则，比如做padding 和 自定义数据类型支持。



### 单进程

在单进程模式下，`DataLoader` **初始化的进程和取数据的进程是一样的** ，因此，数据加载可能会阻止计算。

但是，当用于在进程之间共享数据的资源（例如共享内存，文件描述符）有限时，或者当整个数据集很小并且可以完全加载到内存中时，此模式可能是首选。

此外，单进程加载通常显示更多可读的错误跟踪，因此**对于调试很有用**。



### 多进程

```DataLoader``` 实例化的时候并不会启动 worker，而是每次 ```DataLoader``` 创建 ```iterator``` 进行迭代时，worker 才会启动。```dataset, collate_fn, worker_init_fn``` 会被传到每个 worker 中，每个 worker 都用独立的进程。

- 对于 map-style 数据，**主线程会用 Sampler** 产生 indice，并将它们送到 worker 里。因此，**shuffle是在主线程做的**。
- 对于 iterable-style 数据，因为每个 worker 都有相同的 data 复制样本，并在各个进程里进行不同的操作，以防止每个进程输出的数据是重复的，所以一般用 `torch.utils.data.get_worker_info()` 来进行辅助处理。



> [!NOTE]
>
> **注意**，worker 进行读文件，解码图片/文本，tokenize 等等操作之后 **不应该返回 cuda Tensor，而是返回 CPU Tensor**。 这是由于 worker 是多个独立子进程，而 CUDA Tensor 跨进程有一套独立的 GPU context，显存所有权，CUDA IPC，生命周期管理的问题。

```python
loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    pin_memory=True
)

for x, y in loader:
    x = x.cuda(non_blocking=True)
    y = y.cuda(non_blocking=True)

    output = model(x)
    loss = criterion(output, y)
```



### 锁页内存 Memory Pinning

主机中的内存，有两种存在方式，一是锁页，二是不锁页，锁页内存存放的内容在任何情况下都不会与主机的虚拟内存（硬盘）进行交换，而不锁页内存在主机内存不足时，数据会存放在虚拟内存中。主机到GPU副本源自固定（页面锁定）内存时，速度要快得多。CPU张量和存储暴露了一种 `pin_memory()` 方法，该方法返回对象的副本，并将数据放在固定的区域中。

而**显卡中的显存全部是锁页内存！当计算机的内存充足的时候，可以设置 `pin_memory=True`。**设置 `pin_memory=True`，则意味着生成的 Tensor 数据最开始是属于内存中的锁页内存，这样将内存的Tensor转义到GPU的显存就会更快一些。同时，由于 `pin_memory` 的作用是将张量返回之前将其复制到 CUDA 固定的内存中，所以只有在 CUDA 环境支持下才有用。

简易流程如下：

1. 先创建一块临时的 pinned memory
2. CPU 将 pageable 内存拷贝到 pinned 内存
3. GPU MDA（Memory Direct Access）将 pinned memory 拷贝到 VRAM

```python
def pin_memory(data, device=None):
    if isinstance(data, torch.Tensor):
        return data.pin_memory()

    if hasattr(data, "pin_memory"):
        return data.pin_memory()

    if isinstance(data, (str, bytes)):
        return data

    if isinstance(data, collections.abc.Mapping):
        try:
            if isinstance(data, collections.abc.MutableMapping):
                clone = copy.copy(data)
                clone.update(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )
                return clone
            else:
                return type(data)(
                    {k: pin_memory(sample, device) for k, sample in data.items()}
                )  
        except TypeError:
            return {k: pin_memory(sample, device) for k, sample in data.items()}

    if isinstance(data, tuple):
        if hasattr(data, "_fields"):  
            return type(data)(*(pin_memory(sample, device) for sample in data))
        return type(data)(pin_memory(sample, device) for sample in data)

    if isinstance(data, collections.abc.Sequence):
        try:
            if isinstance(data, collections.abc.MutableSequence):

                clone = copy.copy(data)  
                for i, item in enumerate(data):
                    clone[i] = pin_memory(item, device)
                return clone
            return type(data)([pin_memory(sample, device) for sample in data])  
        except TypeError:
            return [pin_memory(sample, device) for sample in data]

    return data
```

```DataLoader(pin_memory=True)``` 默认只认识常见结构，如 ```Tensor, dict, list, tuple```。如果 ```collate_fn``` 返回的是一个自定义对象，那么 DataLoader 不知道这个类里面哪些字段需要 pinned memory ，因此，需要在这个自定义类里实现 ```pin_memory``` 方法。

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

class SimpleCustomBatch:

    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(sample.inp.is_pinned())  # True
    print(sample.tgt.is_pinned())  # True
```



### 预取 prefetch

预取是指当前 batch 还没被模型用完之前，DataLoader 已经提前准备后面的 batch

DataLoader 通过指定 ```prefetch_factor``` 来进行数据的预取

```python
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        ...
        self._reset(loader, first_iter=True)

    def _reset(self, loader, first_iter=False):
        ...
        # prime the prefetch loop
        for _ in range(self._prefetch_factor * self._num_workers):
            self._try_put_index()
```

for 循环会调用 dataloader 的 ```__iter__(self)``` 方法，以此获得迭代器来遍历 dataset

```python
class DataLoader(Generic[T_co]):
    ...
    def __iter__(self) -> '_BaseDataLoaderIter':

        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()
```

可以看见，dataloader 调用了 ```self._get_iterator()``` 方法，根据 ```num_worker``` 获得迭代器，并指示进行单进程还是多进程

对于单线程调用逻辑有：

```
DataLoader.__iter__()
        |
        v
self._get_iterator()
        |
        v
_SingleProcessDataLoaderIter
        |
        v
_BaseDataLoaderIter.__next__()
        |
        v
self._next_data()
        |
        v
self._next_index()
        |
        v
next(self._sampler_iter)
        |
        v
next(iter(self._index_sampler))
        |
        v
	获得 index
        |
        v
self._dataset_fetcher.fetch(index)
        |
        v
	获得 data
```

对多线程

```
DataLoader.__iter__()
        |
        v
self._get_iterator()
        |
        v
_MultiProcessingDataLoaderIter
        |
        v
创建 worker 进程
        |
        v
创建 index_queue / worker_result_queue
        |
        v
主进程从 sampler 取 index
        |
        v
next(self._sampler_iter)
        |
        v
获得 index
        |
        v
把 index 放入 index_queue
        |
        v
worker 从 index_queue 取 index
        |
        v
worker 调用 dataset_fetcher.fetch(index)
        |
        v
worker 获得 data
        |
        v
worker 执行 collate_fn
        |
        v
把 batch 放入 worker_result_queue
        |
        v
主进程 __next__()
        |
        v
self._next_data()
        |
        v
从 worker_result_queue 取 batch
        |
        v
如果 pin_memory=True，进入 pin_memory_thread
        |
        v
返回 batch/data
```

有以下这些比较重要的 flag 参数来协调各个 worker 之间的工作

- `_send_idx`: 发送索引，用来记录这次要放 index_queue 中 batch 的 idx
- `_rcvd_idx`: 接受索引，记录要从 data_queue 中取出的 batch 的 idx
- `_task_info`: 存储将要产生的 data 信息的 dict，key为 task idx（由 0 开始的整形索引），value 为 `(worker_id,)` 或 `(worker_id, data)`，分别对应数据 未取 和 已取 的情况
- `_tasks_outstanding`: 整形，代表已经准备好的 task/batch 的数量（可能有些正在准备中）



每个 worker 一次产生一个 batch 的数据，返回 batch 数据前放入下一个批次要处理的数据下标。

dataloader 初始化的时候，每个 worker 的 index_queue 默认会放入两个 batch 的 index，从 ```index_queue``` 中取出要处理的下标。







## nn.Module

### 常用接口

#### ```__init__ ```

在 nn.Module 的 `__init__` 函数中，会首先调用 ```torch._C._log_api_usage_once("python.nn_module")```

在此之后，nn.Module 初始化了一系列重要的成员变量。这些变量初始化了在模块 forward、 backward 和权重加载等时候会被调用的的 hooks，也定义了 parameters 和 buffers

```python
self.training = True  # 控制 training/testing 状态
self._parameters = OrderedDict()  # 在训练过程中会随着 BP 而更新的参数
self._buffers = OrderedDict()  # 在训练过程中不会随着 BP 而更新的参数
self._non_persistent_buffers_set = set()
self._backward_hooks = OrderedDict()  # Backward 完成后会被调用的 hook
self._forward_hooks = OrderedDict()  # Forward 完成后会被调用的 hook
self._forward_pre_hooks = OrderedDict()  # Forward 前会被调用的 hook
self._state_dict_hooks = OrderedDict()  # 得到 state_dict 以后会被调用的 hook
self._load_state_dict_pre_hooks = OrderedDict()  # load state_dict 前会被调用的 hook
self._modules = OrderedDict()  # 子神经网络模块
```

继承 nn.Module 的神经网络在实现自己的 ```__init__ ``` 函数时，一定要先调用 ```super().__init__()``` 这样才能正确初始化上述代码中的成员变量。



#### 状态转换

- 训练与测试

nn.Module 通过 **self.traing** 来区分训练和测试两种状态，使得模块可以在训练和测试时有不同的 forward 行为。nn.Module 通过 self.train() 和 self.eval() 来修改训练和测试状态。

```model.train()``` 实际上是通过 ```model.training = True``` 

```model.eval()``` 本质上是 ```model.train(False)```

 一个大模型 model，实际上是遍历 model 的子节点然后递归的将 ```child.train(mode=True/False)```

例如：

在目标检测和分割这类任务中，batch size 往往很小，比如每张 GPU 只有 1~2 张图，用 BatchNorm 继续当前的 batch 统计均值和方差会非常稳定，常见做法是 **冻结 backbone 中的 BN 层** 

```python
def train(self, mode=True):
    # 正常的调用父类的 train(mode), 整个 ResNet 进入训练模式
    super(ResNet, self).train(mode)
    # 冻结指定 stage 的参数
    self._freeze_stages()
    if mode and self.norm_eval:
        for m in self.modules():
    		# 当某个 module 是 BatchNorm, 就单独调用 eval()        
            if isinstance(m, _BatchNorm):
                m.eval()
```



#### 梯度处理

nn.Module 有两个相关的函数实现，分别是 ```requires_grad```, ```zero_grad``` 函数，他们都调用了 ```self.parameters()``` 来访问所有的参数，并修改参数的 ```requires_grad``` 状态或者清理参数的梯度。

```python
def requires_grad_(self, requires_grad: bool = True) -> Self:

        for p in self.parameters():
            p.requires_grad_(requires_grad)
        return self

def zero_grad(self, set_to_none: bool = True) -> None:
    
    if getattr(self, "_is_replica", False):
        warnings.warn(
            "Calling .zero_grad() from a module created with nn.DataParallel() has no effect. "
            "The parameters are copied (in a differentiable manner) from the original module. "
            "This means they are not leaf nodes in autograd and so don't accumulate gradients. "
            "If you need gradients in your forward method, consider using autograd.grad instead.",
            stacklevel=2,
        )

    for p in self.parameters():
        if p.grad is not None:
            if set_to_none:
                p.grad = None
            else:
                if p.grad.grad_fn is not None:
                    p.grad.detach_()
                else:
                    p.grad.requires_grad_(False)
                p.grad.zero_()

```



#### 参数的转移或转换

```nn.Module``` 实现了以下常用函数将模块转变成 float16 等类型，转移到 CPU/GPU 上

1. CPU：将所有 parameters 和 buffer 转移到 CPU 上
2. type：将所有 parameters 和 buffer 转变成另一个类型
3. CUDA：将所有 parameters 和 buffer 转移到 GPU 上
4. float：将所有浮点类型的 parameters 和 buffer 转变成 float32 类型
5. double：将所有浮点类型的 parameters 和 buffer 转变成 double 类型
6. half：将所有浮点类型的 parameters 和 buffer 转变成 float16 类型
7. bfloat16：将所有浮点类型的 parameters 和 buffer 转变成 bfloat16 类型
8. to：移动模块或/和改变模块的类型



这些函数都是通过 ```self._apply(function)``` 实现的， function 一般是 lambda 表达式或其他自定义函数。```self._apply()``` 函数实际上做了如下3件事情，最终将 function 完整的应用于整个模块：

- 通过 ```self.children()``` 进行递归的调用
- 对 ```self._parameters``` 中的参数及其 gradient 通过 function 进行处理
- 对 ```self._buffers``` 中的 buffer 逐个通过 function 来进行处理



```python
def _apply(self, fn, recurse=True):
        if recurse:
            for module in self.children():
                module._apply(fn)

        from torch._subclasses.fake_tensor import FakeTensor

        def compute_should_use_set_data(tensor, tensor_applied) -> bool:
            # 判断是否可以沿用旧行为：直接用 tensor.data = tensor_applied 替换底层数据。
        	# 这里要求原 tensor 和转换后的 tensor 类型兼容。
            if torch._has_compatible_shallow_copy_type(
                tensor, tensor_applied
            ) and not isinstance(tensor_applied, FakeTensor):
            	# 新版 PyTorch 提供了 future flag。
            	# 如果用户没有开启“转换时覆盖 module 参数”的未来行为，
            	# 就继续使用旧行为：param.data = param_applied。
                return not torch.__future__.get_overwrite_module_params_on_conversion()
            else:
                return False

        # 另一个 future flag：
    	# 是否使用 swap_tensors 的方式替换参数。
    	# 这是新版 PyTorch 为 tensor subclass / FakeTensor 等场景准备的更安全机制。 
        should_use_swap_tensors = (
            torch.__future__.get_swap_module_params_on_conversion()
        )

        for key, param in self._parameters.items():
            if param is None:
                continue

            # 参数是 autograd 里的 leaf tensor。
        	# 设备/类型转换不应该被 autograd 记录到计算图里，
        	# 所以这里用 no_grad。
            with torch.no_grad():
                param_applied = fn(param)
                
            # 判断是否可以用 param.data = param_applied 的方式原地替换数据。
            p_should_use_set_data = compute_should_use_set_data(param, param_applied)

            p_should_use_swap_tensors = (
                should_use_swap_tensors
                or is_traceable_wrapper_subclass(param_applied)
                or isinstance(param, FakeTensor)
            )

            param_grad = param.grad
            if p_should_use_swap_tensors:
                # 情况 1 使用 swap_tensors 替换参数内容
                try:
                    if param_grad is not None:
                        # 暂时移除 grad
                        # 因为访问 param.grad 会增加底层 Tensor 的 use_count
                        # 可能导致 swap_tensors 失败
                        param.grad = None
                        
                    # 把转换后的 tensor 重新包装成 Parameter，
                	# 并保持原来的 requires_grad 设置。
                    param_applied = torch.nn.Parameter(
                        param_applied,
                        requires_grad=param.requires_grad,
                    )
                    
                    # 交换 param 和 param_applied 的底层 tensor 内容。
                	# 这样可以尽量保持原 Parameter 对象身份不变。
                    torch.utils.swap_tensors(param, param_applied)
                except Exception as e:
                    # swap 失败则把原来的 grad 恢复
                    if param_grad is not None:
                        param.grad = param_grad
                    raise RuntimeError(
                        f"_apply(): Couldn't swap {self._get_name()}.{key}"
                    ) from e
                out_param = param
            elif p_should_use_set_data:
                # 情况 2 旧行为，直接替换 param.data
                # 只替换数据，保留原来的 Parameter 对象
                param.data = param_applied
                out_param = param
            else:
                
                # 情况 3 创建一个新的 Parameter 替换旧 Paramenter
                if not isinstance(param, Parameter):
                    raise AssertionError("param must be a Parameter")
                if not param.is_leaf:
                    raise AssertionError("param must be a leaf tensor")

                # 用转换后的 tensor 创建新的 Parameter，
            	# 并保留原来的 requires_grad。
                out_param = Parameter(param_applied, param.requires_grad)
                self._parameters[key] = out_param

            # 如果原参数有 grad，也要对 grad 执行同样的转换。
        	# 例如参数从 CPU 到 CUDA，grad 也必须从 CPU 到 CUDA。    
            if param_grad is not None:
                with torch.no_grad():
                    grad_applied = fn(param_grad)
                    
                # 判断 grad 是否可以使用 .data 替换。
                g_should_use_set_data = compute_should_use_set_data(
                    param_grad, grad_applied
                )
                if p_should_use_swap_tensors:
                    # 如果参数本身用了 swap_tensors，
                	# 那么 grad 也尽量用 swap_tensors。
                    grad_applied.requires_grad_(param_grad.requires_grad)
                    try:
                        torch.utils.swap_tensors(param_grad, grad_applied)
                    except Exception as e:
                        raise RuntimeError(
                            f"_apply(): Couldn't swap {self._get_name()}.{key}.grad"
                        ) from e
                        
                    # 把转换后的 grad 放回参数。    
                    out_param.grad = param_grad
                    
                elif g_should_use_set_data:
                    # 如果可以沿用旧方式，就直接替换 grad.data。
                    if out_param.grad is None:
                        raise AssertionError("out_param.grad must not be None")
                    out_param.grad.data = grad_applied
                    
                else:
                    # 否则直接设置新的 grad
                    if not param_grad.is_leaf:
                        raise AssertionError("param_grad must be a leaf tensor")
                        
                    # 保留原 grad 的 requires_grad 设置。
                    out_param.grad = grad_applied.requires_grad_(
                        param_grad.requires_grad
                    )

        # 处理 buffer
        # buffer 是模型状态，但不是可训练参数
        # 例如 BatchNorm 的 running_mean、running_var、num_batches_tracked
        for key, buf in self._buffers.items():
            if buf is not None:
                # 对 buffer 也应用同样的转换函数
                # 例如 model.cuda() 时，running_mean 也要转到 CUDA
                self._buffers[key] = fn(buf)
		
        # 支持链式调用
        return self
```

总结这个流程的核心就是：

```
递归 children
    ↓
转换 parameter
    ↓
转换 parameter.grad
    ↓
转换 buffer
    ↓
返回 self
```



#### apply 函数

```nn.Module``` 还实现了一个 apply 函数，与 _apply() 函数不同的是，apply 函数只是简单地递归调用了 ```self.children()``` 去处理自己及其子模块

```python
def apply(self: T, fn: Callable[['Module'], None]) -> T:
    for module in self.children():
        module.apply(fn)
    fn(self)
    return self
```

从外形上：

- ```_apply``` 单前导下划线，一般是 python 对类的私有接口，仅对内部使用
- ```apply``` 是公有的

从作用上：

- ```apply``` 作用于 Module 本身，递归访问每个子模块，常用于初始化
- ```_apply``` 作用于 Tensor：parameter / grad / buffer，递归转换参数和 buffer

例如初始化所有 Linear 层：

```python
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

def init_module(m): # m for model
    if isinstance(m, nn.Linear):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

model.apply(init_module)
```



### 属性的增删改查

#### 属性设置

Pytorch 并不是简单的将变量存储在 ```__dict__``` 里，而是根据对象类型，把它注册到不同的内部字典中。Module 内部三大核心字典

- self._modules

  ```python
  self.conv = nn.Conv2d(...)
  # then
  self._modules["conv"] = Conv2d(...)
  ```

- self._parameters

  可训练参数

  ```python
  self.weight = nn.Parameter(...)
  # Then
  self._parameters["weight"] = Parameter(...)
  ```

- self._buffers

  非训练状态 Tensor

  例如 BatchNorm 的 running_mean, running_var



有三个注册函数提供对这些属性的修改：

```add_module```

```python
def add_module(self, name: str, module: Optional["Module"]) -> None:
        
    if not isinstance(module, Module) and module is not None:
        raise TypeError(f"{torch.typename(module)} is not a Module subclass")
    elif not isinstance(name, str):
        raise TypeError(
            f"module name should be a string. Got {torch.typename(name)}"
        )
    elif hasattr(self, name) and name not in self._modules:
        raise KeyError(f"attribute '{name}' already exists")
    elif "." in name:
        raise KeyError(f'module name can\'t contain ".", got: {name}')
    elif name == "":
        raise KeyError('module name can\'t be empty string ""')
        
    # 新版 PyTorch 这里可能会执行全局 module registration hook
    # 用于调试、框架扩展等场景        
    for hook in _global_module_registration_hooks.values():
        output = hook(self, name, module)
        if output is not None:
            module = output
    self._modules[name] = module
```

注意，其实 ```add_module``` 也有一个和其它两种注册方法对齐的别名：

```python
def register_module(self, name: str, module: Optional["Module"]) -> None:
    r"""Alias for :func:`add_module`."""
    self.add_module(name, module)
```



```register_buffer```

```python
def register_buffer(
        self, name: str, tensor: Tensor | None, persistent: bool = True
    ) -> None:
        
    if persistent is False and isinstance(self, torch.jit.ScriptModule):
        raise RuntimeError("ScriptModule does not support non-persistent buffers")

    # 检查是否已经调用 super().__init__ 进行初始化    
    if "_buffers" not in self.__dict__:
        raise AttributeError("cannot assign buffer before Module.__init__() call")
    elif not isinstance(name, str):
        raise TypeError(
            f"buffer name should be a string. Got {torch.typename(name)}"
        )
    elif "." in name:
        raise KeyError('buffer name can\'t contain "."')
    elif name == "":
        raise KeyError('buffer name can\'t be empty string ""')
    elif hasattr(self, name) and name not in self._buffers:
        raise KeyError(f"attribute '{name}' already exists")
    elif tensor is not None and not (
        isinstance(tensor, torch.Tensor) or hasattr(tensor, "__torch_function__")
    ):
        raise TypeError(
            f"cannot assign '{torch.typename(tensor)}' object to buffer '{name}' "
            "(torch Tensor or None required)"
        )
    else:
        
        # 新版 PyTorch 这里可能会执行全局 buffer registration hook
        for hook in _global_buffer_registration_hooks.values():
            output = hook(self, name, tensor)
            if output is not None:
                tensor = output
        self._buffers[name] = tensor
        
        # 进入 state_dict 的 buffer 会随着模型checkpoint 保存/加载
        # 不进入的 buffer 只是运行时临时状态，不会被保存
        if persistent:
            # persistent=True：
            # buffer 会进入 state_dict，被保存
            self._non_persistent_buffers_set.discard(name)
        else:
            # persistent=False：
            # buffer 不进入 state_dict
            self._non_persistent_buffers_set.add(name)
```

```self.register_buffer``` 是给模块添加 buffer 的唯一方式



```register_parameter```

```python
def register_parameter(self, name: str, param: Parameter | None) -> None:

    # 检查是否已经调用 super().__init__ 进行字典初始化
    if "_parameters" not in self.__dict__:
        raise AttributeError(
            "cannot assign parameter before Module.__init__() call"
        )

    elif not isinstance(name, str):
        raise TypeError(
            f"parameter name should be a string. Got {torch.typename(name)}"
        )
    elif "." in name:
        raise KeyError('parameter name can\'t contain "."')
    elif name == "":
        raise KeyError('parameter name can\'t be empty string ""')
    elif hasattr(self, name) and name not in self._parameters:
        raise KeyError(f"attribute '{name}' already exists")

    if param is None:
        # 可以注册 None
        # 常见于 bias=False 的情况
        self._parameters[name] = None
    elif not isinstance(param, Parameter):
        raise TypeError(
            f"cannot assign '{torch.typename(param)}' object to parameter '{name}' "
            "(torch.nn.Parameter or None required)"
        )
    elif param.grad_fn:
        # Parameter 必须是 leaf tensor
        # 不能是某个计算结果，例如 torch.randn(3) * 2
        # 换句话说就是必须是计算图的根部可训练参数
        raise ValueError(
            f"Cannot assign non-leaf Tensor to parameter '{name}'. Model "
            f"parameters must be created explicitly. To express '{name}' "
            "as a function of another Tensor, compute the value in "
            "the forward() method."
        )
    else:
        
        # 新版 PyTorch 这里可能会执行全局 parameter registration hook
        for hook in _global_parameter_registration_hooks.values():
            output = hook(self, name, param)
            if output is not None:
                param = output
        self._parameters[name] = param
```



日常开发过程中，更常见的用法时直接 ```self.xxx = xxx``` 的方式来增加或修改子模块，parameters，buffers以及一些其它的 attribute（初始化Module对象的时候也是如此），源码有：

```python
def __setattr__(self, name: str, value: Union[Tensor, "Module"]) -> None:
    # __setattr__ 会拦截 self.xxx = value 这种赋值操作
    # PyTorch 用它来自动判断 value 是 Parameter、Module、Buffer 还是普通属性

    def remove_from(*dicts_or_sets) -> None:
        # 如果同一个 name 曾经注册在其他容器里，需要先删除
        # 例如原来 self.x 是 Module，现在重新赋值为 Parameter，
        # 就要从 _modules 里移除 x
        for d in dicts_or_sets:
            if name in d:
                if isinstance(d, dict):
                    del d[name]
                else:
                    d.discard(name)

    # 取出当前 module 的 _parameters 字典
    # 如果还没调用 Module.__init__()，这里会是 None
    params = self.__dict__.get("_parameters")

    if isinstance(value, Parameter):
        # 情况 1：赋值的是 nn.Parameter
        # 例如 self.weight = nn.Parameter(...)

        if params is None:
            # 说明 super().__init__() 还没调用，不能注册 parameter
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call"
            )

        # 从普通属性、buffer、module 等位置删除同名对象，避免重复注册
        remove_from(
            self.__dict__,
            self._buffers,
            self._modules,
            self._non_persistent_buffers_set,
        )

        # 注册到 self._parameters[name]
        self.register_parameter(name, value)

    elif params is not None and name in params:
        # 情况 2：这个 name 已经是一个 parameter
        # 此时只允许重新赋值为 None，不允许赋值成普通 Tensor 或其他类型

        if value is not None:
            raise TypeError(
                f"cannot assign '{torch.typename(value)}' as parameter '{name}' "
                "(torch.nn.Parameter or None expected)"
            )

        # 用 None 替换已有 parameter
        self.register_parameter(name, value)

    else:
        # 取出当前 module 的 _modules 字典
        modules = self.__dict__.get("_modules")

        if isinstance(value, Module):
            # 情况 3：赋值的是 nn.Module
            # 例如 self.conv = nn.Conv2d(...)

            if modules is None:
                # 说明 super().__init__() 还没调用
                raise AttributeError(
                    "cannot assign module before Module.__init__() call"
                )

            # 从普通属性、parameter、buffer 等位置删除同名对象
            remove_from(
                self.__dict__,
                self._parameters,
                self._buffers,
                self._non_persistent_buffers_set,
            )

            # 新版 PyTorch 支持全局 module registration hook
            # 普通使用时可以先忽略
            for hook in _global_module_registration_hooks.values():
                output = hook(self, name, value)
                if output is not None:
                    value = output

            # 注册到 self._modules[name]
            modules[name] = value

        elif modules is not None and name in modules:
            # 情况 4：这个 name 已经是一个子模块
            # 此时只允许重新赋值为 None，不允许赋值为其他类型

            if value is not None:
                raise TypeError(
                    f"cannot assign '{torch.typename(value)}' as child module '{name}' "
                    "(torch.nn.Module or None expected)"
                )

            # 同样走全局 hook
            for hook in _global_module_registration_hooks.values():
                output = hook(self, name, value)
                if output is not None:
                    value = output

            # 用 None 替换已有 module
            modules[name] = value

        else:
            # 取出当前 module 的 _buffers 字典
            buffers = self.__dict__.get("_buffers")

            if isinstance(value, Buffer) or buffers is not None and name in buffers:
                # 情况 5：赋值的是 Buffer，或者这个 name 原本已经是 buffer
                # Buffer 是新版 PyTorch 引入的显式 buffer 类型
                # 旧写法一般是 self.register_buffer(...)

                if value is not None and not (
                    isinstance(value, torch.Tensor)
                    or hasattr(value, "__torch_function__")
                ):
                    # buffer 必须是 Tensor、Buffer、支持 torch function 的对象，或者 None
                    raise TypeError(
                        f"cannot assign '{torch.typename(value)}' as buffer '{name}' "
                        "(torch.nn.Buffer, torch.Tensor or None expected)"
                    )

                if isinstance(value, Buffer):
                    # 如果 value 是显式 Buffer 对象，
                    # persistent 信息从 value.persistent 读取
                    persistent = value.persistent
                else:
                    # 否则根据 _non_persistent_buffers_set 判断它是否 persistent
                    persistent = name not in self._non_persistent_buffers_set

                if (
                    getattr(self.register_buffer, "__func__", None)
                    is torch.nn.Module.register_buffer
                ):
                    # 如果 register_buffer 没被子类重写，
                    # 直接调用标准 register_buffer
                    self.register_buffer(name, value, persistent)

                else:
                    # 如果子类重写了 register_buffer，
                    # 需要检查它是否支持 persistent 参数
                    sign = inspect.signature(self.register_buffer)

                    if "persistent" in sign.parameters:
                        self.register_buffer(name, value, persistent)
                    else:
                        if not persistent:
                            # 子类 register_buffer 不支持 persistent，
                            # 就不能注册 non-persistent buffer
                            raise RuntimeError(
                                "Registering a non-persistent buffer "
                                "on a Module subclass that implements "
                                "register_buffer() without the persistent "
                                "argument is not allowed."
                            )

                        self.register_buffer(name, value)

            else:
                # 情况 6：普通 Python 属性
                # 例如 self.name = "resnet"
                # 或 self.num_layers = 50
                super().__setattr__(name, value)
```

- 在新增 ```self._parameters```，```self._modules``` 时，会先调用 ```remove_from``` 函数，从其余的私有属性中删除对应的 name，这说明 ```self.dict```，```self._buffers```，```self._parameters```，```self._modules ``` 中的属性应该是互斥的
- 除了其他普通的 attribute，**最终 parameters 还是会在 `__setattr__` 中通过 register_parameter 来增加**，但是子神经网络模块和 buffer 是直接修改的 ```self._modules``` 和 ```self._buffers```
- ==self.xxx = torch.Tensor() 是一种不被推荐的行为，因为这样新增的 attribute 既不属于 ```self._parameters```，也不属于 ```self._buffers```== 而是被视作普通的 attribute，那么进行状态转换（如model.cuda(), model.to(...)）的时候会被遗漏而出现device，type 不匹配的 bug



#### 属性删除

属性的删除通过重载 ```__delattr__``` 来实现

```python
def __delattr__(self, name) -> None:
    if name in self._parameters:
        del self._parameters[name]
    elif name in self._buffers:
        del self._buffers[name]
        self._non_persistent_buffers_set.discard(name)
    elif name in self._modules:
        del self._modules[name]
    else:
        super().__delattr__(name)
```

```__delattr__``` 会挨个检查 ```self._parameters``` ，```self._buffers```, ```self._modules```  和普通的 attributes 并将 name 从中删除



#### 常见的属性访问

```nn.Module``` 中的常用函数包括下面8个， 他们都会返回一个迭代器用于访问模块中的 buffer，parameter，子模块等

```python
def _named_members(
    self, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True
):
    # 通用的“命名成员遍历器”。
    # parameters / buffers 都会复用这个函数。

    memo = set()
    # memo 用来去重，避免同一个 Parameter / Buffer 被重复返回。

    modules = (
        self.named_modules(prefix=prefix, remove_duplicate=remove_duplicate)
        if recurse
        else [(prefix, self)]
    )
    # 如果 recurse=True，递归遍历当前 module 及所有子 module。
    # 如果 recurse=False，只处理当前 module 自己。

    for module_prefix, module in modules:
        # module_prefix 是模块名前缀，例如 "layer1.conv1"
        # module 是对应的 Module 对象。

        members = get_members_fn(module)
        # 取出当前 module 的成员。
        # 例如 module._parameters.items() 或 module._buffers.items()

        for k, v in members:
            # k 是成员名，例如 "weight"
            # v 是成员对象，例如 Parameter 或 Buffer Tensor。

            if v is None or v in memo:
                # None 不返回。
                # 已经见过的对象也不重复返回。
                continue

            if remove_duplicate:
                memo.add(v)
                # 记录已经返回过的对象。

            name = module_prefix + ("." if module_prefix else "") + k
            # 拼接完整名字。
            # 例如 module_prefix="layer1.0", k="weight"
            # 得到 "layer1.0.weight"

            yield name, v
            # 返回完整名字和对应成员。


def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    # 返回模型中的 Parameter，不带名字。
    # 常用于 optimizer = SGD(model.parameters(), ...)

    for _name, param in self.named_parameters(recurse=recurse):
        yield param
        # 直接复用 named_parameters，只丢掉名字。


def named_parameters(
    self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
) -> Iterator[tuple[str, Parameter]]:
    # 返回模型中的 Parameter，带完整名字。
    # 例如 "layer1.0.conv.weight"

    gen = self._named_members(
        lambda module: module._parameters.items(),
        # 对每个 module，取它自己的 _parameters。

        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    )

    yield from gen
    # 把 _named_members 生成的内容继续 yield 出去。


def buffers(self, recurse: bool = True) -> Iterator[Tensor]:
    # 返回模型中的 buffer，不带名字。
    # 例如 BatchNorm.running_mean / running_var。

    for _, buf in self.named_buffers(recurse=recurse):
        yield buf
        # 复用 named_buffers，只丢掉名字。


def named_buffers(
    self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
) -> Iterator[tuple[str, Tensor]]:
    # 返回模型中的 buffer，带完整名字。

    gen = self._named_members(
        lambda module: module._buffers.items(),
        # 对每个 module，取它自己的 _buffers。

        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    )

    yield from gen


def children(self) -> Iterator["Module"]:
    # 返回当前 module 的直接子模块，不递归。
    # 不返回名字，只返回 module 对象。

    for _name, module in self.named_children():
        yield module
        # 复用 named_children，只丢掉名字。


def named_children(self) -> Iterator[tuple[str, "Module"]]:
    # 返回当前 module 的直接子模块，带名字。
    # 只看 self._modules 的第一层，不递归。

    memo = set()
    # 用来避免同一个 module 对象被重复返回。

    for name, module in self._modules.items():
        # self._modules 保存当前 module 直接注册的子模块。

        if module is not None and module not in memo:
            # 跳过 None。
            # 跳过重复引用的 module。

            memo.add(module)
            yield name, module
            # 返回直接子模块名和子模块对象。


def modules(self, remove_duplicate: bool = True) -> Iterator["Module"]:
    # 返回当前 module 及其所有子模块，不带名字。
    # 是递归遍历。

    for _, module in self.named_modules(remove_duplicate=remove_duplicate):
        yield module
        # 复用 named_modules，只丢掉名字。


def named_modules(
    self,
    memo: set["Module"] | None = None,
    prefix: str = "",
    remove_duplicate: bool = True,
):
    # 返回当前 module 及其所有子模块，带完整名字。
    # 例如 "", "layer1", "layer1.0", "layer1.0.conv"

    if memo is None:
        memo = set()
        # memo 用来去重，避免共享 module 被重复返回。

    if self not in memo:
        # 如果当前 module 没被访问过，才返回。

        if remove_duplicate:
            memo.add(self)
            # 记录当前 module，防止重复返回。

        yield prefix, self
        # 先返回当前 module 自身。
        # 最外层模型的 prefix 通常是 ""。

        for name, module in self._modules.items():
            # 遍历当前 module 的直接子模块。

            if module is None:
                continue
                # 跳过 None 子模块。

            submodule_prefix = prefix + ("." if prefix else "") + name
            # 拼接子模块完整名字。

            yield from module.named_modules(
                memo, submodule_prefix, remove_duplicate
            )
            # 递归进入子模块，继续返回更深层的 module。
```

| 函数                 | 功能                                                         |
| -------------------- | ------------------------------------------------------------ |
| `_named_members()`   | 通用遍历工具，被 `named_parameters()` 和 `named_buffers()` 复用 |
| `parameters()`       | 返回所有参数，不带名字                                       |
| `named_parameters()` | 返回所有参数，带名字                                         |
| `buffers()`          | 返回所有 buffer，不带名字                                    |
| `named_buffers()`    | 返回所有 buffer，带名字                                      |
| `children()`         | 返回当前模块的**直接子模块**，不递归，不带名字               |
| `named_children()`   | 返回当前模块的**直接子模块**，不递归，带名字                 |
| `modules()`          | 返回当前模块及所有子模块，递归，不带名字                     |
| `named_modules()`    | 返回当前模块及所有子模块，递归，带名字                       |



```nn.Module``` 重载了 ```__dir__``` 函数，将 ```self._modules, self._parameters, self._buffers``` 中的 attributes 给暴露出来

```python
def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        keys = module_attrs + attrs + parameters + modules + buffers

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)
```



还有一种常见的属性访问时通过 module.attribute 来进行的。这种调用等价于 ```getattr(module, 'atribute')``` 。在 nn.Module 中，我们常使用 ```model.conv``` 调用，其实等价于 ```getattr(model, "conv")```，但是，很多属性并不直接放在普通 ```__dict__``` 里，而是放在 ```self._parameters, self._buffers, self._module```。所以 nn.Module 也重载了 ```__getattr__```

```python
def __getattr__(self, name: str) -> Union[Tensor, "Module"]:
        if "_parameters" in self.__dict__:
            _parameters = self.__dict__["_parameters"]
            if name in _parameters:
                return _parameters[name]
        if "_buffers" in self.__dict__:
            _buffers = self.__dict__["_buffers"]
            if name in _buffers:
                return _buffers[name]
        if "_modules" in self.__dict__:
            modules = self.__dict__["_modules"]
            if name in modules:
                return modules[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )

```

于是查找顺序变为：

1. 类及父类定义的 属性/方法
2. 实例自己的 ```__dict__```
3. 都找不到就调用 ```__getattr__```



### Forward & Backward

#### Hooks

类似 ```torch.autograd``` 中的 hook，在 Forward & Backward 里引入 hook 机制的作用是： **在前向和反向传播的过程中插入自定义函数，用来观察或修改中间信息**。而 ```nn.Module``` 中的 hook 有 **全局 hook** 和 **局部 hook**

- 全局 hook

  注册在整个 ```nn.Module``` 系统上，只要模型某个 module 执行对应的前向/反向传播，这个全局 hook 都可能被触发。被触发时 hook 会修改与其对应的 OrderedDict。

  | 注册函数                           | 保存位置                    |
  | ---------------------------------- | --------------------------- |
  | `register_module_backward_hook`    | `_global_backward_hooks`    |
  | `register_module_forward_pre_hook` | `_global_forward_pre_hooks` |
  | `register_module_forward_hook`     | `_global_forward_hooks`     |

  以 ```register_module_forward_hook``` 源码为例

  ```python
  def register_module_forward_hook(
      hook: Callable[..., None],
      *,
      with_kwargs: bool = False,
      always_call: bool = False,
  ) -> RemovableHandle:
      # 注册一个“全局 forward hook”。
      # 这个 hook 会作用于所有 nn.Module 实例的 forward 过程。
  
      handle = RemovableHandle(
          _global_forward_hooks, 
          extra_dict=_global_forward_hooks_always_called
      )
      # 创建一个可移除的 handle。
      # 之后可以通过 handle.remove() 删除这个 hook。
      # _global_forward_hooks 是真正保存 hook 的全局 OrderedDict。
      # extra_dict 表示 remove 时也要同步清理额外字典。
  
      _global_forward_hooks[handle.id] = hook
      # 把 hook 保存到全局 forward hook 字典中。
      # key 是 handle.id，value 是用户传入的 hook 函数。
  
      if with_kwargs:
          _global_forward_hooks_with_kwargs[handle.id] = True
          # 如果 with_kwargs=True，
          # 表示这个 hook 可以接收 forward 中的 keyword arguments。
  
      if always_call:
          _global_forward_hooks_always_called[handle.id] = True
          # 如果 always_call=True，
          # 即使 forward 过程中抛出异常，也尽量调用这个 hook。
          # 常用于调试或清理逻辑。
  
      return handle
      # 返回 handle，方便之后 remove：
      # handle.remove()
  ```

  例子：

  ```python
  import torch
  import torch.nn as nn
  from torch.nn.modules.module import (
      register_module_forward_pre_hook,
      register_module_forward_hook,
      register_module_backward_hook,  
      # 旧接口，不推荐新代码继续使用
      # 新接口名为 global_full_backward_hook
  )
  
  model = nn.Sequential(
      nn.Linear(2, 2),
      nn.ReLU(),
      nn.Linear(2, 1)
  )
  
  def global_forward_pre_hook(module, inputs):
      print(f"[forward_pre] {module.__class__.__name__}")
      print("  input:", inputs)
  
  def global_forward_hook(module, inputs, output):
      print(f"[forward] {module.__class__.__name__}")
      print("  output:", output)
  
  def global_backward_hook(module, grad_input, grad_output):
      print(f"[backward] {module.__class__.__name__}")
      print("  grad_input:", grad_input)
      print("  grad_output:", grad_output)
  
  h1 = register_module_forward_pre_hook(global_forward_pre_hook)
  h2 = register_module_forward_hook(global_forward_hook)
  h3 = register_module_backward_hook(global_backward_hook)
  
  x = torch.randn(1, 2, requires_grad=True)
  
  y = model(x)
  loss = y.sum()
  
  print("\nloss:", loss)
  
  loss.backward()
  
  h1.remove()
  h2.remove()
  h3.remove()
  ```

  值得注意的是 旧的接口 `register_module_backward_hook` 和新的接口 `register_module_full_backward_hook` 都可以使用 （pytorch 2.12.0）但他们有如下区别

  - 旧接口 `register_module_backward_hook`

    旧 hook 是挂在 module 输出的某些 `grad_fn` 上。问题在于，如果一个module 的 forward 内部有多个 autograd 节点，或者输出/输入结构比较复杂，它可能只在其中一部分梯度流上触发，因此，**拿到的 `grad_input/grad_output` 不一定完整**

  - 新接口 `register_module_full_backward_hook`

    等 module 的输入/输出相关梯度准备好后再触发 hook

- 局部 hook

  只注册在某一个具体的 module 实例上，而不是作用于所有 module。和全局 hook 类似的，也是通过3个函数来管理自己的3个属性并维护3个 attribute，他们的类型也是 OrderedDict

  | 注册函数名                      | 保存位置                  |
  | ------------------------------- | ------------------------- |
  | `register_forward_pre_hook()`   | `self._forward_pre_hooks` |
  | `register_forward_hook()`       | `self._forward_hooks`     |
  | `register_full_backward_hook()` | `self._backward_hooks`    |

  一个 local hook 的示例：

  ```python
  import torch
  import torch.nn as nn
  
  model = nn.Sequential(
      nn.Linear(2, 2),
      nn.ReLU(),
      nn.Linear(2, 1)
  )
  
  # 只 hook 第一个 Linear
  layer = model[0]
  
  def forward_pre_hook(module, inputs):
      print("[pre]")
      print(module)
      print("input:", inputs)
  
  def forward_hook(module, inputs, output):
      print("[forward]")
      print(module)
      print("output:", output)
  
  def backward_hook(module, grad_input, grad_output):
      print("[backward]")
      print(module)
      print("grad_input:", grad_input)
      print("grad_output:", grad_output)
  
  h1 = layer.register_forward_pre_hook(forward_pre_hook)
  
  h2 = layer.register_forward_hook(forward_hook)
  
  h3 = layer.register_full_backward_hook(backward_hook)
  
  x = torch.randn(1, 2, requires_grad=True)
  
  y = model(x)
  
  loss = y.sum()
  
  loss.backward()
  
  h1.remove()
  h2.remove()
  h3.remove()
  ```

  与全局 hook 的新旧接口类似， `register_backward_hook` 也是旧接口了 ，`register_full_backward_hook` 是更新的，更推荐的接口 （pytorch 3.12.0）



#### 运行逻辑

通常 `nn.Module` 在被调用的时候，一般是以 `module(input)` 的形式

```
module(input)
    |
    v
self.__call__()
    |
    v
self._call_impl()
    |
    v
_global_forward_pre_hooks
    |
    v
self._forward_pre_hooks
    |
    v
self.forward(...) / self._slow_forward(...)
    |
    v
_global_forward_hooks
    |
    v
self._forward_hooks
    |
    v
注册/挂载 backward hooks 到 autograd graph
    |
    v
返回 output
    |
    v
loss.backward()
    |
    v
触发 _global_backward_hooks / self._backward_hooks
```



`_call_impl` 的代码实现如下

```python
def _call_impl(self, *args, **kwargs):
    # 1. 决定真正要调用的是 forward 还是 _slow_forward
    # tracing / JIT 状态下走 _slow_forward，否则走 self.forward
    forward_call = (
        self._slow_forward 
        if torch._C._get_tracing_state() 
        else self.forward
    )

    # 2. 如果没有任何 hook，直接调用 forward，走最快路径
    if not (
        self._backward_hooks
        or self._backward_pre_hooks
        or self._forward_hooks
        or self._forward_pre_hooks
        or _global_backward_pre_hooks
        or _global_backward_hooks
        or _global_forward_hooks
        or _global_forward_pre_hooks
    ):
        return forward_call(*args, **kwargs)

    result = None
    called_always_called_hooks = set()

    def inner():
        nonlocal result, args, kwargs

        # 3. 收集 backward hooks
        full_backward_hooks, non_full_backward_hooks = [], []
        backward_pre_hooks = []

        if self._backward_pre_hooks or _global_backward_pre_hooks:
            backward_pre_hooks = self._get_backward_pre_hooks()

        if self._backward_hooks or _global_backward_hooks:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        # 4. 执行 forward_pre_hooks
        # 顺序：global forward pre hooks -> local forward pre hooks
        if _global_forward_pre_hooks or self._forward_pre_hooks:
            for hook_id, hook in (
                *_global_forward_pre_hooks.items(),
                *self._forward_pre_hooks.items(),
            ):
                if hook_id in self._forward_pre_hooks_with_kwargs:
                    # with_kwargs=True 的 pre hook:
                    # hook(self, args, kwargs) -> (new_args, new_kwargs)
                    args, kwargs = hook(self, args, kwargs)
                else:
                    # 普通 pre hook:
                    # hook(self, args) -> new_args 或 None
                    new_args = hook(self, args)
                    if new_args is not None:
                        if not isinstance(new_args, tuple):
                            new_args = (new_args,)
                        args = new_args

        # 5. 如果有 full backward hook / backward pre hook，
        # 先创建 BackwardHook 对象，并给输入挂 hook
        bw_hook = None
        if full_backward_hooks or backward_pre_hooks:
            bw_hook = BackwardHook(
                self,
                full_backward_hooks,
                backward_pre_hooks
            )
            args = bw_hook.setup_input_hook(args)

        # 6. 真正执行 forward
        result = forward_call(*args, **kwargs)

        # 7. 执行 forward_hooks
        # 顺序：global forward hooks -> local forward hooks
        if _global_forward_hooks or self._forward_hooks:
            for hook_id, hook in (
                *_global_forward_hooks.items(),
                *self._forward_hooks.items(),
            ):
                if hook_id in self._forward_hooks_always_called:
                    called_always_called_hooks.add(hook_id)

                if hook_id in self._forward_hooks_with_kwargs:
                    # with_kwargs=True:
                    # hook(self, args, kwargs, result)
                    hook_result = hook(self, args, kwargs, result)
                else:
                    # 普通 forward hook:
                    # hook(self, args, result)
                    hook_result = hook(self, args, result)

                # forward hook 可以修改输出
                if hook_result is not None:
                    result = hook_result

        # 8. 如果使用 full backward hook，
        # 给输出挂 hook，使 backward 时能触发 full backward hooks
        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # 9. 如果使用旧版 non-full backward hook，
        # 找到 result 中的一个 Tensor，然后把 hook 挂到 grad_fn 上
        if non_full_backward_hooks:
            var = result

            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next(
                        v for v in var.values()
                        if isinstance(v, torch.Tensor)
                    )
                else:
                    var = var[0]

            grad_fn = var.grad_fn

            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    grad_fn.register_hook(
                        _WrappedHook(hook, self)
                    )

        return result

    # 10. torch.compile 正在编译时，直接执行 inner
    if torch.compiler.is_compiling():
        return inner()

    try:
        # 11. 正常执行完整调用流程
        return inner()

    except Exception:
        # 12. 如果 forward 报错，仍然尝试执行 always_call=True 的 forward hooks
        # 主要用于调试、日志、资源清理
        ...
        raise


# 新版中 __call__ 指向 _wrapped_call_impl，
# 而 _wrapped_call_impl 内部最终会走 _call_impl。
__call__: Callable[..., Any] = _wrapped_call_impl
```



### 模块存取

#### Hooks

`nn.Module`  还有两个相关的 hook 是关于模型参数的加载和储存的

- `_register_state_dict_hook`

  将 hook 存入 `self._state_dict_hooks`，在 `model.state_dict()` 生成模型状态字典时，插入一个自定义函数，对保存过程进行修改或扩展

  ```python
  def _register_state_dict_hook(self, hook):
  
      # 检查 hook 是否已经通过公开接口注册过
      if getattr(hook, "_from_public_api", False):
          raise RuntimeError(
              "Cannot register the same function as the state dict post hook that was "
              "previously registered via register_state_dict_post_hook"
          )
      handle = RemovableHandle(self._state_dict_hooks)
      self._state_dict_hooks[handle.id] = hook
      return handle
  ```

  

- `_register_load_state_dict_pre_hook`

  注册 `load_state_dict` 前置 hook，在执行 `model.load_state_dict(state_dict)` 真正加载参数之前，先调用用户自定义函数，对 `state_dict` 做检查，修改或兼容处理

  ```python
  def _register_load_state_dict_pre_hook(self, hook, with_module=False):
         
      handle = RemovableHandle(self._load_state_dict_pre_hooks)
      self._load_state_dict_pre_hooks[handle.id] = _WrappedHook(
          hook, self if with_module else None
      )
      return handle
  ```



#### `state_dict()` 和 `load_state_dict()` 的底层机制

- `state_dict()`

  总的来说 **`state_dict()` 本质是递归遍历整个 module 树，把每个模块自己的 parameters 和 persistent buffers 按层级名字收集到同一个字典里。**

  ```python
  import torch
  import torch.nn as nn
  
  class MyModel(nn.Module):
      def __init__(self):
          super().__init__()
  
          # parameter：会进入 state_dict
          self.linear = nn.Linear(2, 1)
  
          # persistent buffer：会进入 state_dict
          self.register_buffer("running_value", torch.ones(1), persistent=True)
  
          # non-persistent buffer：不会进入 state_dict
          self.register_buffer("temp_cache", torch.zeros(1), persistent=False)
  
  model = MyModel()
  
  sd = model.state_dict()
  
  print(sd)
  
  # OrderedDict([
  # ('running_value', tensor([1.])), 
  # ('linear.weight', tensor([[-0.1089,  0.3872]])), 
  # ('linear.bias', tensor([0.3867]))
  # ])
  ```

  来到其源码

  ```python
  def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
      # state_dict 用来导出当前 module 及其所有子模块的状态。
      # 主要包括：
      # 1. parameters
      # 2. persistent buffers
      # 最终常用于 torch.save(model.state_dict(), path)
  
      if len(args) > 0:
          # 兼容旧版位置参数写法。
          # 新版推荐使用关键字参数，例如 destination=..., prefix=...
          warnings.warn(
              "Positional args are being deprecated, use kwargs instead. Refer to "
              "https://pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.state_dict"
              " for details.",
              FutureWarning,
              stacklevel=2,
          )
  
          if destination is None:
              destination = args[0]
              # 旧写法中第一个位置参数对应 destination。
  
          if len(args) > 1 and prefix == "":
              prefix = args[1]
              # 旧写法中第二个位置参数对应 prefix。
  
          if len(args) > 2 and keep_vars is False:
              keep_vars = args[2]
              # 旧写法中第三个位置参数对应 keep_vars。
  
      if destination is None:
          destination = OrderedDict()
          # destination 是最终保存所有状态的字典。
  
          destination._metadata = OrderedDict()
          # _metadata 用来保存每个 module 的版本信息。
  
      local_metadata = dict(version=self._version)
      # 当前 module 的版本号。
      # 用于之后 load_state_dict 时做版本兼容处理。
  
      if hasattr(destination, "_metadata"):
          destination._metadata[prefix[:-1]] = local_metadata
          # 把当前 module 的 metadata 保存到 destination._metadata。
          # prefix[:-1] 是当前 module 的名字路径。
          # 例如 prefix="layer1.0."，则 key 是 "layer1.0"。
  
      for hook in self._state_dict_pre_hooks.values():
          hook(self, prefix, keep_vars)
          # state_dict 生成前的 hook。
          # 可以在真正保存前执行一些自定义逻辑。
  
      self._save_to_state_dict(destination, prefix, keep_vars)
      # 保存当前 module 自己的 parameters 和 persistent buffers。
      # 注意：这里只保存当前层，不递归保存子模块。
      # 递归逻辑在下面的 for module in self._modules 中完成。
  
      for name, module in self._modules.items():
          # 遍历当前 module 的直接子模块。
  
          if module is not None:
              module.state_dict(
                  destination=destination,
                  prefix=prefix + name + ".",
                  keep_vars=keep_vars,
              )
              # 递归调用子模块的 state_dict。
              # prefix 会逐层累加。
              # 例如：
              # 当前 prefix=""
              # 子模块 name="layer1"
              # 子模块 prefix="layer1."
              # 最终 key 可能是 "layer1.weight"。
  
      for hook in self._state_dict_hooks.values():
          # state_dict 生成后的 hook。
          # 可以修改最终 destination。
  
          hook_result = hook(self, destination, prefix, local_metadata)
  
          if not getattr(hook, "_from_public_api", False):
              # 内部 API 注册的 hook 允许返回新的 destination。
  
              if hook_result is not None:
                  destination = hook_result
  
          else:
              # 通过公开 API 注册的 post-hook 不允许返回新对象。
              # 它必须原地修改 destination，并返回 None。
  
              if hook_result is not None:
                  raise RuntimeError("state_dict post-hook must return None")
  
      return destination
      # 返回完整 state_dict。
  ```

  我们可以将 `state_dict` 的实现流程总结如下：

  1. 调用 `model.state_dict()`
  2. 创建 `destination = OrderedDict()`
  3. 保存当前 module 的**版本信息** `_version` 到 `metadata`
  4. 执行 `state_dict_pre_hooks`
  5. `_save_to_state_dict()`
     保存当前 module 自己的 parameters 和 persistent buffers
  6. 遍历 `self._modules`
     递归调用每个子模块的 state_dict()
  7. 执行 `state_dict_hooks` / `post_hooks`
  8. 返回完整 destination

  ==用户还可以通过重载 `_save_to_state_dict` 来满足特定需求== ，例如

  ```python
  import torch
  import torch.nn as nn
  
  class MyModule(nn.Module):
      def __init__(self):
          super().__init__()
          self.weight = nn.Parameter(torch.tensor([1.0, 2.0]))
          
          # 普通属性，默认不会进入 state_dict
          self.scale = torch.tensor([10.0])
  
      def _save_to_state_dict(self, destination, prefix, keep_vars):
          # 先调用父类逻辑，保存 parameters 和 persistent buffers
          super()._save_to_state_dict(destination, prefix, keep_vars)
  
          # 再额外保存普通属性 scale
          destination[prefix + "scale"] = self.scale if keep_vars else self.scale.detach()
  
  
  m = MyModule()
  sd = m.state_dict()
  
  print(sd)
  
  # OrderedDict([
  #	('weight', tensor([1., 2.])), 
  #   ('scale', tensor([10.]))
  #]) 
  ```

  `self.scale` 本来是普通 Tensor 属性不会被保存，重载之后也加入到 OrderedDict 中了，**注意！编写重载函数的时候要 `super()._save_to_state_dict`**

  

- `load_from_state_dict`

  真正把 checkpoint 中参数和 buffer 加载回到当前 module，例如

  ```python
  import torch
  import torch.nn as nn
  
  class MyModule(nn.Module):
      def __init__(self):
          super().__init__()
  
          # 普通可训练参数，会自动进入 state_dict
          self.weight = nn.Parameter(torch.tensor([1.0]))
  
          # 普通 Tensor 属性，默认不会进入 state_dict
          self.my_scale = torch.tensor([0.0])
  
      def _save_to_state_dict(self, destination, prefix, keep_vars):
          # 先调用父类逻辑：
          # 保存当前 module 自己的 parameters 和 persistent buffers
          super()._save_to_state_dict(destination, prefix, keep_vars)
  
          # 手动把普通属性 my_scale 保存进 state_dict
          # prefix 用于处理子模块命名，例如 "layer1.scale"
          destination[prefix + "scale"] = (
              self.my_scale if keep_vars else self.my_scale.detach()
          )
  
      def _load_from_state_dict(
          self,
          state_dict,
          prefix,
          local_metadata,
          strict,
          missing_keys,
          unexpected_keys,
          error_msgs,
      ):
          # 自定义 key
          key = prefix + "scale"
  
          # 如果 checkpoint 里有 scale，就手动加载
          if key in state_dict:
              print("loading custom scale...")
  
              # clone 一份，避免直接共享 checkpoint 中的 Tensor 对象
              self.my_scale = state_dict[key].clone()
  
              # 关键：
              # 加载完自定义 key 后，要从 state_dict 中移除
              # 否则 strict=True 时会被认为是 unexpected key
              state_dict.pop(key)
  
          # 再调用父类逻辑：
          # 加载 parameters 和 persistent buffers，例如 weight
          super()._load_from_state_dict(
              state_dict,
              prefix,
              local_metadata,
              strict,
              missing_keys,
              unexpected_keys,
              error_msgs,
          )
  
  
  # 1. 创建模型并修改状态
  m1 = MyModule()
  # 修改 parameter
  m1.weight.data[:] = 999
  # 修改普通 Tensor 属性
  m1.my_scale[:] = 123
  
  # 2. 保存 state_dict
  sd = m1.state_dict()
  
  print("saved state_dict:")
  print(sd)
  
  # 3. 创建新模型
  m2 = MyModule()
  print("\nbefore load:")
  print("weight:", m2.weight)
  print("my_scale:", m2.my_scale)
  
  # 4. 加载 state_dict
  m2.load_state_dict(sd)
  
  # 5. 结果
  print("\nafter load:")
  print("weight:", m2.weight)
  print("my_scale:", m2.my_scale)
  
  # saved state_dict:
  # OrderedDict([('weight', tensor([999.])), ('my_scale', tensor([123.]))])
  
  # before load:
  # weight: Parameter containing:
  # tensor([1.], requires_grad=True)
  # my_scale: tensor([0.])
  # loading custom scale...
  
  # after load:
  # weight: Parameter containing:
  # tensor([999.], requires_grad=True)
  # my_scale: tensor([123.])
  ```

  值得注意的是，因为 `my_scale` 是自定义的普通属性，PyTorch 不能自动识别（因为不属于 `_parameter, _buffer, _submodule`），所以后面加载逻辑检查 `state_dict` 中 `my_scale` 这个key不能被识别，因此在 `strict=True` 时会出现 `Unexpected key(s) in state_dict: "my_scale"` 所以我们需要将其移除 `state_dict.pop(key)`

  

  具体两者的源码实现如下：

  ```python
  def _load_from_state_dict(
      self,
      state_dict,
      prefix,
      local_metadata,
      strict,
      missing_keys,
      unexpected_keys,
      error_msgs,
  ) -> None:
      # 只加载“当前 module 自己”的 parameter 和 persistent buffer。
      # 不递归加载子模块；递归由 load_state_dict() 负责。
  
      for hook in self._load_state_dict_pre_hooks.values():
          # 加载前 hook：
          # 可以在真正加载前修改 state_dict、missing_keys、unexpected_keys 等。
          hook(
              state_dict,
              prefix,
              local_metadata,
              strict,
              missing_keys,
              unexpected_keys,
              error_msgs,
          )
  
      persistent_buffers = {
          k: v
          for k, v in self._buffers.items()
          if k not in self._non_persistent_buffers_set
      }
      # 只保留 persistent=True 的 buffer。
      # persistent=False 的 buffer 不参与 checkpoint 加载。
  
      local_name_params = itertools.chain(
          self._parameters.items(),
          persistent_buffers.items(),
      )
      # 当前 module 需要加载的本地状态：
      # parameters + persistent buffers。
  
      local_state = {k: v for k, v in local_name_params if v is not None}
      # 去掉值为 None 的 parameter / buffer。
  
      assign_to_params_buffers = local_metadata.get("assign_to_params_buffers", False)
      # assign=True 时：
      # 用 checkpoint 里的 Tensor 直接替换当前 module 中的 param/buffer。
      # assign=False 时：
      # 使用 copy_ 把 checkpoint 的值复制到当前 param/buffer 中。
  
      use_swap_tensors = torch.__future__.get_swap_module_params_on_conversion()
      # 新版 PyTorch 的 tensor swap 加载机制。
      # 主要用于 tensor subclass / future 行为兼容。
  
      for name, param in local_state.items():
          # 遍历当前 module 自己的所有 parameter 和 persistent buffer。
  
          key = prefix + name
          # checkpoint 中完整 key。
          # 例如 prefix="layer1.", name="weight" -> key="layer1.weight"
  
          if key in state_dict:
              # checkpoint 中存在当前参数。
  
              input_param = state_dict[key]
              # 从 checkpoint 中取出对应 Tensor。
  
              if not torch.overrides.is_tensor_like(input_param):
                  # checkpoint 中对应值必须是 Tensor-like。
                  error_msgs.append(f"{key}: checkpoint value is not Tensor-like")
                  continue
  
              is_param_lazy = torch.nn.parameter.is_lazy(param)
              # LazyModule 的参数可能还没有真实 shape，因此 shape 检查要特殊处理。
  
              if (
                  not is_param_lazy
                  and len(param.shape) == 0
                  and len(input_param.shape) == 1
                  and input_param.shape[0] == 1
              ):
                  # 历史兼容：
                  # 旧版 PyTorch 可能把 scalar tensor 存成 shape=(1,)。
                  # 新版中 scalar 是 shape=()，这里做兼容转换。
                  input_param = input_param[0]
  
              if not is_param_lazy and input_param.shape != param.shape:
                  # 当前模型中的 shape 必须和 checkpoint 中的 shape 一致。
                  error_msgs.append(f"{key}: shape mismatch")
                  continue
  
              if (
                  param.is_meta
                  and not input_param.is_meta
                  and not assign_to_params_buffers
              ):
                  # 当前参数在 meta device 上，而 checkpoint 参数是真实 Tensor。
                  # 如果 assign=False，copy_ 到 meta 参数其实不会真正 materialize。
                  # 所以这里给出警告。
                  warnings.warn(f"{key}: copying non-meta tensor to meta parameter is no-op")
  
              try:
                  with torch.no_grad():
                      # 加载权重不应该被 autograd 记录，所以使用 no_grad。
  
                      if use_swap_tensors:
                          # 路径 1：使用新版 swap_tensors 机制加载。
                          # param.module_load 可以定义模块自定义加载逻辑。
                          new_input_param = param.module_load(
                              input_param,
                              assign=assign_to_params_buffers,
                          )
  
                          if id(new_input_param) == id(input_param) or id(new_input_param) == id(param):
                              # module_load 不应该直接返回输入本身。
                              raise RuntimeError("module_load returned input itself")
  
                          if isinstance(param, torch.nn.Parameter):
                              # 如果当前对象是 Parameter，
                              # 加载后的对象也需要保持 Parameter 语义。
                              if not isinstance(new_input_param, torch.nn.Parameter):
                                  new_input_param = torch.nn.Parameter(
                                      new_input_param,
                                      requires_grad=param.requires_grad,
                                  )
                              else:
                                  new_input_param.requires_grad_(param.requires_grad)
  
                          # 交换当前 param 和新 param 的底层 tensor。
                          torch.utils.swap_tensors(param, new_input_param)
                          del new_input_param
  
                      elif assign_to_params_buffers:
                          # 路径 2：assign=True。
                          # 直接把 checkpoint 中的 Tensor/Parameter 赋值给当前 module。
                          # 注意：这种方式会改变当前 module 中参数对象本身。
  
                          if isinstance(param, torch.nn.Parameter):
                              if not isinstance(input_param, torch.nn.Parameter):
                                  input_param = torch.nn.Parameter(
                                      input_param,
                                      requires_grad=param.requires_grad,
                                  )
                              else:
                                  input_param.requires_grad_(param.requires_grad)
  
                          setattr(self, name, input_param)
                          # 这里会触发 Module.__setattr__，
                          # 从而把 input_param 重新注册为 parameter 或 buffer。
  
                      else:
                          # 路径 3：默认路径。
                          # 不替换 Parameter 对象，只把 checkpoint 的数值复制进当前对象。
                          param.copy_(input_param)
  
              except Exception as ex:
                  # 加载失败时记录错误信息，最后由 load_state_dict 统一抛出。
                  action = "swapping" if use_swap_tensors else "copying"
                  error_msgs.append(f"{key}: error while {action}: {ex}")
  
          elif strict:
              # strict=True 时，如果当前 module 需要的 key 在 checkpoint 中不存在，
              # 记录为 missing key。
              missing_keys.append(key)
  
      extra_state_key = prefix + _EXTRA_STATE_KEY_SUFFIX
      # extra_state 是 module 额外自定义状态，
      # 通过 get_extra_state / set_extra_state 保存和加载。
  
      if (
          getattr(self.__class__, "set_extra_state", Module.set_extra_state)
          is not Module.set_extra_state
      ):
          # 如果子类重写了 set_extra_state，说明它支持加载 extra_state。
  
          if extra_state_key in state_dict:
              self.set_extra_state(state_dict[extra_state_key])
          elif strict:
              missing_keys.append(extra_state_key)
  
      elif strict and (extra_state_key in state_dict):
          # 如果当前 module 不支持 extra_state，
          # 但 checkpoint 里有 extra_state，则认为是 unexpected key。
          unexpected_keys.append(extra_state_key)
  
      if strict:
          # strict=True 时，检查 checkpoint 里是否有当前 module 不需要的 key。
  
          for key in state_dict:
              if key.startswith(prefix) and key != extra_state_key:
                  # 只检查属于当前 prefix 下的 key。
  
                  input_name = key[len(prefix):].split(".", 1)
                  # 去掉 prefix 后，判断它是当前 module 的本地状态，
                  # 还是某个子模块的状态。
  
                  if len(input_name) > 1:
                      # 说明 key 形如 "child.xxx"。
                      # 如果 child 不是当前 module 的子模块，则 unexpected。
                      if input_name[0] not in self._modules:
                          unexpected_keys.append(key)
  
                  elif input_name[0] not in local_state:
                      # 说明 key 是当前 module 的本地 key。
                      # 如果它不在 local_state 中，则 unexpected。
                      unexpected_keys.append(key)
  ```

  ```python
  def load_state_dict(
      self,
      state_dict: Mapping[str, Any],
      strict: bool = True,
      assign: bool = False,
  ):
      # 加载整个 module 树的 checkpoint。
      # 它负责递归遍历所有子模块；
      # 真正加载每个 module 自己状态的是 _load_from_state_dict()。
  
      if not isinstance(state_dict, Mapping):
          # state_dict 必须是类似 dict 的对象。
          raise TypeError("state_dict must be dict-like")
  
      missing_keys: list[str] = []
      unexpected_keys: list[str] = []
      error_msgs: list[str] = []
      # 这三个列表会在递归加载过程中不断被子模块填充。
      # 最后统一判断是否报错。
  
      metadata = getattr(state_dict, "_metadata", None)
      # state_dict 可能带有 _metadata，
      # 里面记录每个 module 的版本信息。
  
      state_dict = OrderedDict(state_dict)
      # 拷贝一份 state_dict。
      # 因为 _load_from_state_dict() 可能会修改它。
  
      if metadata is not None:
          state_dict._metadata = metadata
          # 保留 metadata。
  
      def load(module, local_state_dict, prefix="") -> None:
          # 递归加载函数。
          # module 是当前要加载的模块。
          # prefix 是当前模块在整个模型中的路径前缀。
  
          local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
          # 取出当前 module 对应的 metadata。
          # prefix="layer1." 时，metadata key 是 "layer1"。
  
          if assign:
              local_metadata["assign_to_params_buffers"] = assign
              # 把 assign 选项传给 _load_from_state_dict。
  
          module._load_from_state_dict(
              local_state_dict,
              prefix,
              local_metadata,
              True,
              missing_keys,
              unexpected_keys,
              error_msgs,
          )
          # 加载当前 module 自己的 parameter 和 persistent buffer。
  
          for name, child in module._modules.items():
              # 递归加载子模块。
  
              if child is not None:
                  child_prefix = prefix + name + "."
                  # 子模块 prefix，例如 "layer1."
  
                  child_state_dict = {
                      k: v
                      for k, v in local_state_dict.items()
                      if k.startswith(child_prefix)
                  }
                  # 只把属于该子模块的 key 传给子模块。
                  # 例如 child_prefix="layer1."，只保留 "layer1.xxx"。
  
                  load(child, child_state_dict, child_prefix)
                  # 递归进入子模块。
  
          incompatible_keys = _IncompatibleKeys(missing_keys, unexpected_keys)
          # 当前已经收集到的 missing/unexpected key。
  
          for hook in module._load_state_dict_post_hooks.values():
              # 加载后 hook。
              # 可以原地修改 incompatible_keys。
              out = hook(module, incompatible_keys)
  
              if out is not None:
                  # post hook 不应该返回新对象。
                  raise AssertionError("post hook should return None")
  
      load(self, state_dict)
      # 从根模块开始递归加载。
  
      del load
      # 删除内部递归函数引用。
  
      if strict:
          # strict=True 时，missing/unexpected 都会被视为错误。
  
          if len(unexpected_keys) > 0:
              error_msgs.insert(0, f"Unexpected keys: {unexpected_keys}")
  
          if len(missing_keys) > 0:
              error_msgs.insert(0, f"Missing keys: {missing_keys}")
  
      if len(error_msgs) > 0:
          # 统一抛出所有错误，而不是遇到一个就立刻中断。
          raise RuntimeError(
              "Error(s) in loading state_dict:\n\t" + "\n\t".join(error_msgs)
          )
  
      return _IncompatibleKeys(missing_keys, unexpected_keys)
      # 返回加载结果。
      # strict=False 时，即使有 missing/unexpected，也不会抛错，
      # 但会在这里返回给用户查看。
  ```

  可以用简易流程图表示：

  ```
      _load_from_state_dict(...)
              |
              v
      执行 load_state_dict_pre_hooks
              |
              v
      收集当前 module 的:
      parameters + persistent buffers
              |
              v
      遍历每个 name, param/buffer
              |
              v
      用 prefix + name 在 state_dict 中找 key
              |
              v
          找到 key?
     |                |
     | yes            | no
     v                v
  检查类型/shape      strict=True -> missing_keys
              |
              v
      加载权重:
      copy_ / assign / swap
              |
              v
      检查 extra_state
              |
              v
      strict=True 时检查 unexpected_keys
  ```

  ```
  load_state_dict(state_dict)
          |
          v
  检查 state_dict 类型
          |
          v
  创建 missing_keys / unexpected_keys / error_msgs
          |
          v
  复制 state_dict，读取 metadata
          |
          v
  load(root_module, prefix="")
          |
          v
  调用 root_module._load_from_state_dict(...)
          |
          v
  递归遍历 root_module._modules
          |
          v
  对子模块 child 调用 load(child, prefix="child.")
          |
          v
  执行 load_state_dict_post_hooks
          |
          v
  strict=True 时检查 missing / unexpected
          |
          v
  有错误则报错，否则返回 IncompatibleKeys
  ```

  两者有如下的区别

  - `_load_from_state_dict()` 是典型的内部接口命名规则，作用于单个module，把当前层自己的 `parameter` 和 `persistent buffer` 从 checkpoint 里加载进来
  - `load_state_dict` 是递归遍历所有子模块，组织加载流程，收集 `missing_keys`/`unexpected_keys`，而真正的子模块参数加载还是通过`_load_from_state_dict()`



#### `_load_from_state_dict()` 小技巧

模型代码升级后，checkpoint 可能会出现：

| 问题           | 例子                                          | 解决方式                 |
| -------------- | --------------------------------------------- | ------------------------ |
| 新版本多了 key | BatchNorm 新增 `num_batches_tracked`          | 加载旧权重时自动补默认值 |
| key 名字变了   | `xxx_offset.weight` 改成 `conv_offset.weight` | 加载时自动改名           |
| 跨项目迁移     | `backbone.` 要变成 `img_backbone.`            | 加载时批量重命名         |

等新版本代码无法直接加载旧版本checkpoint的问题。这些问题被称之为 **BC-breaking** 也就是 **Backward Compatibility Breaking**。PyTorch 可以通过 `_version` 和 `_load_from_state_dict` 来处理这些问题。

下面给出一个 `_NormBase` 类避免 BC-breaking 的方式。假设 Normalization layers 在某个新版本中引入了 `num_batches_tracked` 这个 key， 给 BN 记录训练过程中经历的 batch 数。

```python
def _load_from_state_dict(
    self,
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # 读取当前 module 保存时的版本号。
    # 如果是很旧的 checkpoint，可能没有 metadata，因此 version=None。
    version = local_metadata.get("version", None)

    # _NormBase 在 version 2 中新增了 num_batches_tracked。
    # 如果 checkpoint 来自 version < 2，里面可能没有这个 key。
    if (version is None or version < 2) and self.track_running_stats:

        # 当前模块中 num_batches_tracked 在 state_dict 里的完整 key。
        # 例如 prefix="bn1."，则 key="bn1.num_batches_tracked"
        num_batches_tracked_key = prefix + "num_batches_tracked"

        # 如果旧 checkpoint 没有这个 key，就手动补一个默认值 0。
        # 这样后面的默认加载逻辑就不会报 missing key。
        if num_batches_tracked_key not in state_dict:
            state_dict[num_batches_tracked_key] = torch.tensor(
                0,
                dtype=torch.long,
            )

    # 调用父类默认加载逻辑：
    # 加载 weight、bias、running_mean、running_var、num_batches_tracked 等。
    super()._load_from_state_dict(
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    )
```
