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

