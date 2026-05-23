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