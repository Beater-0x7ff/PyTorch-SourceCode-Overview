每个 GPU 复制一份模型，将一批样本分成多份输入，各个模型并行计算。 数学上，因为求导以及加和都是线性的，因此数据并行是成立的。

设一个 batch 有 n 个样本，一共有 k 个 GPU，每个 GPU分到 $m_j$ 个样本（假设样本等分）。那么总损失函数对于参数 w 的导数就有


$$
\begin{aligned}
\frac{\partial Loss}{\partial w}
&=
\frac{\partial \left[
\frac{1}{n}
\sum_{i=1}^{n}
l(x_i,y_i)
\right]}
{\partial w}
\\
&=
\frac{1}{n}
\sum_{i=1}^{n}
\frac{\partial l(x_i,y_i)}
{\partial w}
\\
&=
\sum_{j=1}^{k}
\frac{m_j}{n}
\cdot
\frac{
\partial \left[
\frac{1}{m_j}
\sum_{i=m_{j-1}+1}^{m_j}
l(x_i,y_i)
\right]
}
{\partial w}
\\
&=
\sum_{j=1}^{k}
\frac{m_j}{n}
\frac{\partial loss_j}{\partial w}
\\
&=
\frac{1}{k}
\sum_{j=1}^{k}
\frac{\partial loss_j}{\partial w}
\end{aligned}
$$

> [!NOTE]
>
> - 数据并行不仅指对训练数据的并行，对 **网络模型梯度，权重参数，优化器状态** 等数据进行并行
> - 负责收集梯度和广播的参数服务器不一定使用 GPU，也可以使用 CPU，但是有以下缺陷
>   - 内存带宽，CPU（DDR5）~50-100GB/s， GPU(HBM) ~800-3000GB/s
>   - 卡间通讯，GPU-CPU通讯使用PCIe ~32 GB/s，GPU-GPU通讯使用NvLink / NVSwitch ~600+ GB/s
>   - 训练优化， **ALL-Reduce + NCLL** 在GPU上进行了带宽，延迟，重叠计算，kernel fusion的优化
> - 参数服务器也可以分布在所有 GPU 节点，每个节点指更新其中一部分梯度



### DP 实现

DP（Data Parallel）的优势就是使用方便，只需要将单卡 module 调用函数改成多卡 

```python
model = nn.DataParallel(model)
```

也可以将参数服务器分布再所有GPU节点，每个节点只**更新其中一部分梯度**

数据并行不仅指对训练数据的并行操作，对 **网络模型梯度，权重参数，优化器状态** 等数据进行并行



#### 流程

```
               Global Batch
                    |
                    v
          +-------------------+
          | batch 切分(split)  |
          +-------------------+
                    |
                    v
 +---------+   +---------+   +---------+
 | GPU 0   |   | GPU 1   |   | GPU 2   |
 | model   |   | model   |   | model   |
 | replica |   | replica |   | replica |
 +---------+   +---------+   +---------+
      |             |              |
      v             v              v
 forward         forward        forward
      |             |              |
      v             v              v
  loss0           loss1          loss2
      |             |              |
      v             v              v
 backward        backward       backward
      |             |              |
      +-------------+--------------+
                    |
                    v
          gradient synchronization
               (AllReduce)
                    |
                    v
            averaged gradients
                    |
                    v
              optimizer.step()
                    |
                    v
            所有 GPU 参数同步更新
```

```torch.nn.DataParallel``` 基于单进程多线程实现，它使用**一个进程来计算模型权重**，在每个**批处理**期间将数据分发到每个GPU

过程如下：

- 将 inputs 从主 GPU 分发到所有 GPU 上。
- 将 model 从主 GPU 分发到所有 GPU 上。
- 每个 GPU 分别独立进行前向传播，得到 outputs。
- 将每个 GPU 的 outputs 发回主 GPU。
- 在主 GPU 上，通过 loss function 计算出 loss，对 loss function 求导，求出损失梯度。
- 计算得到的梯度分发到所有 GPU 上。
- 反向传播计算参数梯度。
- 将所有梯度回传到主 GPU，通过梯度更新模型权重。
- repeat



#### 实现

```python
class DataParallel(Module, Generic[T]):

    def __init__(
        self,
        module: T,
        device_ids: Sequence[int | torch.device] | None = None,
        output_device: int | torch.device | None = None,
        dim: int = 0,
    ) -> None:
        super().__init__()

        # 记录 API 使用情况，内部统计用。
        torch._C._log_api_usage_once("torch.nn.parallel.DataParallel")

        device_type = _get_available_device_type()

        if device_type is None or device_type == "mps":
            # 如果没有可用 GPU，或者是 MPS，则退化为普通单设备 module。
            self.module = module
            self.device_ids = []
            return

        if device_ids is None:
            # 如果用户没指定 GPU，则使用所有可见 GPU。
            device_ids = _get_all_device_indices()

        if device_ids is None:
            raise RuntimeError("no available devices were found")

        if output_device is None:
            # 默认把最终输出 gather 到第一张 GPU。
            output_device = device_ids[0]

        # dim 表示按哪个维度切分输入 batch，通常是 dim=0。
        self.dim = dim

        self.module = module

        # 将 device_ids 统一转换成 GPU index。
        self.device_ids = [_get_device_index(x, True) for x in device_ids]

        # 输出聚合到哪张 GPU。
        self.output_device = _get_device_index(output_device, True)

        # 主模型必须位于 device_ids[0] 上。
        self.src_device_obj = torch.device(device_type, self.device_ids[0])

        if device_type == "cuda":
            # 检查多张 GPU 性能是否差异过大。
            _check_balance(self.device_ids)

        if len(self.device_ids) == 1:
            # 如果只有一张 GPU，直接把模型移动到该 GPU。
            self.module.to(self.src_device_obj)

    def forward(self, *inputs: Any, **kwargs: Any) -> Any:
        # DataParallel 的核心执行逻辑：
        # scatter -> replicate -> parallel_apply -> gather

        with torch.autograd.profiler.record_function("DataParallel.forward"):
            # profiler 标记，用于性能分析。

            if not self.device_ids:
                # 没有多 GPU 时，直接调用原始模型。
                return self.module(*inputs, **kwargs)

            for t in chain(self.module.parameters(), self.module.buffers()):
                # 检查原始模型的所有参数和 buffer 是否都在主 GPU 上。
                # DataParallel 要求 module 先放在 device_ids[0]。
                if t.device != self.src_device_obj:
                    raise RuntimeError(
                        "module must have its parameters and buffers "
                        f"on device {self.src_device_obj} (device_ids[0]) but found one of "
                        f"them on device: {t.device}"
                    )

            # 1. scatter：
            # 把输入 inputs / kwargs 按 dim 切分并分发到多张 GPU。
            inputs, module_kwargs = self.scatter(inputs, kwargs, self.device_ids)

            if not inputs and not module_kwargs:
                # 如果没有输入，构造一个空输入，保证后续流程能执行。
                inputs = ((),)
                module_kwargs = ({},)

            if len(self.device_ids) == 1:
                # 单 GPU 情况下不用复制模型，直接 forward。
                return self.module(*inputs[0], **module_kwargs[0])

            # 2. replicate：
            # 把原始模型复制到每个参与计算的 GPU 上。
            replicas = self.replicate(self.module, self.device_ids[: len(inputs)])

            # 3. parallel_apply：
            # 多个 GPU 并行执行各自 replica 的 forward。
            outputs = self.parallel_apply(replicas, inputs, module_kwargs)

            # 4. gather：
            # 把多个 GPU 的输出聚合到 output_device。
            return self.gather(outputs, self.output_device)

    def replicate(self, module: T, device_ids: Sequence[int | torch.device]) -> list[T]:
        # 将 module 复制到多个 device 上。
        # not torch.is_grad_enabled() 表示在 no_grad/inference 时可使用 detach 形式复制。
        return replicate(module, device_ids, not torch.is_grad_enabled())

    def scatter(
        self,
        inputs: tuple[Any, ...],
        kwargs: dict[str, Any] | None,
        device_ids: Sequence[int | torch.device],
    ) -> Any:
        # 将输入和 kwargs 按 self.dim 切分到不同 GPU。
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

    def parallel_apply(
        self, replicas: Sequence[T], inputs: Sequence[Any], kwargs: Any
    ) -> list[Any]:
        # 在多张 GPU 上并行调用每个 replica 的 forward。
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[: len(replicas)]
        )

    def gather(self, outputs: Any, output_device: int | torch.device) -> Any:
        # 将各 GPU 输出聚合到 output_device。
        return gather(outputs, output_device, dim=self.dim)
```

关键函数有 scatter, replicate, parallel_apply 和 gather，我们逐个学习



##### scatter

数据并行的原因是因为数据量过于庞大，单张卡很难一次加载这么多数据进入其中，所以我们期望将数据拆分，就像是一本口算的题目太多了，我们给每个小朋友一人派五页口算的任务，这样大家一起做完一本口算也就做完了。

- ```DataParallel.scatter``` 函数，实际上封装的 ```scatter_kwargs```，源码如下

  ```python
  def scatter_kwargs(
      inputs: tuple[Any, ...],
      kwargs: dict[str, Any] | None,
      target_gpus: Sequence[int | torch.device],
      dim: int = 0,
  ) -> tuple[tuple[Any, ...], tuple[dict[str, Any], ...]]:
  
      # 注意这里的 scatter 是 torch.nn.parallel.scatter_gather.scatter
      # 不是调用 scatter_kwargs 的 DataParallel.scatter
      scattered_inputs = scatter(inputs, target_gpus, dim) if inputs else []
      scattered_kwargs = scatter(kwargs, target_gpus, dim) if kwargs else []
      
      
      # 补齐，让每张 GPU 都有一组完整的（args, kwargs）
      if len(scattered_inputs) < len(scattered_kwargs):
          scattered_inputs.extend(
              () for _ in range(len(scattered_kwargs) - len(scattered_inputs))
          )
      elif len(scattered_kwargs) < len(inputs):
          scattered_kwargs.extend(
              {} for _ in range(len(scattered_inputs) - len(scattered_kwargs))
          )
      return tuple(scattered_inputs), tuple(scattered_kwargs)
  ```

- `torch.nn.parallel.scatter_gather.scatter` 的源码如下

  ```python
  def scatter(inputs, target_gpus, dim=0):
  
      # scatter_map 递归地把输入拆分到多个 GPU。
      def scatter_map(obj):
  
          if isinstance(obj, torch.Tensor):
              # Tensor 真正执行切分：
              # 沿 dim 维度 split 后发送到 target_gpus。
              return Scatter.apply(target_gpus, None, dim, obj)
  
          if _is_namedtuple(obj):
              # namedtuple:
              # 对每个字段递归 scatter，
              # 再重新构造 namedtuple。
              return [
                  type(obj)(*args)
                  for args in zip(*map(scatter_map, obj), strict=False)
              ]
  
          if isinstance(obj, tuple) and len(obj) > 0:
              # tuple:
              # 对 tuple 中每个元素递归 scatter，
              # 再按 GPU 重新组合。
              return list(zip(*map(scatter_map, obj), strict=False))
  
          if isinstance(obj, list) and len(obj) > 0:
              # list:
              # 递归 scatter 后重新组装 list。
              return [list(i) for i in zip(*map(scatter_map, obj), strict=False)]
  
          if isinstance(obj, dict) and len(obj) > 0:
              # dict:
              # 对每个 key/value 递归 scatter，
              # 再按 GPU 重新构造 dict。
              return [
                  type(obj)(i)
                  for i in zip(*map(scatter_map, obj.items()), strict=False)
              ]
  
          # 非 Tensor 对象（如 int/str/None 等）：
          # 不切分，直接复制到每张 GPU。
          return [obj for _ in target_gpus]
  
      try:
          # 对整个 inputs 递归 scatter。
          res = scatter_map(inputs)
  
      finally:
          # 手动断开 scatter_map 的递归引用，
          # 帮助 Python GC 回收闭包。
          scatter_map = None
  
      return res
  ```

  对于 Tensor 的处理 `return Scatter.apply(target_gpus, None, dim, obj)`，又调用了一个名为 `Scatter` 的类，这是 `torch.nn.parallel._functions.Scatter` 类

- `torch.nn.parallel._functions.Scatter` 源码如下

  ```python
  class Scatter(Function):
      # forward 负责把 input 切分并分发到多个 GPU；
      # backward 负责把各 GPU 返回的梯度重新 gather 回原设备。
  
      @staticmethod
      def forward(ctx, target_gpus, chunk_sizes, dim, input):
          # 将 target_gpus 统一转换成设备编号。
          target_gpus = [_get_device_index(x, True) for x in target_gpus]
  
          # 保存切分维度，backward 时 gather 梯度要用。
          ctx.dim = dim
  
          # 记录输入原本所在设备。
          # 如果 input 在 CPU 上，则记录为 -1。
          ctx.input_device = input.get_device() if input.device.type != "cpu" else -1
  
          streams = None
  
          # 如果 accelerator 可用，且 input 来自 CPU，
          # 则为每个目标 GPU 获取一个后台 stream。
          # 这样 CPU -> GPU 拷贝可以异步进行。
          if torch.accelerator.is_available() and ctx.input_device == -1:
              streams = [_get_stream(torch.device(device)) for device in target_gpus]
  
          # 记录是否是 complex tensor。
          is_complex = input.is_complex()
  
          # 真正执行 scatter：
          # 沿 ctx.dim 将 input 切成若干 chunk，
          # 并发送到 target_gpus。
          outputs = comm.scatter(input, target_gpus, chunk_sizes, ctx.dim, streams)
  
          # 如果输入是 complex tensor，
          # scatter 后需要恢复 complex view。
          if is_complex:
              outputs = tuple(torch.view_as_complex(o) for o in outputs)
  
          if streams is not None:
              # 如果使用了后台 stream，
              # 需要让当前主 stream 等待对应 copy stream 完成。
              for i, output in enumerate(outputs):
                  with torch.accelerator.device_index(target_gpus[i]):
                      main_stream = torch.accelerator.current_stream()
  
                      # 当前主 stream 等待 scatter copy stream。
                      main_stream.wait_stream(streams[i])
  
                      # 告诉 allocator：output 这个 Tensor 会被 main_stream 使用。
                      # 防止异步拷贝尚未完成时内存被错误复用。
                      output.record_stream(main_stream)
  
          # 返回分发到不同 GPU 的输出 tuple。
          return outputs
  
      @staticmethod
      def backward(ctx, *grad_output):
          # forward 有 4 个输入：
          # target_gpus, chunk_sizes, dim, input
          # 其中前三个不是 Tensor，不需要梯度，所以返回 None。
          # input 的梯度需要从多个 GPU 的 grad_output 聚合回来，
          # 因此调用 Gather.apply。
  
          return (
              None,
              None,
              None,
              Gather.apply(ctx.input_device, ctx.dim, *grad_output),
          )
  ```

  核心流程

  ```
  forward:
      input
        -> comm.scatter
        -> 多个 GPU 上的 input chunks
  
  backward:
      多个 GPU 上的 grad_output
        -> Gather.apply
        -> 原 input 设备上的 grad_input
  ```

  `forward` 函数中还有一个 `comm.scatter` 函数，但是是 C++ 实现，我们就不考虑了



##### replicate

当我们多卡进行数据并行的时候将模型的全部权重分发到每个卡中，那么就使用到 `torch.nn.parallel.replicate` ，源码如下

- `torch.nn.parallel.replicate`

  ```python
  def replicate(
      network: T,
      devices: Sequence[int | torch.device],
      detach: bool = False,
  ) -> list[T]:
  
      if not _replicatable_module(network):
          # 检查这个 module 是否可以被复制。
          # 例如 ScriptModule 中混入普通 Python module 的情况可能不支持。
          raise RuntimeError(
              "Cannot replicate network where python modules are children of ScriptModule"
          )
  
      if not devices:
          # 没有目标设备，直接返回空列表。
          return []
  
      # 将 device 转换成标准 device index。
      # 例如 cuda:0 -> 0
      devices = [_get_device_index(x, True) for x in devices]
  
      # replica 数量等于设备数量。
      num_replicas = len(devices)
  
      # 收集原始模型的所有 parameters。
      params = list(network.parameters())
  
      # 建立 parameter -> index 的映射。
      # 后面复制 module 结构时，可以通过原 param 找到对应的拷贝。
      param_indices = {param: idx for idx, param in enumerate(params)}
  
      # 将所有 parameters 广播到多个 device 上。
      # param_copies[j][i] 表示第 j 个 device 上第 i 个 parameter 的拷贝。
      param_copies = _broadcast_coalesced_reshape(params, devices, detach)
  
      # 收集原始模型的所有 buffers。
      buffers = list(network.buffers())
  
      # requires_grad=True 且 detach=False 的 buffer，需要保留 autograd 关系。
      buffers_rg: list[torch.Tensor] = []
  
      # 不需要梯度的 buffer，或者 detach=True 时的 buffer。
      buffers_not_rg: list[torch.Tensor] = []
  
      for buf in buffers:
          if buf.requires_grad and not detach:
              buffers_rg.append(buf)
          else:
              buffers_not_rg.append(buf)
  
      # 建立 buffer -> index 的映射。
      buffer_indices_rg = {buf: idx for idx, buf in enumerate(buffers_rg)}
      buffer_indices_not_rg = {buf: idx for idx, buf in enumerate(buffers_not_rg)}
  
      # 广播需要梯度的 buffers。
      buffer_copies_rg = _broadcast_coalesced_reshape(buffers_rg, devices, detach=detach)
  
      # 广播不需要梯度的 buffers。
      # 这里强制 detach=True，因为这些 buffer 不需要参与反向传播。
      buffer_copies_not_rg = _broadcast_coalesced_reshape(
          buffers_not_rg, devices, detach=True
      )
  
      # 收集整个 module tree 中的所有 module。
      # 包括 network 自己和所有子模块。
      modules = list(network.modules())
  
      # module_copies[j] 保存第 j 个 device 上的所有 module replica。
      module_copies: list[list[Module]] = [[] for _ in devices]
  
      # 原 module -> index 的映射。
      # 用于恢复父子模块关系。
      module_indices: dict[Module, int] = {}
  
      # 第一步：只复制 module 对象本身。
      # 注意：这一步还没有正确连接子模块、参数和 buffer。
      for i, module in enumerate(modules):
          module_indices[module] = i
  
          for j in range(num_replicas):
              # 创建当前 module 的 data parallel replica。
              replica = module._replicate_for_data_parallel()
  
              # replica 上的参数副本不再作为真正 Parameter 存在于 _parameters 中。
              # 因此用 _former_parameters 保存一份，兼容 DDP / state_dict 等逻辑。
              replica._former_parameters = OrderedDict()
  
              # 保存到第 j 个 device 的 module 列表里。
              module_copies[j].append(replica)
  
      # 第二步：重新连接每个 replica 的子模块、参数和 buffer。
      for i, module in enumerate(modules):
  
          # 2.1 复制子模块结构
          for key, child in module._modules.items():
              if child is None:
                  # 如果原模块的某个子模块是 None，
                  # replica 上也设为 None。
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
                      replica._modules[key] = None
              else:
                  # 找到 child 在 modules 列表中的 index。
                  module_idx = module_indices[child]
  
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
  
                      # 将当前 replica 的子模块指向同一 device 上对应的 child replica。
                      # 例如 GPU0 上的 parent 指向 GPU0 上的 child。
                      setattr(replica, key, module_copies[j][module_idx])
  
          # 2.2 复制 parameters
          for key, param in module._parameters.items():
              if param is None:
                  # 如果原参数是 None，例如 bias=False，
                  # replica 上也设为 None。
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
                      replica._parameters[key] = None
              else:
                  # 找到当前 param 在参数列表中的 index。
                  param_idx = param_indices[param]
  
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
  
                      # 取出第 j 个 device 上对应的 parameter copy。
                      param_copy = param_copies[j][param_idx]
  
                      # 把参数副本挂到 replica 上。
                      # 注意：param_copy 通常是 Tensor，不一定作为 leaf Parameter 注册。
                      setattr(replica, key, param_copy)
  
                      # 保存到 _former_parameters。
                      # 这是 DataParallel 中为了兼容 parameter 访问而保留的结构。
                      replica._former_parameters[key] = param_copy
  
          # 2.3 复制 buffers
          for key, buf in module._buffers.items():
              if buf is None:
                  # 如果 buffer 是 None，replica 上也设为 None。
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
                      replica._buffers[key] = None
              else:
                  # 根据 buffer 是否需要梯度，选择对应的 buffer copy 列表。
                  if buf.requires_grad and not detach:
                      buffer_copies = buffer_copies_rg
                      buffer_idx = buffer_indices_rg[buf]
                  else:
                      buffer_copies = buffer_copies_not_rg
                      buffer_idx = buffer_indices_not_rg[buf]
  
                  for j in range(num_replicas):
                      replica = module_copies[j][i]
  
                      # 把第 j 个 device 上的 buffer copy 设置到 replica。
                      setattr(replica, key, buffer_copies[j][buffer_idx])
  
      # 返回每个 device 上的根 module replica。
      # module_copies[j][0] 是第 j 个 device 上的 network 根模块。
      return [cast(T, module_copies[j][0]) for j in range(num_replicas)]
  ```

  核心流程：

  ```
  replicate(network, devices)
      |
      |-- 收集 parameters / buffers / modules
      |-- 广播 parameters 到各 GPU
      |-- 广播 buffers 到各 GPU
      |-- 复制 module 对象
      |-- 重建每个 replica 的:
      |       子模块连接
      |       参数引用
      |       buffer 引用
      |
      |-- 返回每张 GPU 上的 model replica
  ```

  其中我们看到在将模型参数广播到其它显卡的时候使用了 `_broadcast_coalesced_reshape`

- `_broadcast_coalesced_reshape`

  ```python
  def _broadcast_coalesced_reshape(
      tensors: Sequence[torch.Tensor],
      devices: Sequence[int | torch.device],
      detach: bool = False,
  ) -> list[list[torch.Tensor]]:
      # 这个函数用于把一组 tensors 广播到多个 device 上。
      # 在 DataParallel.replicate() 中，它用于复制 parameters 和 buffers。
  
      from torch.nn.parallel._functions import Broadcast
  
      if len(tensors) == 0:
          # 如果没有 tensor 需要广播，直接返回空列表。
          return []
  
      if detach:
          # detach=True：
          # 不需要保留 autograd 关系。
          # 常用于 no_grad 推理，或者不需要梯度的 buffer。
  
          complex_mask = [
              not isinstance(t, torch.nn.UninitializedParameter) and t.is_complex()
              for t in tensors
          ]
          # 记录哪些 tensor 是 complex tensor。
          # 因为 broadcast_coalesced 对 complex tensor 的处理可能需要恢复 complex view。
  
          outputs = comm.broadcast_coalesced(tensors, devices)
          # 真正执行广播。
          # coalesced 表示会把多个小 tensor 合并后再广播，
          # 减少通信次数，提高效率。
  
          for device_outputs in outputs:
              # device_outputs 表示某一个 device 上收到的一组 tensor。
              for i, is_complex in enumerate(complex_mask):
                  if is_complex:
                      # 如果原 tensor 是 complex，
                      # 广播后恢复成 complex tensor。
                      device_outputs[i] = torch.view_as_complex(device_outputs[i])
  
          return outputs
          # 返回每个 device 上的 tensor copies。
  
      else:
          # detach=False：
          # 需要保留 autograd 关系。
          # 例如 DataParallel 训练时复制 parameter，
          # 后续 backward 要能把各 replica 的梯度回传到原始 parameter。
  
          tensor_copies = Broadcast.apply(devices, *tensors)
          # 使用 autograd.Function Broadcast。
          # forward：把 tensors 广播到多个 device
          # backward：把多个 device 上的梯度 reduce 回原 tensor
  
          return [
              list(tensor_copies[i : i + len(tensors)])
              for i in range(0, len(tensor_copies), len(tensors))
          ]
          # Broadcast.apply 返回的是一个扁平 tuple：
          # (
          #   tensor0_gpu0, tensor1_gpu0,
          #   tensor0_gpu1, tensor1_gpu1,
          #   ...
          # )
          #
          # 这里重新 reshape 成：
          # [
          #   [tensor0_gpu0, tensor1_gpu0],
          #   [tensor0_gpu1, tensor1_gpu1],
          # ]
  ```

  在这个内部的广播函数开头就引入了  `torch.nn.parallel._functions.Broadcast`



- `torch.nn.parallel._functions.Broadcast`

  ```python
  class Broadcast(Function):
      # Broadcast 是一个自定义 autograd.Function。
      # forward 负责把多个输入 tensor 广播到多个 GPU；
      # backward 负责把各 GPU 上产生的梯度重新加回原始输入设备。
  
      @staticmethod
      def forward(ctx, target_gpus, *inputs):
          # target_gpus: 目标 GPU 列表
          # *inputs: 需要被广播的 tensor 列表，例如模型参数列表
  
          if not all(i.device.type != "cpu" for i in inputs):
              # Broadcast 这里要求输入 tensor 已经在 GPU 上。
              # CPU tensor 的分发通常走 Scatter，而不是 Broadcast。
              raise AssertionError("Broadcast function not implemented for CPU tensors")
  
          # 将 target_gpus 统一转换成设备编号。
          target_gpus = [_get_device_index(x, True) for x in target_gpus]
  
          # 保存目标 GPU，backward 时可能需要相关上下文。
          ctx.target_gpus = target_gpus
  
          if len(inputs) == 0:
              # 没有输入 tensor，直接返回空 tuple。
              return ()
  
          # 保存输入 tensor 的数量。
          # backward 时需要知道每个 GPU 上有多少个 tensor。
          ctx.num_inputs = len(inputs)
  
          # 保存原始输入所在设备。
          # backward 时要把梯度 reduce 回这个设备。
          ctx.input_device = inputs[0].get_device()
  
          # 记录哪些输入是 complex tensor。
          ctx.complex_mask = [inp.is_complex() for inp in inputs]
  
          # 真正执行广播。
          # coalesced 表示把多个小 tensor 合并通信，减少通信次数。
          #
          # outputs 的结构大致是：
          # [
          #   [input0_gpu0, input1_gpu0, ...],
          #   [input0_gpu1, input1_gpu1, ...],
          #   ...
          # ]
          outputs = comm.broadcast_coalesced(inputs, ctx.target_gpus)
  
          # 如果有 complex tensor，广播后恢复 complex view。
          for device_outputs in outputs:
              for i, is_complex in enumerate(ctx.complex_mask):
                  if is_complex:
                      device_outputs[i] = torch.view_as_complex(device_outputs[i])
  
          non_differentiables = []
          # ctx.needs_input_grad 对应 forward 的每个输入是否需要梯度。
          # forward 输入包括：
          #   target_gpus, input0, input1, ...
          # 所以这里用 [1:] 跳过 target_gpus。
          for idx, input_requires_grad in enumerate(ctx.needs_input_grad[1:]):
              if not input_requires_grad:
                  # 如果某个原始输入不需要梯度，
                  # 那么它在所有 GPU 上的 broadcast copy 也标记为不可导。
                  non_differentiables.extend(output[idx] for output in outputs)
  
          # 告诉 autograd：这些输出不需要反向传播。
          ctx.mark_non_differentiable(*non_differentiables)
  
          # 将 outputs 从二维结构打平成 tuple 返回。
          #
          # 原结构：
          # [
          #   [input0_gpu0, input1_gpu0],
          #   [input0_gpu1, input1_gpu1],
          # ]
          #
          # 返回：
          # (
          #   input0_gpu0, input1_gpu0,
          #   input0_gpu1, input1_gpu1,
          # )
          return tuple(chain.from_iterable(outputs))
  
      @staticmethod
      def backward(ctx, *grad_outputs):
          # grad_outputs 是 forward 返回的每个 broadcast copy 对应的梯度。
          #
          # 例如 forward 输出：
          # (
          #   p0_gpu0, p1_gpu0,
          #   p0_gpu1, p1_gpu1,
          # )
          #
          # backward 收到：
          # (
          #   grad_p0_gpu0, grad_p1_gpu0,
          #   grad_p0_gpu1, grad_p1_gpu1,
          # )
  
          # ReduceAddCoalesced 会把不同 GPU 上对应 tensor 的梯度相加，
          # 并 reduce 回 ctx.input_device。
          #
          # 输出 grads 对应原始 inputs 的梯度：
          # (
          #   grad_input0,
          #   grad_input1,
          #   ...
          # )
          grads = ReduceAddCoalesced.apply(
              ctx.input_device,
              ctx.num_inputs,
              *grad_outputs
          )
  
          # forward 的第一个输入 target_gpus 不是 Tensor，不需要梯度，返回 None。
          # 后面的 grads 对应每个原始 input tensor 的梯度。
          return (None,) + grads
  ```

  

##### parallel_apply

简单来说 `parallel_apply` 的作用就是在多张 GPU 上同时调用多个模型副本，并行执行 forward 的函数

```python
def parallel_apply(
    modules: Sequence[Module],
    inputs: Sequence[Any],
    kwargs_tup: Sequence[dict[str, Any]] | None = None,
    devices: Sequence[int | torch.device | None] | None = None,
) -> list[Any]:
    # parallel_apply 的作用：
    # 在多个 device 上并行执行多个 module replica 的 forward。
    # modules[i] + inputs[i] + kwargs_tup[i] + devices[i]
    # 对应第 i 个 replica 的一次 forward。

    if len(modules) != len(inputs):
        # 每个 module replica 必须对应一份 input。
        raise AssertionError(...)

    if kwargs_tup is not None:
        if len(modules) != len(kwargs_tup):
            # 每个 module replica 也必须对应一份 kwargs。
            raise AssertionError(...)
    else:
        # 如果没有 kwargs，则给每个 replica 补一个空 dict。
        kwargs_tup = (cast(dict[str, Any], {}),) * len(modules)

    if devices is not None:
        if len(modules) != len(devices):
            # 每个 module replica 必须对应一个 device。
            raise AssertionError(...)
    else:
        # 如果没有显式给 devices，后面会尝试从 input tensor 推断 device。
        devices = [None] * len(modules)

    # 将 device 统一转换成设备编号。
    devices = [_get_device_index(x, True) for x in devices]

    # 为每个 device 获取当前 stream。
    # 每个 replica 会在对应 device 的 stream 上执行 forward。
    streams = [torch.accelerator.current_stream(x) for x in devices]

    if not torch.accelerator.is_available():
        # 没有可用加速设备则不能并行执行。
        raise AssertionError("No available accelerator found.")

    # 当前 accelerator 类型，例如 "cuda"。
    device_type = torch.accelerator.current_accelerator().type

    # 多线程写 results 时需要加锁，避免竞态条件。
    lock = threading.Lock()

    # results 用来保存每个 replica 的输出或异常。
    results = {}

    # 记录当前主线程的 grad/autocast 状态。
    # worker 线程里要恢复这些状态。
    grad_enabled, autocast_enabled = (
        torch.is_grad_enabled(),
        torch.is_autocast_enabled(),
    )

    def _worker(
        i: int,
        module: Module,
        input: Any,
        kwargs: dict[str, Any],
        device: int | torch.device | None = None,
        stream: torch.Stream | None = None,
    ) -> None:
        # 每个 worker 线程负责执行一个 replica 的 forward。

        # 恢复主线程的 grad 状态。
        # 例如外层在 no_grad() 下，这里也要 no_grad。
        torch.set_grad_enabled(grad_enabled)

        if device is None:
            # 如果没有显式 device，就尝试从 input 中找一个 Tensor 来推断 device。
            t = get_a_var(input)

            if t is None:
                # 如果 input 里没有 Tensor，就无法推断 device。
                with lock:
                    results[i] = ExceptionWrapper(...)
                return

            # 使用输入 Tensor 所在的 device。
            device = t.get_device()

        if isinstance(device, torch.device):
            # torch.device("cuda:0") -> 0
            device = device.index

        if stream is None:
            # 如果没传 stream，就取当前 device 的默认 current stream。
            stream = torch.accelerator.current_stream(device)

        try:
            with (
                # 切换到当前 worker 对应的 device。
                torch.accelerator.device_index(device),

                # 在对应 stream 上执行 forward。
                stream,

                # 恢复主线程 autocast 状态。
                # 如果外层启用了 AMP，这里也启用 AMP。
                torch.amp.autocast(device_type, enabled=autocast_enabled),
            ):
                if not isinstance(input, (list, tuple)):
                    # forward 统一用 *input 调用。
                    # 如果 input 不是 list/tuple，就包装成单元素 tuple。
                    input = (input,)

                # 真正执行当前 replica 的 forward。
                output = module(*input, **kwargs)

            with lock:
                # 保存当前 replica 的输出。
                results[i] = output

        except Exception:
            with lock:
                # 如果当前 replica forward 报错，
                # 不立刻在子线程里抛出，而是包装起来放入 results。
                results[i] = ExceptionWrapper(
                    where=f"in replica {i} on device {device}"
                )

    if len(modules) > 1:
        # 多个 replica 时，为每个 replica 创建一个 Python thread。
        threads = [
            threading.Thread(
                target=_worker,
                args=(i, module, input, kwargs, device, stream),
            )
            for i, (module, input, kwargs, device, stream) in enumerate(
                zip(modules, inputs, kwargs_tup, devices, streams, strict=True)
            )
        ]

        # 启动所有 worker 线程。
        for thread in threads:
            thread.start()

        # 等待所有 worker 线程执行完成。
        for thread in threads:
            thread.join()

    else:
        # 只有一个 replica 时，不创建线程，直接执行。
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0], streams[0])

    outputs = []

    for i in range(len(inputs)):
        # 按 replica 顺序取回输出。
        output = results[i]

        if isinstance(output, ExceptionWrapper):
            # 如果某个 worker 中发生异常，
            # 在主线程中重新抛出，方便用户看到完整错误。
            output.reraise()

        outputs.append(output)

    # 返回所有 replica 的 forward 结果。
    return outputs
```



##### gather

并行计算完成后接下来我们需要将结果收集到 device[0]

```python
# torch.nn.parallel.scatter_gather.py
def gather(outputs: Any, target_device: int | torch.device, dim: int = 0) -> Any:
    # 将多个 GPU 上的 outputs 聚合到 target_device 上。
    # 它不仅能处理 Tensor，也能递归处理 dict / tuple / namedtuple 等结构。

    def gather_map(outputs):
        # outputs 是来自多个 replica 的输出列表。
        # 例如 [out_gpu0, out_gpu1, out_gpu2]

        out = outputs[0]
        # 用第一个输出判断整体结构类型。

        if isinstance(out, torch.Tensor):
            # 如果输出是 Tensor，真正调用 Gather.apply 合并 Tensor。
            return Gather.apply(target_device, dim, *outputs)

        if out is None:
            # 如果输出是 None，直接返回 None。
            return None

        if isinstance(out, dict):
            # 如果输出是 dict，要求每个 GPU 的输出 dict 结构一致。
            if not all(len(out) == len(d) for d in outputs):
                raise ValueError("All dicts must have the same number of keys")

            # 对 dict 中每个 key 对应的值递归 gather。
            return type(out)((k, gather_map([d[k] for d in outputs])) for k in out)

        if _is_namedtuple(out):
            # 如果输出是 namedtuple，逐字段递归 gather，
            # 再重新构造成同类型 namedtuple。
            return type(out)._make(map(gather_map, zip(*outputs, strict=True)))

        # 其他 tuple / list 等结构：
        # zip(*outputs) 把多个 GPU 的同位置元素对齐，
        # 然后递归 gather。
        return type(out)(map(gather_map, zip(*outputs, strict=True)))

    try:
        # 对整个 outputs 递归 gather。
        res = gather_map(outputs)

    finally:
        # 断开递归函数自身引用，帮助 GC 回收闭包。
        gather_map = None

    return res

# torch.nn.parallel._functions.py
class Gather(Function):
    # Gather 是自定义 autograd.Function。
    # forward：把多个 GPU 的 Tensor 输出合并到 target_device。
    # backward：把 target_device 上的梯度再 scatter 回原来的 GPU。

    @staticmethod
    def forward(ctx, target_device, dim, *inputs):
        # target_device: 聚合输出的目标设备
        # dim: 沿哪个维度拼接
        # *inputs: 来自多个 GPU 的 Tensor 输出

        if not all(i.device.type != "cpu" for i in inputs):
            # Gather 这里要求输入来自 GPU。
            raise AssertionError("Gather function not implemented for CPU tensors")

        if target_device == "cpu":
            # 允许将结果 gather 到 CPU。
            ctx.target_device = "cpu"
        else:
            # 将 target_device 转换为标准 GPU index。
            target_device = _get_device_index(target_device, True)
            ctx.target_device = target_device

        # 保存拼接维度，backward 时 scatter 梯度要用。
        ctx.dim = dim

        # 保存每个输入 Tensor 原来所在的 GPU。
        # backward 时要把梯度发回这些 GPU。
        ctx.input_gpus = tuple(i.get_device() for i in inputs)

        if all(t.dim() == 0 for t in inputs) and dim == 0:
            # 特殊情况：
            # 如果每个 GPU 输出都是 scalar，不能直接沿 dim=0 concat。
            # 所以先 view(1)，把 scalar 变成 shape=(1,)。
            inputs = tuple(t.view(1) for t in inputs)

            warnings.warn(
                "Was asked to gather along dimension 0, but all "
                "input tensors were scalars; will instead unsqueeze "
                "and return a vector.",
                stacklevel=2,
            )

            # 记录这个情况，backward 时需要把梯度再 squeeze 回 scalar。
            ctx.unsqueezed_scalar = True
        else:
            ctx.unsqueezed_scalar = False

        # 保存每个输入在 dim 维度上的长度。
        # backward 时需要按这些长度把 grad_output 切回去。
        ctx.input_sizes = tuple(i.size(ctx.dim) for i in inputs)

        # 记录是否是 complex tensor。
        is_complex = len(inputs) > 0 and inputs[0].is_complex()

        # 真正执行 gather：
        # 把多个 GPU 上的 inputs 沿 dim 拼接到 target_device。
        output = comm.gather(inputs, ctx.dim, ctx.target_device)

        if is_complex:
            # 如果是 complex tensor，恢复 complex view。
            output = torch.view_as_complex(output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output 是 gather 后输出 Tensor 的梯度。
        # backward 要把这个梯度按照 forward 时的 input_sizes
        # 切分回各个原始 GPU。

        scattered_grads = Scatter.apply(
            ctx.input_gpus,
            ctx.input_sizes,
            ctx.dim,
            grad_output,
        )
        # Scatter.apply 会把 grad_output 沿 ctx.dim 切成多份，
        # 分发回 ctx.input_gpus。

        if ctx.unsqueezed_scalar:
            # 如果 forward 时把 scalar 临时 view 成 shape=(1,)，
            # backward 时要把梯度再取回 scalar。
            scattered_grads = tuple(g[0] for g in scattered_grads)

        # forward 输入是：
        # target_device, dim, *inputs
        #
        # target_device 和 dim 不需要梯度，所以返回 None, None。
        # 后面的 scattered_grads 对应每个 input 的梯度。
        return (None, None) + scattered_grads
```



#### 开销分析

- **单进程多线程的问题**

  DataParallel 使用的单进程多线程方便了信息的交换，但是受困于 **GIL**（Global Interpreter Lock），会带来性能开销，速度很慢，而且只能 **单机多卡**。同时不能使用 Apex进行混合精度训练

- 通讯开销

  数据集需要先拷贝到主进程，然后再进行分片到每个设备。每次结果输出后又要 gather 到 GPU0 上 

- GPU 利用效率低

  由于只有 GPU0 收集输出的梯度，更新再分发，其负载会更大，而且于此同时其它 GPU 还会空等

- 不支持模型并行



### 分布式数据并行 PyTorch DDP

分布式数据并行(`torch.nn.DistributedDataParallel`)，基于**多进程**进行实现的，每个进程都有**独立的优化器**，执行自己的更新过程。每个进程都执行相同的任务，并且每个进程都与所有其他进程通信。进程之间只传递梯度，网络通信不再是瓶颈。



DDP 的过程如下：

- 首先将 rank=0 进程中的模型参数**广播**到其他进程；
- 然后，每个 DDP 进程都会创建一个 **local Reducer** 来负责梯度同步。
- 各个进程从磁盘加载batch数据并将它们传递到其GPU，然后完成前向传播和loss计算
- 各个GPU开始反向传播，然后部分梯度计算完毕立刻 **All-Reduce**, 不等待全部计算完毕。这个过程直到全部梯度同步完成
- 每一层的梯度不依赖于前一层，所以**梯度的 All-Reduce 和后向过程同时计算**，以进一步缓解网络瓶颈。
- 在后向过程的最后，每个节点都得到了平均梯度，这样各个 GPU 中的模型参数保持同步 。



**与 DP 的区别**

1. 执行模型

   `DataParallel` 采用单进程多线程方式，由一个主进程控制多张 GPU；

   `DistributedDataParallel` 推荐采用多进程方式，通常每张 GPU 对应一个独立进程。

2. 通信方式

   `DataParallel` 的通信主要围绕主 GPU 展开，包括输入分发、输出收集、梯度回传以及参数复制，因此主 GPU 容易成为瓶颈。

   `DistributedDataParallel` 则通过 AllReduce 在各进程之间同步梯度。以 Ring AllReduce 为例，每张 GPU 的通信量约为：

$$
2 \cdot \frac{N-1}{N} \cdot P
$$

   其中 N 是 GPU 数量，P 是梯度大小。当 N 增大时，该通信量趋近于 2P，因此 DDP 的扩展性通常优于 DP。

3. 参数同步方式

   `DataParallel` 通常由主 GPU 保存原始参数，梯度最终回传到主 GPU 后完成参数更新，下一轮 forward 再将参数复制到其它 GPU。

   `DistributedDataParallel` 则要求各进程初始参数一致，并在 backward 过程中同步梯度。由于每张 GPU 获得相同的平均梯度，并执行相同的 optimizer step，因此各模型副本在训练过程中保持一致。

示例

```python
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


local_rank = int(os.environ["LOCAL_RANK"])

torch.cuda.set_device(local_rank)
dist.init_process_group("gloo")

model = nn.Linear(10, 1).cuda(local_rank)
model = DDP(model, device_ids=[local_rank])

optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

x = torch.randn(16, 10).cuda(local_rank)
y = torch.randn(16, 1).cuda(local_rank)

for step in range(5):
    optimizer.zero_grad()

    pred = model(x)
    loss = ((pred - y) ** 2).mean()

    loss.backward()
    optimizer.step()

    if dist.get_rank() == 0:
        print(f"step {step}, loss = {loss.item():.4f}")

dist.destroy_process_group()
# torchrun --nproc_per_node=num_of_gpus autograd.py
```



#### Ring AllReduce

所有的 GPU 围成一个环，每张 GPU 只和左右邻居通信，通过多轮 **传递** 和 **累加** 最终让所有 GPU 都得到完整平均梯度

Reference： https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/

（非常好的帖子）

![GPUs arranged in a logical ring](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/ring-gpus.png)



作者假设计算的目标是对一个大型浮点数数组中的所有元素逐个求和（累加梯度）。系统中有 N 个 GPU，每个 GPU 都有一个大小相同的数组，并且在 allreduce 操作结束后，每个 GPU 都应该有一个大小相同的数组，其中包含原始数组中数字的总和。首先，GPU 将数组分割成 N 个较小的块（N 是环中 GPU 的数量）

![Partitioning of an array into N chunks](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/array-partition.png)

接下来，GPU 会进行 N-1 次的 scatter-reduce。在每次迭代中，GPU 会将其一个数据块发送给右侧的 GPU，并从左侧的 GPU 接受一个数据块，并将这两个数据块累加到该数据块中。每次迭代中，每个 GPU 发送和接收的数据块都不同。第 n 个 GPU 首先发送数据块 n 并接收数据块 n-1，然后从那里开始反向进行，每次迭代都发送它在前一次迭代中接收到的数据块。（这个步骤可以看出来 GPU 在进行发送和接接收的时候不是同一块数据，也就是说可以并行减少空等）

scatter-reduce 第一次迭代就有

![分散归约算法第一次迭代中的数据传输](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/scatter-reduce-iteration-1.png)

scatter-reduce 第二次迭代

![第一次 scatter-reduce 迭代完成后的中间和](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/scatter-reduce-iteration-2.png)

最终

![所有分散归约转移后的最终状态](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/scatter-reduce-iteration-done.png)

> [!NOTE]
>
> 上述迭代过程是 scatter-reduce 的过程
>
> 但是 scatter-reduce 和 ring allreduce 的过程 **完全相同**
>
> 不同之处在于，ring allreduce 中 GPU 接收到的数据不是累加的，而是 **直接覆盖**
>
> 这个不同就会导致经过 ring allreduce 后的各 GPU 中数据如下图
>
> ![所有 allgather 转移后的最终状态](https://andrew.gibiansky.com/blog/machine-learning/baidu-allreduce/images/allgather-iteration-done.png)



**通信成本分析**

Ring AllReduce 分为两个阶段

- Scatter-Reduce

  每张 GPU 发送一次，接收一次，每次传输大小为 $\frac{K}{N}$ 的梯度

- AllGather

  也是每张 GPU 发送一次，接收一次，每次传输大小为 $\frac{K}{N}$ 的梯度

因此可以得出


$$
Data \ Transferred = 2\frac{N-1}{N}K
$$

可以看到随着 N 增大，这个开销是收敛到 2K 的 （其实应该是 4K，但是许多人认为只有 send 时存在开销所以忽略了接收的开销）关键在于，这**与 GPU 的数量无关**。

由于所有传输都是在离散迭代中同步进行的，因此 allreduce 的速度受限于环中**相邻 GPU 之间最慢（带宽最低）的连接**。如果为每个 GPU 选择合适的邻居，则该算法**在带宽方面是最优的**，并且是执行 allreduce 的最快算法（假设延迟成本相对于带宽可以忽略不计）。一般来说，如果节点上的所有 GPU 在环中彼此相邻（全连接），则该算法运行效果最佳；这可以最大限度地减少网络争用，否则网络争用可能会显著降低 GPU-GPU 连接的有效带宽。



#### DDP 的核心优化

##### Gradient Bucketing

DDP 在构建阶段会为模型参数注册 **autograd hook**

反向传播过程中，当某个参数的梯度计算完成时，对应的 hook 会通知 Reducer 这个梯度已经就绪

Reducer 不会等待整个 backward 结束，而是将多个梯度按照 bucket 分组

每个 bucket 默认大小由 `bucket_cap_mb = 25MB`

这些 bucket 通常按照 parameters 的反向顺序构建的，因为 backward 是从后往前，所以后层的梯度会更早填满 bucket 从而更早启动通讯



##### Overlap Communication

当某个 bucket 中所有梯度都就绪的时候，Reducer 会立刻 **异步启动 AllReduce**

而 backward 和 AllReduce 互不冲突可以并行操作

这样利用了通讯的空等时间



#### 实现结构

![ddp_code.png](https://github.com/user-attachments/assets/b6fe5258-f1f1-4c73-8e1d-6fd96406faa2)



Reference: PyTorch Distributed: Experiences on Accelerating Data Parallel Training

 https://arxiv.org/abs/2006.15704

![ddp_algorithm](https://github.com/Beater-0x7ff/PyTorch-SourceCode-Overview/blob/main/pytorch%E6%BA%90%E7%A0%81%E8%A7%A3%E6%9E%90/figures/ddp_algorithm.svg)

这段伪代码在讲 **DDP 如何在 backward 中同步梯度**

- **constructor**

  初始化的时候做三件事

  - rank 0 把模型参数广播给其它进程
  - 按反向传播顺序构建 gradient buckets
  - 给每个参数注册 autograd hook

- **forward**

  如果 `self.find_unused_parameters = True`，DDP 会在 forward 结束时 traverse autograd graph 找到所有没用过的parameters 并标记为 ready。这一步开销很大，但是有时计算动态图会改变，所以很必要

- **autograd_hook**

  当某个参数的梯度计算完成后

  - 找到这个参数属于哪个 bucket
  - 找到它在 bucket 中的位置 offset
  - 把 param.grad 复制到 bucket 对应的位置
  - 如果 bucket 中所有梯度都 ready，就标记 bucket ready
  - 按顺序对 ready bucket 启动异步 AllReduce
  - 所有 bucket 都ready 后，等待 AllReduce 完成，将得到的梯度写道 param.grad



#### 通讯

因为 DDP 依赖 c10d 的 ProcessGroup 进行通信，所以开始前我们先要有个 ProcessGroup 实例。这步可以通过 `torch.distributed.init_process_group` 实现。



#### 构建

 现在的 `class DistributedDataParallel` 太长了，读者可以前往 [DDP 源码](https://github.com/pytorch/pytorch/blob/6f8064045c77de01e541de37490867bebd6e466e/torch/nn/parallel/distributed.py#L466) 阅读，这里我们改为摘取初始化主线结构进行讲解

```
DistributedDataParallel.__init__()
    |
    |-- 检查 process_group / device_mesh
    |-- 收集需要同步的 parameters
    |-- 检查 device_ids、参数设备、requires_grad
    |-- 设置 bucket_cap_mb / find_unused_parameters 等配置
    |-- init_sync:
    |       |-- 校验各进程参数 shape 一致
    |       |-- 从 rank 0 广播参数和 buffer
    |
    |-- _build_params_for_reducer()
    |       |-- 收集 reducer 需要管理的参数
    |
    |-- _ddp_init_helper()
            |-- 计算 bucket assignment
            |-- 创建 Reducer
            |-- 注册 reducer/autograd hook
```

可以看到每个 DDP 进程都会创建本地 Reducer 在 backward 时管理梯度

```python
self.reducer = dist.Reducer(

    # 需要被 DDP 管理和同步梯度的参数列表
    parameters,

    # 每个 bucket 内包含哪些 parameter
    # reverse order 是为了匹配 backward 顺序
    list(reversed(bucket_indices)),

    # 每个 bucket 的大小限制
    # 控制 bucket 如何切分
    list(reversed(per_bucket_size_limits)),

    # 当前使用的通信组
    # reducer 会通过它执行 allreduce
    self.process_group,

    # bucket 总大小上限（字节）
    # 对应 bucket_cap_mb
    self.bucket_bytes_cap,

    # 是否检测 unused parameter
    # 如果某些参数没有参与 forward/backward
    # reducer 需要特殊处理
    self.find_unused_parameters,

    # 是否让 grad 直接作为 bucket view
    # 可以减少 grad copy
    self.gradient_as_bucket_view,

    # parameter -> parameter name
    # 主要用于 debug/log/error message
    param_to_name_mapping,

    # 第一个 bucket 的特殊大小限制
    # 用于尽早启动 overlap communication
    self._bucket_config.first_bucket_bytes_cap,

    # 是否跳过 unused parameter 的 allreduce
    self.skip_all_reduce_unused_params,

    # 是否使用 python reducer
    # 通常默认 False（C++ reducer 更快）
    self._use_python_reducer,

    # bucket rebuild 时使用的 bucket size 配置
    bucket_size_limits_for_rebuilding,

    # 是否批量进行 grad copy
    # 减少小 tensor copy 开销
    self.batched_grad_copy,
)
```

而这里的 Reducer 是通过 `torch.csrc.distributed/c10d/reducer.cpp` 实现的，构建 Reducer 时， 除了各种初始化，最重要的一步就是

```cpp
const auto variable_count = params_.size();
grad_accumulators_.resize(variable_count);

for (const auto variable_index : c10::irange(variable_count)) {
    auto& variable = params_[variable_index];

    auto grad_accumulator =
        torch::autograd::impl::grad_accumulator(variable);

    hooks_.emplace_back(
        grad_accumulator->add_post_hook(
            std::make_unique<LambdaPostHook>(
                [this, variable_index](...) {
                    this->autograd_hook(variable_index);
                    return outputs;
                }
            )
        ),
        grad_accumulator
    );

    if (find_unused_parameters_) {
        gradAccToVariableMap_[grad_accumulator.get()] = variable_index;
    }

    grad_accumulators_[variable_index] = std::move(grad_accumulator);
}
```

OpenMMLab 大佬的教程中使用的 c++ reducer.cpp 在 PyTorch 2.12.0 版本已经被重构为上面的代码。旧版 Reducer 还在兼容 **一个 DDP 进程内可能有多个 model replica** 的结构；新版 Reducer 更偏向现代推荐用法，即 **一进程一 GPU / 一进程一个本地模型副本**，所以内部结构从二维变成了一维。



其中，`autograd_hook` 的作用就是在 `parameter.grad` 产生时被触发，`Reducer` 就知道参数 ready 了，之后再检查 bucket 也 ready 了就可以进行 AllReduce 了。源码如下：

```cpp
void Reducer::autograd_hook(size_t index) {
  std::lock_guard<std::mutex> lock(this->mutex_);
  if (!first_autograd_hook_called_) {
    first_autograd_hook_called_ = true;
    num_bwd_calls_++;
  }

  if (dynamic_graph_find_unused() || static_graph_first_iteration()) {

    auto& variable = get_param_from_index(index);
    runGradCallbackForVariable(variable, [&](auto& grad) {
      if (grad.defined()) {
        local_used_map_[static_cast<int64_t>(index)] = 1;
      }
      return false;
    });
  }

  if (static_graph_first_iteration()) {
    numGradHooksTriggeredMap_[index] += 1;
    return;
  }

  if (!expect_autograd_hooks_) {
    return;
  }

  grad_ready_order_indices_.push_back(static_cast<int64_t>(index));

  if (!has_marked_unused_parameters_) {
    has_marked_unused_parameters_ = true;
    for (const auto& unused_index : unused_parameters_) {
      mark_variable_ready(unused_index);
    }
  }

  if (static_graph_after_first_iteration()) {
    REDUCER_CHECK(
        numGradHooksTriggeredMapPerIteration_[index] > 0,
        logger_,
        "Your training graph has changed in this iteration, ",
        "e.g., one parameter is unused in first iteration, but ",
        "then got used in the second iteration. this is not ",
        "compatible with static_graph set to True.");
    if (--numGradHooksTriggeredMapPerIteration_[index] == 0) {
      if (should_rebuild_buckets()) {
        push_rebuilt_params(index);
      }
      mark_variable_ready(index);
    }
  } else {
    if (should_rebuild_buckets()) {
      push_rebuilt_params(index);
    }
    mark_variable_ready(index);
  }
}
```



**前向传播**

```python
    def forward(self, *inputs, **kwargs):
        with torch.autograd.profiler.record_function("DistributedDataParallel.forward"):
            inputs, kwargs = self._pre_forward(*inputs, **kwargs)
            output = (
                self.module.forward(*inputs, **kwargs)
                if self._delay_all_reduce_all_params
                else self._run_ddp_forward(*inputs, **kwargs)
            )
            return self._post_forward(output)
```

旧版的前向传播完整的实现了前向传播的计算，而新版更加模块化，被拆解成了 `_pre_forward()`， `_run_ddp_forward()` ，`_post_forward()`

```python
def _run_ddp_forward(self, *inputs, **kwargs):
    # 真正执行被 DDP 包装的原始 module.forward。
    # DDP 自己的前后处理逻辑分别放在 _pre_forward 和 _post_forward 中。

    if self._use_python_reducer:
        # 如果使用 Python reducer，直接调用原模型。
        return self.module(*inputs, **kwargs)
    else:
        # 进入 DDP forward 上下文，主要用于标记当前线程正在执行 DDP forward。
        with self._inside_ddp_forward():
            return self.module(*inputs, **kwargs)


def _pre_forward(self, *inputs, **kwargs):
    # DDP forward 前的准备工作：
    # lazy init、Reducer 状态准备、bucket rebuild、buffer 同步、输入搬运等。

    if self._use_python_reducer:
        # Python reducer 路径下，不走 C++ reducer 的 forward 前处理。
        return inputs, kwargs

    if (
        not self._lazy_init_ran
        and not torch.compiler.is_compiling()
    ):
        # 第一次 forward 前执行 lazy initialization。
        # 例如 optimizer-in-backward 相关 hook 的设置。
        self._lazy_init()

    if self._delay_all_reduce_all_params:
        # 如果所有参数都走 delayed allreduce，就跳过普通 DDP reducer 流程。
        return inputs, kwargs

    if (
        torch.is_grad_enabled()
        and self.require_backward_grad_sync
    ):
        # 只有训练模式且需要梯度同步时，才通知 reducer 准备 forward/backward 统计。
        if self.logger is None:
            raise AssertionError(
                "self.logger must not be None"
            )

        # 记录运行时日志信息。
        self.logger.set_runtime_stats_and_log()

        # 通知 reducer：一次新的 forward 开始。
        self.reducer.prepare_for_forward()

    # Join mode 相关：
    # 用于处理不同 rank 输入数量不一致时，已经结束的 rank 仍能参与必要通信。
    work = Join.notify_join_context(self)

    if work:
        # 把 forward 阶段的通信句柄交给 reducer。
        # 后续可用于动态 world size / join mode 下的梯度缩放。
        self.reducer._set_forward_pass_work_handle(
            work,
            self._divide_by_initial_world_size,
        )

    if (
        torch.is_grad_enabled()
        and self.reducer._rebuild_buckets()
    ):
        # 根据真实 backward 梯度到达顺序重建 bucket。
        # 这样后续 bucket 更容易和 backward 顺序匹配，实现通信计算重叠。
        logger.info(
            "Reducer buckets have been rebuilt in this iteration."
        )

        self._has_rebuilt_buckets = True

    if self._check_sync_bufs_pre_fwd():
        # 如果配置要求在 forward 前同步 buffer，则执行同步。
        # 典型 buffer 包括 BatchNorm 的 running_mean / running_var。
        self._sync_buffers()

    if self._join_config.enable:
        # Join mode 下通知其它 rank：
        # 当前 rank 本轮 backward 是否需要同步梯度。
        self._check_global_requires_backward_grad_sync(
            is_joined_rank=False
        )

    if self.device_ids:
        # 如果 DDP 管理具体 device_ids，
        # 将输入移动到当前进程对应的 device 上。

        moved_inputs, moved_kwargs = _to_kwargs(
            inputs,
            kwargs,
            torch.device(
                self.device_type,
                self.device_ids[0],
            ),
            self.use_side_stream_for_tensor_copies,
        )

        args, kwargs = moved_inputs[0], moved_kwargs[0]

        if self.mixed_precision is not None:
            # 如果启用 DDP mixed precision，
            # 将 forward 输入 cast 到指定低精度 dtype。
            args, kwargs = _cast_forward_inputs(
                self.mixed_precision.param_dtype,
                *args,
                **kwargs,
            )

        return args, kwargs

    else:
        # device_ids=None 时，DDP 不主动移动输入。
        # 用户需要确保 input/module 已经在正确设备上。

        if self.mixed_precision is not None:
            # 同样处理 mixed precision 输入 cast。
            inputs, kwargs = _cast_forward_inputs(
                self.mixed_precision.param_dtype,
                *inputs,
                **kwargs,
            )

        return inputs, kwargs


def _post_forward(self, output):
    # DDP forward 后的处理：
    # buffer 同步、准备 backward、unused parameter 检测、DDPSink 包装输出等。

    if self._use_python_reducer:
        # Python reducer 路径直接返回输出。
        return output

    if self._delay_all_reduce_all_params:
        # delayed allreduce 模式下，清理延迟梯度 buffer 后返回。
        self._clear_grad_buffer()
        return output

    if self._check_sync_bufs_post_fwd():
        # 如果配置要求在 forward 后同步 buffer，则执行同步。
        self._sync_buffers()

    if (
        torch.is_grad_enabled()
        and self.require_backward_grad_sync
    ):
        # 如果本轮需要 backward 梯度同步，
        # 下一轮 forward 前也需要保证参数/状态同步。
        self.require_forward_param_sync = True

        if (
            self.find_unused_parameters
            and not self.static_graph
        ):
            # 如果启用 unused parameter 检测，
            # 从 output 中找出 tensor，并让 reducer 遍历 autograd graph。
            # 没出现在图中的参数会被提前标记 ready。
            self.reducer.prepare_for_backward(
                list(_find_tensors(output))
            )

        else:
            # 不检测 unused parameter 时，直接准备 backward。
            self.reducer.prepare_for_backward([])

    else:
        # no_grad 或 no_sync 情况下，不需要下一轮 forward 参数同步。
        self.require_forward_param_sync = False

    if (
        self.find_unused_parameters
        and not self.static_graph
    ) or (
        self.static_graph
        and not self._static_graph_delay_allreduce_enqueued
    ):
        # 在 unused parameter 检测或 static graph 特殊路径下，
        # DDP 会通过 DDPSink 包装输出，使 backward 能正确触发 reducer 逻辑。

        (
            output_tensor_list,
            treespec,
            output_is_rref,
        ) = _tree_flatten_with_rref(output)

        # 用于保存最终输出结构中的 tensor。
        output_placeholders = [
            None
            for _ in range(len(output_tensor_list))
        ]

        for i, output in enumerate(output_tensor_list):

            if (
                torch.is_tensor(output)
                and output.grad_fn is None
            ):
                # 没有 grad_fn 的 tensor 不参与 DDPSink，
                # 避免破坏不需要梯度的输出。
                output_placeholders[i] = output

        # 让需要梯度的输出经过 DDPSink。
        # 这样即使部分 output 没被 loss 使用，
        # reducer 也能得到正确的 undefined grad 信号。
        passthrough_tensor_list = _DDPSink.apply(
            weakref.ref(self),
            *output_tensor_list,
        )

        for i in range(len(output_placeholders)):

            if output_placeholders[i] is None:

                output_placeholders[i] = (
                    passthrough_tensor_list[i]
                )

        # 按原始结构重建 output。
        output = _tree_unflatten_with_rref(
            output_placeholders,
            treespec,
            output_is_rref,
        )

    # forward 结束时清理 delayed grad buffer / grad views。
    self._clear_grad_buffer()

    return output
```



**反向传播**

这一部分我们主要研究 DDP 究竟是怎么启动 AllReduce 的，这就要看一下 Bucket 的定义和用法了（新版的 Bucket 被放到了 `torch.csrc.distributed.c10d.reducer.hpp`）主要是几个 `mark_*_ready`

```cpp
struct Bucket {

    at::Tensor gradients;
    
    std::vector<at::Tensor> bucket_views_in;
    std::vector<at::Tensor> bucket_views_out;

    std::vector<at::Tensor> variables;

    std::vector<size_t> offsets;
    std::vector<size_t> lengths;

    std::vector<c10::IntArrayRef> sizes_vec;

    size_t pending;

    std::vector<size_t> variable_indices;

    c10::intrusive_ptr<at::ivalue::Future> future_work;

    bool is_complex_bucket = false;

    bool expect_sparse_gradient = false;

    std::optional<at::Tensor> sparse_tensor_indices = std::nullopt;

    std::vector<size_t> deferred_copy_indices;
  };
```



看到 `mark_variable_ready` 源码

```cpp
void Reducer::mark_variable_ready(size_t variable_index) {
  // 检查参数 index 是否合法。
  REDUCER_CHECK(
      variable_index < variable_locators_.size(),
      logger_,
      "Out of range variable index.");

  // 检查该参数在本轮 backward 中是否被重复标记 ready。
  checkAndRaiseMarkedTwiceError(variable_index);

  // 记录该参数在当前 iteration 中已经 ready。
  perIterationReadyParams_.insert(variable_index);

  // 记录该参数梯度 ready 的时间，用于 DDP runtime stats。
  backward_stats_[variable_index] =
      current_time_in_nanos() - backward_compute_start_time_;

  // 只要有参数被标记 ready，本轮 backward 最后就必须执行 finalize_backward。
  require_finalize_ = true;

  // 根据 variable_index 找到该参数属于哪个 bucket，
  // 以及它在 bucket 内部的位置。
  const auto& bucket_index = variable_locators_[variable_index];

  // 获取对应 bucket。
  auto& bucket = buckets_[bucket_index.bucket_index];

  // 设置梯度缩放因子。
  // 通常是 world_size；join mode 下可能是动态参与训练的 rank 数。
  set_divide_factor();

  // 根据 bucket 是否是 sparse gradient bucket，
  // 选择不同的 grad 写入逻辑。
  if (bucket.expect_sparse_gradient) {
    // sparse gradient 不能和 dense gradient 一样 flatten 到大 bucket 中。
    mark_variable_ready_sparse(variable_index);
  } else {
    // dense gradient 会被 copy / alias 到 bucket view 中。
    mark_variable_ready_dense(variable_index);
  }

  // 当前参数已经 ready，因此该 bucket 中待完成参数数量减 1。
  // 如果 pending 变成 0，说明 bucket 内所有参数梯度都已经 ready。
  if (--bucket.pending == 0) {

    // 如果启用了 batched_grad_copy_，
    // 之前延迟的 grad -> bucket copy 会在 bucket ready 时统一执行。
    if (batched_grad_copy_) {
      flush_deferred_copies(bucket, bucket_index.bucket_index);
    }

    // 标记 bucket ready，并尝试按顺序启动该 bucket 的 allreduce。
    mark_bucket_ready(bucket_index.bucket_index);
  }

  // 如果 next_bucket_ 已经走到 buckets_.size()，
  // 说明所有 bucket 的 allreduce 都已经被启动。
  if (next_bucket_ == buckets_.size()) {

    // 如果启用了动态 unused parameter 检测，
    // 需要同步 local_used_map，用于判断参数是否在所有 rank 上都未使用。
    if (dynamic_graph_find_unused()) {
      all_reduce_local_used_map();
    }

    // 将 finalize_backward 注册为 autograd engine 的 callback。
    // 它会在 backward 主流程结束后执行。
    torch::autograd::Engine::get_default_engine().queue_callback([this] {
      std::lock_guard<std::mutex> lock(this->mutex_);

      // 记录 backward 计算结束时间。
      if (should_collect_runtime_stats()) {
        record_backward_compute_end_time();
      }

      // 确认所有 bucket 的通信都已经被启动。
      TORCH_INTERNAL_ASSERT(next_bucket_ == buckets_.size());

      // static graph 模式下，如果需要 rebuild bucket，
      // 还要把 unused parameters 也加入 rebuild 参数列表。
      if (static_graph_after_first_iteration() && should_rebuild_buckets()) {
        for (const auto& unused_index : unused_parameters_) {
          push_rebuilt_params(unused_index);
        }
      }

      // 等待 allreduce 完成，并将 bucket 中的结果写回 param.grad。
      this->finalize_backward();
    });
  }
}
```



`mark_bucket_ready` 源码有

```cpp
void Reducer::mark_bucket_ready(size_t bucket_index) {
  // 确保当前 ready 的 bucket 不在 next_bucket_ 之前。
  // DDP 要求 bucket 按顺序启动 allreduce。
  TORCH_INTERNAL_ASSERT(bucket_index >= next_bucket_);

  // 如果当前 bucket 不是下一个应该 reduce 的 bucket，
  // 即使它已经 ready，也先不启动通信。
  // 需要等待前面的 bucket 先完成 ready。
  if (bucket_index > next_bucket_) {
    return;
  }

  // 从 next_bucket_ 开始，连续检查已经 ready 的 bucket。
  // 只要当前 bucket.pending == 0，就可以启动它的 allreduce。
  for (
      ;
      next_bucket_ < buckets_.size() &&
      buckets_[next_bucket_].pending == 0;
      next_bucket_++
  ) {
    // 记录 ready bucket 数量。
    num_buckets_ready_++;

    // 第一个 bucket ready 时，记录 backward 通信开始时间。
    if (num_buckets_ready_ == 1 && should_collect_runtime_stats()) {
      record_backward_comm_start_time();
    }

    // 获取当前要 reduce 的 bucket。
    auto& bucket = buckets_[next_bucket_];

    // 如果这个 bucket 不需要跳过 allreduce，
    // 就真正启动 bucket 通信。
    if (!should_skip_all_reduce_bucket(bucket)) {
      // 对当前 bucket 启动 allreduce 或自定义 comm hook。
      all_reduce_bucket(bucket);

      // 记录实际被 reduce 的 bucket 数量。
      num_buckets_reduced_++;
    }
  }
}
```



刚刚我们在 `autograd_hook` 时提到，DDP 在构建时会为每个 parameter 的 `grad_accumulator` 注册 `autograd hook`。当某个参数的梯度在 backward 中计算完成后，对应 hook 会被触发，并通知 Reducer 当前梯度已经 ready。Reducer 会将梯度写入对应 bucket，并在整个 bucket 中所有梯度都 ready 后立刻启动异步 allreduce，从而实现 backward 计算与通信的 overlap，提高训练效率。

但是如果模型训练过程中只使用部分 subgraph（例如条件分支、MoE、多任务等），某些参数可能不会参与当前 iteration 的 autograd graph，此时对应 hook 永远不会触发，bucket 的 pending 计数也无法归零，最终导致 DDP 卡死。

**因此 `find_unused_parameters=True` 会在 forward 后额外遍历 autograd graph，提前找出未使用参数并将其标记为 ready，保证 bucket 能正常完成归约。** 这就解释了为什么上文提到的 `find_unused_parameters=True` 开销很大，但是仍要启动的原因。

DDP 本身只负责同步梯度，`optimizer.step()` 是各个 rank 独立执行的；由于所有 rank 的初始参数和 allreduce 后的梯度都相同，因此最终参数更新结果也会保持一致。

而 `no_sync()` 则允许多轮 backward 只累积梯度而不执行 allreduce，从而实现 gradient accumulation 并减少通信开销。

```python
@contextmanager
def no_sync(self):

    old_require_backward_grad_sync = self.require_backward_grad_sync
    self.require_backward_grad_sync = False
    try:
        yield
    finally:
        self.require_backward_grad_sync = old_require_backward_grad_sync
```
