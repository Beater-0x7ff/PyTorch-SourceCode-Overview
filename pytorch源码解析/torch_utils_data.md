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