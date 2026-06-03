# cpp_extension c++/CUDA 算子实现和调用

PyTorch 的逻辑是用 Python 编写高层逻辑，而用 C++/CUDA 编写高性能算子，通过 `torch.utils.cpp_extension` 编译成 Python 可调用模块





## 扩展的调用方式

PyTorch 将拓展分成两类理解

- Python 直接调用 `pybind` 函数           `ext_module.nms(...)`
- 注册为 PyTorch Custom Operator         `torch.ops.my_namespace.my_op(...)`  (更推荐)



PyTorch 官方文档指出：如果只是通过 pybind 暴露 C++ 函数，虽然可以运行，但它不一定能很好的组合进 `autograd`, `torch.compile`, `vmap` 等 PyTorch 子系统，更完整的融入 PyTorch 的方案是使用 operator registration API





## setup.py

```python
from setuptools import setup
from torch.utils.cpp_extension import CppExtension, CUDAExtension, BuildExtension

setup(
    # 扩展模块的包名，安装后可以通过 import extension_cpp 导入
    name="extension_cpp",

    # 定义需要编译的扩展模块
    ext_modules=[
        CppExtension(
            # 编译生成的 Python 扩展模块名
            name="extension_cpp",

            # 参与编译的 C++ 源文件
            # 这里表示只编译 muladd.cpp
            sources=["muladd.cpp"],
        )
    ],

    # 指定使用 PyTorch 提供的 BuildExtension 来编译扩展
    # 它会自动处理 PyTorch 头文件路径、库路径、编译参数等
    cmdclass={"build_ext": BuildExtension},
)
```

如果是导入 CUDA 文件则使用

```python
CUDAExtension(
    name="extension_cuda",
    sources=["muladd.cpp", "muladd_kernel.cu"],
)
```

`CppExtension` / `CUDAExtension` 本质上仍然是对 `setuptools.Extension` 的封装，会自动加入 PyTorch 的 include 路径，library 路径和必要编译参数。PyTorch 官方文档仍然把 `torch.utils.cpp_extension` 作为构建 C++/CUDA 拓展的标准入口



## PYBIND11_MODULE

`PYBIND11_MODULE` 是 python 与 c++ 的桥梁

旧的写法是

```python
#include <torch/extension.h>

Tensor nms(Tensor boxes, Tensor scores, float iou_threshold, int offset);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("nms", &nms, "nms (CPU/CUDA)",
          py::arg("boxes"),
          py::arg("scores"),
          py::arg("iou_threshold"),
          py::arg("offset"));
}
```

这段代码是将 `Tensor nms(...)`  注册成 Python 模块里的函数 `ext_module.nms`

其中 `TORCH_EXTENSION_NAME` 会由编译命令自动定义，例如 `-DTORCH_EXTENSION_NAME=_ext`

所以在 Python 中最后会被转变成 `mmcv._ext.nms`



**现在更推荐的方式是** 使用 `TORCH_LIBRARY`, `TORCH_LIBRARY_IMPL` 例如

```python
#include <torch/library.h>

torch::Tensor mymuladd_cpu(torch::Tensor a, torch::Tensor b, double c);

TORCH_LIBRARY(my_ops, m) {
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
}

TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
}
```

Python 端调用

```python
torch.ops.my_ops.mymuladd(a, b, 1.0)
```

这种方式比单纯 pybind11 更像 PyTorch 原生算子，能更好地接入 dispatcher、autograd、fake tensor、`torch.compile` 等机制。官方 Custom Operators 页面也强调了这一点



## 算子的具体实现

### cpp 算子实现

把分发逻辑交给 Pytorch dispatcher 处理

```c++
TORCH_LIBRARY_IMPL(my_ops, CPU, m) {
    m.impl("mymuladd", &mymuladd_cpu);
}

TORCH_LIBRARY_IMPL(my_ops, CUDA, m) {
    m.impl("mymuladd", &mymuladd_cuda);
}
```

这样 Python 无需自己判断 kernel 选择

- 如果 x 在 CPU 就走 CPU kernel

- 如果在 CUDA, 就走 CUDA kernel



在算子的实现，官方更推荐使用 `#include <torch/torch.h>`, `#include <ATen/ATen.h>`, 例如

```c++
#include <torch/extension.h>

torch::Tensor mymuladd_cpu(torch::Tensor a, torch::Tensor b, double c) {
    TORCH_CHECK(a.device().is_cpu(), "a must be a CPU tensor");
    TORCH_CHECK(b.device().is_cpu(), "b must be a CPU tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have the same shape");

    return a * b + c;
}
```



### CUDA 算子实现

PyTorch 拓展中常用 `cudaStream_t stream = at::cuda::getCurrentCUDAStream();`

这样 CUDA kernel 会接入 PyTorch 当前 stream

CUDA kernel 例如

```c++
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void mymuladd_kernel(
    const float* a,
    const float* b,
    float* out,
    float c,
    int64_t n
) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        out[idx] = a[idx] * b[idx] + c;
    }
}

torch::Tensor mymuladd_cuda(torch::Tensor a, torch::Tensor b, double c) {
    TORCH_CHECK(a.is_cuda(), "a must be CUDA tensor");
    TORCH_CHECK(b.is_cuda(), "b must be CUDA tensor");
    TORCH_CHECK(a.sizes() == b.sizes(), "a and b must have same shape");
    TORCH_CHECK(a.dtype() == torch::kFloat32, "only float32 is supported");

    auto out = torch::empty_like(a);

    int64_t n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    mymuladd_kernel<<<blocks, threads, 0, stream>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<float>(c),
        n
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return out;
}
```

其中
| 概念             | 含义                     |
| -------------- | ---------------------- |
| `blockIdx`     | 当前 block 在 grid 中的位置   |
| `threadIdx`    | 当前 thread 在 block 中的位置 |
| `blockDim`     | 每个 block 的线程维度         |
| `gridDim`      | grid 的 block 维度        |
| `cudaStream_t` | 当前 CUDA stream         |



==本篇写的非常粗略，意图给读者一个大概的实现自定义算子的流程==
