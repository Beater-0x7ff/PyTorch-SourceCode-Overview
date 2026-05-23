# PyTorch源码略解

## 声明

本仓库的整体结构设计、部分代码实现、部分文字描述与思路参考并借鉴自以下优秀教程：

https://zhuanlan.zhihu.com/p/328674159

非常感谢原作者的开源分享与工作。

本人在自学 PyTorch 源码的过程中，发现目前网络上较系统的源码分析教程相对较少，且很多内容已经是 3~4 年前的版本。随着 PyTorch 的持续更新，部分接口、调用逻辑以及底层实现已经发生了变化。因此，本项目在参考上述教程的基础上，进行了大量的：

- 代码更新与适配
- 内容重写与补充
- 行间注释添加
- 示例与实验补充
- 调用结构与源码逻辑整理

希望能够帮助读者更好地理解 PyTorch 的底层实现，代码结构，调用逻辑。

**本项目完全开源，仅供学习交流与研究用途，禁止商业用途**

如果您在文章、项目或其它内容中引用了本仓库的内容，请同时注明原教程与本仓库出处。

本人目前仍然是 PyTorch 源码学习阶段的初学者，项目中的部分理解、表述或代码实现可能存在错误、不严谨或遗漏之处。如发现问题，欢迎通过 Issues 或 Discussions 提出反馈与建议。

希望大家能够互相学习、共同进步。

目前已完成：

- [x] torch.autograd：梯度计算与计算图机制
- [x] BatchNorm & SyncBatchNorm：BN 与多卡同步 BN
- [x] torch.utils.data：DataLoader 与数据处理流程
- [x] nn.Module：模块系统、Hook、state_dict、参数管理

计划更新：

- [ ] DistributedDataParallel (DDP) 模型并行和分布式训练
- [ ] torch.optim 优化算法接口
- [ ] torch.cuda.amp 自动混合精度
- [ ] cpp_extension C++/CUDA 算子实现和调用


## Main Topics

详见：[pytorch.md](./pytorch.md)
