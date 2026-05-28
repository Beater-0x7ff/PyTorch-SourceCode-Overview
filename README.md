# PyTorch源码略解

## 声明

本仓库的整体结构设计、部分代码实现、部分文字描述与思路参考并借鉴自以下优秀教程：

- https://zhuanlan.zhihu.com/p/328674159

非常感谢原作者的开源分享与工作。

在自学 PyTorch 源码的过程中，我发现目前网络上较系统的源码分析教程相对较少，且很多内容已经是 3~4 年前的版本。随着 PyTorch 的持续更新，部分接口、调用逻辑以及底层实现已经发生了变化。

因此，本项目在参考上述教程的基础上，进行了大量：

- 接口更新与版本适配
- 内容重写与补充
- 行间源码注释添加
- 调用流程与执行逻辑整理
- 示例与实验补充
- 部分源码行为验证

希望能够帮助读者更好地理解 PyTorch 的底层实现，代码结构，调用逻辑。

> **本项目完全开源，仅供学习交流与研究用途，禁止商业用途。**

如果您在文章、项目或其它内容中引用了本仓库的内容，请同时注明原教程出处，本仓库地址

---

## 反馈与讨论

本人目前仍然处于 PyTorch 源码学习阶段。

项目中的部分理解、表述或代码实现可能存在错误，

如果您发现问题，欢迎通过：Issues，Discussions 进行反馈与讨论。

希望大家能够互相学习、共同进步。

---

# 当前进度

## 已完成

- [x] [torch.autograd：梯度计算与计算图机制](https://github.com/ZhuangshengLi/PyTorch-SourceCode-Overview/wiki/torch.autograd)

- [x] [BatchNorm & SyncBatchNorm：BN 与多卡同步 BN](https://github.com/ZhuangshengLi/PyTorch-SourceCode-Overview/wiki/BN-&-SyncBN)

- [x] [torch.utils.data：DataLoader 与数据处理流程](https://github.com/ZhuangshengLi/PyTorch-SourceCode-Overview/wiki/torch.utils.data)

- [x] [nn.Module：模块系统、Hook、state_dict、参数管理](https://github.com/ZhuangshengLi/PyTorch-SourceCode-Overview/wiki/nn.Module)

- [x] [DistributedDataParallel (DDP)](https://github.com/ZhuangshengLi/PyTorch-SourceCode-Overview/wiki/数据并行)

---

## 计划更新

- [ ] torch.optim  
      优化器与参数更新机制

- [ ] torch.cuda.amp  
      自动混合精度（AMP）

- [ ] cpp_extension  
      C++ / CUDA 自定义算子实现与调用


## 参考版本

- **Python**: 3.10
- **PyTorch**: 2.12.0


