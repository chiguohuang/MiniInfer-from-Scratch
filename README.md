# MiniInfer

从零实现 DDP（数据并行）的核心机制，多卡与纯 CPU 均可运行。**当前目标只有这一件事**；后续的方向（更多并行技术、分布式推理等）还没有想好，等 DDP 做完再规划。

不依赖任何分布式框架，用最少量的代码拆解数据并行的核心机制：每一步都从「最小可运行示例」开始，先把通信和同步逻辑写对，再逐步叠加工程优化。

## Key Features

* 从最小正确实现开始，逐个拆解 DDP 的核心机制：进程组初始化、数据切分、梯度 all-reduce、梯度分桶与通信/计算重叠；
* 先跑通「每步全量同步梯度」的朴素版本，再演进到带分桶与反向 hook 的版本，看清 `torch.nn.parallel.DistributedDataParallel` 每一层封装在做什么；
* 代码支持**多卡设备**和**纯 CPU** 运行，哪怕只用一台笔记本，也可以跑通全部示例，足够理解实现逻辑。

## Roadmap

- [ ] `src/naive-ddp`：朴素 DDP 最小实现——进程组初始化、`DistributedSampler` 数据切分、每步训练结束后 all-reduce 梯度，并与单卡训练对齐 loss，验证数值一致性；
- [ ] `src/bucketed-ddp`：工程化演进——梯度分桶与 `autograd` 反向 hook，把通信与反向计算重叠，对比朴素版本的加速效果。

> DDP 之后的方向（张量并行、流水线并行、ZeRO、分布式推理等）还没想好，确定后再更新这里的规划。

## Quick Start

> 第一个示例完成后补充具体的运行命令。规划中的使用方式与 [MiniDDP](https://github.com/Pearblossom-M/MiniDDP-from-Scratch) 保持一致：

```bash
git clone https://github.com/chiguohuang/MiniInfer-from-Scratch.git
cd MiniInfer-from-Scratch/src/naive-ddp
torchrun --standalone --nproc_per_node=2 train.py
```

纯 CPU 运行时，将入口脚本的 `use_cpu` 参数设置为 `True` 即可，无需任何代码改动。

## Environment

* Python ≥ 3.9
* PyTorch ≥ 2.0（示例只依赖 `torch` 与其自带的 `torchrun`，无需安装其他包）

## Tutorial

详细教程撰写中，将随代码逐章更新，与 `src/` 下各示例一一对应。

## References

* [MiniDDP-from-Scratch](https://github.com/Pearblossom-M/MiniDDP-from-Scratch)——与当前目标最接近的同类项目，建议对照阅读；
* [PyTorch Distributed Overview](https://docs.pytorch.org/tutorials/distributed/home.html)——分布式通信原语的官方参考。
