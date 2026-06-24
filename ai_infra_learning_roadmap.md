# AI Infra / 大模型基础设施学习路线

适用对象：已有一定编程能力的前后端工程人员，希望转向 AI Infra、LLM Serving、训练平台、推理引擎、GPU 平台、分布式训练工程等岗位。

参考岗位：B 站校园招聘截图中的「模型开发工程师 AI Infra（实习）」。

截图岗位发布时间：2026-06-22。

当前整理日期：2026-06-23。

重要提醒：截图里的实习岗位要求「仅限 2026-09 至 2030-06 毕业的大学生」。如果不符合这个实习身份，仍然可以按这条路线准备类似的正式岗或社招岗，例如 AI Infra 工程师、LLM Serving 工程师、推理引擎工程师、训练平台工程师、GPU 平台工程师。

---

## 1. 岗位本质

这个岗位的核心不是普通 AI 应用开发，也不是只做算法建模，而是做大模型背后的系统工程：

- 大模型训练框架
- 大模型推理框架
- GPU 算子优化
- 分布式通信优化
- GPU 集群资源调度
- 训练和推理服务稳定性
- 大规模 checkpoint、故障恢复和监控
- 存储、网络、计算资源的协同优化

一句话概括：

> AI Infra 工程师负责让大模型在多 GPU、多节点、大规模请求和长时间训练场景下跑得更快、更稳、更便宜。

你已有前后端工程经验，这是一个优势。因为 AI Infra 不是纯算法岗位，它非常需要工程能力：

- 写服务
- 做 API
- 设计任务系统
- 调试线上问题
- 处理并发
- 观测指标
- 做部署和运维
- 读复杂源码
- 定位性能瓶颈

但你需要补齐以下能力：

- PyTorch 和深度学习框架
- Transformer 和 LLM 基础
- GPU 架构和 CUDA/Triton
- 分布式训练
- vLLM、SGLang、TensorRT-LLM 等推理框架
- NCCL/RDMA/NVLink 等通信和硬件知识
- GPU 集群调度、存储和容错

---

## 2. 岗位能力地图

```text
AI Infra / LLM Infra
├── 编程与系统基础
│   ├── Python 工程化
│   ├── C++ 系统编程
│   ├── Linux
│   ├── 并发与异步
│   └── 性能 profiling
├── 深度学习基础
│   ├── PyTorch
│   ├── autograd
│   ├── optimizer
│   ├── mixed precision
│   └── torch.profiler
├── LLM 原理
│   ├── Transformer decoder-only
│   ├── Attention
│   ├── RoPE
│   ├── RMSNorm
│   ├── SwiGLU
│   ├── MHA / MQA / GQA
│   └── KV Cache
├── 推理 Infra
│   ├── vLLM
│   ├── SGLang
│   ├── TensorRT-LLM
│   ├── Triton Inference Server
│   ├── continuous batching
│   ├── PagedAttention
│   ├── prefix cache
│   └── quantization
├── 训练 Infra
│   ├── PyTorch DDP
│   ├── FSDP
│   ├── DeepSpeed ZeRO
│   ├── Megatron-LM
│   ├── tensor parallel
│   ├── pipeline parallel
│   ├── data parallel
│   ├── context parallel
│   └── expert parallel
├── GPU 与算子
│   ├── NVIDIA GPU 架构
│   ├── CUDA
│   ├── Triton Language
│   ├── Attention kernel
│   ├── LayerNorm / Softmax / Matmul
│   └── kernel profiling
├── 通信与网络
│   ├── NCCL / HCCL
│   ├── all-reduce
│   ├── all-gather
│   ├── reduce-scatter
│   ├── RDMA / RoCE / InfiniBand
│   ├── NVLink
│   └── NVSwitch
└── 集群工程
    ├── Docker
    ├── Kubernetes
    ├── GPU scheduling
    ├── checkpoint
    ├── fault tolerance
    ├── monitoring
    └── distributed storage
```

---

## 3. 学习优先级

### 第一优先级：必须掌握

这些是进入 AI Infra 的基本门槛：

- Python 工程化
- Linux 基础
- PyTorch 基础
- Transformer 基础
- LLM 推理流程
- vLLM 部署和压测
- DDP/FSDP 基础
- GPU 显存和性能分析

### 第二优先级：强竞争力

这些决定你是否像真正的 AI Infra 候选人：

- vLLM 源码阅读
- SGLang 运行时理解
- Triton 自定义算子
- CUDA 基础 kernel
- DeepSpeed ZeRO
- Megatron-LM 并行策略
- NCCL 通信原理
- 推理服务 P95/P99 延迟优化

### 第三优先级：长期深入

这些适合在项目和实习中逐步深入：

- TensorRT-LLM 插件和 engine
- MoE 训练与推理
- FP8/INT8/INT4 量化
- 多机多卡网络拓扑优化
- 大规模 checkpoint 和故障恢复
- GPU 资源调度系统
- 存储系统如 GPFS、Lustre、JuiceFS

---

## 4. 推荐主线

对已有前后端经验的人，最现实的路径是：

```text
后端工程能力
  -> LLM 推理服务
  -> vLLM/SGLang 性能优化
  -> PyTorch 分布式训练
  -> Triton/CUDA 算子优化
  -> Megatron/DeepSpeed 大规模训练框架
  -> GPU 集群和通信优化
```

不要一开始就直接啃 Megatron-LM 和 CUDA Attention kernel。那样学习曲线非常陡，容易卡住。

最适合你的切入方向是：

> LLM 推理基础设施工程师。

原因：

- 和后端服务开发最接近
- 能快速做出可展示项目
- 岗位需求多
- vLLM/SGLang/Triton Inference Server 都可以在中小规模环境中实践
- 后续可以自然延伸到 GPU、分布式训练和算子优化

---

## 5. 阶段路线

下面按 10 个月设计。如果时间紧，可以压缩到 3 个月版本，见后面的「快速投递路线」。

---

## 阶段 0：理解岗位地图，1 周

### 目标

建立对 AI Infra 的全局认识，知道自己要学什么、为什么学、每块技术解决什么问题。

### 需要理解的问题

- 为什么大模型训练需要多卡？
- 为什么显存总是不够？
- 为什么 LLM 推理不只是调用模型？
- prefill 和 decode 为什么性能特征不同？
- KV Cache 为什么会成为推理系统核心问题？
- 为什么分布式训练会受网络通信影响？
- 为什么 checkpoint 会拖慢训练？
- 为什么 AI Infra 工程师要懂 C++、CUDA、NCCL？

### 输出物

整理一页自己的 AI Infra 知识地图，至少包含：

- Training Infra
- Inference Infra
- GPU Compute
- Communication
- Storage
- Scheduling
- Stability

### 阶段验收

你能用自己的话解释：

- vLLM 解决什么问题
- DDP 和 FSDP 的区别
- KV Cache 是什么
- NCCL 在训练里做什么
- CUDA/Triton 算子优化为什么重要

---

## 阶段 1：Python、C++、Linux 系统基础，3-5 周

### 目标

把前后端工程能力升级到系统工程能力。AI Infra 里大量问题不是业务逻辑问题，而是性能、并发、内存、进程、网络、文件系统和 GPU 资源问题。

### Python 必学内容

- `asyncio`
- `threading`
- `multiprocessing`
- `queue`
- `subprocess`
- `argparse`
- `logging`
- `pytest`
- `pydantic`
- `FastAPI`
- Python package 结构
- type hints
- profiling：`cProfile`、`py-spy`

### Python 需要达到的能力

你应该能独立写出：

- 一个异步 API 服务
- 一个批处理任务队列
- 一个后台 worker
- 一个压测脚本
- 一个性能指标采集模块
- 一个命令行工具

### C++ 必学内容

- 指针、引用、对象生命周期
- RAII
- move semantics
- `unique_ptr` / `shared_ptr`
- STL
- template 基础
- lambda
- 多线程
- CMake
- gdb
- perf
- pybind11

### C++ 目标

不要求一开始写出复杂高性能 C++ 框架，但至少要能：

- 看懂 PyTorch extension
- 看懂 vLLM/TensorRT-LLM 中的 C++ 部分
- 写简单 C++ 模块
- 用 pybind11 暴露 C++ 函数给 Python
- 用 gdb 定位 crash

### Linux 必学内容

- 进程和线程
- 文件描述符
- socket
- epoll 基础
- mmap
- page cache
- NUMA 基础
- CPU/GPU 利用率排查
- `top`
- `htop`
- `iostat`
- `nvidia-smi`
- `dmesg`
- `strace`
- `lsof`
- `ss`
- `netstat`

### 阶段项目 1：简化版推理网关

实现一个小系统：

- FastAPI 接收请求
- 请求进入队列
- 后台 scheduler 聚合 batch
- worker 模拟模型推理
- 支持并发压测
- 输出 QPS、P50、P95、P99 延迟
- 支持 graceful shutdown

这个项目先不接真实模型，重点是理解「请求调度」和「batch 聚合」。

### 阶段验收

你能解释：

- 异步和多线程的区别
- 多进程为什么适合绕开 GIL
- P95/P99 延迟怎么统计
- batch size 增大为什么可能提升吞吐但恶化延迟
- 如何定位一个服务的 CPU、I/O、网络瓶颈

---

## 阶段 2：PyTorch 和深度学习基础，4-6 周

### 目标

从工程视角理解模型训练和推理是如何运行的。

你不需要先成为算法研究员，但必须理解模型、显存、梯度、优化器和 profiler。

### PyTorch 必学内容

- Tensor
- broadcasting
- autograd
- `torch.nn.Module`
- `torch.optim`
- Dataset / DataLoader
- forward / backward
- loss
- optimizer step
- mixed precision
- `torch.cuda`
- `torch.compile`
- `torch.profiler`
- checkpoint 保存和加载

### 深度学习基础

重点掌握：

- MLP
- embedding
- softmax
- cross entropy
- LayerNorm
- RMSNorm
- attention
- Transformer block
- AdamW
- learning rate schedule
- overfitting / underfitting

CNN 和 RNN 可以了解，不是这条路线的核心。

### 显存分析必须理解

训练时显存主要来自：

- model parameters
- gradients
- optimizer states
- activations
- temporary buffers
- CUDA context

推理时显存主要来自：

- model weights
- KV Cache
- activation temporary buffers
- batch 中请求的中间状态

### 阶段项目 2：Tiny Transformer 训练

实现一个小型 Transformer：

- 使用 PyTorch
- 用小文本数据集训练
- 实现 tokenizer 或使用简单字符级 tokenizer
- 支持训练、验证、保存 checkpoint
- 输出 loss 曲线
- 用 `torch.profiler` 分析耗时
- 比较不同 batch size 和 sequence length 的显存变化

### 阶段验收

你能回答：

- `loss.backward()` 做了什么
- optimizer state 为什么占显存
- activation checkpointing 为什么能省显存
- sequence length 增大对 attention 成本有什么影响
- mixed precision 为什么能提升速度并降低显存

---

## 阶段 3：LLM 原理，3-4 周

### 目标

理解 LLaMA 类 decoder-only 模型是如何训练和推理的。

### 必学概念

- tokenizer
- BPE / SentencePiece
- decoder-only Transformer
- causal mask
- multi-head attention
- MQA / GQA
- RoPE
- RMSNorm
- SwiGLU
- KV Cache
- prefill
- decode
- sampling
- temperature
- top-k
- top-p
- repetition penalty
- LoRA / QLoRA
- SFT
- RLHF / DPO 基础

### 系统视角下的重点问题

- 一个 token 是怎么被生成的？
- prefill 阶段为什么更像大矩阵计算？
- decode 阶段为什么更容易受访存和调度影响？
- KV Cache 为什么会占用大量显存？
- 多轮对话为什么需要 prefix cache？
- 长上下文为什么会放大显存和计算压力？

### 阶段项目 3：最小 LLM 推理流程

可以基于 Hugging Face Transformers 或自己写的小模型：

- 加载一个小型语言模型
- 实现 prompt 输入
- 实现 token-by-token 生成
- 实现 streaming 输出
- 比较有无 KV Cache 的速度差异
- 记录显存变化
- 支持 temperature、top-k、top-p

### 阶段验收

你能解释：

- prefill 和 decode 的区别
- KV Cache 的 shape 大概是什么
- MHA/MQA/GQA 的区别
- RoPE 的作用
- 为什么推理服务需要 continuous batching

---

## 阶段 4：推理 Infra 主线，6-8 周

### 目标

从普通后端工程师切入 AI Infra 的关键阶段。重点掌握 vLLM/SGLang/TensorRT-LLM 的使用、压测、调参和部分源码。

---

### 4.1 vLLM

vLLM 是最优先学习的推理框架。

重点概念：

- PagedAttention
- Continuous Batching
- KV Cache block
- scheduler
- block manager
- prefill/decode
- OpenAI-compatible API server
- tensor parallel serving
- prefix caching
- speculative decoding
- metrics

你要会做：

- 部署 vLLM server
- 接入本地模型
- 用 OpenAI API 格式请求
- 做 streaming 输出
- 写压测脚本
- 统计 tokens/s、QPS、P50、P95、P99
- 观察 GPU 利用率和显存
- 调整 max model len、max num seqs、max num batched tokens
- 分析吞吐和延迟的 tradeoff

### 阶段项目 4：vLLM 推理服务压测平台

功能要求：

- vLLM server
- OpenAI-compatible API
- 压测脚本
- 支持不同并发数
- 支持不同输入长度和输出长度
- 支持流式输出
- 统计 P50/P95/P99
- 统计 tokens/s
- 统计 GPU 显存和利用率
- 输出 Markdown 性能报告

性能报告模板：

```text
模型：
GPU：
vLLM 版本：
并发数：
输入 token 长度：
输出 token 长度：
QPS：
tokens/s：
P50：
P95：
P99：
GPU 利用率：
显存占用：
瓶颈分析：
调参动作：
调参结果：
```

简历写法：

> 基于 vLLM 构建 LLM 推理服务与压测平台，支持 OpenAI-compatible API、流式输出、并发压测和 GPU 指标采集；针对不同输入输出长度、并发数和 batching 参数进行吞吐/延迟分析，定位 KV Cache 显存占用和 decode 阶段性能瓶颈。

---

### 4.2 SGLang

SGLang 重点看 serving runtime 和 cache 设计。

重点概念：

- serving runtime
- RadixAttention
- prefix cache
- structured generation
- constrained decoding
- multi-turn serving
- 与 vLLM 的差异

你要能比较：

- vLLM 更适合什么场景
- SGLang 更适合什么场景
- 多轮对话和 agent workflow 对推理系统有什么压力
- prefix cache 如何减少重复计算

---

### 4.3 TensorRT-LLM

TensorRT-LLM 学习成本更高，建议先会用，再逐步读。

重点概念：

- engine build
- runtime
- plugin
- kernel fusion
- quantization
- tensor parallel
- paged KV cache
- inflight batching
- FP16 / BF16 / INT8 / FP8

阶段目标：

- 能把一个模型转换并部署
- 能理解 engine build 和 runtime 的关系
- 能解释 TensorRT-LLM 与 vLLM 的定位差异
- 能看懂部分 plugin 和 kernel 相关代码

---

## 阶段 5：分布式训练，8-10 周

### 目标

理解大模型训练为什么需要 DP/TP/PP/FSDP/ZeRO/Megatron，以及这些技术如何降低显存压力和提升训练吞吐。

---

### 5.1 PyTorch DDP

必学概念：

- process group
- rank
- world size
- local rank
- data parallel
- gradient all-reduce
- NCCL backend
- `torchrun`
- distributed sampler
- gradient synchronization

你要理解：

普通 DDP 中，每张 GPU 都有完整模型副本。每张卡处理不同数据，反向传播后用 all-reduce 同步梯度。

### 阶段项目 5：DDP 小模型训练

要求：

- 单机多卡 DDP
- 对比单卡和多卡吞吐
- 记录 GPU 利用率
- 使用 mixed precision
- 保存 checkpoint
- 模拟训练中断后恢复

验收问题：

- DDP 为什么需要 all-reduce
- batch size 如何按 GPU 数变化
- 为什么多卡不一定线性加速
- 如果一个 rank 卡住会发生什么

---

### 5.2 FSDP

必学概念：

- parameter sharding
- gradient sharding
- optimizer state sharding
- all-gather
- reduce-scatter
- CPU offload
- activation checkpointing
- mixed precision

你要理解：

FSDP 解决的是 DDP 中每张卡都保存完整模型、梯度和 optimizer state 的问题。FSDP 会把这些状态切分到多张 GPU 上，用通信换显存。

验收问题：

- FSDP 为什么比 DDP 省显存
- FSDP forward 前为什么需要 all-gather
- backward 后为什么需要 reduce-scatter
- FSDP 和 ZeRO-3 有什么相似点

---

### 5.3 DeepSpeed ZeRO

必学概念：

- ZeRO-1：切 optimizer state
- ZeRO-2：切 optimizer state 和 gradients
- ZeRO-3：切 optimizer state、gradients 和 parameters
- offload
- activation checkpointing
- pipeline parallel

你要能比较：

```text
DDP:
  每张卡完整保存模型、梯度和 optimizer state。

FSDP:
  PyTorch 原生参数切分方案。

DeepSpeed ZeRO:
  DeepSpeed 提供的大规模训练优化方案，ZeRO-3 与 FSDP 思路接近。
```

---

### 5.4 Megatron-LM

Megatron-LM 是训练 Infra 的重点，但建议最后学。

必学概念：

- tensor parallel
- pipeline parallel
- sequence parallel
- context parallel
- expert parallel
- MoE
- distributed optimizer
- activation recomputation
- communication overlap

并行策略理解：

```text
DP: Data Parallel，数据并行
TP: Tensor Parallel，张量并行，把单层计算切到多张 GPU
PP: Pipeline Parallel，流水线并行，把不同层放到不同 GPU
CP: Context Parallel，把长上下文维度切分
EP: Expert Parallel，MoE 中把专家切到不同 GPU
```

验收问题：

- TP 为什么通信频繁
- PP 为什么会有 bubble
- DP 为什么显存压力大
- CP 主要解决什么问题
- EP 为什么常见于 MoE

### 阶段项目 6：小规模 Megatron/DeepSpeed 训练实验

即使没有大集群，也可以做小规模实验：

- 小模型
- 小数据集
- 1-2 张 GPU
- 开启/关闭 activation checkpointing
- 对比 DDP/FSDP/ZeRO 显存
- 输出训练性能分析报告

---

## 阶段 6：CUDA / Triton 算子优化，8-12 周

### 目标

理解 GPU 是如何执行计算的，并能用 Triton 或 CUDA 写出基础高性能算子。

岗位截图明确提到了：

- Attention
- MoE
- 算子融合
- KV Cache
- 量化推理
- CUDA/Triton 高性能实现

所以算子优化是非常强的加分项。

---

### 6.1 CUDA 基础

必学概念：

- thread
- block
- grid
- warp
- SM
- register
- shared memory
- global memory
- memory coalescing
- occupancy
- stream
- event
- kernel launch overhead

基础 kernel 练习：

- vector add
- matrix transpose
- reduction
- prefix sum
- naive matmul
- tiled matmul
- softmax
- layernorm

验收问题：

- warp 是什么
- shared memory 为什么快
- memory coalescing 为什么重要
- occupancy 是什么
- 为什么 kernel fusion 可以提升性能

---

### 6.2 Triton Language

Triton 更适合作为算子优化入门工具。

必学内容：

- program id
- block programming model
- `tl.load`
- `tl.store`
- mask
- block size
- num warps
- benchmarking
- correctness test

### 阶段项目 7：Triton 算子库

建议实现顺序：

1. vector add
2. layernorm
3. softmax
4. matmul
5. fused SiLU/GLU
6. top-k sampling 简化版
7. attention 简化版

每个算子都要包含：

- PyTorch baseline
- Triton 实现
- correctness test
- benchmark
- 不同 shape 下的性能比较
- 简短性能分析

简历写法：

> 使用 Triton 实现并优化 LayerNorm、Softmax、Matmul 和简化 Attention 算子，对比 PyTorch baseline 进行 correctness test 和 benchmark，分析 block size、num warps、memory access pattern 对性能的影响。

---

## 阶段 7：通信、网络和集群，6-8 周

### 目标

理解多机多卡训练和推理的通信瓶颈。

岗位截图提到：

- H/NCCL
- RDMA
- RoCE
- InfiniBand
- NVLink
- NVSwitch

这些都是大规模 GPU 集群的关键技术。

---

### 7.1 NCCL

必学内容：

- all-reduce
- all-gather
- reduce-scatter
- broadcast
- ring all-reduce
- tree all-reduce
- NCCL backend
- NCCL debug
- NCCL environment variables

训练中的通信模式：

```text
DDP:
  backward 后 all-reduce gradients。

FSDP:
  forward 前 all-gather parameters。
  backward 后 reduce-scatter gradients。

TP:
  单层内部经常发生 all-reduce 或 all-gather。

PP:
  stage 之间传 activation 和 gradient。

EP:
  MoE token dispatch 和 combine 会产生通信。
```

验收问题：

- all-reduce 和 reduce-scatter 的区别
- 为什么 TP 对网络要求高
- 为什么跨机 TP 更难
- NCCL hang 怎么排查
- `NCCL_DEBUG=INFO` 有什么用

---

### 7.2 网络和硬件

需要了解：

- PCIe
- NVLink
- NVSwitch
- InfiniBand
- RoCE
- RDMA
- GPUDirect RDMA
- bandwidth vs latency
- topology

你要能解释：

- 为什么单机 8 卡训练比跨机训练简单
- 为什么 NVLink 比 PCIe 更适合 GPU 间通信
- 为什么 RDMA 可以降低 CPU 参与
- 为什么网络拓扑会影响 TP/PP/DP 选择

---

### 7.3 Kubernetes / GPU 调度

你已有后端工程经验，这块是优势。

必学内容：

- Docker
- Kubernetes
- GPU device plugin
- node selector
- taints / tolerations
- resource quota
- priority class
- job queue
- gang scheduling
- Volcano
- Kueue
- Ray
- Slurm 基础

### 阶段项目 8：简化 GPU 任务调度系统

可以先模拟，不一定需要真实大集群。

功能：

- 用户提交训练/推理任务
- 任务声明需要 N 张 GPU
- 支持任务排队
- 支持优先级
- 支持失败重试
- 支持任务状态查询
- 支持任务日志
- 支持 Prometheus 指标

简历写法：

> 设计并实现面向训练/推理任务的 GPU 资源调度原型，支持多租户任务提交、优先级排队、失败重试和资源使用统计，为训练任务提供基础可观测性和容错能力。

---

## 阶段 8：存储、Checkpoint 和稳定性，3-5 周

### 目标

理解大规模训练为什么需要可靠的 checkpoint 和自动恢复。

截图里提到：

- checkpoint 优化
- 故障自动检测
- 自动恢复
- 大规模训练数月稳定性
- GPFS / Lustre / JuiceFS

### 必学问题

- checkpoint 保存什么
- checkpoint 为什么慢
- 分布式 checkpoint 怎么保存
- optimizer state 为什么很大
- 训练中断后如何恢复
- rank 挂了怎么办
- 节点故障怎么处理
- checkpoint 太频繁有什么问题
- checkpoint 太少有什么风险
- 大规模数据加载为什么受存储带宽影响

checkpoint 通常包含：

- model parameters
- optimizer states
- scheduler states
- random states
- dataloader state
- global step
- distributed training metadata

### 阶段项目 9：DDP/FSDP 训练容错系统

功能：

- 定期 checkpoint
- 保存 optimizer state
- 保存 global step
- 异常中断后自动恢复
- 模拟训练失败
- 从最近 checkpoint 继续训练
- 记录恢复耗时

验收问题：

- 为什么只保存模型参数不够
- optimizer state 缺失会怎样
- 分布式训练中每个 rank 如何保存 checkpoint
- checkpoint 写入如何影响训练吞吐

---

## 6. 10 个月学习计划

| 时间 | 重点 | 目标产出 |
|---|---|---|
| 第 1 个月 | Python/C++/Linux + PyTorch 基础 | 简化推理网关、Tiny Transformer |
| 第 2 个月 | Transformer/LLM 原理 | 最小 LLM 推理流程、KV Cache 实验 |
| 第 3 个月 | vLLM 部署和压测 | vLLM 推理服务性能报告 |
| 第 4 个月 | SGLang/TensorRT-LLM 入门 | vLLM vs SGLang 对比报告 |
| 第 5 个月 | DDP/FSDP/DeepSpeed | 分布式训练实验 |
| 第 6 个月 | Megatron-LM 并行策略 | DP/TP/PP 实验和源码笔记 |
| 第 7 个月 | Triton 算子优化 | LayerNorm/Softmax/Matmul benchmark |
| 第 8 个月 | CUDA 基础和 Attention | 简化 Attention kernel |
| 第 9 个月 | NCCL/网络/存储/调度 | GPU 调度原型或通信实验 |
| 第 10 个月 | 简历、面试、项目整理 | 作品集、简历、面试题库 |

---

## 7. 3 个月快速投递路线

如果目标是尽快投递相关实习或初级岗位，不要平均用力，按下面顺序来：

### 第 1-2 周：PyTorch + Transformer 基础

输出：

- Tiny Transformer
- 训练脚本
- loss 曲线
- profiler 简单分析

### 第 3-4 周：LLM 推理流程

输出：

- Hugging Face 小模型推理
- streaming 输出
- KV Cache 速度对比
- 显存变化记录

### 第 5-6 周：vLLM 服务与压测

输出：

- vLLM server
- OpenAI API 调用
- 压测脚本
- P50/P95/P99 报告
- tokens/s 报告

### 第 7-8 周：SGLang 或 TensorRT-LLM 入门

输出：

- 另一个推理框架的部署记录
- 与 vLLM 的对比
- 场景差异总结

### 第 9-10 周：DDP/FSDP

输出：

- 单机多卡训练实验
- DDP/FSDP 显存和吞吐对比
- checkpoint 恢复

### 第 11-12 周：Triton 算子

输出：

- LayerNorm
- Softmax
- Matmul
- benchmark 表格

3 个月后，简历上至少应该有：

- 一个 vLLM 推理服务项目
- 一个 PyTorch DDP/FSDP 项目
- 一个 Triton 算子优化项目
- 一份性能分析报告

---

## 8. 最推荐做的 4 个简历项目

### 项目 1：LLM 推理服务性能优化平台

技术栈：

- Python
- FastAPI
- vLLM
- Prometheus/Grafana 或自写 metrics
- Locust、wrk 或自写压测脚本
- Docker

功能：

- OpenAI-compatible API
- 流式输出
- 并发压测
- 不同输入长度测试
- 不同输出长度测试
- 不同并发数测试
- GPU 利用率记录
- 显存记录
- P50/P95/P99 统计
- tokens/s 统计
- 自动生成 Markdown 报告

可以深入的优化点：

- batch 参数调优
- max model len 调优
- KV Cache 显存分析
- prefix cache 效果分析
- prefill/decode 分离统计
- 长上下文场景压测

---

### 项目 2：Triton 自定义算子优化

技术栈：

- PyTorch
- Triton
- CUDA profiling 工具

实现：

- LayerNorm
- Softmax
- Matmul
- 简化 Attention
- Top-k sampling

产出：

- correctness test
- benchmark
- 性能曲线
- shape sensitivity 分析
- 和 PyTorch baseline 对比

---

### 项目 3：分布式训练实验框架

技术栈：

- PyTorch DDP
- FSDP
- DeepSpeed
- NCCL
- checkpoint

功能：

- 单卡训练
- DDP 多卡训练
- FSDP 参数切分
- mixed precision
- activation checkpointing
- checkpoint 保存与恢复
- 训练吞吐和显存对比

---

### 项目 4：简化 GPU 任务调度系统

技术栈：

- Python 或 Go
- FastAPI 或 Gin
- Redis/PostgreSQL
- Kubernetes
- Prometheus

功能：

- 提交训练/推理任务
- 声明 GPU 需求
- 优先级队列
- 失败重试
- 状态查询
- 日志查看
- 资源使用统计

这个项目能把你已有的后端能力和 AI Infra 结合起来。

---

## 9. 源码阅读顺序

不要从第一行开始读源码。每次带着问题读。

### 第一优先级：vLLM

重点模块：

- API server
- scheduler
- block manager
- worker
- attention backend
- sampling
- metrics

带着这些问题读：

- 请求进入后如何排队
- scheduler 如何组成 batch
- KV Cache block 如何分配
- 显存不足时怎么处理
- prefill 和 decode 如何调度
- tensor parallel 在哪里发生

---

### 第二优先级：SGLang

重点模块：

- runtime
- scheduler
- radix cache
- serving engine
- structured generation

带着这些问题读：

- prefix cache 如何设计
- 多轮对话如何复用前缀
- structured generation 如何约束输出
- 和 vLLM 的 scheduler 有什么差异

---

### 第三优先级：PyTorch FSDP

重点问题：

- 参数如何切分
- forward 前如何 all-gather
- backward 后如何 reduce-scatter
- checkpoint 如何保存
- mixed precision 如何处理

---

### 第四优先级：DeepSpeed

重点问题：

- ZeRO-1/2/3 如何实现
- offload 如何工作
- pipeline parallel 如何调度
- activation checkpointing 如何集成

---

### 第五优先级：Megatron-LM

重点问题：

- tensor parallel 如何切 Linear
- pipeline parallel 如何切 layer
- sequence parallel 解决什么
- context parallel 解决什么
- distributed optimizer 如何工作
- MoE expert parallel 如何 dispatch token

---

### 第六优先级：TensorRT-LLM

重点问题：

- engine build 做了什么
- runtime 如何管理请求
- KV Cache 如何管理
- quantization 如何接入
- plugin 和 kernel 在哪里

---

## 10. 面试题清单

### Python / C++

- Python GIL 是什么？
- 多线程和多进程有什么区别？
- `asyncio` 适合什么场景？
- C++ RAII 是什么？
- `unique_ptr` 和 `shared_ptr` 区别？
- move semantics 解决什么问题？
- 如何排查 C++ 内存泄漏？
- 如何做性能 profiling？

### PyTorch

- autograd 原理是什么？
- forward/backward 过程中保存了什么？
- optimizer state 为什么占显存？
- mixed precision 为什么能省显存？
- FP16 和 BF16 有什么区别？
- activation checkpointing 原理是什么？
- `torch.profiler` 怎么用？
- `torch.compile` 大致解决什么问题？

### Transformer / LLM

- Attention 复杂度是多少？
- KV Cache 的作用是什么？
- prefill 和 decode 有什么区别？
- MHA/MQA/GQA 区别是什么？
- RoPE 的作用是什么？
- RMSNorm 和 LayerNorm 有什么区别？
- MoE 为什么需要 expert parallel？

### 推理系统

- vLLM 为什么快？
- PagedAttention 解决什么问题？
- continuous batching 是什么？
- P95/P99 延迟如何优化？
- 吞吐和延迟如何权衡？
- 长上下文为什么难？
- prefix cache 有什么用？
- TensorRT-LLM 和 vLLM 的定位有什么区别？
- 如何做量化推理？

### 分布式训练

- DP/TP/PP 区别是什么？
- all-reduce 是什么？
- all-gather 是什么？
- reduce-scatter 是什么？
- DDP 如何同步梯度？
- FSDP 为什么省显存？
- ZeRO-1/2/3 区别？
- TP 为什么通信重？
- PP 为什么有 pipeline bubble？
- FSDP 和 ZeRO-3 有什么相似点？

### GPU / 通信

- CUDA thread/block/grid 是什么？
- warp 是什么？
- shared memory 有什么用？
- memory coalescing 为什么重要？
- kernel fusion 为什么能加速？
- NCCL 做什么？
- NVLink 和 PCIe 区别？
- RDMA 是什么？
- RoCE 和 InfiniBand 有什么区别？
- 为什么跨机训练比单机训练难？

### 集群和稳定性

- checkpoint 保存哪些内容？
- checkpoint 为什么会影响训练吞吐？
- 训练任务失败后如何恢复？
- 如何监控 GPU 利用率？
- 如何设计 GPU 任务队列？
- 如何处理任务优先级？
- 如何定位分布式训练 hang？

---

## 11. 简历表达模板

### vLLM 项目

> 基于 vLLM 构建 LLM 推理服务与压测平台，支持 OpenAI-compatible API、流式输出、并发压测和 GPU 指标采集；针对不同输入输出长度、并发数和 batching 参数进行吞吐/延迟分析，定位 KV Cache 显存占用和 decode 阶段性能瓶颈。

### Triton 算子项目

> 使用 Triton 实现并优化 LayerNorm、Softmax、Matmul 和简化 Attention 算子，对比 PyTorch baseline 进行 correctness test 和 benchmark，分析 block size、num warps、memory access pattern 对性能的影响。

### 分布式训练项目

> 基于 PyTorch DDP/FSDP 构建小规模分布式训练实验框架，对比 DDP、FSDP、activation checkpointing 和 mixed precision 在吞吐、显存和稳定性上的差异，并实现训练中断后的 checkpoint 恢复机制。

### GPU 调度项目

> 设计并实现面向训练/推理任务的 GPU 资源调度原型，支持多租户任务提交、优先级排队、失败重试和资源使用统计，为训练任务提供基础可观测性和容错能力。

---

## 12. 每周学习节奏建议

如果每天能投入 3-4 小时，可以这样安排：

```text
周一到周三：
  学概念 + 写小实验

周四到周五：
  做项目功能

周六：
  压测、debug、整理性能数据

周日：
  写复盘文档、补源码阅读、准备面试题
```

每周必须有一个可见产出：

- 一段可运行代码
- 一张 benchmark 表
- 一份源码笔记
- 一个问题排查记录
- 一篇项目复盘

不要只看视频和文章。这个方向最看重工程落地。

---

## 13. 没有 GPU 怎么办

没有 GPU 也可以先推进一部分：

- Python 推理网关
- Tiny Transformer CPU 训练
- LLM 推理服务接口封装
- 压测系统
- GPU 调度系统模拟
- 源码阅读
- PyTorch DDP CPU/gloo 小实验
- Triton/CUDA 理论和代码阅读

但要真正竞争 AI Infra，后期最好获得 GPU 环境：

- 学校实验室资源
- 公司实习资源
- 低成本云 GPU
- 朋友/团队共享机器
- 本地消费级 NVIDIA GPU

优先用小模型、小 batch、小数据集做实验，不要一上来追求训练大模型。

---

## 14. 官方资料入口

优先看官方文档和源码：

- vLLM 文档：https://docs.vllm.ai/
- SGLang 文档：https://docs.sglang.io/
- PyTorch DDP 教程：https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
- PyTorch FSDP 文档：https://docs.pytorch.org/docs/stable/fsdp.html
- DeepSpeed 官网：https://www.deepspeed.ai/
- Megatron-LM 源码：https://github.com/NVIDIA/Megatron-LM
- TensorRT-LLM 文档：https://nvidia.github.io/TensorRT-LLM/
- Triton Language 文档：https://triton-lang.org/main/index.html
- NVIDIA CUDA C++ Programming Guide：https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- NVIDIA NCCL 文档：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- NVIDIA Triton Inference Server 文档：https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/

---

## 15. 最终能力验收

当你准备投 AI Infra 相关岗位时，至少要能做到：

### 工程能力

- 能写稳定的 Python 服务
- 能设计压测和指标采集
- 能读复杂开源项目源码
- 能写清楚性能分析报告
- 能定位 CPU/GPU/网络/存储瓶颈

### 推理能力

- 能部署 vLLM
- 能做推理压测
- 能解释 KV Cache
- 能解释 continuous batching
- 能分析 P95/P99 延迟
- 能比较 vLLM、SGLang、TensorRT-LLM

### 训练能力

- 能写 DDP 训练
- 能理解 FSDP
- 能解释 ZeRO
- 能理解 DP/TP/PP/CP/EP
- 能做 checkpoint 恢复

### GPU 能力

- 能解释 GPU memory hierarchy
- 能写简单 CUDA kernel
- 能用 Triton 写基础算子
- 能做 benchmark
- 能解释 kernel fusion

### 通信和集群能力

- 能解释 NCCL collective
- 能理解 RDMA/RoCE/InfiniBand/NVLink
- 能理解 GPU 调度
- 能设计基础任务队列
- 能理解大规模训练稳定性问题

---

## 16. 对你的定位建议

你目前是有前后端基础的工程人员，最适合的定位不是「从零转算法」，而是：

> 后端/系统工程背景的 LLM Infra 工程师。

推荐投递关键词：

- AI Infra
- LLM Infra
- LLM Serving
- 推理引擎
- 大模型推理优化
- 训练平台
- 分布式训练
- GPU 平台
- 模型服务平台
- 机器学习系统工程师

优先准备的作品集：

1. vLLM 推理服务压测平台
2. PyTorch DDP/FSDP 分布式训练实验
3. Triton 算子优化 benchmark
4. GPU 任务调度原型

这 4 个项目能覆盖岗位截图中的大部分关键词，也能体现你从前后端工程向 AI Infra 转型的合理路径。

