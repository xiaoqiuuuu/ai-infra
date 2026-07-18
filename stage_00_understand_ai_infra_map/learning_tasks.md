# Stage 00: 理解岗位地图

## Goal

建立 AI Infra 的全局地图，知道训练、推理、GPU、通信、存储、调度和稳定性分别解决什么工程问题。

本阶段不要求深入源码或实现 CUDA kernel，重点是能把岗位要求翻译成具体系统问题。

## Tasks

1. 阅读路线图中的“岗位本质”“岗位能力地图”“推荐主线”和“阶段 0”。
2. 在 `knowledge_map.md` 中整理一页自己的 AI Infra 知识地图。
3. 用自己的话解释 Training Infra、Inference Infra、GPU Compute、Communication、Storage、Scheduling、Stability。
4. 回答本文件里的验收问题，不需要长篇，每题 3-6 句话即可。

## Reference Materials

本阶段目标是建立地图，不是深入实现。先按“必读”完成，再根据卡住的概念看“选读”。

### Required

1. 本仓库路线图：`../ai_infra_learning_roadmap.md`
   - 读：“岗位本质”“岗位能力地图”“推荐主线”“阶段 0”。
   - 解决的问题：先知道 AI Infra 岗位到底由哪些系统问题组成。

2. vLLM PagedAttention 论文：https://arxiv.org/abs/2309.06180
   - 读：Abstract、Introduction 和系统问题描述，不要求读完所有实验。
   - 解决的问题：理解为什么 LLM serving 的核心瓶颈之一是 KV Cache 和显存管理。

3. vLLM Paged Attention 设计文档：https://docs.vllm.ai/en/latest/design/paged_attention/
   - 读：整体设计说明，重点关注 KV cache blocks、sequence、block table 这些概念。
   - 解决的问题：回答“vLLM 主要解决什么推理系统问题”。

4. Hugging Face Transformers Caching 文档：https://huggingface.co/docs/transformers/main/en/cache_explanation
   - 读：Caching、Attention matrices、Cache storage implementation。
   - 解决的问题：回答“KV Cache 是什么，为什么推理阶段要缓存 K/V”。

5. PyTorch DDP 教程：https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
   - 读：What is DDP 和 Basic Use Case 部分。
   - 解决的问题：理解 DDP 为什么要在多 GPU 之间同步梯度。

6. PyTorch FSDP2 教程：https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
   - 读：FSDP 的目标和 sharding 思路，不要求先跑代码。
   - 解决的问题：回答“DDP 和 FSDP 的核心区别是什么”。

### Optional

1. NVIDIA NCCL Overview：https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/overview.html
   - 读：Overview 和 collectives 相关描述。
   - 解决的问题：回答“NCCL 在多 GPU 训练中做什么”。

2. NVIDIA CUDA Programming Guide：https://docs.nvidia.com/cuda/cuda-programming-guide/index.html
   - 读：Introduction 和 Programming Model 的前几节。
   - 解决的问题：理解为什么 GPU 编程和普通 CPU 程序不同。

3. Triton Tutorials：https://triton-lang.org/main/getting-started/tutorials/index.html
   - 读：Vector Addition 和 Matrix Multiplication 的标题与代码结构即可。
   - 解决的问题：知道 Triton 是用 Python 风格写 GPU kernel 的工具。

4. Kubernetes Schedule GPUs：https://kubernetes.io/docs/tasks/manage-gpus/scheduling-gpus/
   - 读：GPU 作为资源被 request/limit 的方式。
   - 解决的问题：理解集群调度为什么会影响 GPU 利用率。

## Output

更新 `knowledge_map.md`，至少包含：

- Training Infra：训练平台要解决什么问题。
- Inference Infra：LLM 推理服务为什么不是简单调用模型。
- GPU Compute：显存、算力和算子优化分别影响什么。
- Communication：多卡训练/推理为什么依赖通信。
- Storage：checkpoint、模型权重、数据集为什么会影响系统性能。
- Scheduling：GPU 资源如何被分配、隔离和复用。
- Stability：长时间训练和线上推理服务如何保证稳定。

## Acceptance Questions

### Core Questions

1. vLLM 主要解决什么推理系统问题？

> 解决推理过程中，碎片多，GPU 空间利用不充分，提升推理速度的问题

2. DDP 和 FSDP 的核心区别是什么？
3. KV Cache 是什么，为什么它会成为 LLM 推理的关键？
4. NCCL 在多 GPU 训练中做什么？
5. CUDA/Triton 算子优化为什么对 AI Infra 重要？

### Classic Interview Questions

以下题目用于扩展面试表达。Stage 00 不要求深入公式或源码，重点是能先给出一句话结论，再从性能、显存、通信或稳定性的角度解释原因。

#### AI Infra 全局认知

1. AI Infra 工程师和算法工程师、普通后端工程师的工作边界分别是什么？
2. 为什么说 AI Infra 的核心目标是让模型跑得更快、更稳、更便宜？这三个目标之间可能有什么冲突？
3. 面对一个“GPU 利用率很低”的问题，你会从计算、通信、数据、调度和服务请求哪些方向排查？

#### 训练 Infra

4. 为什么大模型训练通常需要多 GPU？仅仅是为了缩短训练时间吗？
5. 数据并行、张量并行和流水线并行分别切分什么？它们各自会引入什么通信？
6. DDP 为什么需要 All-Reduce？如果某张 GPU 比其他 GPU 慢，会发生什么？
7. FSDP/ZeRO 为什么能够降低单卡显存占用？它用什么代价换取显存？
8. 混合精度训练为什么能加速并节省显存？为什么还需要关注数值稳定性？

#### 推理 Infra

9. LLM 推理中的 prefill 和 decode 有什么区别？为什么它们的性能瓶颈不同？
10. continuous batching 和传统 static batching 有什么区别？它为什么能提高吞吐量？
11. PagedAttention 解决了传统 KV Cache 管理中的什么问题？它和操作系统分页有什么相似之处？
12. 推理服务中的吞吐量、首 Token 延迟（TTFT）和每 Token 延迟（TPOT）为什么不能只优化一个？
13. 量化为什么能降低推理成本？它可能对精度、吞吐和硬件兼容性产生什么影响？

#### GPU 与算子

14. GPU 为什么适合深度学习计算？哪些任务不一定适合放到 GPU 上？
15. 什么是计算受限（compute-bound）和访存受限（memory-bound）？如何初步判断一个算子属于哪一种？
16. 显存容量、显存带宽和 GPU 算力分别会限制哪些工作负载？
17. 算子融合为什么可能提升性能？它主要减少了哪些开销？
18. CUDA kernel 性能差时，你会优先检查哪些指标或现象？

#### 通信与网络

19. All-Reduce、All-Gather、Reduce-Scatter 分别完成什么数据交换？常见于哪些并行方式？
20. PCIe、NVLink、NVSwitch 和 InfiniBand/RoCE 分别用于什么范围的通信？
21. 为什么分布式训练规模扩大后，性能通常不能随 GPU 数量线性增长？
22. 什么是通信与计算重叠？它为什么能够提升分布式训练效率？

#### 存储、调度与稳定性

23. checkpoint 中通常需要保存哪些状态？只保存模型权重为什么可能无法完整恢复训练？
24. 为什么保存 checkpoint 可能导致所有 GPU 等待？有哪些常见缓解思路？
25. GPU 调度中的独占、时间切片和 MIG 分别适合什么场景？
26. 为什么 GPU 已经分配成功，任务仍可能出现利用率低或 OOM？
27. 大规模训练中某个节点或 GPU 故障后，系统应该如何发现、隔离和恢复？
28. 训练系统和在线推理系统关注的稳定性指标有什么不同？

#### 场景分析题

29. 一个推理服务 GPU 利用率只有 30%，但请求延迟很高，你会如何建立排查顺序？
30. 增加 GPU 后训练吞吐量几乎没有提升，你认为最可能的瓶颈有哪些？
31. 推理服务频繁 OOM，但监控显示平均显存仍有空余，可能是什么原因？
32. 训练任务每隔几小时就整体卡住，但进程没有退出，你会重点检查哪些组件和指标？
33. 如果只能选择一个优化目标，你会如何根据离线批处理、在线聊天和长上下文服务分别选择吞吐量、TTFT 或 TPOT？

## Next Checkpoint

当 `knowledge_map.md` 能清楚回答以上问题后，进入 Stage 01：Python、C++、Linux 系统基础。
