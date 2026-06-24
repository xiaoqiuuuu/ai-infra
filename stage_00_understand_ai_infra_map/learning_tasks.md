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

1. vLLM 主要解决什么推理系统问题？
2. DDP 和 FSDP 的核心区别是什么？
3. KV Cache 是什么，为什么它会成为 LLM 推理的关键？
4. NCCL 在多 GPU 训练中做什么？
5. CUDA/Triton 算子优化为什么对 AI Infra 重要？

## Next Checkpoint

当 `knowledge_map.md` 能清楚回答以上问题后，进入 Stage 01：Python、C++、Linux 系统基础。
