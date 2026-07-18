我看了 [job2.md](/Users/weiqt/Public/MyProject/ai-infra/job2.md:1)。这个岗位本质上不是纯 AI Infra，而是：

> 商业场景里的 LLM Post-training + Agent 算法落地岗，AI Infra 是加分项。

先确认一个硬条件：岗位写的是 2027 届，毕业时间在 **2026 年 9 月到 2027 年 8 月** 之间。如果身份符合，就按下面准备。

**准备优先级**

1. **LLM 基础**
   你要能讲清楚 decoder-only Transformer、attention、KV Cache、prefill/decode、LoRA/QLoRA、SFT、RLHF、DPO 这些概念。
   重点不是背定义，而是能解释：为什么这些技术能改善商家问答、诊断、推荐和长程对话效果。

2. **Post-training 主线**
   这是岗位核心，对应 [job2.md](/Users/weiqt/Public/MyProject/ai-infra/job2.md:6) 和 [job2.md](/Users/weiqt/Public/MyProject/ai-infra/job2.md:11)。
   必须准备：
   - SFT 数据怎么构造、清洗、配比
   - RM/reward model 怎么训练
   - RLHF/PPO/DPO/GRPO 的区别
   - reward hacking、长度偏置、偏好数据噪声怎么处理
   - 如何评估模型是否真的更会“经营诊断”和“说服商家”

3. **Agent 主线**
   这是岗位另一个核心，对应 [job2.md](/Users/weiqt/Public/MyProject/ai-infra/job2.md:5) 和 [job2.md](/Users/weiqt/Public/MyProject/ai-infra/job2.md:7)。
   准备这些：
   - ReAct / plan-and-execute / tool calling
   - 多轮对话状态管理
   - RAG 与工具调用的边界
   - 长程任务拆解
   - Agent 评测：任务成功率、工具调用准确率、幻觉率、业务指标提升

4. **商业广告场景理解**
   你至少要懂一些基础指标：
   - GMV、ROI、CTR、CVR、CPA、ROAS
   - 投放预算、出价、转化漏斗
   - 商家问题诊断：曝光少、点击低、转化差、成本高、素材疲劳

5. **AI Infra 加分项**
   你当前仓库路线偏 AI Infra，但这个岗位里 Infra 不是主战场。保留这些就够：
   - vLLM 部署和压测
   - batch inference / latency / throughput
   - KV Cache 为什么影响推理成本
   - SFT/RLHF 训练为什么吃显存和算力

**最推荐做的项目**

做一个“商家经营诊断 Agent”作为主项目：

- 输入：商家经营数据、广告投放数据、商品信息、历史对话
- 工具：查询 ROI、CTR、CVR、预算消耗、行业 benchmark
- 能力：回答问题、诊断原因、给优化建议、生成投放方案
- Post-training：用 LoRA/QLoRA 做 SFT，再用 DPO 或偏好数据做对齐
- 评测：准备 50-100 个场景题，统计诊断准确率、工具调用正确率、建议可执行性

这一个项目能覆盖岗位里的 Agent、Post-training、工具调用、经营理解、评测和落地。

**接下来 4 周安排**

第 1 周：补 LLM + Post-training 基础
输出：一页笔记，讲清 SFT、RM、RLHF、DPO、PPO、GRPO 的区别。

第 2 周：做商家 Agent 原型
输出：能调用 mock 数据工具，完成经营状态理解和问题诊断。

第 3 周：做 SFT 小实验
输出：用商家问答/诊断样例微调一个小模型，记录数据格式、loss、效果案例。

第 4 周：做评测和优化
输出：一份评测报告，包括失败案例、幻觉问题、工具调用错误、改进方案。

你仓库里当前 `stage_00_understand_ai_infra_map/knowledge_map.md` 还是空的。今天建议先补这个文件，但内容要偏向这个岗位：把 `Inference Infra`、`LLM Post-training`、`Agent`、`Evaluation` 四块写清楚。然后下一步直接开始商家 Agent 项目，而不是先深挖 CUDA/Triton。