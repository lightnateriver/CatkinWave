# CatkinWave 多模态升级：面向多模态的投机推理加速优化方案

## CatkinWave Multimodal Upgrade: Optimization Scheme for Multimodal Speculative Inference Acceleration

## 项目简介

### 中文

CatkinWave 以柳絮漫波为意象，构建跨模态特征扩散与时序波联动机制，优化图文语义对齐与投机候选生成策略，重构多模态推理调度逻辑，适配图文联合任务，显著提升投机推理命中率与多模态吞吐加速性能。

### English

Inspired by the wave of drifting catkins, CatkinWave builds cross-modal feature diffusion and temporal wave linkage. It optimizes image-text semantic alignment and speculative candidate generation, reconstructs multimodal reasoning scheduling, and effectively improves hit rate and throughput acceleration for multimodal tasks.

## 项目目标

- 梳理 vLLM 当前 DFlash 技术方案与数据链路。
- 分析 DFlash 在多模态场景中的现有限制，尤其是视觉信息未有效注入 draft model 的问题。
- 设计一组尽量复用现有 vLLM 推理主链路、又能逐步增强多模态能力的升级方案。
- 为后续工程实现提供文档基线、接口边界和改造优先级。

## 文档结构

- [docs/dflash_design.md](docs/dflash_design.md)
  - DFlash 在 vLLM 中的技术方案、运行时链路与上下文 KV 预写机制说明。
- [docs/multimodal_upgrade_schemes.md](docs/multimodal_upgrade_schemes.md)
  - CatkinWave 多模态升级方案，包含 4 个候选路线、数据流字符图、改动点与推荐顺序。
- [docs/ddtree_vs_dflash.md](docs/ddtree_vs_dflash.md)
  - DDTree 与 DFlash 的异同分析，以及 CatkinWave 面向多线并行验证的推荐技术路线。

## 当前判断

- 当前 vLLM 中的 DFlash 已具备完整的文本 speculative decoding 主链路。
- 在多模态路径上，`mm_embed_inputs` 已经可以从 runner 层传入 proposer 入口，但 DFlash 本身尚未真正消费视觉 embeddings。
- 因此，多模态升级的重点不是重写整条 speculative decode 框架，而是在尽量保持 DFlash 主体结构不变的前提下，把视觉信息有层次地注入 draft model。

## 推荐演进路线

- 第一阶段：接通 query 侧视觉 embedding 复用，最小化改动实现多模态可用。
- 第二阶段：把视觉摘要融入 target hidden state 到 draft hidden state 的投影链路，提升语义条件化能力。
- 第三阶段：视效果与复杂度权衡，再考虑视觉 memory 与文本 memory 的统一或双路建模。

## 适用范围

- 图文问答
- OCR 与图表理解
- 多图推理
- 依赖视觉上下文的长文本生成

## 状态

当前仓库以设计与方案文档为主，后续可在此基础上继续补充：

- 原型实现
- 性能对比实验
- 命中率与吞吐评测
- 后端适配说明
