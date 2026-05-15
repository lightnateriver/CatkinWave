# CatkinWave 多模态升级：面向多模态的投机推理加速优化方案

## CatkinWave Multimodal Upgrade: Optimization Scheme for Multimodal Speculative Inference Acceleration

## 项目简介

### 中文

CatkinWave 以柳絮漫波为意象，构建跨模态特征扩散与时序波联动机制，优化图文语义对齐与投机候选生成策略，重构多模态推理调度逻辑，适配图文联合任务，显著提升投机推理命中率与多模态吞吐加速性能。

### English

Inspired by the wave of drifting catkins, CatkinWave builds cross-modal feature diffusion and temporal wave linkage. It optimizes image-text semantic alignment and speculative candidate generation, reconstructs multimodal reasoning scheduling, and effectively improves hit rate and throughput acceleration for multimodal tasks.

## 项目目标

- 梳理 DFlash 与 DDTree 的详细技术方案，以及它们在 vLLM 中的接入边界。
- 分析 DFlash 单线验证与 DDTree tree attention 依赖各自带来的工程约束。
- 设计一组基于 DFlash 与 DDTree 思想的 CatkinWave 备选改进方案，用于后续筛选和论证。
- 为后续工程实现提供文档基线、接口边界和迭代方向。

## 文档结构

- [docs/dflash_vs_ddtree.md](docs/dflash_vs_ddtree.md)
  - DFlash 与 DDTree 的详细方案、DFlash 在 vLLM 中的数据流、DDTree 若合入 vLLM 的改动点，以及两者优劣和适用场景分析。
- [docs/catkinwave_design.md](docs/catkinwave_design.md)
  - CatkinWave 的候选设计池，保留多种基于 DFlash 和 DDTree 的改进路线，用于后续筛选和论证。

## 当前判断

- 当前 vLLM 中的 DFlash 已具备完整的 block diffusion drafter 主链路，但 verifier 仍然是单线的。
- DDTree 的主要价值在于多候选并行验证思想，而不只是 tree 本身。
- tree attention 在现有 GPU 路径上有局部基础设施，但没有现成 DDTree verifier 集成；在 NPU / vllm-ascend 侧则缺少完整成熟支持。
- 因此 CatkinWave 更适合保留 DFlash drafter，并探索标准 causal verifier 友好的并行多线验证方案。

## 推荐演进路线

- 第一阶段：围绕 Prefix-Slab Shared-Prefix Batch Verification 收敛 verifier 方案，先解决 DFlash 单线验证问题。
- 第二阶段：叠加 Delayed Branching、Hazard-aware Branching、Adaptive Budget 等 candidate builder 增强策略。
- 第三阶段：在 verifier 方案稳定后，再决定是否继续引入更完整的多模态 drafter 注入路线。

## 适用范围

- 面向 speculative decoding 的 drafter / verifier 升级设计
- GPU / NPU 兼顾的工程路线评估
- 后续多模态 speculative inference 加速方案选型

## 状态

当前仓库以设计与方案文档为主，后续可在此基础上继续补充：

- 原型实现
- 性能对比实验
- 命中率与吞吐评测
- 后端适配说明
