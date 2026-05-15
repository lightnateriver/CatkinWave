# CatkinWave Design

## 1. 文档定位

这份文档不是拍板某一个唯一方案，而是保留一组可论证、可筛选、可逐步迭代的候选设计。

目标很明确：

```text
CatkinWave 要比 DFlash 更好，
解决 DFlash 的单线验证问题；

CatkinWave 也要比 DDTree 更容易落地，
解决 DDTree 对 tree attention 的强依赖问题；

同时尽量复用 vLLM 现有推理主链路和高性能 attention backend。
```


## 2. 设计目标

### 2.1 要解决的问题

#### 问题 A：DFlash 单线验证

DFlash 的问题不是 drafter 不够好，而是：

```text
最终只验证一条线
```

一旦前 1~4 个 token 里有一个错，后续整段 block 基本白算。

#### 问题 B：DDTree verifier 过重

DDTree 虽然解决了单线问题，但代价是：

- tree attention
- 不规则 verifier mask
- branch cache compaction
- 更重的 scheduler / request orchestration

这使它很难成为第一阶段主干工程方案。

### 2.2 CatkinWave 的核心目标

所以 CatkinWave 的核心目标可以表述为：

```text
保留 DFlash 的高效 block drafter；
吸收 DDTree 的多候选并行验证思想；
把 verifier 设计成标准 causal attention 友好的形式。
```


## 3. 设计原则

### 3.1 drafter 尽量不重写

优先保留 vLLM 当前 DFlash 主干：

- target hidden states
- combine_hidden_states
- projected_context_states
- precompute_and_store_context_kv
- query-only drafting

因为这部分已经是 DFlash 最有价值、也最稳定的部分。

### 3.2 verifier 尽量复用标准 causal attention

优先复用：

- FlashAttention-2
- FlashInfer
- Triton causal attention
- 未来的标准 NPU causal backend

避免第一阶段引入：

- tree attention verifier
- 不规则 branch-local sparse mask

### 3.3 candidate builder 可以更灵活

相较于 verifier，candidate builder 是更适合做创新的地方。

因为它只依赖：

- draft logits
- 位置风险
- budget 调度

而不会直接触碰最底层 attention backend。

### 3.4 路线要支持后续多模态融合

CatkinWave 的 verifier 升级路线和多模态 draft 注入路线，应该尽量解耦。

也就是说：

- verifier 升级不应该阻碍视觉信息后续注入
- 多模态扩展不应该推翻 verifier 设计


## 4. 候选方案池

下面保留多种可能路线，用于后续筛选。


## 5. 方案 A：Shared-Prefix Batch Verification

### 5.1 核心思想

这是 CatkinWave 最基础、也最值得保留的方案。

不用 tree attention，不构造 verifier tree，而是：

```text
从 draft logits 中构造 K 条线性候选路径；
把这 K 条路径当成 batch 中的 K 个 verifier 分支；
用标准 causal attention 一次性并行验证。
```

### 5.2 数据流

```text
target prefill
    |
    v
DFlash drafter
    |
    v
draft logits q1...qL
    |
    v
candidate path builder
    |
    +--> path 1
    +--> path 2
    +--> ...
    +--> path K
    |
    v
Shared-Prefix Batch Verification
    |
    | K virtual branches
    | standard causal attention
    v
choose best branch / longest accepted prefix
```

### 5.3 工程优势

- verifier 不需要 tree attention
- 可以复用现有 GPU / NPU causal attention 路径
- 和 vLLM 的 paged KV cache 更容易兼容
- 是所有候选方案里最稳的主干框架

### 5.4 问题

- 不同路径之间存在重复 suffix 计算
- 没有 DDTree 那么极致地共享分支结构


## 6. 方案 B：Prefix-Slab Verification

### 6.1 核心思想

不是对整条 block 做多线验证，而是只对前 `m` 个最关键的 token 做多线并行验证。

因为 DFlash 的大多数损失都来自：

```text
block 前几位出错
```

所以可以把问题缩成：

```text
先解决 early rejection
```

### 6.2 数据流

```text
draft logits q1...qL
    |
    v
build K candidate prefixes of length m
    |
    v
batch verify these prefixes
    |
    v
choose best accepted prefix
    |
    v
continue with base DFlash suffix or next round
```

### 6.3 优点

- verifier 成本更低
- 适合先在 vLLM 里做第一阶段落地
- 可以解决大量 early rejection 问题

### 6.4 问题

- 如果错误主要发生在 block 深处，收益有限

### 6.5 判断

这是 CatkinWave v1 最适合作为主线落地的方案。


## 7. 方案 C：Full-Block Shared-Prefix Verification

### 7.1 核心思想

这是方案 A 的完整版本：

- 不只验证前缀
- 而是验证完整 block 的多条候选线

### 7.2 优点

- 更接近 DDTree “多候选验证”的完整收益
- 不需要 tree attention

### 7.3 问题

- verifier 计算量明显高于 Prefix-Slab
- 对 batch 组织和 slot 复用要求更高

### 7.4 判断

适合作为 CatkinWave v2，而不是第一阶段。


## 8. 方案 D：Delayed Branching

### 8.1 核心思想

不要一开始就分支，而是：

```text
先走一段 greedy trunk；
在第一个明显分歧的位置之后再开始分支。
```

### 8.2 原因

DDTree 给出的一个重要启发是：

```text
浅层位置往往 target 与 draft 更一致；
真正需要探索的是更靠后、分歧更大的位置。
```

### 8.3 作用方式

这个方案不是独立 verifier，而更像 candidate builder 的增强策略，可以和 A/B/C 叠加。

### 8.4 判断

很值得保留，尤其适合和 Prefix-Slab 结合。


## 9. 方案 E：Hazard-aware Branching

### 9.1 核心思想

分支位置不平均分配，而是优先选：

- 更靠前
- 风险更高

的位置。

例如定义：

```text
risk_i = 1 - q_i(top1)
```

或者：

```text
risk_i = entropy_i
```

再用：

```text
priority_i = risk_i * decay(i)
```

进行排序。

### 9.2 优势

- 非常符合 speculative decoding 的真实收益结构
- 不需要动 verifier backend
- 只动 candidate builder，就可能带来明显 acceptance 改善

### 9.3 判断

强烈建议保留，是 CatkinWave 很值得吸收的关键思想。


## 10. 方案 F：Adaptive Budget

### 10.1 核心思想

不同请求不应该固定使用同一个：

- 候选数 K
- prefix 长度 m
- 分支预算 B

更合理的是根据：

- 熵
- top-1 margin
- 历史接受率

做动态调整。

### 10.2 适合方式

可叠加在 A/B/C/D/E 之上。

### 10.3 判断

适合作为第二阶段增强，而不是第一阶段必做项。


## 11. 方案 G：Iterative Deepening Verification

### 11.1 核心思想

不是一次性验证完整候选，而是逐层扩展、逐层验证：

```text
depth 1 batch verify
 -> depth 2 batch verify
 -> depth 3 batch verify
 ...
```

### 11.2 优点

- 不需要 tree attention
- 可以更细粒度控制分支扩张

### 11.3 缺点

- verifier forward 次数变多
- 调度复杂
- 和 vLLM 当前 spec decode 主流程偏差较大

### 11.4 判断

保留为研究路线，不建议第一阶段主推。


## 12. 方案 H：Multi-Modal Aware CatkinWave

### 12.1 核心思想

在 verifier 升级之外，CatkinWave 还可以叠加多模态 draft 注入能力。

这一条线不直接决定 verifier 形态，但会影响 drafter 质量。

可复用的多模态候选方案包括：

- Query-only visual embed
- Visual-aware hidden state fusion
- Dual-memory DFlash
- Visual cross-attention adapter

### 12.2 和本设计的关系

这一层和 verifier 方案应该尽量解耦。

也就是说：

```text
先确定 verifier 是 Shared-Prefix / Prefix-Slab / Full-Block 哪一类；
再决定视觉信息如何进入 DFlash drafter。
```


## 13. 推荐筛选顺序

### 第一优先级

- 方案 A：Shared-Prefix Batch Verification
- 方案 B：Prefix-Slab Verification
- 方案 D：Delayed Branching
- 方案 E：Hazard-aware Branching

这是最值得优先保留并论证的组合。

### 第二优先级

- 方案 C：Full-Block Shared-Prefix Verification
- 方案 F：Adaptive Budget

### 第三优先级

- 方案 G：Iterative Deepening Verification
- Tree Attention 原始 DDTree verifier


## 14. 当前最推荐的组合

如果现在就让我给一个最靠谱、最适合后续演进的设计组合，我会建议：

```text
CatkinWave-v1
= DFlash drafter
+ Delayed Branching
+ Hazard-aware candidate selection
+ Prefix-Slab Shared-Prefix Batch Verification
```

原因是：

- 解决 DFlash 单线验证问题
- 避开 DDTree 的 tree attention verifier 复杂度
- 更容易复用现有高性能 causal attention backend
- 改动集中在 candidate builder 和 verifier orchestration
- 对后续多模态扩展更友好


## 15. 后续如何继续筛选

后续方案筛选建议围绕以下维度做：

### 15.1 acceptance length 提升

重点看：

- 是否显著减少 early rejection
- 平均接受长度是否稳定提升

### 15.2 verifier 成本

重点看：

- 额外 verifier token 数
- verifier batch 利用率
- 是否能继续使用 FA2 / FlashInfer / causal backend

### 15.3 vLLM 改动量

重点看：

- 是否需要新 attention backend
- 是否需要大改 scheduler / cache 管理
- 是否能增量落地

### 15.4 多模态兼容性

重点看：

- verifier 升级是否阻碍后续视觉信息接入 drafter


## 16. 总结

CatkinWave 的价值，不在于机械地“把 DFlash 和 DDTree 拼起来”，而在于：

```text
保留 DFlash 的高效 block drafter，
吸收 DDTree 的多线验证思想，
重写一个更适合 vLLM / GPU / NPU 演进的 verifier 组织方式。
```

因此，这份文档里保留的候选方案池，真正值得优先推进的是：

```text
Shared-Prefix Batch Verification
    + Prefix-Slab
    + Delayed Branching
    + Hazard-aware Branching
```

这组路线最稳，也最适合后续继续筛选论证。
