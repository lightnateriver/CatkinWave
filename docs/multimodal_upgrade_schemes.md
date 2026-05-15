# CatkinWave 多模态升级方案设计

## 1. 背景

当前 vLLM 中的 DFlash 已经具备完整的文本 speculative decoding 主链路，但在多模态路径上存在一个明显缺口：

- `gpu_model_runner` 已经能够收集 `mm_embed_inputs`，并传给 drafter 的 `propose(...)` 入口。
- 普通 EAGLE 多模态路径会在 `build_model_inputs_first_pass()` 中消费这些 multimodal embeddings。
- DFlash 当前没有真正消费 `mm_embed_inputs`，而是在 `build_model_inputs_first_pass()` 里固定返回 `inputs_embeds=None`。
- 因此，当前 DFlash 的 draft model 主要依赖 text token 与 target model aux hidden states，视觉信息没有系统性进入 draft path。

换句话说，当前 DFlash 的多模态问题不是“完全没有视觉编码器”，而是：

```text
视觉信息已经被 target path 感知
但是没有沿着 draft path 被稳定注入和消费
```

CatkinWave 的目标，就是在尽量少改 vLLM 推理主链路的前提下，把视觉信息注入到 DFlash draft model。


## 2. 设计约束

为了让方案真正可落地，这里明确几个约束：

### 2.1 尽量少改 vLLM 主链路

优先复用现有能力：

- `gpu_model_runner._gather_mm_embeddings()`
- `mm_features` 与 `mm_position`
- `model.embed_input_ids(..., multimodal_embeddings=..., is_multimodal=...)`
- 现有 DFlash 的 `target hidden states -> combine_hidden_states -> precompute_and_store_context_kv`

### 2.2 保留 DFlash 的性能核心

尽量不要破坏 DFlash 当前最重要的加速结构：

```text
target hidden states
    -> projected_context_states
    -> precompute_and_store_context_kv
    -> draft model only runs query block
```

### 2.3 多模态注入要分层次推进

不是所有方案都要一步到位。更合理的方式是：

- 先打通“能用”的最小路径
- 再增强“上下文记忆中的视觉条件”
- 最后评估是否需要更强但更重的双路记忆或 cross-attention


## 3. 当前 DFlash 与多模态数据流

先把当前状态画清楚，后面的 4 个方案都建立在这个基线之上。

### 3.1 当前数据流

```text
image/video input
      |
      v
target multimodal encoder / projector
      |
      v
target model sees visual information
      |
      +--> target hidden_states / aux_hidden_states
      |
      +--> runner gathers mm_embed_inputs
                |
                v
           propose(..., mm_embed_inputs)
                |
                v
        DFlash build_model_inputs_first_pass()
                |
                +--> currently ignores mm_embed_inputs
                +--> inputs_embeds = None
                |
                v
        draft model runs text-side DFlash path only
```

### 3.2 当前问题归纳

当前问题可以拆成两类：

1. query 注入缺失

```text
query block [next_token, mask, ...]
没有消费真实的 multimodal embeddings
```

2. context 注入缺失

```text
target aux hidden states 被投影到 draft hidden size
但这条链路没有显式融合视觉摘要或视觉记忆
```


## 4. 方案总览

### 4.1 四个候选方案

- 方案 A：Query-Only Visual Embed Reuse
- 方案 B：Visual-Aware Hidden State Fusion
- 方案 C：Dual-Memory DFlash
- 方案 D：Cross-Attention Visual Adapter

### 4.2 从改动小到大的排序

```text
方案 A < 方案 B < 方案 C < 方案 D
```

### 4.3 推荐顺序

```text
短期 MVP: 方案 A
中期主线: 方案 B
中长期增强: 方案 C
研究型扩展: 方案 D
```


## 5. 方案 A：Query-Only Visual Embed Reuse

### 5.1 核心思路

这是改动最小的方案。

不改 DFlash 的 context KV 预写逻辑，只改 query block 的输入构造：

- 继续沿用当前 DFlash 的 `target hidden states -> precompute_and_store_context_kv`
- 让 query block 在进入 draft model 前，像 EAGLE 多模态路径一样消费 `mm_embed_inputs`
- 也就是说，只把视觉信息注入 query 输入，不改 context memory

### 5.2 数据流字符图

```text
image/video
    |
    v
target mm encoder / projector
    |
    v
mm embeddings
    |
    +--> runner._gather_mm_embeddings()
             |
             v
        mm_embed_inputs
             |
             v
   DFlash build_model_inputs_first_pass()
             |
             | embed_input_ids(
             |   input_ids=query_input_ids,
             |   multimodal_embeddings=mm_embeds,
             |   is_multimodal=is_mm_embed
             | )
             v
        query_inputs_embeds
             |
             v
        draft model forward(query only)
             |
             v
        draft logits / speculative tokens

in parallel:

target aux hidden states
      |
      v
combine_hidden_states
      |
      v
projected_context_states
      |
      v
precompute_and_store_context_kv
```

### 5.3 数据调用链路

```text
SchedulerOutput
   -> gpu_model_runner._gather_mm_embeddings()
   -> DFlashProposer.propose(..., mm_embed_inputs=...)
   -> DFlashProposer.build_model_inputs_first_pass(...)
   -> DFlashQwen3ForCausalLM.embed_input_ids(...)
   -> DFlashQwen3Model.forward(inputs_embeds=query_inputs_embeds)
```

### 5.4 需要改哪些地方

- `DFlashProposer.build_model_inputs_first_pass()`
  - 参考 EAGLE 多模态分支，消费 `mm_embed_inputs`
- `DFlashQwen3ForCausalLM.embed_input_ids()`
  - 透传 `multimodal_embeddings` 和 `is_multimodal`
- `DFlashQwen3Model.embed_input_ids()`
  - 如果底层 draft model 结构需要，扩展为支持多模态 embedding 替换

### 5.5 优点

- 改动最小
- 基本不动 DFlash 的核心加速路径
- 完全复用现有 `mm_embed_inputs` 接口
- 实现成本最低，适合先验证收益

### 5.6 局限

- 视觉信息只进入 query，不进入 context memory
- 对强视觉依赖任务，提升可能有限
- 更像“让 draft 看见图像 token 的输入”，还不是“让 DFlash 记住视觉上下文”

### 5.7 适用建议

适合做第一版 MVP，用来回答两个问题：

- 多模态 DFlash 在现有架构下是否有正收益
- 视觉信息只注入 query 端时，收益是否已经足够


## 6. 方案 B：Visual-Aware Hidden State Fusion

### 6.1 核心思路

这是最推荐的主线方案。

保留 DFlash 当前主结构不变，只在 `combine_hidden_states()` 附近加一个轻量视觉条件分支。

做法是：

- 从 target path 或 encoder path 得到一个低成本视觉摘要 `v_summary`
- 与多个 aux hidden states 拼接后的特征一起融合
- 再投影到 draft hidden size
- 继续沿用 DFlash 的 context KV 预写机制

换句话说：

```text
把视觉信息注入 projected_context_states
而不是只注入 query token embedding
```

### 6.2 视觉摘要的形式

可以有几种轻量实现：

- 全局平均池化：`v_global`
- Resampler 输出的少量 summary token，再池化
- 选取 CLS / pooled token
- 少量 learnable visual anchors 与原视觉 token 做聚合

为了尽量少改推理路径，建议优先选：

```text
单向量或少量向量摘要
```

### 6.3 数据流字符图

```text
image/video
    |
    v
target mm encoder / projector
    |
    +--> visual token embeddings
    |         |
    |         v
    |    visual summarizer / pooling
    |         |
    |         v
    |      v_summary
    |
    +--> target model forward
              |
              v
        aux_hidden_states (K tensors)
              |
              v
        concat aux states
              |
              | cat / gate / FiLM with v_summary
              v
  multimodal_hidden_fusion_features
              |
              v
     combine_hidden_states / fc_mm
              |
              v
    projected_context_states
              |
              v
   precompute_and_store_context_kv
              |
              v
   draft model query-only forward
```

### 6.4 数据调用链路

```text
mm encoder output
   -> visual summary builder
   -> DFlashProposer.propose(..., mm_summary=...)

target aux hidden states
   -> torch.cat(..., dim=-1)
   -> DFlashQwen3ForCausalLM.combine_hidden_states(
          aux_concat,
          mm_summary
      )
   -> projected_context_states
   -> precompute_and_store_context_kv
```

### 6.5 可选融合方式

可以从轻到重分三档：

1. 直接拼接

```text
[aux_concat, v_summary] -> linear
```

2. 门控融合

```text
g = sigmoid(W[v_summary])
projected = g * aux_proj + (1-g) * visual_proj
```

3. FiLM / bias conditioning

```text
aux_proj = gamma(v_summary) * aux_proj + beta(v_summary)
```

第一版建议选 1 或 2。

### 6.6 需要改哪些地方

- runner 新增 `mm_summary` 构造与传递
- `DFlashQwen3ForCausalLM.combine_hidden_states()` 扩展接口
- `DFlashQwen3Model` 增加融合投影层
- 保持 `precompute_and_store_context_kv()` 不变

### 6.7 优点

- 对 vLLM 主链路改动仍然较小
- 不用改 attention backend
- 不用改 KV cache 语义
- 视觉信息真正进入 DFlash 的 context states
- 保住了 DFlash 当前最核心的性能结构

### 6.8 局限

- 视觉信息是压缩表达，细粒度局部视觉细节可能损失
- 需要定义一个稳定的视觉摘要接口
- 效果依赖 summary 质量

### 6.9 适用建议

这是最平衡的路线，适合作为 CatkinWave 的主线升级方案。


## 7. 方案 C：Dual-Memory DFlash

### 7.1 核心思路

这个方案让 draft model 同时拥有两类上下文 memory：

- 文本 context memory
- 视觉 context memory

文本 context memory 仍然来自 target aux hidden states 投影。
视觉 context memory 则来自视觉 token embeddings 再投影到 draft hidden size。

两者最终统一进入 DFlash 的 context KV 预写流程。

### 7.2 基本想法

把视觉 token 当作一组额外 context states：

```text
visual_context_states : [S_v, H_d]
text_context_states   : [S_t, H_d]

unified_context_states = concat(visual_context_states, text_context_states)
```

随后统一生成：

- `context_positions`
- `context_slot_mapping`

然后走一套 `precompute_and_store_context_kv()`

### 7.3 数据流字符图

```text
image/video
    |
    v
target mm encoder / projector
    |
    v
visual token embeddings
    |
    | visual projector
    v
visual_context_states [S_v, H_d]

target aux hidden states
    |
    v
combine_hidden_states
    |
    v
text_context_states [S_t, H_d]

visual_context_states
      +
text_context_states
      |
      v
unified_context_states [S_v + S_t, H_d]
      |
      +--> unified_positions
      +--> unified_slot_mapping
      |
      v
precompute_and_store_context_kv
      |
      v
draft KV cache now stores:
    visual memory + text memory
      |
      v
query block forward
```

### 7.4 数据调用链路

```text
visual embeddings
   -> visual projector
   -> visual_context_states

aux hidden states
   -> combine_hidden_states
   -> text_context_states

merge contexts
   -> unified_context_states
   -> DFlash precompute_and_store_context_kv
   -> draft attention reads merged memory
```

### 7.5 位置与 slot 的处理策略

这是这个方案的关键难点。

有两种主要做法：

1. 复用原始 multimodal 占位位置

```text
视觉 context token 使用它们在原 prompt 中的真实位置
```

2. 额外插入虚拟 prefix 区间

```text
给视觉 context token 分配一个单独的 prefix 区域
```

更推荐第 1 种，因为它和现有 `mm_position` 语义更一致，也更容易与 target path 对齐。

### 7.6 需要改哪些地方

- 新增 `visual_context_states` 构造逻辑
- DFlash proposer 扩展 context buffers，支持视觉和文本双来源 context
- `context_positions` 与 `context_slot_mapping` 合并逻辑
- 可能需要 attention metadata 更细致地标记 visual/text context 范围

### 7.7 优点

- 视觉信息真正进入 DFlash context memory
- 表达能力明显强于视觉摘要方案
- 保留了“先写 KV，再跑 query”的 DFlash 主理念

### 7.8 局限

- 上下文 token 变长，KV 预写成本增加
- 位置和 slot 组织更复杂
- 多图、多区域、多 patch 时，视觉 memory 膨胀风险较大

### 7.9 适用建议

如果方案 B 收益不够，而任务又强依赖局部视觉细节，这个方案值得进入实现评估。


## 8. 方案 D：Cross-Attention Visual Adapter

### 8.1 核心思路

这个方案不把视觉信息塞进 DFlash 的 self-attention context memory，而是给 draft model 增加一层很薄的 visual cross-attention adapter。

也就是说：

- 文本依赖仍然通过 DFlash context KV 提供
- 视觉依赖通过单独 cross-attention memory 提供

### 8.2 结构理解

```text
self-attention memory 负责文本上下文
cross-attention memory 负责视觉上下文
```

这比“把视觉 token 强行并到 self-attention context 里”语义更干净。

### 8.3 数据流字符图

```text
target aux hidden states
    |
    v
combine_hidden_states
    |
    v
text_context_states
    |
    v
precompute_and_store_context_kv
    |
    v
draft self-attention text memory

in parallel:

image/video
    |
    v
target mm encoder / projector
    |
    v
visual memory projector
    |
    v
visual_memory [S_v, H_v]
    |
    v
draft visual cross-attention adapter

query block
   |
   v
draft layer:
   self-attn on text memory
   -> visual cross-attn adapter
   -> mlp
   -> next layer
```

### 8.4 数据调用链路

```text
target aux hidden states
   -> text context KV prewrite

visual embeddings
   -> visual memory builder
   -> cached visual memory

draft forward(query only)
   -> self-attn(text KV cache)
   -> visual cross-attn adapter(visual memory)
   -> logits
```

### 8.5 实现方式

可以有两种挂载方式：

1. 只挂在前 1 到 2 层

```text
减小额外计算量
```

2. 挂在每层但参数很小

```text
提升视觉感知深度，但开销更高
```

更推荐第一种，用作高收益低风险版本。

### 8.6 需要改哪些地方

- 扩展 draft model 结构
- 增加 visual memory cache / buffer
- 在 forward 中插入 adapter
- 可能需要新的权重格式与初始化策略

### 8.7 优点

- 文本与视觉 memory 解耦，语义最清晰
- 对复杂视觉任务潜力最大
- 不需要把视觉 token 强行模拟成普通文本 context token

### 8.8 局限

- 模型结构改动最大
- 推理时新增 cross-attention 计算
- 对工程接入和权重设计要求最高

### 8.9 适用建议

更适合作为中长期研究路线，而不是第一版工程落地方向。


## 9. 四个方案的对比

### 9.1 对比维度

```text
维度 1: 对 vLLM 主链路改动大小
维度 2: 对 DFlash 性能主结构影响大小
维度 3: 视觉信息注入深度
维度 4: 实现复杂度
维度 5: 多模态效果上限
```

### 9.2 对比结论

- 方案 A
  - 改动最小
  - 效果上限相对有限
  - 最适合 MVP
- 方案 B
  - 改动较小
  - 效果与复杂度平衡最好
  - 最适合作为主线
- 方案 C
  - 视觉建模更完整
  - 需要处理 unified context 的位置与 slot
  - 适合在方案 B 之后增强
- 方案 D
  - 表达力最强
  - 工程复杂度最高
  - 适合作为研究型升级


## 10. 推荐落地路线

### 10.1 第一阶段：方案 A

目标：

```text
最小改动接通多模态 DFlash
```

验证：

- 是否能稳定跑通多模态 spec decode
- query 侧视觉注入是否已有显著收益

### 10.2 第二阶段：方案 B

目标：

```text
让视觉信息进入 projected_context_states
```

验证：

- 命中率提升是否明显高于方案 A
- 吞吐损失是否仍然可控

### 10.3 第三阶段：按收益决定 B -> C 或 B -> D

如果发现问题主要是：

- 视觉细节表达不足
  - 优先走方案 C

- 视觉与文本交互需要更强适配
  - 再评估方案 D


## 11. CatkinWave 的建议拍板

如果从“尽量少改当前 vLLM 推理逻辑”这个目标出发，推荐结论很明确：

### 推荐优先级

1. 方案 A：先上线，快速验证多模态 DFlash 是否有稳定收益
2. 方案 B：作为主线升级，强化视觉条件化能力
3. 方案 C：作为效果增强版本，补强视觉上下文记忆
4. 方案 D：作为长期研究路线

### 一句话总结

```text
CatkinWave 的首选路线不是重写 DFlash，
而是在保留 DFlash “context KV 预写 + query block 并行提案” 核心结构的前提下，
逐步把视觉信息从 query 侧、context 侧、memory 侧分层注入。
```

这条路线对现有 vLLM 改动最可控，也最有机会兼顾工程可落地性和多模态收益。
