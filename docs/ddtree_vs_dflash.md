# DDTree vs DFlash 以及 CatkinWave 推荐路线

## 1. 文档目标

这份文档回答三个问题：

1. DDTree 的技术方案到底是什么。
2. 它和当前 vLLM 中的 DFlash 有哪些相同点、不同点。
3. 如果 CatkinWave 的目标是：
   - 解决 DFlash 单线验证的问题；
   - 吸收 DDTree 的并行多线验证思想；
   - 避免 DDTree 对 tree attention 的依赖；
   - 尽量复用现有高性能 verifier backend，例如 FlashAttention-2、FlashInfer、标准 causal attention；
   - 同时考虑后续多模态升级和 vLLM 工程落地，
   
   那么哪条技术路线最靠谱。


## 2. 先给结论

一句话总结：

```text
DFlash 的核心价值是 block diffusion drafter；
DDTree 的核心价值是多分支并行验证思想；
CatkinWave 最适合走的路线，不是直接复刻 DDTree 的 tree attention，
而是把 DDTree 的“多候选并行验证”改写成
vLLM / FA2 / NPU 友好的 Shared-Prefix Batch Verification。
```

更进一步说：

```text
CatkinWave = DFlash drafter
           + Hazard-aware / Delayed-Branch candidate builder
           + Shared-Prefix Batch Verification
           + Prefix-slab / Two-stage verification (第一阶段更推荐)
```


## 3. DDTree 的方案到底是什么

### 3.1 DDTree 不是另一个完全不同的 drafter

DDTree 仓库虽然名字叫 `DDTree`，但它内部的 draft model 实际上仍然是 DFlash 风格的 draft model。

它保留了以下关键点：

- 仍然使用 target model 导出的多层 hidden states。
- 仍然对这些 target hidden states 做拼接与投影。
- 仍然使用 block diffusion 方式一次生成一个 block 的 draft logits。
- 仍然使用非自回归的 block 内并行建模。

所以 DDTree 不是“抛弃 DFlash”，而是：

```text
在 DFlash block drafter 之上，
把单路径验证升级成树形多路径验证。
```

### 3.2 DDTree 的真正创新点

DDTree 的真正创新点不在 drafter，而在 verifier 输入的组织方式。

普通 DFlash：

```text
draft logits
   -> 采样出一条 block 路径
   -> target 验证这一条线
```

DDTree：

```text
draft logits
   -> 构造一棵高概率 prefix tree
   -> target 一次验证整棵树
   -> 沿 posterior 走出最长接受路径
```

### 3.3 DDTree 的数据流

```text
target model prefill
    |
    v
target hidden states / aux hidden states
    |
    v
DFlash-like draft model
    |
    v
draft logits for a block
    |
    v
tree builder
    |
    +--> node_token_ids
    +--> node_depths
    +--> parents / child_maps
    +--> visibility mask
    |
    v
tree compiler
    |
    +--> verify_input_ids
    +--> verify_position_ids
    +--> tree attention mask
    |
    v
target verifier (tree attention)
    |
    v
posterior over tree nodes
    |
    v
follow accepted tree path
    |
    v
compact target KV cache
```


## 4. DDTree 和 vLLM DFlash 的相同点

### 4.1 两者都属于 speculative decoding

这点最基础，但也是最重要的共同点：

- 都有 draft side
- 都有 verify side
- 都是 target model 最终决定接受与拒绝

### 4.2 两者都依赖 target hidden states

二者都不是简单 token-only 的小模型 speculative decoding。

它们都通过 target model 的多层 hidden states 来条件化 draft model。

### 4.3 两者都属于 block proposal

两者都不是：

```text
一次 draft 一个 token
```

而是：

```text
一次 draft 一个 block
```

### 4.4 两者都需要 draft side 非因果式 block 内建模

无论是 vLLM DFlash 还是 DDTree 中的 DFlash 风格 draft model，本质都要求：

```text
在同一 block 内并行提出多个 speculative token
```

这也是为什么它们都不是标准单线 causal-only 小模型。


## 5. DDTree 和 vLLM DFlash 的核心差异

### 5.1 最大差异：DFlash 验证一条线，DDTree 验证一棵树

这是最根本的区别。

#### DFlash

```text
candidate block:
  y1 y2 y3 y4 ... yL

target verifier:
  验证这条路径
```

#### DDTree

```text
candidate tree:
          root
       /   |   \
      a    b    c
     / \        |
    d   e       f

target verifier:
  一次验证整棵树
```

所以 DDTree 并不是“更强的 drafter”，而是“更激进的 verifier 组织方式”。

### 5.2 DDTree 需要 tree attention，DFlash 不需要

DFlash verifier 验证的是标准线性 token 序列，因此 verifier 端可以保持标准 causal 语义。

DDTree verifier 验证的是树结构，因此：

- 每个节点可见的祖先不同；
- 不同分支彼此不可见；
- 不能用标准 causal mask；
- 需要不规则 tree mask。

这直接导致：

```text
DDTree verifier backend 比 DFlash verifier backend 更难接入
```

### 5.3 DDTree 需要分支 cache compaction，DFlash 基本不需要

DDTree 在 target verifier 里一次 append 了整棵树的多个节点，但最终只接受其中一条路径。

因此需要：

```text
把没选中的分支从 target KV cache 里压缩掉
```

这是一个明显高于 DFlash 的工程复杂度。

### 5.4 当前 vLLM DFlash 更偏向“context KV 预写”，DDTree 仓库实现更偏向“显式条件 attention”

这一点对 CatkinWave 后续设计很关键。

#### vLLM DFlash

更偏向：

```text
target hidden states
  -> projected_context_states
  -> precompute_and_store_context_kv
  -> draft model 只跑 query block
```

#### DDTree 仓库中的 DFlash

更偏向：

```text
target_hidden
  直接参与 attention 里的 K/V 计算
```

也就是说，二者在“draft model 如何消费 target hidden states”上，工程实现风格并不完全一样。


## 6. 为什么 DDTree 不能直接成为 CatkinWave 的落地路线

### 6.1 tree attention 会显著抬高后端适配成本

如果直接照搬 DDTree 到 vLLM，至少会新增以下成本：

- verifier attention backend 需要支持 tree attention mask
- 需要支持不规则可见性矩阵
- 需要处理分支型 cache compaction
- 需要处理动态 tree shape
- 很难直接复用现有 FA2 / FlashInfer / Triton causal verifier 路径

这和 CatkinWave 当前“尽量复用现有高性能推理算子”的目标是相冲突的。

### 6.2 对 NPU / 静态图 / 图编译后端更不友好

即便不只看 Ascend，这类后端通常都会偏好：

- 固定 shape
- 规则 batch
- 标准 causal mask

而 DDTree 的 tree attention 恰好是反方向：

- 每轮树结构不一样
- mask 不规则
- 分支数和深度动态变化

所以如果 CatkinWave 还希望保留后续 NPU 落地空间，那么直接走 DDTree tree attention 不是最优选择。

### 6.3 它改 verifier 太深，不适合第一阶段迭代

CatkinWave 当前最现实的目标，不是“把最复杂的 verifier 做出来”，而是先完成：

```text
从单线验证升级到并行多线验证
但 verifier 仍尽量使用标准 causal attention
```


## 7. 对你给出的候选思路的判断

下面从“和 vLLM 的适配度、改动量、后续迭代演进性”三个维度判断。

### 7.1 直接上 Tree Attention

#### 判断

不建议作为 CatkinWave 第一阶段主路线。

#### 原因

- 改动最大
- 和 vLLM verifier backend 耦合最深
- 对 FA2 / FlashInfer 复用最差
- 对 NPU 适配最不友好

#### 定位

可以保留为长期研究分支，但不建议作为主干工程方案。

### 7.2 Shared-Prefix Batch Verification

#### 判断

非常靠谱，是 CatkinWave 最值得吸收的主框架。

#### 核心思想

把 DDTree 的“多候选并行验证”保留下来，但不做 tree attention。

改成：

```text
构造 K 条线性候选路径
把它们作为 batch 中的 K 个样本
用标准 causal attention 一次验证
```

#### 和 vLLM 的适配点

这是最关键的地方。

在 vLLM 里，prefix KV cache 本来就是通过 block table 管理的。  
因此对于同一个请求的 K 条候选路径，可以把它们视作：

```text
K 个“共享前缀 block table”的虚拟分支请求
```

也就是说：

- 已提交的 prefix blocks 可以共享引用；
- 只有待验证的新增 suffix token 需要单独临时 slot / block；
- 不需要复制整段 prefix KV；
- verifier attention 仍是标准 causal。

这和 DDTree 的“共享树前缀”虽然不是同一种形式，但在工程上达到了类似目标：

```text
共享前缀，不共享不规则分支 attention mask
```

#### 优点

- 最容易复用现有 FA2 / FlashInfer / Triton verifier 路径
- 不需要 tree attention
- 和 paged KV cache 的共享前缀思路天然兼容
- 对后续 NPU 适配更友好

#### 缺点

- 多条路径之间存在重复 suffix 计算
- 理论上没有 DDTree tree attention 那么极致

#### 结论

这是 CatkinWave 最值得采用的 verifier 主框架。

### 7.3 Prefix-Slab / Two-Stage Verification

#### 判断

比完整 shared-prefix full-block verification 更适合作为第一阶段落地。

#### 核心思想

不是一上来对 K 条完整 block 路径全长验证，而是：

1. 先对前 `m` 个高风险位置做 K 条候选 prefix 并行验证
2. 选出最优 prefix
3. 再接回单路径 DFlash 或下一轮推进

#### 为什么它特别适合 CatkinWave v1

因为 DFlash 的单线问题，往往最先出现在 block 前几位：

```text
前 1~4 个 token 一旦错了，后面整个 block 都没意义
```

所以：

```text
只在早期高风险段展开分支
```

通常就能覆盖大部分收益来源。

#### 优点

- 比 full-block 多路径验证更省 verifier 计算
- 仍然只用标准 causal attention
- 非常适合先在 vLLM 里做可控迭代

#### 结论

这是 CatkinWave 第一阶段最推荐落地的版本。

### 7.4 Iterative Deepening Verification

#### 判断

可以做，但不建议第一阶段优先。

#### 原因

- verifier 要多次 forward
- 调度与 batch 组织更复杂
- 和 vLLM 当前单轮 spec verify 主流程偏差更大

#### 定位

更像后续研究型扩展，而不是第一版落地方案。

### 7.5 Top-k Set Verification

#### 判断

不建议作为主线方案。

#### 原因

- 容易破坏无损性
- 概率校正复杂
- 和现有 speculative decoding 的 accept/reject 语义不自然

#### 定位

可作为研究方向，但不适合作为当前 CatkinWave 主设计。


## 8. 树构建要不要保留？保留什么，不保留什么

DDTree 给 CatkinWave 的启发，真正值得保留的不是 tree attention，而是：

```text
“不要只相信 top-1 单线候选”
```

以及：

```text
“预算应该优先花在早期高风险位置”
```

所以 CatkinWave 更适合保留以下思想：

### 8.1 Delayed Branching

不要一开始就大分支。

更合理的是：

```text
前几个确定性强的位置走 greedy trunk
在第一个明显分歧的位置之后再开始分支
```

### 8.2 Hazard-aware Branching

分支位置不应该平均分配，而应该根据风险来选。

例如：

```text
risk_i = 1 - q_i(top1)
```

或者：

```text
risk_i = entropy_i
```

然后优先在：

- 靠前
- 风险高

的位置展开 top-2 / top-3 分支。

### 8.3 Depth-decayed / Early-weighted Scoring

浅层错误的代价明显更高，所以路径评分不能只看原始路径概率。

CatkinWave 更建议：

```text
score(path) = Σ λ_i log q_i(t_i)
```

其中：

```text
λ_1 > λ_2 > ... > λ_L
```

让早期位置的决策更重要。

### 8.4 Adaptive Budget

不是所有请求都需要 K=8 或 K=16。

更合理的是：

- 低熵请求：K 小、m 小
- 高熵请求：K 大、m 大

这样可以减少“对简单请求过度分支”的浪费。


## 9. 最推荐的 CatkinWave 技术路线

这里给出我认为最靠谱的方案。

### 9.1 方案名称

建议把第一阶段主线定为：

```text
CatkinWave-v1
= Hazard-aware Prefix-Slab Shared-Prefix Batch Verification
```

如果压缩成一句话：

```text
用 DFlash 做 block drafter；
用 DDTree 的风险分支思想生成少量多线候选；
用标准 causal verifier 做 batch 并行验证；
只先验证前几个最关键的位置。
```

### 9.2 数据流

```text
target prefill
    |
    v
DFlash drafter
    |
    v
block logits q1...qL
    |
    +--> risk / entropy estimation
    |
    +--> delayed branch selection
    |
    v
build K prefix candidates of length m
    |
    v
Shared-Prefix Batch Verification
    |
    | K virtual branches
    | shared committed prefix blocks
    | independent temporary suffix slots
    v
target verifier with standard causal attention
    |
    v
choose best branch / longest accepted prefix
    |
    v
commit accepted tokens
    |
    v
continue DFlash or next round
```

### 9.3 关键点

#### Drafter 保持 DFlash 主体不变

不建议在第一阶段重写 draft model。

直接保留：

```text
target hidden states
  -> combine_hidden_states
  -> projected_context_states
  -> precompute_and_store_context_kv
  -> query block drafting
```

这样 draft side 的收益、稳定性、后续多模态改造都不被打断。

#### Candidate builder 改成“小 beam + 风险优先”

建议：

- 先找 trunk 深度 `d*`
- 只在前 `W` 个位置中挑 top `M` 个高风险位置
- 每个位置只放 top-2 / top-3
- 构造 `K=4~8` 条候选 prefix

这本质上是：

```text
不用树，
用一个很小的 early-branch beam 来近似 tree 的收益
```

#### Verifier 用标准 causal batch

这是整个 CatkinWave 的工程核心。

每个候选分支都视作：

```text
一个普通线性序列
```

因此：

- verifier 不需要 tree attention
- verifier 可以继续复用 FA2 / FlashInfer / 标准 causal backend
- 未来也更容易适配 NPU

### 9.4 这个方案如何在 vLLM 里落地

这里按改动量给出落点。

#### 改动最小的部分：draft side

基本不动：

- `vllm/v1/spec_decode/dflash.py`
- `vllm/model_executor/models/qwen3_dflash.py`

只需要在 drafter 输出阶段，多暴露一些候选信息，例如：

- 每个位置的 top-k token ids
- 每个位置的 top-k logits / logprobs

而不是只保留最终单条 sampled path。

#### 改动核心：verifier 前的 candidate builder

建议新增一个 CatkinWave candidate builder 模块，输入：

```text
draft logits
```

输出：

```text
K 条候选 prefix / candidate paths
```

它的算法建议是：

```text
Delayed-Branch + Hazard-aware + Small Beam
```

#### 改动核心：shared-prefix branch verifier

在 `gpu_model_runner` 的 speculative verify 阶段引入“虚拟分支请求”概念：

- 同一原始请求派生 K 个 verifier branches
- 这些 branch 共享已提交 prefix block table
- 各自拥有独立的待验证新 token slot

这样可以在不复制整段 prefix KV 的情况下完成 batch verify。

#### 尽量不动的部分：attention backend

第一阶段目标是：

```text
不新增 tree attention backend
```

只复用：

- FA2
- FlashInfer
- Triton causal
- 其他标准 causal verifier backend

这会极大降低工程风险。


## 10. 建议的迭代路线

### Phase 1：Prefix-Slab Shared-Prefix Batch Verification

#### 目标

先解决：

```text
DFlash 单线验证
```

而且做到：

```text
不引入 tree attention
```

#### 建议配置

```text
block_size L = 16
prefix_slab m = 4
candidate count K = 4
branch search window W = 8
branch positions M = 2
```

#### 适合原因

- 改动最小
- 最先解决 early rejection
- easiest to integrate into vLLM

### Phase 2：Full-Block Shared-Prefix Batch Verification

在 Phase 1 稳定后，再把验证范围从前 `m` 个 token 扩展到整个 block。

适合：

- GPU 上 verifier batch 利用率高
- early rejection 解决后，仍希望进一步延长 acceptance length

### Phase 3：Adaptive Budget + Delayed Branch Beam

在 Phase 2 之后，把 candidate builder 做聪明：

- 自适应 K
- 自适应 m
- 自适应 trunk 深度
- 风险分支评分更稳

### Phase 4：研究分支

如果未来确实值得，再探索：

- tree attention backend
- 训练式 branch scorer
- 多模态条件分支评分器


## 11. 和多模态 CatkinWave 的关系

这条路线还有一个很大的优点：

```text
多模态注入问题
和
并行多线验证问题
基本可以解耦
```

也就是说：

- draft side 如何注入视觉信息
- verifier side 如何并行验证多条候选线

是两条可以并行演进的路线。

这样 CatkinWave 的整体路线就很清晰：

### 路线 A：多模态 draft 注入

- Query-only visual embed
- Visual-aware hidden state fusion
- Dual-memory DFlash

### 路线 B：多线 verifier 升级

- Prefix-slab shared-prefix batch verification
- Full-block shared-prefix batch verification
- Adaptive delayed-branch beam

二者最终可以合流成：

```text
多模态 DFlash drafter
   + 多线并行 verifier
```


## 12. 最终建议

如果从：

- 和 vLLM 现有实现的兼容性
- 改动量
- 后续迭代演进性
- FA2 / FlashInfer / NPU 友好度

这四个维度一起看，我的建议是：

### 不推荐作为主线

- 直接照搬 DDTree tree attention
- Top-k set verification
- 第一阶段就做 iterative deepening

### 推荐作为主线

```text
CatkinWave-v1:
Hazard-aware Prefix-Slab Shared-Prefix Batch Verification
```

### 推荐作为 v2 / v3 增强

- Full-block shared-prefix batch verification
- Adaptive delayed-branch beam
- Entropy / hazard-aware budget scheduling

### 核心判断

CatkinWave 不应该是：

```text
把 DDTree 原样搬进 vLLM
```

而应该是：

```text
吸收 DDTree 的多线并行验证思想，
但把 verifier 组织方式重写成
标准 causal attention 友好的 shared-prefix batch verification。
```

这条路线最稳，也最适合后续持续演进。
