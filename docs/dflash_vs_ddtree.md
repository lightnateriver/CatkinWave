# DFlash 与 DDTree 对比设计

## 1. 文档目的

这份文档聚焦四件事：

1. 详细说明 DFlash 的方案，尤其是它在 vLLM 里的实现链路。
2. 详细说明 DDTree 的方案，以及它和 DFlash 的真正关系。
3. 分析如果要把 DDTree 的思想合入 vLLM，需要改哪些模块、哪里最难。
4. 对比两种方案的优劣、适用场景，并明确 tree attention 在 GPU / NPU 上的现状。


## 2. 先给结论

一句话概括：

```text
DFlash 解决的是“如何高效地产生一个 block proposal”；
DDTree 解决的是“如何不只验证单条 block 路径，而是并行验证多条高概率候选”。
```

更准确一点：

```text
DFlash = block diffusion drafter + 单线验证
DDTree = block diffusion drafter + 树形多线验证
```

所以 DDTree 不是“另一个完全不同的 drafter”，而是：

```text
在 DFlash 式 drafter 之上，
把 verifier 从单线扩展成多分支验证。
```


## 3. DFlash 方案详解

### 3.1 DFlash 的核心思想

DFlash 不是普通的小模型 token-by-token speculative decoding。

它的关键思想是：

1. target model 先正常前向。
2. 从 target model 抽取若干层 hidden states。
3. 把这些 hidden states 投影成 draft model 可以直接使用的上下文表示。
4. 不让 draft model 重新自回归跑完整个前缀，而是把上下文预先转成 draft 每层的 context KV cache。
5. draft model 本轮只处理一个小 query block，例如：

```text
[next_token, mask, mask, mask, ...]
```

6. draft model 一次并行提出多个 speculative token。
7. target model 验证这一条线性 block 路径。

所以 DFlash 的本质是：

```text
target hidden states 变成 draft context memory
draft model 只跑 query block
```

### 3.2 DFlash 在 vLLM 中的实现入口

在 vLLM 中，DFlash 通过 speculative decoding 配置接入：

```python
speculative_config = {
    "method": "dflash",
    ...
}
```

关键入口在：

- `vllm/config/speculative.py`
- `vllm/v1/spec_decode/dflash.py`
- `vllm/model_executor/models/qwen3_dflash.py`
- `vllm/v1/worker/gpu_model_runner.py`

其中有一个很关键的配置点：

```text
dflash_config.target_layer_ids
```

它决定了：

- 从 target model 的哪些层抽 hidden states
- 一共抽多少层 hidden states
- `combine_hidden_states()` 的输入维度是多少

如果 `target_layer_ids = [l1, l2, l3, l4]`，那么就表示：

```text
每个 token 会从 target model 取 4 份 hidden state
```

随后在最后一维拼接成：

```text
[S, 4 * H_t]
```

再投影到 draft hidden size：

```text
[S, H_d]
```

### 3.3 DFlash 在 vLLM 中的数据链路

下面是 vLLM 当前 DFlash 的核心数据流。这里的 `K` 就是 `target_layer_ids` 里选中的层数。

```text
target model forward
    |
    +--> final hidden_states
    +--> aux_hidden_states from target_layer_ids
    |        shape of each: [S, H_t]
    |        count: K
    |
    v
gpu_model_runner
    |
    | cat aux_hidden_states on last dim
    v
concat_hidden_states [S, K * H_t]
    |
    | combine_hidden_states()
    v
projected_context_states [S, H_d]
    |
    v
DFlashProposer
    |
    +--> build query block: [next_token, mask, ...]
    +--> build context/query positions
    +--> build context/query slot mappings
    +--> set non-causal attention metadata
    |
    v
precompute_and_store_context_kv()
    |
    | convert projected_context_states into
    | per-layer K/V of the draft model
    v
draft KV cache is prefilled
    |
    v
draft model forward(query block only)
    |
    v
draft logits
    |
    v
sample one linear candidate block
    |
    v
target model verify this single path
```

如果把这条链路写得更贴近代码，可以展开成下面这个版本：

```text
draft model hf_config.dflash_config.target_layer_ids
    |
    v
gpu_model_runner._get_eagle3_aux_layers_from_config()
    |
    v
target model.set_aux_hidden_state_layers(layer_ids)
    |
    v
target model forward
    |
    +--> hidden_states                     [S, H_t]
    +--> aux_hidden_states[0]             [S, H_t]
    +--> aux_hidden_states[1]             [S, H_t]
    +--> ...
    +--> aux_hidden_states[K-1]           [S, H_t]
    |
    v
gpu_model_runner
    |
    | torch.cat([h0, h1, ..., hK-1], dim=-1)
    v
concat_hidden_states                      [S, K * H_t]
    |
    | draft_model.combine_hidden_states()
    |   -> fc projection
    v
projected_context_states                  [S, H_d]
    |
    v
DFlashProposer.set_inputs_first_pass()
    |
    +--> target_token_ids                 [S]
    +--> target_positions                 [S]
    +--> next_token_ids                   [B]
    +--> query block = [next_token, mask, ..., mask]
    +--> causal = False
    |
    v
DFlashQwen3ForCausalLM.precompute_and_store_context_kv()
    |
    | one fused KV projection for all draft layers
    | RMSNorm + fused linear + RoPE + direct KV cache write
    v
draft per-layer KV cache is ready
    |
    v
draft model forward(query tokens only)
    |
    v
draft logits for block positions q1...qL
    |
    v
top-1 / sampled linear block y1...yL
    |
    v
target verifier accepts or rejects this single path
```

### 3.4 DFlash 在 vLLM 中的关键实现点

#### A. target hidden states 的抽取

target model 会导出多个指定层位点的 hidden states。  
这些层位点来自：

- `eagle_aux_hidden_state_layer_ids`
- 或 DFlash 配置里的 `dflash_config.target_layer_ids`

因此：

```text
一共多少个 hidden state
= target_layer_ids 的长度
```

这些 hidden states 会在 `gpu_model_runner` 中按最后一维拼接，再通过 `combine_hidden_states()` 投影到 draft hidden size。

#### B. context KV 预写

vLLM 版 DFlash 最关键的一点是：

```text
不是让 draft model 自己把上下文再跑一遍，
而是直接把 target-derived context states 写成 draft 每层的 KV cache。
```

这也是它和很多普通 draft model speculative decoding 的根本区别。

更具体地说：

```text
projected_context_states
    -> RMSNorm
    -> fused KV linear projection for all draft layers
    -> K-norm / RoPE
    -> direct write into each draft layer KV cache
```

也就是说，`combine_hidden_states()` 负责的是：

```text
把 target hidden features 压到 draft hidden size
```

而 `precompute_and_store_context_kv()` 负责的是：

```text
把这些 draft-size context states 真正变成 draft 每层可消费的 KV memory
```

#### C. query-only forward

预写完成后，draft model 本轮只跑：

```text
[next_token, mask, mask, ...]
```

所以计算量集中在：

- query token 数量很小
- context 只作为 memory 读取

#### D. verifier 是单线的

无论 draft model 生成 block 的方式多么并行，当前 DFlash 最终仍然只验证：

```text
一条线性 block 路径
```

也就是：

```text
y1 y2 y3 ... yL
```

这是它的最大限制。

这里有一个非常关键的消费关系：

```text
target hidden states
    -> 先被 draft 侧吸收为 context KV
    -> 只在本轮 drafter proposal 阶段消费

draft logits q1...qL
    -> 被采样成一条 block 路径
    -> 交给 target verifier 消费

target verifier
    -> 并不会直接消费那 K 份 raw hidden states
    -> 它消费的是线性 candidate block token ids
```


## 4. DDTree 方案详解

### 4.1 DDTree 的本质

DDTree 的论文名是：

```text
Accelerating Speculative Decoding with Block Diffusion Draft Trees
```

从这个名字就能看出来，它不是替换 DFlash，而是在 DFlash block diffusion draft 的基础上，引入：

```text
draft tree
```

也就是说，DDTree 的核心创新不在 drafter，而在 verifier 侧的候选组织方式。

### 4.2 DDTree 的主链路

```text
target prefill
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
    +--> parent/child structure
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
posterior over the tree
    |
    v
follow longest accepted path
    |
    v
compact target KV cache
```

### 4.3 DDTree 的树是怎么来的

DDTree 的输入仍然是 DFlash 式的 block logits：

```text
q1, q2, q3, ..., qL
```

其中每个 `qi` 是 block 第 i 个位置的 token 分布。

DDTree 不再只取每个位置的 top-1，拼成一条路径，而是：

1. 对每个位置取 top-k 候选；
2. 用 heap / best-first 搜索去扩展高概率 prefix；
3. 在预算 `B` 下保留一棵高概率 draft tree；
4. target verifier 一次验证这棵树。

### 4.4 DDTree 相对 DFlash 的真正提升点

DDTree 要解决的是：

```text
DFlash 只相信单条 top-1 block 路径，
一旦前几个 token 里有一个错了，后面的 block 都作废。
```

DDTree 的思路是：

```text
不要只验证一条线；
同时把多个高概率 continuation 暴露给 target verifier。
```


## 5. DFlash 和 DDTree 的相同点

### 5.1 都属于 speculative decoding

两者都保留了：

- drafter 提 proposal
- target model 做验证
- 接受 / 拒绝由 target model 决定

### 5.2 都是 block-level speculative decoding

两者都不是传统的逐 token speculative decoding，而是：

```text
一次 draft 一个 block
```

### 5.3 都依赖 target hidden states 条件化

两者都不是简单的 token-only 小模型 speculative decoding，而是：

```text
用 target model 的中间表示来增强 draft model
```

### 5.4 都需要 block 内并行建模

也就是：

```text
同一个 block 内不是严格 causal 串行生成
```


## 6. DFlash 和 DDTree 的关键差异

### 6.1 最大差异：单线验证 vs 多线验证

#### DFlash

```text
candidate block:
  y1 y2 y3 ... yL

target verify:
  只验证这一条线
```

#### DDTree

```text
candidate structure:
  一棵 prefix tree

target verify:
  一次验证整棵树
```

### 6.2 DFlash 不需要 tree attention，DDTree 需要

这就是最核心的工程分水岭。

#### DFlash verifier

- 线性序列
- 标准 causal attention 语义
- 可以直接复用现有高性能 attention backend

#### DDTree verifier

- 不同节点共享不同祖先
- 不同分支彼此不可见
- 必须使用不规则 tree attention mask

因此 DDTree verifier 比 DFlash verifier 难很多。

### 6.3 DDTree 需要 KV cache compaction

因为 DDTree verifier 一次性 append 了树中的多条分支，但最后只接受其中一条路径，所以必须：

```text
把 target KV cache 中无效分支压缩掉
```

这在工程上明显比 DFlash 更复杂。

### 6.4 当前 vLLM DFlash 的实现方式和 DDTree 仓库里的 DFlash 实现方式并不完全一样

这是一个容易被忽略但很关键的点。

#### vLLM 版 DFlash

更强调：

```text
projected_context_states
   -> precompute_and_store_context_kv
   -> draft model query-only forward
```

#### DDTree 仓库中的 DFlash

更偏向：

```text
target_hidden 直接进入 attention 中参与 K/V 构造
```

所以：

```text
DDTree 的思想可以吸收，
但不能简单按仓库代码逐文件拷进 vLLM。
```


## 7. tree attention 在 GPU / NPU 上的现状

### 7.1 GPU 侧现状

当前 vLLM 主仓里**确实存在 tree attention 相关代码**：

- `vllm/v1/attention/backends/tree_attn.py`
- `TreeAttentionBackend`
- `TreeAttentionMetadataBuilder`

这说明：

```text
GPU 路径上已经存在局部 tree attention 能力
```

但这并不等价于：

```text
vLLM 已经完整支持 DDTree 式 verifier
```

原因是：

- 现有 tree attention 主要是为了 speculative draft tree 路径服务；
- 还没有现成的“DDTree verifier + cache compaction + branch request orchestration”整套方案；
- 要支持 DDTree，不只是有一个 tree attention backend 就够了。

因此对 GPU 更准确的表述应该是：

```text
GPU 侧存在 tree attention 基础设施，
但没有现成可直接复用的 DDTree 完整 verifier 路线。
```

### 7.2 NPU / vllm-ascend 侧现状

在 `vllm-ascend` 里，我们只看到零散 tree attention 相关痕迹，例如：

- speculative decode 路径里的 TODO：

```text
TODO(wenlong): get more than one token for tree attention
```

但没有看到：

- 完整 tree attention backend
- 完整 DDTree verifier 实现
- 完整 branch cache compaction 路径

因此对 NPU 的结论可以明确写成：

```text
NPU / vllm-ascend 侧目前没有完整成熟的 tree attention verifier 支持。
```

### 7.3 对 CatkinWave 的含义

这意味着：

```text
如果 CatkinWave 想走生产可落地路线，
不能把 tree attention 当成第一阶段必需前提。
```


## 8. 如果要把 DDTree 合入 vLLM，需要怎么改

这一节是重点。

结论先说：

```text
如果按 DDTree 原始思路合入 vLLM，
真正难的不是 candidate tree builder，
而是 verifier 侧的 branch orchestration、tree attention metadata、cache compaction，以及它们和现有 speculative decode 主流程的耦合。
```

### 8.1 需要新增 / 改造的模块

#### A. Candidate tree builder

现有 DFlash 输出的是：

```text
一条 block 路径
```

如果合入 DDTree，则 drafter 阶段还需要额外输出：

- 每个位置的 top-k logits / top-k token ids
- candidate tree 结构
- parent / child / depth 信息

也就是说要新增：

```text
draft logits -> draft tree builder
```

这一层主要会落在：

- `vllm/v1/spec_decode/*`
- proposer 内部的 draft result 组织逻辑
- candidate budget / top-k / parent-child 编码逻辑

#### B. Verifier 输入组织层

现在 vLLM speculative verify 更偏向：

```text
一个请求 -> 一条候选线
```

DDTree 合入后，需要支持：

```text
一个请求 -> 一棵候选树
```

因此需要一个新的 verifier request orchestration 层来组织：

- verify_input_ids
- verify_position_ids
- branch-local attention semantics
- 验证结果回填

这层本质上需要把当前更偏“线性 speculative token append”的输入组织方式，扩展成：

```text
一个 request
    -> 多个 branch node
    -> 每个 node 有 parent
    -> 每个 node 有独立可见祖先集合
```

这和现在 `query_start_loc`、`slot_mapping`、`block_table_tensor` 主要面向线性 query append 的假设并不完全一致。

#### C. Tree attention metadata

虽然 vLLM 已有 `TreeAttentionBackend`，但 DDTree verifier 真正需要的是：

- verifier tree visibility mask
- verifier branch-local slot mapping
- verifier 场景下的 block table / query_start_loc 组织

也就是说：

```text
现有 tree attention 基础设施可以复用一部分，
但 verifier metadata builder 仍需要单独设计。
```

更准确地说，当前本地 vLLM 里的 tree attention 代码，更多体现为：

- tree bias / mask 的构造能力
- drafting 场景下的 tree metadata builder

但 DDTree 需要的是：

- verifier tree 的 token flatten 规则
- verifier tree 的 node-to-parent 映射
- verifier 结束后 accepted path 的映射与裁剪

这并不是把现有 backend 直接接上就能完成的。

#### D. KV cache compaction

这是 DDTree 合入 vLLM 时最难的一块之一。

因为 vLLM 当前 cache 管理天然偏向：

```text
接受一条线性 continuation
```

而 DDTree verifier 会在一次验证里 append 多个分支节点，最后只保留一条路径。

所以需要新增：

- 分支节点临时 slot 管理
- 未接受节点的清理 / 回收
- target KV cache 的路径压缩

这里的难点在于，vLLM 当前 speculative decoding 的很多缓存复用逻辑，默认接受结果仍然是：

```text
从原请求尾部继续向前延长的一条线
```

而 DDTree verifier 的结果则是：

```text
一次写入很多 branch node
最后只保留其中一条 root-to-leaf path
```

因此需要有一个显式的：

```text
branch append -> accepted path select -> cache compact
```

收束过程。

#### E. Scheduler / request state

当前 vLLM scheduler 里的请求状态通常还是“一个请求一条生成线”。

DDTree 真要原样合入，request state 也要临时支持：

```text
同一请求的一次 verify 中存在多个 draft branches
```

这会增大调度复杂度。

如果再进一步拆，DDTree 若原样合入 vLLM，至少会同时影响下面几层：

```text
draft proposal layer
    -> 输出 tree candidate

verification input layer
    -> 组织 flattened tree tokens

attention metadata layer
    -> 组织 verifier tree visibility

KV lifecycle layer
    -> branch write / compact / recycle

request state layer
    -> branch-aware speculative state
```

这也是为什么我把它判断为“大改动量”。

### 8.2 改动量判断

如果按原生 DDTree 方式合入，我会把改动量判断为：

```text
大
```

因为它会同时动到：

- spec decode proposer 输出
- verifier request 组织
- attention metadata
- KV cache 管理
- scheduler / request state

这已经不只是“加一个新 speculative method”的级别了。


## 9. 两种方案的优劣和适用场景

### 9.1 DFlash

#### 优点

- verifier 简单
- 更容易复用现有高性能 causal attention backend
- 更容易落地到 GPU / NPU / vLLM 现有推理架构
- drafter 设计优雅，context KV 预写效率高
- 工程复杂度相对可控

#### 缺点

- 只验证一条线
- 一旦前几个 token 出错，整个 block 的后续候选浪费
- 容易出现 early rejection

#### 适合场景

- 想快速落地 speculative decoding
- 后端希望保持稳定
- 要求尽量少改现有推理框架
- GPU / NPU 都要考虑

### 9.2 DDTree

#### 优点

- 真正解决“单线验证”的问题
- 能同时暴露多条高概率 continuation
- acceptance length 理论上更优
- 更适合高不确定性、分支多的生成任务

#### 缺点

- 依赖 tree attention
- verifier 组织复杂
- 需要 cache compaction
- 更难落地到 NPU
- 即使 GPU 上有 tree attention 基础设施，也不代表 DDTree 可以低成本直接接入

#### 适合场景

- 主要面向 GPU
- 可以接受更高实现复杂度
- 更看重 verifier 质量极限而不是工程简洁性
- 有能力做 deeper backend integration


## 10. CatkinWave 从这两者应该继承什么

如果从工程和演进角度看，CatkinWave 最应该继承的是：

### 从 DFlash 继承

- block diffusion drafter
- target hidden state conditioning
- projected context states
- context KV 预写
- query-only draft forward

### 从 DDTree 继承

- 不要只验证单条路径
- 预算应该优先投给早期高风险位置
- 多候选并行验证比单路径更有潜力
- delayed branching / hazard-aware branching / small beam 这些 candidate builder 思想非常值得保留


## 11. CatkinWave 最推荐的方向

如果只从这份文档角度收一个结论，我的判断是：

```text
CatkinWave 不应该直接走原始 DDTree tree attention 路线，
而应该保留 DFlash drafter，
并把 DDTree 的“多线并行验证”思想
改写成标准 causal verifier 友好的 shared-prefix batch verification。
```

这样可以同时达到：

- 解决 DFlash 单线验证问题
- 避开 DDTree 对 tree attention 的强依赖
- 更容易复用现有 FA2 / FlashInfer / Triton / NPU 标准 verifier 路径

这条路线会在 `catkinwave_design.md` 中展开。
