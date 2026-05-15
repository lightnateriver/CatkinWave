# DFlash in vLLM: 设计与数据链路说明

## 1. 结论先行

在 vLLM 里，DFlash 不是一个新的 attention backend，而是一种接入到 `speculative decoding` 框架中的 proposer 方案。配置入口是：

```python
speculative_config = {
    "method": "dflash",
    ...
}
```

它的核心技术路线是：

1. target model 先正常前向，并额外导出若干个指定层位点的 hidden state。
2. 这些 hidden state 不是直接送进 draft model 做普通自回归，而是先拼接、再投影到 draft hidden size。
3. 投影后的结果不走 draft model 的普通 token-by-token 前缀计算，而是被直接当作 draft model 每一层的 context K/V 来源。
4. draft model 本轮 forward 只处理一个很小的 query block：`[next_token, mask, mask, ...]`。
5. 由于 query block 是并行建模的，所以 attention metadata 必须支持 `causal=False`。

一句话概括：

```text
target model 提供“上下文表征”
    -> DFlash 把它预写成 draft model 的 context KV cache
    -> draft model 只对 query block 做一次并行前向
    -> 输出多个 speculative token 候选
```


## 2. 关键代码位置

下面这些文件基本覆盖了 DFlash 的主体实现：

- `vllm/config/speculative.py`
  - 识别 `method == "dflash"`，并打开 `parallel_drafting`
- `vllm/transformers_utils/configs/eagle.py`
  - 把 draft model 的 architecture 改写成 `DFlash...`
- `vllm/v1/worker/gpu_model_runner.py`
  - 运行时装配 target model、aux hidden states、drafter 与 verifier
- `vllm/v1/spec_decode/eagle.py`
  - 复用 spec decode 主框架，并在 DFlash 路径上做专门分支
- `vllm/v1/spec_decode/dflash.py`
  - DFlash proposer 主体
- `vllm/v1/spec_decode/utils.py`
  - `copy_and_expand_dflash_inputs_kernel`
- `vllm/model_executor/models/qwen3_dflash.py`
  - DFlash draft model 的实现，尤其是 context KV 预计算与预写 cache
- `vllm/model_executor/models/qwen3_5.py`
  - target model 产出 aux hidden states 的方式
- `vllm/model_executor/models/interfaces.py`
  - aux hidden state 记录的通用 mixin


## 3. 名词和符号

为了讲清楚数据流，先定义几个符号：

```text
B   = batch size，请求数
N   = target model 的 decoder layer 总数
K   = 选出来的 target hidden state 个数
S_i = 第 i 个请求的有效 context token 数
S   = sum_i S_i，整个 batch 的 context token 总数
M   = num_speculative_tokens
Q   = B * (1 + M)，本轮 draft model 实际 forward 的 query token 总数
H_t = target model hidden size
H_d = draft model hidden size
L   = draft model 的 attention layer 数
```

几个最重要的张量形状：

```text
aux_hidden_states[j]     : [S, H_t]       ，第 j 个 target hidden state
concat_hidden_states     : [S, K * H_t]
projected_context_states : [S, H_d]
context_positions        : [S]
context_slot_mapping     : [S]
query_input_ids          : [Q]
query_positions          : [Q]
query_slot_mapping       : [Q]
draft_hidden_states      : [Q, H_d]
draft_logits             : [Q, V_d] 或 [Q, V_t]
```


## 4. 总体数据流

### 4.1 高层流程图

```text
+--------------------+
| target model       |
| 正常前向           |
+---------+----------+
          |
          | 1) 输出最终 hidden_states
          | 2) 输出 aux_hidden_states(list)
          v
+--------------------+
| gpu_model_runner   |
| 选择 DFlash 路径   |
+---------+----------+
          |
          | 按最后一维拼接 K 个 aux hidden states
          v
+-------------------------------+
| DFlash draft model            |
| combine_hidden_states (fc)    |
| [S, K*H_t] -> [S, H_d]        |
+---------------+---------------+
                |
                | 作为 context states
                v
+-----------------------------------------------+
| DFlashProposer                                |
| 1. 构造 query block: [next_token, mask... ]   |
| 2. 分离 context/query 的 positions/slots      |
| 3. 设置 non-causal attention metadata         |
+----------------+------------------------------+
                 |
                 | projected_context_states
                 v
+------------------------------------------------------+
| DFlash draft model.precompute_and_store_context_kv   |
| 把 context states 直接转换成每层 KV 并写入 KV cache    |
+----------------+-------------------------------------+
                 |
                 | query_input_ids/query_positions
                 v
+------------------------------------------------------+
| DFlash draft model.forward(query block only)         |
| 每层 attention 读取:                                 |
| - 已预写入 cache 的 context KV                       |
| - 当前 query token 自己算出来的 query KV             |
+----------------+-------------------------------------+
                 |
                 v
+--------------------+
| draft logits       |
| speculative tokens |
+--------------------+
```

### 4.2 最关键的一句话

DFlash 的本质不是“让 draft model 重新走完整个上下文”，而是：

```text
把 target model 产生的上下文表征，直接改造成 draft model 每层的上下文 KV cache；
draft model 本轮只负责 query block。
```


## 5. target model 侧：hidden state 从哪里来

### 5.1 先说层编号语义

这一步很容易搞混。vLLM 里 aux hidden state 的层编号，不完全等于“第几层模块对象”。

在 `Qwen3NextModel.forward()` 里，记录点有两个位置：

```text
layer_id = 0
    表示 embedding 输出、进入第 0 层 decoder 之前

layer_id = i (1 <= i <= N)
    表示第 i-1 个 decoder layer 执行完成后的输出边界
```

也就是说：

```text
0  -> embedding output
1  -> layer 0 output
2  -> layer 1 output
...
N  -> layer N-1 output
```

这是由 target model 的 forward 逻辑决定的：

```text
embedding
  -> _maybe_add_hidden_state(..., layer_idx=0)
for each decoder layer:
  run layer
  -> _maybe_add_hidden_state(..., layer_idx + 1)
```

而真正被保存进去的值也不是裸 `hidden_states`，而是：

```text
value = hidden_states + residual   (如果 residual 存在)
value = hidden_states              (如果 residual 不存在)
```

所以 aux hidden state 更接近一个“层边界的残差融合表示”。

### 5.2 默认取多少个 hidden state

取多少个 hidden state，优先级如下：

```text
优先 1: dflash_config.target_layer_ids
优先 2: eagle_aux_hidden_state_layer_ids
优先 3: target model 的默认配置
```

对 Qwen3.5 系列，默认是：

```python
(2, num_layers // 2, num_layers - 3)
```

也就是默认取 3 个 hidden state，因此：

```text
K = 3
```

如果显式配置了：

```python
dflash_config = {
    "target_layer_ids": [...],
}
```

那么：

```text
K = len(target_layer_ids)
```

### 5.3 target model 输出的数据结构

如果启用了 DFlash，`gpu_model_runner` 会要求 target model 额外返回 aux hidden states。于是 target model 输出不再只是：

```text
hidden_states
```

而会变成：

```text
(hidden_states, aux_hidden_states)
```

其中：

```text
hidden_states      : [S, H_t] 或本轮对应 token 数的同型张量
aux_hidden_states  : list[Tensor]
aux_hidden_states[j].shape = [S, H_t]
len(aux_hidden_states) = K
```

### 5.4 target model 侧字符流程图

```text
input_ids / inputs_embeds
        |
        v
  embedding output -----------------------------+
        |                                       |
        | if 0 in aux_hidden_state_layers       |
        +------> aux_hidden_states[?] = [S,H_t] |
        |                                       |
        v                                       |
  decoder layer 0                               |
        |                                       |
        | if 1 in aux_hidden_state_layers       |
        +------> aux_hidden_states[?] = [S,H_t] |
        |                                       |
        v                                       |
  decoder layer 1                               |
        |                                       |
        | if 2 in aux_hidden_state_layers       |
        +------> aux_hidden_states[?] = [S,H_t] |
        |                                       |
       ...                                     ...
        |                                       |
        v                                       |
  decoder layer N-1                             |
        |                                       |
        | if N in aux_hidden_state_layers       |
        +------> aux_hidden_states[?] = [S,H_t] |
        |                                       |
        v                                       |
  final hidden_states = [S,H_t] <---------------+
```


## 6. target hidden state 如何变成 DFlash context states

### 6.1 拼接

在 `gpu_model_runner` 中，如果 `use_aux_hidden_state_outputs=True`，会先把多个 aux hidden states 沿最后一维拼接：

```text
aux_hidden_states[0] : [S, H_t]
aux_hidden_states[1] : [S, H_t]
...
aux_hidden_states[K-1] : [S, H_t]

torch.cat(..., dim=-1)
    ->
concat_hidden_states : [S, K * H_t]
```

### 6.2 投影到 draft hidden size

随后进入 `DFlashQwen3ForCausalLM.combine_hidden_states()`。

如果 `use_aux_hidden_state=True`，会通过一个全连接层：

```text
fc: [K * H_t] -> [H_d]
```

于是：

```text
[S, K * H_t] -> [S, H_d]
```

这个结果就是后续 DFlash 使用的 context states。

这一步非常关键，因为后面 draft model 的 context KV 预计算，需要的输入维度必须匹配 draft model 自己的 hidden size，而不是 target model 的 hidden size。

### 6.3 一个容易误解的点

代码里这个变量后面仍然叫 `target_hidden_states`，但从语义上说：

```text
它在进入 DFlashProposer 之前，通常已经不再是 target hidden size 空间里的向量了；
它已经被投影成 draft hidden size 空间里的 context states。
```

为了避免混淆，本文后面统一称它为：

```text
projected_context_states
```

### 6.4 这一段的字符流程图

```text
aux_hidden_states (K 个)

  h0: [S, H_t]
  h1: [S, H_t]
  ...
  hK-1: [S, H_t]

        cat on dim=-1
               |
               v
     concat_hidden_states
         [S, K * H_t]
               |
               | combine_hidden_states()
               | fc: K*H_t -> H_d
               v
    projected_context_states
             [S, H_d]
```


## 7. DFlashProposer：如何构造 query block

### 7.1 DFlash 不把整段 context 当成本轮 forward 输入

在 DFlash 里：

- context token 不进入 draft model 本轮 `input_ids`
- context 只用于生成每层的 K/V cache
- 本轮真正进入 draft model 的只有 query block

### 7.2 query block 长什么样

每个请求的 query block 形如：

```text
[next_token, mask, mask, mask, ...]
```

长度为：

```text
1 + M
```

其中：

- `next_token` 是 target model 刚采出来的那个 token，作为 query block 的条件起点
- 后面的 `mask` 是并行 speculative 位置

因此整个 batch 的 query token 总数是：

```text
Q = B * (1 + M)
```

### 7.3 为什么 DFlash 不需要 `mask_hidden`

EAGLE 某些并行路径会使用 `mask_hidden`。而 DFlash 不这么做，它要求 draft model 配置中提供：

```text
dflash_config.mask_token_id
```

然后用这个真实 token 的 embedding 作为 masked query 的输入起点。

所以 DFlash 是：

```text
mask token embedding
```

不是：

```text
learned mask hidden vector
```

### 7.4 query / context 的 buffer 分离

`DFlashProposer` 会显式维护两套 buffer：

- context
  - `context_positions_buffer`
  - `context_slot_mapping_buffer`
- query
  - `input_ids`
  - `positions`
  - `slot_mapping_buffer`

原因是：

```text
context KV 是预写 cache 的；
query token 才是本轮真正执行 model.forward 的 token。
```

### 7.5 构造输入时 kernel 做了什么

`copy_and_expand_dflash_inputs_kernel` 会一次完成以下事情：

1. 拷贝 context positions
2. 计算 query positions
3. 写 query input ids：`[next_token, mask, mask, ...]`
4. 分别计算 context 和 query 的 slot mapping
5. 生成后续用于 sample 的 `token_indices_to_sample`

### 7.6 这一步的字符流程图

```text
per request:

context token ids / positions
        +
next_token
        +
mask_token_id x M

        |
        v
+-------------------------------------------+
| copy_and_expand_dflash_inputs_kernel      |
+-------------------+-----------------------+
                    |
                    +--> context_positions     : [S_i]
                    +--> context_slot_mapping  : [S_i]
                    +--> query_input_ids       : [1+M]
                    +--> query_positions       : [1+M]
                    +--> query_slot_mapping    : [1+M]
                    +--> token_indices_sample  : [M]
```


## 8. 为什么必须是 non-causal attention

DFlash 在构造新的 `CommonAttentionMetadata` 时，直接设置：

```text
causal = False
```

原因不是简单的“为了快”，而是因为 query block 不是普通自回归串行结构。

每个请求的 query block 包含：

```text
[next_token, mask_1, mask_2, ..., mask_M]
```

这些 query token 在本轮里是一起送进 draft model 的，而不是一个一个递推出来的。因此 backend 必须支持：

```text
在同一个 query block 内做 non-causal attention
```

同时又要能读取已经预写进去的 context KV。

换句话说，DFlash 需要 backend 支持的是：

```text
context: 来自预写 cache
query  : 当前批次新算出来
mask   : non-causal
```

这也是为什么 `vllm-ascend` 如果没有完整 non-causal 这条链路，就很难“原样支持” DFlash。


## 9. draft model：context states 如何变成每层 KV cache

这一段是 DFlash 技术实现里最核心的优化点。

### 9.1 目标

输入：

```text
projected_context_states : [S, H_d]
context_positions        : [S]
context_slot_mapping     : [S]
```

输出目标：

```text
对 draft model 的每一层 attention：
  生成该层的 K_ctx / V_ctx
  并直接写到该层 KV cache 的正确 slot 上
```

### 9.2 不是逐层普通 forward，而是“批量预计算 KV”

`precompute_and_store_context_kv()` 不会把 context states 当作普通 token 送入每层 forward。它做的是：

1. 收集并融合所有层的 KV projection 权重
2. 一次大 GEMM 算出所有层的 K/V
3. 对 K 做 RMSNorm 和 RoPE
4. 逐层把结果写入 cache

### 9.3 预构建 fused 权重

在加载完权重后，DFlash 会做一次 `_build_fused_kv_buffers()`，把所有层的关键参数提取出来：

- 所有层的 KV projection weight，拼成一个大矩阵
- 所有层的 K norm weight
- 统一的 RoPE 参数
- 每层内部 `Attention` 对象的引用

这一步的目的是减少 runtime 时的 Python 层循环和小算子数量。

### 9.4 runtime 预写 cache 的详细步骤

给定：

```text
projected_context_states : [S, H_d]
```

执行步骤如下：

1. 对 `projected_context_states` 做 hidden norm

```text
[S, H_d] -> normed_context_states : [S, H_d]
```

2. 用融合后的大矩阵做一次线性投影，得到所有层的 K/V

```text
normed_context_states
    x fused_kv_weight
    ->
all_kv_flat
```

3. reshape / permute 成 layer-major 布局

```text
all_kv
shape = [2, L, S, num_kv_heads, head_dim]

all_k = all_kv[0]
all_v = all_kv[1]
```

4. 对每一层的 K 做 RMSNorm

```text
all_k[i] -> all_k_normed[i]
```

5. 对所有层的 K 统一做 RoPE

```text
all_k_normed -> all_k_final
```

6. 逐层调用 `do_kv_cache_update()`，按 `context_slot_mapping` 直接写入 cache

```text
for each layer i:
    kv_cache[layer_i][context_slot_mapping] = (K_i, V_i)
```

### 9.5 这一段的字符流程图

```text
projected_context_states [S, H_d]
            |
            | hidden_norm
            v
  normed_context_states [S, H_d]
            |
            | one fused GEMM over all layers
            v
  all_kv_flat
            |
            | reshape/permute
            v
  all_kv [2, L, S, num_kv_heads, head_dim]
      |                     |
      |                     +--> all_v [L, S, nkv, hd]
      v
  all_k [L, S, nkv, hd]
      |
      | per-layer RMSNorm on K
      v
  all_k_normed [L, S, nkv, hd]
      |
      | fused RoPE
      v
  all_k_final [L, S, nkv, hd]
      |
      | for layer = 0..L-1
      v
  do_kv_cache_update(layer_i, K_i, V_i, context_slot_mapping)
      |
      v
  draft layer i 的 context KV cache 已就绪
```


## 10. draft model.forward：query token 如何消费这些 KV

### 10.1 此时 context 已经不再是 token 序列输入

当 `precompute_and_store_context_kv()` 完成后，draft model 每层的 attention 已经有了 context 部分的 KV。

所以接下来送入 `DFlashQwen3Model.forward()` 的不是完整前缀，而只是：

```text
query_input_ids : [Q]
query_positions : [Q]
```

也就是：

```text
[next_token, mask, mask, ...] x B
```

### 10.2 每层 attention 实际做了什么

`DFlashQwen3Attention.forward()` 对 query hidden states 做正常的：

- qkv projection
- q norm / k norm
- RoPE
- attention
- o_proj

但语义上与普通 causal decoder 不同：

1. 本轮 `q/k/v` 只针对 query block 计算
2. attention 读取的上下文部分，来自已经存在于 KV cache 里的 context KV
3. 当前 query token 的 k/v 也会写入自己的 query slot
4. attention metadata 是 `causal=False`，因此 query block 内部可以并行交互

可以把它理解成：

```text
context memory 已经提前灌进 cache
query block 本轮只负责“在这块 memory 上做一次并行查询”
```

### 10.3 这一段的字符流程图

```text
query_input_ids [Q]
      |
      | embedding
      v
query_embeds [Q, H_d]
      |
      v
for each draft layer i:

  query_hidden_states [Q, H_d]
        |
        | qkv_proj / norm / rope
        v
   q_i, k_i, v_i   (仅 query block)
        |
        | attention backend 读取:
        |   - 预写入 cache 的 context KV
        |   - 当前 query 自己的 KV
        |   - metadata.causal = False
        v
   query_output_i [Q, H_d]

最终:
draft_hidden_states [Q, H_d]
```


## 11. logits、采样与 verifier

### 11.1 logits 头

draft model 最终输出：

```text
draft_hidden_states : [Q, H_d]
```

随后通过 `lm_head + logits_processor` 得到：

```text
draft_logits : [Q, V_d]
```

如果 draft vocab 与 target vocab 不完全一致，还会通过 `draft_id_to_target_id` 做一次映射，扩展到 target vocab 空间。

### 11.2 sample 哪些位置

DFlash 并不是对 `Q = B*(1+M)` 个 query 位置都当作新 speculative token。

真正需要 sample 的是：

```text
mask 对应的那 M 个位置
```

也就是每个请求里的：

```text
[mask_1, mask_2, ..., mask_M]
```

`next_token` 本身是 target model 已经给出的条件 token，不是本轮 speculative 预测目标。

### 11.3 verifier 不变

DFlash 改的是 proposer 这一侧，不是 verifier 这一侧。

后续仍然是：

```text
draft model 提 proposals
    ->
target model 按 spec decode 规则验证
    ->
accept / reject
```

因此从 vLLM 整体架构上看，DFlash 是：

```text
“替换 proposal 生成方式”
而不是
“替换整个 speculative decoding 框架”
```


## 12. 一张完整的数据链路总图

```text
Stage A: target model 产出上下文表征
------------------------------------

input_ids
   |
   v
target embedding
   |
   +--> optional aux hidden state at layer_id=0
   |
   v
target layer 0
   |
   +--> optional aux hidden state at layer_id=1
   |
   v
target layer 1
   |
   +--> optional aux hidden state at layer_id=2
   |
  ...
   |
   v
target layer N-1
   |
   +--> optional aux hidden state at layer_id=N
   |
   v
final hidden_states


Stage B: 选层并拼接
-------------------

selected aux_hidden_states = [h0, h1, ..., hK-1]
each hj shape = [S, H_t]

h0 || h1 || ... || hK-1
        |
        v
concat_hidden_states [S, K*H_t]
        |
        | fc projection
        v
projected_context_states [S, H_d]


Stage C: 构造 DFlash query block
--------------------------------

per request:
    context prefix
    +
    next_token
    +
    mask x M

kernel outputs:
    context_positions     [S]
    context_slot_mapping  [S]
    query_input_ids       [Q]
    query_positions       [Q]
    query_slot_mapping    [Q]
    token_indices_sample  [B*M]

metadata:
    causal = False


Stage D: 预写 draft context KV
------------------------------

projected_context_states [S,H_d]
        |
        | hidden_norm
        | fused KV GEMM over L layers
        | K RMSNorm
        | fused RoPE
        v
for each draft layer i:
    K_ctx_i [S,nkv,hd]
    V_ctx_i [S,nkv,hd]
    -> write to KV cache at context_slot_mapping


Stage E: draft 只跑 query block
-------------------------------

query_input_ids [Q]
    -> embed
    -> draft layer 0
    -> draft layer 1
    -> ...
    -> draft layer L-1
    -> hidden_states [Q,H_d]
    -> logits [Q,V_d or V_t]
    -> sample mask positions only


Stage F: verifier
-----------------

draft proposals
    -> target model verification
    -> accept / reject
    -> next iteration
```


## 13. DFlash 与普通 draft model / EAGLE3 的差异

### 13.1 对比普通 draft model speculative decoding

普通 draft model 路径更像：

```text
小模型重新按 token 顺序跑前缀，再往后猜几个 token
```

DFlash 则是：

```text
target model 提供上下文表征
-> 直接转成 draft model 的 context KV
-> draft model 只跑 query block
```

所以 DFlash 更像“memory-conditioned block proposal”，而不是“小模型复读完整前缀”。

### 13.2 对比 EAGLE3

两者相同点：

- 都可以使用 target model 的 aux hidden states
- 都可能先做 `combine_hidden_states`

两者不同点：

- EAGLE3 主要还是围绕 hidden-state-conditioned drafting 展开
- DFlash 更进一步，把这些状态直接转换成 draft model 各层 context KV
- DFlash 显式要求 `non-causal` attention metadata
- DFlash query block 里使用真实 `mask_token_id` embedding，而不是 `mask_hidden`


## 14. 为什么这套方案对后端要求高

如果要在另一个后端完整实现 DFlash，至少要补齐以下能力：

1. target model 支持导出 aux hidden states
2. draft model 支持 `combine_hidden_states`
3. draft model 支持“context states -> per-layer KV cache”的预计算与预写
4. attention backend 支持 `causal=False`
5. backend 支持按 slot mapping 直接更新 KV cache
6. spec decode 主循环能把 context/query 分离成两条数据路径

缺少其中任意一项，DFlash 都很难只靠“改一点点 proposer 逻辑”就跑起来。


## 15. 实现要点总结

最后用最短的话收一下：

```text
target model:
    产出多个层位点的 aux hidden states

runner:
    按最后一维拼接这些 hidden states

draft model:
    用 fc 把拼接后的 target states 投影到 draft hidden size

DFlash proposer:
    把投影结果当作 context states
    把 [next_token, mask...] 当作 query block
    设置 non-causal metadata

precompute_and_store_context_kv:
    把 context states 直接变成每层 KV，并写入 KV cache

draft forward:
    只跑 query block
    读取预写 context KV
    并行产生 speculative proposals

verifier:
    仍然走 vLLM 原有 accept/reject 逻辑
```

这就是 DFlash 在 vLLM 中的完整技术方案和数据链路。
