# 调试日志说明

## 添加的日志位置

### 1. EngineCore 初始化流程 (vllm/v1/engine/core.py)

#### `_initialize_kv_caches` 函数
- 开始 KV 缓存初始化
- 获取 KV cache specs 数量
- 开始内存 profiling
- 内存 profiling 完成时间和可用显存大小
- 开始 KV 缓存初始化和模型 warmup
- KV 缓存初始化和 warmup 完成时间

#### `EngineCoreProc.__init__` 函数
- 开始 EngineCoreProc 初始化
- 开始 handshakes
- Handshakes 完成时间
- 调用 `super().__init__()`（初始化模型和 KV cache）
- `super().__init__()` 完成时间
- 开始输入/输出线程

#### `_perform_handshake` 函数
- 开始 handshake
- 调用 startup_handshake
- Handshake 完成，yield addresses
- 准备 READY 消息
- 发送 READY 消息
- READY 消息发送完成时间

### 2. FlashInfer GDN 内核 (vllm/model_executor/layers/mamba/gdn_linear_attn.py)

#### `fi_chunk_gated_delta_rule` 函数
- 开始 FlashInfer GDN prefill 内核
- 导入 flashinfer.gdn_prefill 成功
- 导入失败错误日志
- 运行 l2norm_fwd
- 准备 tensors (squeeze/contiguous)
- 调用 chunk_gated_delta_rule_fi (JIT compiled kernel)
- 内核执行完成时间
- 返回 output with/without final_state
- 总执行时间

#### `_warmup_prefill_kernels` 函数
- 开始 GDN prefill kernel warmup
- Kernels already warmed up（如果已经 warmup 过）
- 创建 dummy tensors
- 运行 fused_gdn_gating
- 创建 state tensor
- 调用 chunk_gated_delta_rule
- Warmup 完成时间（成功）
- Warmup 失败堆栈（带异常信息）

## 日志前缀

所有新增日志都带有方括号前缀，方便搜索：
- `[_initialize_kv_caches]` - KV 缓存初始化
- `[EngineCoreProc.__init__]` - EngineCore 进程初始化
- `[_perform_handshake]` - Handshake 过程
- `[fi_chunk_gated_delta_rule]` - FlashInfer GDN 内核执行
- `[_warmup_prefill_kernels]` - GDN 内核 warmup

## 使用方法

运行 vLLM 时设置 DEBUG 日志级别：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve Qwen/Qwen3.5-0.8B 2>&1 | tee vllm_debug.log
```

或者只查看 INFO 级别（新增日志大多是 INFO 级别）：

```bash
VLLM_LOGGING_LEVEL=INFO vllm serve Qwen/Qwen3.5-0.8B 2>&1 | tee vllm_info.log
```

## 预期日志流程

正常的启动流程应该看到：

1. `[EngineCoreProc.__init__] Starting EngineCoreProc initialization`
2. `[EngineCoreProc.__init__] Starting handshakes`
3. `[_perform_handshake] Starting handshake with front-end`
4. `[_perform_handshake] Handshake completed, yielding addresses`
5. `[EngineCoreProc.__init__] Handshakes completed in X.XX seconds`
6. `[EngineCoreProc.__init__] Calling super().__init__() (this initializes model and KV cache)`
7. `[_initialize_kv_caches] Starting KV cache initialization`
8. `[_initialize_kv_caches] Starting memory profiling (determine_available_memory)...`
9. `[_initialize_kv_caches] Memory profiling completed in X.XX seconds`
10. `[_initialize_kv_caches] Available GPU memory for KV cache: XX.XX GiB`
11. `[_initialize_kv_caches] Starting KV cache initialization and model warmup...`
12. `[_initialize_kv_caches] KV cache initialization and model warmup completed in X.XX seconds`
13. `[EngineCoreProc.__init__] super().__init__() completed in X.XX seconds`
14. `[_perform_handshake] Preparing READY message`
15. `[_perform_handshake] Sending READY message`
16. `[_perform_handshake] READY message sent in X.XX seconds`

如果使用 FlashInfer GDN 后端，还会看到：

17. `[_warmup_prefill_kernels] Starting GDN prefill kernel warmup for layer <layer_name>`
18. `[_warmup_prefill_kernels] Creating dummy tensors for warmup`
19. `[_warmup_prefill_kernels] Running fused_gdn_gating`
20. `[_warmup_prefill_kernels] Creating state tensor`
21. `[_warmup_prefill_kernels] Calling chunk_gated_delta_rule...`
22. `[fi_chunk_gated_delta_rule] Starting FlashInfer GDN prefill kernel`
23. `[fi_chunk_gated_delta_rule] Imported flashinfer.gdn_prefill successfully`
24. ...（更多 FlashInfer 内核执行日志）
25. `[_warmup_prefill_kernels] GDN prefill kernel warmup completed in X.XX seconds`

## 问题定位

如果启动卡住，查看最后一条日志：

- 卡在 `[_initialize_kv_caches] Starting memory profiling...` → 内存 profiling hang 住
- 卡在 `[_initialize_kv_caches] Starting KV cache initialization and model warmup...` → KV 缓存初始化或模型 warmup hang 住
- 卡在 `[_warmup_prefill_kernels] Calling chunk_gated_delta_rule...` → FlashInfer GDN 内核执行 hang 住
- 卡在 `[fi_chunk_gated_delta_rule] Calling chunk_gated_delta_rule_fi...` → FlashInfer JIT 内核 hang 住
- 没有看到 `[_perform_handshake] Sending READY message` → EngineCore 在 super().__init__() 之后、发送 READY 之前 hang 住

## 禁用 FlashInfer GDN

如果确认是 FlashInfer GDN 问题，使用 Triton 后端：

```bash
vllm serve Qwen/Qwen3.5-0.8B --gdn-prefill-backend triton
```
