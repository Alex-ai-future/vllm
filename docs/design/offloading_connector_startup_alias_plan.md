# Offloading Connector 启动优化：Rank-local Contiguous Virtual Alias

状态：已实现生产路径；支持的 CUDA 交错布局直接使用 alias，异常直接失败；待真实 TP=8 多 GPU 合入验证

日期：2026-08-06

相关代码：

- `vllm/v1/kv_offload/cpu/shared_offload_region.py`
- `vllm/v1/kv_offload/cpu/gpu_worker.py`
- `vllm/v1/kv_offload/cpu/host_memory.py`
- `benchmarks/benchmark_pin_cuda_host_memory.py`

## 1. 结论

在推理阶段的内存布局性能优先的前提下，不改变现有 shared mmap 的
block-major/interleaved 布局。rank-local contiguous virtual alias 只在 Linux +
NVIDIA CUDA 上启用；ROCm 继续使用原有 full registration 路径。

这个 alias 不复制 KV 数据，也不改变实际推理使用的 CPU tensor 的地址、shape
或 stride。它只在 worker 初始化时建立，用于把当前 rank 分散在多个 block row
中的页面映射成一个连续的虚拟地址范围，然后用一次
`cudaHostRegister()` 注册这个范围。

16 GiB、TP=8、单 rank 的 H800 生产路径实测结果（包含
`SharedOffloadRegion` 创建和注册，不包含完整模型启动）：

| 方案 | 总初始化时间 | mmap + population | host registration | 注册字节数 |
| --- | ---: | ---: | ---: | ---: |
| 当前整段注册 | 7.13 s | 0.84 s | 6.29 s | 16 GiB |
| contiguous alias | 1.91 s | 0.85 s | 1.06 s | 2 GiB |

本次实验中 alias 比当前实现少约 5.22 s，总初始化约 3.7 倍加速；
registration 阶段约 6 倍加速，单 rank 的注册字节数减少 8 倍。这个数字是
单 rank、单次测量，不能直接视为多 GPU 生产环境的最终 wall-time 承诺，但已
经证明大内存下该方向具有数秒级绝对收益。

## 2. 当前问题

`SharedOffloadRegion` 使用一个共享的 mmap 文件保存 CPU offload buffer。
当前布局按 block row 组织，每个 row 内包含所有 rank 的 slot：

```text
block 0: rank 0 | rank 1 | ... | rank 7
block 1: rank 0 | rank 1 | ... | rank 7
block 2: rank 0 | rank 1 | ... | rank 7
```

设：

- `B`：整个 shared mmap 的大小；
- `W`：`TP * PCP`，通常等于 rank 数；
- `S`：一个 rank 在每个 block row 中的 slot 大小；
- `N`：block 数；
- `row_stride = W * S`；
- `B = N * row_stride`。

当前 `SharedOffloadRegion.pin_cuda_host_memory()` 注册整个 mmap：

```python
cudaHostRegister(base_ptr, region.total_size_bytes, 0)
```

因此每个 rank 都向 CUDA driver 注册 `B` 字节，即使该 rank 实际只访问约
`B / W` 字节。TP=8、16 GiB shared mmap 时，单 rank 注册 16 GiB；所有
rank 的注册请求还会在不同 CUDA context 中重复发生。

`MADV_POPULATE_WRITE` 仍然需要 population 当前 rank 的页面，但它在 16 GiB
测试中约占 0.9 s，远小于当前 5.4 s 的 full-range host registration。

## 3. Virtual alias 原理

### 3.1 Alias 的虚拟布局

为 rank 0 建立一段新的连续虚拟地址空间：

```text
alias view:
rank 0 block 0 | rank 0 block 1 | rank 0 block 2 | ...
```

alias 的第 `block` 段映射到原 mmap 的：

```text
source_offset = block * row_stride + rank * S
alias_offset  = block * S
```

这是一组 file-backed virtual mappings，不会复制文件内容，也不会额外分配
一份 2 GiB 的 KV 数据。alias 的长度为：

```text
alias_size = N * S = B / W
```

建立 alias 后只执行一次：

```python
cudaHostRegister(alias_base, alias_size, 0)
```

原有的 CPU tensor 仍然从原 mmap 创建，仍然使用：

```text
shape  = (num_blocks, tensor_page_size)
stride = (row_stride, 1)
```

因此不会修改 KV cache 的物理 layout。secondary tier 和 scheduler 继续使用
原始 interleaved view；GPU transfer handler 在 alias 成功时额外使用一个只读写
alias 的 contiguous view，以便 Triton host-pointer kernel 访问已经注册的 VA。

### 3.2 原始 view、alias view 与 DMA 的关系

alias 和原始 view 指向相同的 file-backed pages。CUDA registration 对 alias
完成后，这些底层 host pages 已经被 driver pin/register。当前 H800 实验中，
使用原始 interleaved view 做 `Tensor.copy_()` host↔GPU 和
`ops.swap_blocks_batch()` 双向 DMA 均成功。对于会在 GPU kernel 中直接解引用
host pointer 的 Triton transfer 路径，不能假设“只注册 alias VA”会自动让原始
interleaved VA 对 kernel 可见；生产实现因此在 alias 模式下构造 transfer-only
contiguous view，并把它传给 handler。该 view 与原始 view 共享相同 file-backed
pages，数据内容和 block 语义不变。

16 GiB alias 实验还执行了三次双向 transfer 校验；另外，生产
`CPUOffloadingWorker` 的 shared-memory transfer 全部 32 个 CUDA 参数组合通过。
其中一次生产类直接校验的 rank-local block 为 256 KiB：

| 方案 | 首次双向 transfer | 平均双向 transfer |
| --- | ---: | ---: |
| 当前整段注册 | 55.52 ms | 18.72 ms |
| contiguous alias | 44.72 ms | 15.19 ms |

生产类直接调用 alias transfer view 的 H2D/D2H 校验分别为 0.14 ms / 0.05 ms，
对应的原始 interleaved view 数据同时校验一致。

这是单 block 的 correctness/smoke 测量，不等价于完整模型的吞吐 benchmark。

## 4. 计划修改

### 4.1 `SharedOffloadRegion`

底层 alias 已抽到
`vllm/v1/kv_offload/cpu/host_memory.py` 的无状态 helper，生产代码和 benchmark
共用这一实现。alias 的指针、Tensor、buffer、view 和 cursor 由
`SharedOffloadRegion` 持有；helper 只负责预留/填充 alias、创建 transfer view
以及释放映射。CUDA registration 状态仍由 `SharedOffloadRegion` 管理：

```python
alias_base, alias_buffer, alias_tensor = create_rank_local_alias(
    fd, num_blocks, row_stride, slot_size, rank
)
cudart.cudaHostRegister(alias_base, num_blocks * slot_size, 0)
```

建立流程：

1. 检查当前平台是否是支持该路径的 Linux CUDA 环境；
2. 检查 `rank is not None`、`num_blocks > 0` 和所有 offset/size 页对齐；
3. 预留 `alias_size` 连续虚拟地址空间；
4. 把每个 block 的 rank slot 映射到 alias 中对应的连续位置；
5. 对整个 alias 调用一次 `cudaHostRegister()`；
6. 保存 alias 映射和 registration 状态，直到所有 transfer 完成；
7. 先调用 `cudaHostUnregister()`，再解除 alias mappings。

推荐使用“先预留、再按地址填充”的方式建立 alias。生产实现不能直接无保护地
使用 `MAP_FIXED` 覆盖未知地址；必须确保固定映射只发生在本次预留的范围内，
并且任意中途失败都能解除已经建立的 mappings。

### 4.2 `gpu_worker.py`

worker 只调用：

```python
mmap_region.pin_cuda_host_memory()
```

平台判断、CUDA/ROCm 分流和 layout path selection 都由
`SharedOffloadRegion` 负责。worker 只通过 `create_next_transfer_view()` 获取
CPU transfer tensor；原始 mmap 的 layout 和 scheduler-side memoryview 不变。

### 4.3 Layout path selection

在 Linux + NVIDIA CUDA 路径中，支持 rank-local alias 的交错布局直接使用 alias。
以下是确定性的 full-range registration 场景：

- layout 不是当前已验证的 rank-local interleaved layout；
- replicated layout、TP=1 或其他没有 rank 间交错的特殊 layout。

非 Linux、非 CUDA/ROCm 平台不进入这条 registration 路径；ROCm 不尝试 alias，
直接使用 full registration。对于已支持的 alias 布局，如果 `mmap`、alias 建立
或 `cudaHostRegister()` 失败，错误直接向上抛出，不静默回退到 full registration。
host-memory helper 会清理已经建立的部分 mapping，避免留下半初始化状态。

### 4.4 不变的部分

以下接口和推理行为不应改变：

- `CanonicalKVCaches`；
- `CanonicalKVCacheTensor`；
- CPU view 的 shape、stride、storage offset；
- shared mmap 的 block-major/interleaved 物理布局；
- scheduler-side `create_kv_memoryview()`；
- secondary tier 对整个 shared region 的访问方式；
- GPU KV cache 的 page/block 编号。

## 5. 性能预期

对于均匀的 `W=8` 布局，可使用下面的近似：

```text
当前注册字节数/worker ≈ B
alias 注册字节数/worker ≈ B / 8
```

基于当前 H800 的 1 GiB 和 16 GiB 测量：

- full registration 约随 `B` 线性增长；
- alias registration 约随 `B / W` 增长；
- `madvise` 仍随当前 rank 实际 population 的页面数增长；
- 大于 16 GiB 时不能假设继续严格线性，driver、pinned-memory limit、
  NUMA 和内存带宽都可能造成拐点。

如果总 engine startup 是几十秒，该优化对总启动时间的比例可能只有几个百分点；
如果瓶颈集中在 offload region 初始化，16 GiB 场景可直接节省数秒。
它不应被描述为 steady-state inference throughput 优化。

## 6. 验证计划

### 6.0 已完成验证

- 16 GiB H800 生产类 full registration 与 rank-local alias 对照；
- 生产类原始 view 的 H2D/D2H 数据校验；
- `test_shared_offload_region.py` + `test_gpu_worker.py`：62 passed, 1 skipped；
- alias 注册失败后的资源清理与错误传播；
- 单文件 benchmark 已收敛为只比较两种 host registration 路径；
- H800 NVL、16 GiB、TP=8 单进程 benchmark：full mmap 均值 5162.69 ms，
  rank-local alias 均值 348.81 ms，节省 4813.89 ms，约 14.80 倍；
- ruff、mypy、SPDX、markdownlint 和 pre-commit hooks。

合并前仍需要在真实一 rank 一 GPU 的 TP=8 拓扑上完成多进程验证。

### 6.1 Standalone benchmark

使用单文件 benchmark，只测 `pin_cuda_host_memory()` 的 host registration 时间，
不加入常规 pytest wall-time assertion：

```bash
.venv/bin/python benchmarks/benchmark_pin_cuda_host_memory.py \
    --num-blocks 8192 \
    --row-stride-mib 2 \
    --world-size 8 \
    --runs 3 \
    --warmup 1
```

脚本需要 CUDA GPU，但不需要 `nvcc`；Python 仍通过 `.venv/bin/python` 运行。
benchmark 使用单进程模拟一个 production worker：默认 shared mmap 为 16 GiB、
TP 交错度为 8，比较该 rank 注册完整 16 GiB 与 rank-local alias 注册 2 GiB。
这里不模拟多进程 contention，因为本优化的收益来自注册地址范围和字节数的减少，
不是内存带宽竞争的变化。
测试规模建议覆盖：256 MiB、1 GiB、4 GiB、8 GiB、16 GiB。只记录：

- full mmap 路径的 `cudaHostRegister` 时间；
- rank-local alias 路径的 `pin_cuda_host_memory()` 时间；
- 两条路径的注册字节数、均值、中位数和加速比。

### 6.2 真实 transfer 路径

需要同时覆盖两种用途：scheduler/secondary tier 继续使用原始 interleaved view；
GPU transfer handler 在 alias 模式使用 contiguous transfer view。不能把原始
interleaved view 当作 direct host-pointer kernel 的 alias 替代品。测试覆盖：

- `ops.swap_blocks_batch()`；
- GPU→CPU store；
- CPU→GPU load；
- 多 block batch；
- dedicated CUDA stream；
- transfer 未完成时 cleanup 的保护逻辑。

### 6.3 多 GPU

必须在真实的一 rank 一 GPU 拓扑上测试 TP=8。当前开发机只有 4 张 H800，
8-rank/4-GPU 测试只能用于观察 registration contention，不能作为最终 inference
性能结论。

多 GPU 测试需要对比：

- worker ready wall time；
- 每个 rank 的 host registration 时间；
- first offload latency；
- steady-state store/load bandwidth；
- P50/P95/P99 transfer latency。

### 6.4 合入门槛

建议满足以下条件后再默认开启：

1. 真实 CUDA GPU 上原始 view 的 store/load 数据正确；
2. `ops.swap_blocks_batch()` 多 block、双向、异步 stream 测试通过；
3. 推理 transfer 的 P50/P95/P99 没有统计显著回退；
4. 16 GiB 场景 registration 成功率不低于当前实现；
5. 已支持 alias 布局的异常能够明确暴露，且失败清理不留下 mapping；
6. 非 CUDA、Tiering、replicated layout 和 `num_blocks == 0` 行为不回归；
7. 现有 CPU offload 测试、ruff、mypy 和 pre-commit 通过。

不建议把绝对 wall-time 阈值放进 CI；性能结果应作为 benchmark/PR 报告保存。

## 7. 方案比较

| 方案 | 是否改变推理布局 | 启动收益 | 推理风险 | 决策 |
| --- | --- | ---: | ---: | --- |
| 改成 rank-major mmap | 是 | 高 | 高 | 不采用 |
| 每 block 一次 registration | 否 | 单 rank 高 | 多 rank 变慢 | 不采用 |
| 删除 `madvise` | 否 | 不确定 | 首次 offload 变慢 | 不单独采用 |
| full registration 与 GPU 初始化重叠 | 否 | 中 | 中 | 备用方案 |
| rank-local contiguous alias | 否 | 高 | 中 | 首选方案 |

## 8. 不在本计划范围内

- 修改 KV cache 的物理布局；
- 重写 `swap_blocks_batch()`；
- 修改 canonical mapping 语义；
- 改变 `shared_by` 协议；
- 为所有平台强行复用 CUDA alias 实现；
- 把性能 benchmark 变成默认 CI 阈值测试。

## 9. 最终决策

实现目标不是让推理使用 alias，而是让 alias 代替整个 shared mmap 成为 CUDA
host registration 的输入。这样可以同时满足：

1. 保持推理阶段最优的现有 interleaved layout；
2. 把每个 rank 的 host registration 字节数降低约 `W` 倍；
3. 避免每 block 调用一次 `cudaHostRegister()`；
4. 在大内存场景获得数秒级启动收益；
5. 在 alias 不适用的布局上安全使用当前 full registration 实现。

当前工作区已经包含 benchmark 和生产实现。实现对支持的 CUDA 交错布局直接使用
alias；只有真实 multi-GPU inference transfer 验证通过后才建议合入主分支。alias
不是可选的 correctness fallback，失败会明确终止该 worker 的初始化。
