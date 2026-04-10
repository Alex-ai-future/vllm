# vLLM IR: A PyTorch-based Functional Intermediate Representation for vLLM

Link to RFC: [\#32358](https://github.com/vllm-project/vllm/issues/32358), link to slides: [02/26/26 vLLM IR](https://docs.google.com/presentation/u/0/d/1k0Zo33KubK7pmhYXg-7G1PjgcEbOhBkSbwiEwUpUyH0/edit) 

Fusion/transformation `torch.compile` passes struggle with custom operators like `RMSNorm`, `Quant`, etc. because they decompose to either a fragile sequence of `torch` ops or a variety of custom kernels, requiring separate handling for each of them. All of those operators are instances of `CustomOp`, which has many of its own issues, mostly stemming from complicated and clunky dispatching logic with low visibility.

vLLM IR is a **functional intermediate representation (IR)** that fills the gap between low-level `torch` ops and vLLM layers like `RMSNorm`, separating the ***semantics*** from the ***implementation*** and ***dispatching***, solving the issues with torch.compile passes and custom op use and dispatching simultaneously. It operates as a **dialect** in the torch FX representation, allowing full interoperability with “regular” torch ops & custom ops, as well as a piecewise migration from the current `CustomOp` approach.

vLLM IR has the following main advantages over the current use of custom ops (& `CustomOp`s):

- Simple and extensible op and implementation registration, in-tree and out-of-tree  
- High-level functional compiler IR for easier compilation passes and OOT compilation backends  
- Single-source-of-truth dispatching and easier kernel dispatching configuration  
- On-demand autotuning of different implementations via `torch.compile`

Importantly, vLLM IR can be added to vLLM in a non-intrusive way. It does not require any model definition changes and it also provides a soft migration for OOT `CustomOp` registration. A more detailed migration guide is towards the end.

The benefits are shared across all vLLM stakeholders. Kernel developers will have an easier way to integrate kernels. Compiler developers will deal with massively improved ergonomics. Other vLLM developers will spend less time debugging which implementation got dispatched to. Most importantly, autotuning and easier dispatching will improve performance for end users.

This document is structured as follows: it begins with a quick overview of the IR, including an end-to-end example for `rms_norm`. This is followed by a comprehensive list of issues with custom op matching and `CustomOp` dispatching, resolved by vLLM IR. In light of those issues, the following section lists important goals for vLLM IR. The next section describes the detailed design and implementation details. The document is finished by a migration guide, possible downsides and alternatives, and open questions.

## Quick overview

In vLLM IR, each op is a pure function whose semantics are defined ***independently*** of its implementations. The semantics of an op are specified as a sequence of torch operations and also serve as the default torch-native implementation. Implementations represent the possible execution  backends for the op and are required to match the declared semantics, making them fully interchangeable. Users of an IR op invoke it directly, and the op itself is responsible for dispatching to an appropriate implementation.

During compilation, the op appears as a custom op in the torch FX graph, which allows performing fusions and other transformations independent of the op implementation selected. Later in the compilation pipeline the op is ***lowered*** to take full advantage of `torch.compile` optimization.

Kernel dispatching is controlled via per-op ***priority lists*** specified in `VllmConfig`, with each platform providing sensible defaults that can be overridden by the user. These priority lists express the order in which implementations are considered, and the IR op is dispatched to the first supported implementation. By default, an implementation is assumed to support all argument combinations, but this can be restricted via an explicit support predicate.

How the priority list is applied depends on the execution mode. When compiling a forward pass, the priority list determines which implementation the IR op is lowered to by checking support using the fake tensors in the compiled graph. During an eager forward pass, the IR op performs dynamic dispatch at call time, checking support by passing the actual runtime arguments to the support predicate.

### Usage example

The snippet below shows an example of the op and kernel registration and use for the layernorm op (currently `RMSNorm`). We first register the `vllm.ir.ops.rms_norm` IR op, register `vllm_c` and `AITER` kernels, and then invoke the IR op in the `RMSNorm` layer code. There are but a few omissions; the design is meant to be extremely simple:

```py
#===========================
# IR declaration in vllm/ir/
#===========================

# vllm/ir/ops/layernorm.py
@ir.register_op("rms_norm")
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float) -> Tensor:
"""Weighted root-mean-square layer normalization"""
  orig_dtype = x.dtype
  x = x.to(torch.float32)
  variance = x.pow(2).mean(dim=-1, keepdim=True)
  x = x * torch.rsqrt(variance + epsilon)
  x = x.to(orig_dtype)
  if weight is not None:
    x = x * weight
  return x

#====================================================
# Custom kernel registrations in vllm/kernels/
#====================================================

# vllm/kernels/vllm_c.py
@ir.ops.rms_norm.register_impl(provider="vllm_c")
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float) -> Tensor:
  output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
  torch.ops._C.rms_norm(output, x, weight, epsilon)
  return output

# vllm/kernels/aiter.py
AITER_SUPPORTED = is_aiter_found()
"""AITER must be installed"""

rms_norm_args_supported = lambda x, w, e: x.dtype in [torch.bfloat16, torch.float16]
"""AITER rms_norm only suppports 16-bit activations"""

# Passing a support predicate via supports_args, also static support
@ir.ops.rms_norm.register_impl(
  provider="aiter", supports_args=rms_norm_args_supported, supported=AITER_SUPPORTED
)
def rms_norm(x: Tensor, weight: Tensor | None, epsilon: float) -> Tensor:
  # NOTE: there's a second layer of custom op here to hide AITER calls from compilation
  return torch.ops.rocm_aiter.rms_norm(x, weight, epsilon)

#=====================================================
# Simplified layer code in vllm/model_executor/layers/
#=====================================================

# This only registers the IR ops
import vllm.ir.ops

# This registers the custom kernels
import vllm.custom_kernels

# RMSNorm interface stays the same, avoiding modifications to model files
# PluggableLayer is the successor to CustomOp but only for OOT layer overrides
class RMSNorm(PluggableLayer):
  def __init__(...):
    self.weight = torch.Parameter(...)
    self.epsilon = ...
    ...

  def forward(x: torch.Tensor, residual: torch.Tensor | None = None):
    if residual is None:
      return vllm.ir.ops.rms_norm(x, self.weight, self.epsilon)
    return vllm.ir.ops.fused_add_rms_norm(x, residual, self.weight, self.epsilon)


#=====================================================
# Kernel selection
#=====================================================

# CUDA: use vllm_c instead of native by default
vllm serve Qwen/Qwen-0.6B --kernel-config.ir_op_priority.rms_norm=vllm_c

# ROCm: use aiter, then vllm_c
vllm serve Qwen/Qwen-0.6B --kernel-config.ir_op_priority.rms_norm=aiter,vllm_c
```

## Issues addressed by vLLM IR {#issues-addressed-by-vllm-ir}

This section enumerates all current issues with custom op compilation and the `CustomOp` abstraction. Feel free to skip this section if you are intimately familiar with those issues and already convinced we need to do better 😀.

### Issues with compilation

**Mutable custom ops pattern matching**  
PyTorch Inductor uses a functional representation internally and does not deal with mutable ops well. They get wrapped into an `auto_functionalized` operator, which makes pattern matching trickier (but possible). `auto_functionalized_v2` is worse because it makes pattern matching impossible. vLLM is hence currently stuck on v1, which means we can’t take advantage of the v2’s memory saving improvements (built specifically for vLLM). There was an [attempt](https://github.com/pytorch/pytorch/pull/164273) to make Inductor support matching mutable ops, but we decided against it due to the complexity, hence this proposal.

**Multiple kernels & fragile native implementation matching**  
Depending on the platform, different kernels will be used which means the FX graph representation for the same operator will be different, even though its semantics are the same. This includes both custom kernels and torch-native implementations. [matcher\_utils.py](https://github.com/vllm-project/vllm/blob/main/vllm/compilation/matcher_utils.py) was an attempt to improve this situation, but it still required significant manual work and introduced fragility in matching native implementations (also described in [\#27072](https://github.com/vllm-project/vllm/issues/27072)). It also imposed additional restrictions on `CustomOp` implementations, requiring them to be purely functional, which required rewriting, diminishing the benefits.

**Custom (megakernel, LLM-based, etc.) compiler backends**  
These custom compiler backends prefer to receive higher-level representation than the low-level torch op FX representation. While manually enabling custom ops could be a solution, those are not functional, polluting the FX graph significantly and making it harder to work with.

**Decomposition autotuning**  
The performance of these operators has a smaller relative impact on E2E performance but the impact is not negligible, and users wanting to squeeze out every last token-per-second care about it. While it would be best if we could have a “single best kernel” for everything, this is not the reality, even for simpler ops. Multiple implementations give users the freedom to achieve the best performance in all cases.

Autotuning is not necessary to select the fastest kernels, but is certainly easier than manually benchmarking different implementations. And if it was easy to get the best performance here, more users would.

### Issues with the `CustomOp` abstraction

`CustomOp` is an outdated abstraction; it served us well but it can no longer support the features that we need. The following list of issues mostly comes from its stateful nature and opaque dispatching logic.

**Holding state**  
All `CustomOp`s are layer objects and hold state (at the very least for dispatching). This prevents them from being used in methods directly as they have to be initialized before the forward pass. While this seems like a reasonable restriction on its face, there is a lot of code that does not use this init-run separation, requiring massive refactoring to add a `CustomOp`. For example, [\#20711 (use QuantFP8 in MoE)](https://github.com/vllm-project/vllm/issues/20711) has been attempted multiple times (never successfully) due to the sheer amount of refactoring required.

**Complex dispatching logic**  
Currently `CustomOp` dispatching is controlled by `CompilationConfig.custom_ops`. If an op is not listed in the config as explicitly enabled (`”+op”` ) or disabled (`”-op”`), it defaults to disabled when using Inductor, and enabled otherwise.

When a `CustomOp` is disabled, it means it’s dispatched to the torch native implementation. That by itself is confusing \- a less-familiar user might mean this op disabled completely and does not run at all. It’s also unintuitive that they are “disabled” by default: when an optimization is disabled by default, that means it carries a risk and enabling it improves performance. Either way the nomenclature here is bad.

**OOT kernel registration**  
Currently OOT kernels can be registered by overriding the whole `CustomOp` instance during instantiation. That means that layer code has to be repeated as well. There is also no way to register more than one kernel so OOT platforms with multiple kernels must do their own dispatching.

**Fused op registration**  
While `CustomOp` dispatching is complex, it is currently possible to dispatch between different kernels. This is not the case for fused ops, where we currently directly insert calls to specific fused kernels into the `fx.Graph`. The only difference between `rms_norm` & `rms_norm_quant` is that the latter cannot appear directly in model forward code because it spans across layers. But dispatching for these custom ops should work the same.

**Term overload**  
This might be a petty issue but the term “custom op” might refer to a torch custom op (which could be a custom kernel or a wrapped python function hidden from Dynamo/Inductor although we usually mean the latter), or a vLLM `CustomOp` subclass. This overloading of the term makes it really confusing to talk about custom ops.

**Testing**  
`CustomOp` implementations are often tested by comparing kernel outputs directly to a reference implementation in torch, which is often written separately, adding yet another implementation. I believe FP8 quantization has 2-3 torch native implementations scattered around the codebase. This adds a significant amount of development friction and can lead to mismatches in semantics. More also in [\#19817](https://github.com/vllm-project/vllm/issues/19817).

**`CustomOp` subclasses and kernel reuse**  
We have many different subclasses of `RoPE` and a few different ones of `RMSNorm`. First, `CustomOp`  subclasses often lead to incorrect dispatching due to slightly different semantics and missing implementations. Additionally, the kernel coverage for these is not 100%: for example, `GemmaRMSNorm` only has a native implementation and no custom cuda/hip implementations. A closer look at the kernel reveals that the two differences (using 32-bit weights instead of 16-bit and adding 1 to the weights) can easily be reconciled with robust semantics for the `rms_norm` op and weight processing of weights. This would allow the use of custom CUDA/HIP/AITER/etc. kernels that are currently only implemented for “regular” `RMSNorm`.

**Single kernel selection**  
Different kernels often exhibit different relative performance for different batch sizes. Static dispatching means we always select the same kernel, which might not be optimal.

**AITER environment variables**  
This probably deserves its own 10-page document, but the tl;dr is that there’s a separate environment variable for almost every AITER kernel, making it statistically almost impossible to achieve the optimal configuration. These variables also don’t have any effect on enabling custom ops, meaning those have to be enabled manually as well for the variables to even take effect. This is a perfect scenario for autotuning.

**Incompatibility with training code**  
For joint training-inference applications like RLHF, the training code has to be separate from inference code, in part due to the static dispatching of kernels. A pure functional definition of ops is a step closer to reconciliation of training and inference model definitions.

## Important higher-level goals for vLLM IR

### Faithfulness to eager mode

The success of PyTorch came largely from its simplicity in eager mode. `torch.compile` continued this success by strictly (apart from slight numerics) adhering to eager mode semantics. Similarly, vLLM IR will have the same semantics in eager and compile modes, resulting in easier prototyping, predictability, and higher overall developer productivity.

### Simple and clear dispatching

While vLLM IR ops will support auto-tuning between implementations, it **has** to be easy to manually specify which implementation should be used for each op. It must also be easy to understand the dispatching logic and which implementations got picked for each op.

### Independence from the rest of vLLM

- Any IR definitions dependent on vLLM can go somewhere else in `vllm/`  
  - Registration can be dynamic so it can be done on either side  
- It’s hard to say now but in the future we might even want to move this to a top level `vllm_ir` folder and distribute it as a separate package. If the LLM inference world (or even LLM-training) standardized on vLLM IR, that can only be good for vLLM 😊.  
  - Only theoretical downside: SGLang could more easily take kernel from us if they adopted vLLM IR but it goes both ways (and happens already, in both directions)  
- Easier for OOT compilers to implement and test compilation of these ops  
- Many unknown uses (build it and they will come)

### Easy extensibility

While most vLLM IR will be defined in `vllm/ir`, it should be easily extensible:

- vLLM custom kernels easily integrated  
- vLLM can define ops with other vLLM dependencies outside `vllm/ir`  
- OOT kernels can just plug in, users can write their own\!  
- OOT models can define custom IR and reuse dispatching & compilation infrastructure  
- OOT platforms can easily plug in kernels and define custom IR ops if needed

### Interoperability with “regular” torch & custom ops

Compilation passes operating on vLLM IR will never assume all ops in the graph are vLLM IR ops and will fully integrate with other custom ops and builtin torch ops. This is similar to the MLIR dialect approach where each dialect only defines relevant operations and can mix with other dialects at any time.

This has multiple benefits:

- Reducing the scope of vLLM IR  
- Allowing piecewise implementation and migration  
- Not inhibiting day-0 model support (ops migrated later)

## Detailed Design

### Folder structure

```
vllm/
├── ir/
│   ├── op.py                     # IrOp class definition
│   ├── op_impl.py                # IrOpImpl class definition
│   ├── contexts.py               # Contexts set during the forward pass
│   ├── ...                       # Also other infra/utility files
│   └── ops/
│       ├── __init__.py           # Imports/registers all IR op definitions
│       ├── layernorm.py          # rms_norm, fused_add_rms_norm
│       ├── activation.py         # silu, gelu, relu, ...
│       ├── quant.py              # quant_fp8, quant_fp4, quant_int8, ...
│       ├── rope.py               # RoPE, mRoPE, ...
│       ├── ssm.py                # SSM-based ops (Mamba/Granite/Nemotron/...)
│       ├── moe.py                # routers, dispatch, etc.
│       ├── fused.py              # rms_norm_quant, silu_mul_quant, ...
│       ├── meta.py               # (optional) non-fwd-pass ops (e.g. metadata prep)
│       └── ...                   # Other IR op files
├── custom_kernels/
│   ├── __init__.py               # Imports (&registers) all custom kernel impls
│   ├── aiter.py                  # AITER impl registration
│   ├── custom.py                 # In-repo C++/CUDA/HIP kernels via torch.ops._C
│   ├── helion/
│   │   ├── kernel.py            # HelionKernelBase class, other infra
│   │   ├── silu_mul_quant.py    # Each helion kernel probably deserves its own file
│   │   ├── allreduce_rms_quant.py # ...
│   │   └── ...
│   └── triton/
│   │   └── ...                  # Any Triton implementations for IR (or other) ops
├── compilation/
│   ├── passes/                   # Custom Inductor passes
│   │   ├── pass_manager.py      # PassManager class and instances 
│   │   ├── ir/                  # IR passes (lowering/dispatch, out variants, ...)
│   │   ├── fusion/              # Other fusion passes
│   │   └── utility/             # Non-IR-related utility passes
│   └── ...                       # Other compilation files (caching, decorators, ...)
└── ...

```

### Op declaration and calling semantics

As shown in the example [near the top](https://docs.google.com/document/d/1takuaA2NVqYIaQ6_89qaLVWBKbrQXGBSZW9-rMam8Go/edit?tab=t.6lqinu8a4ett#heading=h.vnt7dsxr8nnn), an IR Op registration is a simple function containing the torch semantics of the op, decorated with `register_op`, which creates an `IrOp` object for the op. `IrOp` handles implementation registration (creating an `IrOpImpl` instance for each) and manages dispatching between them. There is a single global `IrOp` instance per-op instead of per-layer.

The semantics of calling the `IrOp` are defined as the exact semantics (barring slight numerical variations) of the native implementation in the declaration. Op implementations should adhere to the semantics of the native implementation.

When compiling, an `IrOp` call will invoke a torch custom op in the `vllm_ir` torch library. This op is opaque to `torch.compile`, meaning it will appear directly in the FX graph. We can then do custom transformations on these ops (fusion, sequence parallelism, etc.), before manually lowering them to the selected implementation. Thus, dynamic dispatching can be avoided completely in this case. The lowering can be done even when Inductor is not used. The compilation pipeline is described in more detail in the next section.

In eager mode, an `IrOp` call will again invoke a `vllm_ir` custom op, which then calls `_inner_call` directly on the `IrOp` object. `_inner_call` then calls dispatches to the appropriate implementation. Additional layers of indirection will be added as needed. Note that this path is also taken when compiling but the ops don’t get lowered explicitly.

If this overhead is unacceptable, `__call__` could invoke `_inner_call` directly, or we could even dispatch to an implementation directly if support does not need to be checked dynamically.

The skeleton of the `IrOp` class and registration code is shown below.

```py
def register_op(name: str) -> Callable[[Any], IrOp]:
  def decorator(f) -> IrOp:
    return IrOp(name, f)
  return decorator

class IrOp:
  name: str
  impls: dict[str, IrOpImpl]
  torch_op: OpOverload = getattr(torch.ops.vllm_ir, name).default
  """Registered in the constructor, points to _inner_call of the IrOp instance"""

  def register_fake(f):
  """Register a fake impl for the vllm_ir torch custom op"""
    ...

  def register_impl(provider: str, supports_args=None):
  """Register an implementation for this custom op"""
    def decorator(f):
      from vllm.ir import IrOpImpl
      self.impls[provider] = IrOpImpl(self, provider, f, supports_args)
      return

    return decorator

  def __call__(*args, **kwargs):
  """Direct call to torch op, could also skip the torch layer if not compiling?"""
    return self.torch_op(*args, **kwargs)
  
  def _inner_call(*args, **kwargs):
  """
  The torch op calls this method, which dispatches to the correct implementation.
  This code is not called during compilation as the ops get lowered manually.
  """
    impl = self._dispatch(*args, **kwargs)
    return impl(*args, **kwargs)
```

Both op and impl registration can be dynamic, meaning that we can make it conditional on any statically-known (non-data-dependent-)condition. For example, the snippet below registers the `rms_norm` and `silu_mul` ops on CUDA & ROCM. This approach scales well as we can put multiple registrations in the same `if` statement, and we can also choose a single version of the kernel for the exact hardware configuration. This reduces the overhead for 

```py
if current_platform.is_cuda_alike()
  @ir.rms_norm.register_impl(provider="vllm_c")
  def rms_norm(x: Tensor, weight: Tensor | None, eps: float) -> Tensor:
    output = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    return torch.ops._C.rms_norm(output, x, weight, epsilon)

  @ir.silu_mul.register_impl(provider="vllm_c")
  def silu_mul(x: Tensor, weight: Tensor) -> Tensor:
    output = torch.empty(x.shape[0]//2, x.shape[1:], device=x.device, dtype=x.dtype)
    return torch.ops._C.silu_mul(output, x, weight)
```

### Compilation pipeline

vLLM IR will truly shine during compilation using torch.compile. It will serve as a higher-level intermediate representation that’s easier to operate on than raw torch ops, and it will later lower to torch and custom ops to take advantage of automatic kernel generation, automatic Inductor fusion, and Inductor memory planning.

The full pipeline consist of the following steps (changes in **bold**):

- Dynamo tracing: this captures the initial FX graph**, which now contains vLLM IR ops**  
  - **If not using Inductor, lower vLLM IR (non-functional)**  
- AOTAutograd: the graph is functionalized  
- IR Fusion and transformations: **custom passes operate on** **vLLM IR ops**  
- **vLLM IR lowering: ops are lowered to their implementation (via dispatching config or autotuning)**  
- Inductor cleanup, defunctionalization, and codegen

IR op lowering works by **tracing the op implementation** with Dynamo and replacing the op with the resulting fx graph. It is crucial that the implementation declaration can be traced with inputs to the IR op (it can still be a closure or a bound method).

When autotuning for vLLM IR ops is enabled, torch.compile’s [autotuning-as-a-service feature](https://github.com/pytorch/pytorch/pull/167617) is used to autotune all possible decompositions and lower into the most performant one. It is not clear to me what stage of compilation currently contains this lowering, and this might need to be reconciled in upstream pytorch according to the needs of vLLM IR.

The exact location of lowering is an ***open question***. While lowering early allows DCE and cleanup leading to lower overhead, it would require properly lowering possibly in-place implementations into a functional FX graph. Lowering later also has the benefit of additional cleanup that might make vLLM IR fusion easier. We could try to manually invoke the Inductor post-grad passes before and after lowering.

### Composite (fused) ops

In vLLM IR, composite ops are ops whose definitions contain other vLLM ops. They behave very similarly as other IR ops; they are simple to declare, have well-defined semantics, and can dispatch & autotune between implementations. They are, however, unlikely to appear in model forward code directly. Instead, they are most commonly a product of fusion in the FX graph. Additional care must also be taken during autotuning as the decomposition of the op might be faster than the provided fused kernels.

```py
@vllm.ir.register(composite=True)
def rms_norm_static_fp8_quant(x: Tensor, weight: Tensor | None, e: float, scale: Tensor) -> Tensor:
  x_norm = vllm.ir.ops.rms_norm(x, weight, epsilon)
  return vllm.ir.ops.static_fp8_quant(x_norm, scale)
```

Initially, fusion passes will work as before, with only slightly modified patterns and replacements. Once we port more of the ops and composite ops over to vLLM IR, we can automate the fusion process more and use the fusion op declaration directly to search for patterns.

To autotune composite ops, we autotune over the union of all implementations of the fused op and the cartesian product of the implementations of ops in its decomposition. For example, if we had 4 kernels for `rms_norm_static_fp8_quant`, 2 kernels for `rms_norm`, and 3 kernels for  `static_fp8_quant`, we’d have 4 \+ 2 \* 3 candidates for autotuning. I don’t anticipate this number ever getting large but if it does, we can easily prune away a lot of these candidates, either manually or by using autotuning information for each op separately. We can also start with just autotuning over the 4 kernels for `rms_norm_static_fp8_quant` and the native impl (which is the combination of native `static_fp8_quant` and `rms_norm`).

A big ***open question*** remains about fusion-aware autotuning: if there are native torch ops to either side of the fused op for some reason, it could be optimal to use native code for each of the ops instead of the fused kernel, as the fused kernel cannot be fused onto while the native decompositions can. For now, this issue will be addressed by just manually fusing all known combinations of IR ops that appear consecutively.

### Implementation dispatch

Dispatching works by extracting the implementation priority list for the op, and dispatching to the first supported implementation. There might be some overhead but it only occurs in eager mode, which shouldn’t be a big issue. For op impls that are always supported, we could also dispatch statically at the start of the forward pass and replace the `_dispatch_and_forward` with the selected for the duration of the forward pass.

`IrOp` can also dispatch to a manually compiled version of the native implementation. This can be helpful in eager contexts with compilation enabled, e.g. when an op is nested within an opaque custom op and hence invisible to compilation, or when it’s used in a part of the model that’s not compiled (like multi-modal models). The lowering and dispatching priority lists can also be separate, allowing different behavior in a compiled context vs. eager.

Priority lists are specified in `KernelConfig`. User-specified lists take priority over platform-defaults; this way the user only specifies the changes to the order instead of the full list. The `KernelConfig` also serves as the single-source-of-truth for the priority lists, making it easy to understand which ops will get selected. Op selection can also be logged during warmup.

```py

# op_priority specified in config
class KernelConfig:
  compile_native: bool
  """Should native ops called directly get wrapped in torch.compile?"""

  ir_op_priority: dict[str, list[str]]
  """vLLM IR op dispatching priority, user override merged with platform defaults"""

  def __post_init__(self):
    # Merge op priority by appending the defaults after user-specified values
    default_ir_op_priority = current_platform.default_ir_op_priority()
    for op in default_ir_op_priority:
      self.ir_op_priority[op] = unique(
          self.ir_op_priority.get(op, []) +
          default_ir_op_priority[op]
      )

# Defaults for CUDA platform
class CUDAPlatform(Platform):
  def default_ir_op_priority(self) -> dict[str, list[str] | str]:
  """Separate dict for eager mode omitted for brevity"""
    return {
      "rms_norm": "native",
      "static_quant_fp8": "native",
      "static_group_quant_fp8": "vllm_c",
      ...
     }

# Defaults for CUDA platform
class ROCmPlatform(Platform):
  def default_ir_op_priority(self) -> dict[str, list[str] | str]:
    return {
      "rms_norm": ["aiter", "vllm_c"],
      "static_quant_fp8": "vllm_c",
      "static_group_quant_fp8": ["aiter", "vllm_c"],
      ...
     }

# In vllm model runner code:
with vllm.ir.set_op_priority(
       op_priority=vllm_config.kernel_config.ir_op_priority,
       compile_native=vllm_config.kernel_config.compile_native,
): 
  model(*args)

# ==========
# In vllm/ir
# ==========
@contextmanager
def set_op_priority(op_priority: dict[str, list[str]], compile_native: bool):
  # Compile the native implementations if enabled
  if compile_native:
    for op in registry:
      op.compile_native(...)

  # Set priority lists 
  with [registry[op].with_priority_list(p) for op, p in op_priority.items()]
    yield

# Dispatching code on the op
class IrOp:
  ...
  @contextmanager
  def set_op_priority(op_priority: list[str]):
    old = self.op_priority
    self.op_priority = op_priority
    yield
    self.op_priority = old

  def _dispatch(*args, **kwargs):
  """
  This function dispatches to the implementation according to the priority list in the
  forward context. If this dispatching mechanism results in unacceptable overheads, we
  can always optimize it & cache parts of it.
  """
    import vllm.ir.context
    selected_impl = None
    for impl_name in self.op_priority:
      # if an impl has a supports_args, check support
      if self.impls[impl_name].supports_args is not None and \
        not self.impls[impl_name].supports_args(*args, **kwargs):
        continue

      selected_impl = self.impls[impl_name]
      break

    return self.native_f if selected_impl is None else selected_impl


```

### Helion

While helion kernels could just get registered as implementations for either atomic or composite IR ops, they will want to do some of their own dispatch between different configurations. vLLM IR can allow the Helion implementation to specify multiple possible configs over which a user can autotune; otherwise, Helion can do its own runtime config selection once called.

This should be done by subclassing `IrOpImpl`. With some inspiration taken from the [Helion RFC](https://github.com/vllm-project/vllm/issues/32219), the `HelionCustomOp` can be adapted into a `HelionIrOpImpl`. Further design work is needed here. There should likely be a layer between the Helion kernel and the op registration for better separation of concerns between `HelionIrOpImpl` and `HelionKernelWrapper`. Both will have a single global instance per kernel, and `HelionKernelWrapper` will not be aware of `HelionIrOpImpl`.

```py
@vllm.custom_kernels.helion.register_kernel
def silu_mul_fp8(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Pure Helion kernel implementation."""
    d = x.shape[-1] // 2
    out = torch.empty(x.shape[:-1] + (d,), device=x.device, dtype=torch.float8_e4m3fn)

    for tile_idx in hl.tile(out.shape):
        a_vals = x[..., :d][tile_idx].to(torch.float32)
        b_vals = x[..., d:][tile_idx]
        silu_result = a_vals * torch.sigmoid(a_vals)
        result = silu_result.to(x.dtype) * b_vals
        scale_val = hl.load(scale, [0])
        out[tile_idx] = (result.to(torch.float32) / scale_val).to(out.dtype)

    return out

# register the helion kernel impl using a helion helper
silu_mul_fp8.register_as_vllm_ir_impl(vllm.ir.ops.silu_mul_static_quant_fp8)
```

### OOT (out-of-tree platforms)

`CustomOp` is currently used by OOT platform plugins to override custom operator implementations via the `CustomOp.register_oot` decorator. This requires rewriting the whole layer, including the logic about managing weights and other state. In vLLM IR, OOT platforms can instead use the functional kernel registration mechanism, using the `<ir_op>.register_impl` decorator just like in-tree kernels.

As we migrate dispatching from `CustomOp` subclasses to vLLM IR, those should become subclasses of a new [`PluggableLayer`](https://github.com/vllm-project/vllm/issues/23786) class to allow continued layer-based overriding for OOT platforms. Where state manipulation does not need to be overridden, OOT platforms will be encouraged to transition to the vLLM IR-based kernel registration mechanism.

### RLHF applications

Numerical reproducibility is crucial for RLHF inference, mostly captured by [***batch invariance***](https://docs.vllm.ai/en/latest/features/batch_invariance/): the size of the batch should not affect the numerics of each particular run. Current vLLM kernels are not batch-invariant by default but can be made so with the `VLLM_BATCH_INVARIANT=1`  
Environment variable flag.

In vLLM IR, we could control batch invariance statically or dynamically. Statically, implementations could specify whether they are batch invariant or not, and dynamically, we could pass a `batch_invariant` bool param to the call to tell the implementation to operate in batch-invariant mode. We will likely need some combination of both to avoid duplicate kernel registrations for kernels that can be batch invariant but with a performance penalty.

Additionally, by removing dispatching code from model layers and removing the dependence on other inference infrastructure, vLLM IR would allow greater unification of inference and training code. It would also allow different dispatching and compilation passes for training which requires different optimizations due to a different performance regime.

## Migration process

1. Improve file structure in `vllm/compilation`  
2. Create initial infrastructure for registration, dispatching, and lowering  
   1. Include one op (likely `rms_norm`)  
3. Migrate `CustomOp` instances to vLLM IR piecemeal  
- Layers themselves become `PluggableLayer` to maintain OOT compatibility  
- Add IR op registration, kernel registration, and lowering logic  
- Replace old ops with the IR op in fusion passes  
- Priority ops: `rms_norm`, `quant` (all flavors), `silu_mul`, `rope`  
4. Follow-ups (concurrent):  
   1. Add automatic fusion  
   2. Add autotuning-as-a-service integration  
   3. Integrate Helion  
   4. Remove AITER environment variables for dispatching  
   5. Mirage integration

Migration of the `-cc.custom_ops` flag is an open question, but I think we should deprecate it completely once all `CustomOp` ops are moved over. In the interim, `-cc.custom_ops+=+rms_norm` can be converted to `--kernel-config.ir_op_priority.rms_norm=vllm_c`, with a deprecation warning emitted.

## Possible downsides

- Dispatching logic:  
  - It’s not that simple but I think it’s better  
  - Any feature-complete dispatching logic will be complex  
- Kernels are no longer in the same place:  
  - This is already not the case, most impls just call a different file with the op/kernel code  
  - This actually reduces the complexity and increases standardization  
- Import time increase  
- Especially for Helion kernels, if all registration   
- We should measure the impact and make it lazy if necessary  
- Abstraction/dispatch performance  
- When compiled, never even called \-\> zero-cost  
- When eager, a hit is possible, but we can optimize this, also eager perf is lower priority anyway

### Alternatives:

- Wrap `CustomOp` with a torch functional custom op when compiling  
  - Cannot automatically do lowering, would need to mess around with instances  
- Kernel abstraction  
  - Does not solve the compilation & autotuning issues  
  - Still requires another layer of dispatching/selection  
    - Or it’s impure reducing the benefits of compilation  
  - Might process weights which makes it ineligible for autotuning  
- Rewrite everything in Helion and rely on automatic fusion  
  - Enormous effort  
  - Still need fusion across layers  
  - Still need to wrap the Helion kernels into ops for easier fusion and matching  
- Do nothing  
  - See [list of problems](#issues-addressed-by-vllm-ir)

## Open Questions

### In-place operations

While most vLLM custom ops exhibit functional semantics (apart from taking outputs as params to avoid allocating internally), there are some that strictly use in-place semantics to save on memory. The main such example is the CUDA custom kernel `fused_add_rms_norm` with the schema:  
`fused_add_rms_norm(input: Tensor!, residual: Tensor!, weight: Tensor, eps: float)`. It writes the output activations and output residual back to the corresponding input tensors. The `RMSNorm` layer behaves functionally and returns its outputs even in the residual case, but those tensors might alias the inputs if the CUDA kernel is dispatched to.

These memory semantics should be preserved by vLLM IR in both eager and compiled execution. Eager is harder to handle as we can perform transformations during compilation. We have a few options here:

1. Relax the semantics of vLLM IR and allow input modification (also current behavior of `RMSNorm`)  
   * This could be conditionally enabled by a flag like `allow_inplace` or `reuse_inputs`  
   * How would the compiler handle aliasing between input and outputs?  
     * Could add clones to maintain functional semantics (can be removed in DCE)  
2. Use an in-place overload  
   * `vllm.ir.ops.fused_add_rms_norm.maybe_inplace(input, residual, ...)`  
   * For implementations that aren’t in-place this might require copying in eager mode unless we also return the outputs (similar to option 1)?

Verdict: version 2, compiler needs separate ops, always return output

### Out variants for non-IR ops

We need the ability to write fusions and transformations on non-IR ops as well. This might be due to platform-specific ops or new ops that have not been standardized in the IR yet (example: `scaled_fp4_quant`). If we want to enable `auto_functionalized_v2`, we have to functionalize these custom ops as well.

To functionalize these ops, we need to be able to register “out” variants for custom kernels. We should be able to achieve this with op overloads and custom transformations initially but it would be great if this could be handled on the torch side. Converting from out to functional should be easy (just calling the) In the future, it would be nice to reduce the boilerplate required so that custom kernels can be registered as simply as possible.

For example, `scaled_fp4_quant`:

```c
// current registration
ops.def("scaled_fp4_quant("
        "Tensor! output, Tensor input, Tensor! output_scale, Tensor input_scale"
        ") -> ()");
ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);

// overload registration - I don't know the syntax for overloads
ops.def("scaled_fp4_quant(Tensor x, Tensor scale) -> (Tensor, Tensor)");
ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant);
// Do we need to register a fake function?
ops.impl("scaled_fp4_quant", torch::kMETA, &scaled_fp4_quant_fake);
// Out overload
ops.def("scaled_fp4_quant.out(Tensor! out, Tensor! out_scale, Tensor input, Tensor input_scale) -> ()");
ops.impl("scaled_fp4_quant.out", torch::kCUDA, &scaled_fp4_quant_out);

// Ideal registration (scaled_fp4_quant has optional out param, which is a tuple)
ops.def("scaled_fp4_quant(Tensor x, Tensor scale) -> (Tensor, Tensor)");
ops.impl("scaled_fp4_quant", torch::kCUDA, &scaled_fp4_quant, &scaled_fp4_quant_fake);
```

We also need to handle attention: without Inductor, the `out` variant needs to be invoked so that the output is allocated in a cudagraph when using piecewise cudagraphs. Then, the op needs to be functionalized for compilation passes. Finally, the `out` variant needs to be used for memory planning and Inductor-partition piecewise cudagraphs. Ideally the lowering could happen automatically on the torch side. However, we could perform this manually, especially because we might need to check backend support for the out variant dynamically.

### Other

- Compilation-related:  
- Lowering:  
  - When should we lower IR to impl?  
    - Functionalization vs DCE  
    - When does custom op autotuning happen?  
  - If impls have impure calls, will they get lowered correctly?  
  - Order of passes:  
    - AOTAutograd: functionalization  
      - Inductor Post-grad: noop, patterns, reordering, DCE, custom passes (like all of the vLLM custom passes),   
        - IR lowering  
        - noop, patterns, reordering, DCE (again)  
          - NB: patterns, reordering shouldn’t matter here (they’re very specific things). The only thing I think we care about is no-op elimination?  
      - Inductor lowering  
        - Custom op autotuning happens here  
      - Inductor scheduling  
        - generates pointwise/reduction fusions  
        - There is some “DCE” behavior here  
      - Inductor codegen  
      - Inductor Autotuning(deferred autotuning with fusion optimizations)  
        - autotuning of inductor-generated kernels  
  - Could we trace the implementations with Dynamo to ensure:  
    - They don’t affect external state  
    - They don’t modify inputs?  
      - Trickier with in-place ops above  
    - Try: Just invoke tracing (`make_fx`: slightly different than Dynamo)  
    - Not critical, nice   
- Can we make sure that `supports_args` does not check the batch size?  
  - This would not be possible from kernel abstraction anyway  
  - Run with Dynamo and check no guards are added?  
    - Use unbacked, could manually replace  
      - Could just look at the guards in `ShapeEnv`  
  - What happens	 if an implementation adds guards?  
    - Not allowed in kernel abstraction anyway  
    - Static sizes don’t have this problem  
- Torch custom op autotuning:   
  - Is it compatible with lowering before the end of post-grad pass?  
    - Fusion-aware autotuning:  
      - If we have a bunch of ops that could be fused, do we need to manually fuse them?  
        For example, rms \+ rope \+ quant, and we have fused rope quant kernels but not all three, so we want to compare perf from full native decomp as well as all fused kernel variations  
- Manual fusion might not be as scalable but I don’t see an alternative  
- Eager/non-lowered:  
  - How do we want to handle with ops outside compile region/context (inside opaque op or MM, but compilation enabled)  
1. Always use compiled native (likely best)  
2. Dispatch to custom if not compiling  
3. Some combination of 2+3 if there’s a need  
- Dynamic dispatch potential issues  
  - Overhead of dynamic dispatch?  
    - Could do static dispatch if `supports_args` is ignored/not present  
  - Cannot easily predict which op is selected if support is conditional  
  - Observability:  
    - Log in debug mode?  
    - Keep track of ops  
    - Could prune the list and emit at the start of the forward pass  
    - Aiter, vllm\_c  
- Op priority at the batch size range level  
  - With compilation, easy  
  - For eager dispatch, harder  
    - Allow dictionary for op priority in addition to list?  
    - How is this encoded in `KernelConfig`?  
- Distributed ops  
  - Can we define distributed semantics in pure torch?  
    - Yes  
  - We can start by declaring these on the vLLM side  
- How extensively do we want to overload the ops?  
  - Example: `quant` vs `quant_fp8` vs `quant_group` vs `static_quant_group_fp8`?  
  - Proposed heuristic: if we’re gonna want to dispatch to different providers they should be different custom ops  
  - Do we want torch overloads?  
    - If so can each overload still have an `.out` variant?  
  - Different op names preferred over overloads  
  - Better to consolidate (conv example)  
  -   
- Criteria for in-tree ops  
  - Proposal: IR op only once multiple implementations (for different platforms) with the same semantics  
  - IR ops can live on vLLM side if declaration dependent on vLLM  
  - Should attention ops be part of the IR?  
    - To clarify, attention will be (already almost is) a functional op, the question is if it should live in `vllm/ir`, and it does not use the `CustomOp` abstraction  
    - Hard to define semantics when there are external effects  
      - Unless we export forward context and metadata to IR  
    - Could just throw?  
    - Future problem, for now keep regular `unified_attention` custom op definitions  
- Can Helion \-\> Helion fuse?  
- What will it take for torch \-\> Helion \-\> torch automatic fusion?  
  - What happens to config dispatch?  
- Integrating “heavy” ops (scaled\_mm, grouped\_gemm, etc.)  
  - Autotuning harder due to weight pre-processing  
    - Later problem  
    - Would be nice to autotune over different async\_tp implementations  
  - Define ops for different semantics  
    - Lower priority  
- AITER-specific concerns:  
  - “Sub-provider” (key): different implementations  
    - Better than different providers to simplify priority list  
    - Need to decide priority between keys?  
    - A way to guarantee that for any args at least one key supports them?  
  - JIT compilation  
    - Current warmup might not hit all shapes  
    - How to warm-up all shapes to avoid JITing after startup  
      - Specify buckets?

:white\_check\_mark:  
