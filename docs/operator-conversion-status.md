# ONNX → WebNN operator conversion status

> **Last edited:** `2026-10-30T13:56:50Z`<br>
> **Checkout:** `main` at `b39e4be`
>
> **Freshness:** Use this document only when this provenance is recent relative to the relevant
> code and commits; otherwise verify the implementation, tests, and Git history before relying
> on it.

Tables are generated from [`src/onnx/ops/catalog.rs`](../src/onnx/ops/catalog.rs): run
`python3 scripts/generate_operator_catalog.py` to update them, or add `--check` to detect drift.
Coverage counts the 198 non-deprecated standard operators in ONNX opset 26. “Supported” means at
least one form executes; see each row's scope for restrictions.

For one model, `onnx2webnn convert --input model.onnx --no-backend-graph-build` prints operator
statistics after folding and specialization, including rejected forms, without building a backend
graph. The Python operator test generator uses the generated catalog JSON through
`scripts/webnn_onnx_ops.py`.

## Status categories

| Category | Meaning |
|----------|---------|
| **Direct WebNN** | At least one supported form primarily maps to a standardized WebNN builder primitive. |
| **Decomposed** | Executes through multiple WebNN primitives because no matching native primitive exists. |
| **Folded/static** | Resolved or materialized while converting. |
| **Pattern-only** | Executes only when a specific static export pattern can be eliminated or inlined. |
| **Unsupported** | No executable lowering exists on this branch. |

Only **Direct WebNN** counts as a native WebNN mapping.

### Estimated performance impact

Impact estimates apply only to **Decomposed** paths. They describe graph structure, not measured
backend performance.

| Tier | Meaning |
|------|---------|
| **minimal** | A small number of cheap elementwise, cast, or shape operations; backend fusion can often remove most overhead. |
| **moderate** | Multiple full-tensor passes, reductions, or materialized intermediates are expected to add noticeable overhead. |
| **high** | The lowering loses a fused, quantized, integer-accumulation, attention, or sparse-dispatch path and can materially increase compute or memory traffic. |
| **not inferred** | Impact varies too strongly with the accepted equation, input count, shape, or backend to estimate from the lowering alone. |

**Performance-limited** flags a decomposed path that loses a major capability, such as sparse
dispatch or low-bit matmul. It is a warning, not a conversion failure.

## Coverage

**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. **Regenerate:** `python3 scripts/generate_operator_catalog.py`.

| Opset-26 standard ONNX coverage | Count | Share |
|---------------------------------|------:|------:|
| Total non-deprecated `ai.onnx` names | 198 | 100% |
| **Executable for at least one supported form** | **129** | **65.2%** |
| └ Direct WebNN/spec-level subset | **96** | **48.5%** |
| └ Decomposed, folded, or pattern-only | **33** | **16.7%** |
| Unsupported | 69 | 34.8% |

The generated Python supported-op data derives from the same Rust catalog.

## Complete opset-26 status matrix

**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. **Regenerate:** `python3 scripts/generate_operator_catalog.py`.

| ONNX operator | Status | Execution scope and WebNN limitation | Estimated perf impact | Performance-limited |
|---------------|--------|--------------------------------------|-----------------------|---------------------|
| `Abs` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Acos` | Unsupported | No current executable lowering. |  |  |
| `Acosh` | Decomposed | Elementwise decomposition; WebNN has no native acosh. | moderate |  |
| `Add` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `AffineGrid` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `And` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ArgMax` | Direct WebNN | Direct WebNN argMax; select_last_index is not implemented. |  |  |
| `ArgMin` | Direct WebNN | Direct WebNN argMin; select_last_index is not implemented. |  |  |
| `Asin` | Unsupported | No current executable lowering. |  |  |
| `Asinh` | Decomposed | Elementwise decomposition; WebNN has no native asinh. | moderate |  |
| `Atan` | Unsupported | No current executable lowering. |  |  |
| `Atanh` | Decomposed | Elementwise decomposition; WebNN has no native atanh. | moderate |  |
| `Attention` | Unsupported | No native WebNN attention; no standard Attention decomposition is implemented. |  |  |
| `AveragePool` | Direct WebNN | Direct WebNN pool; 1-D is emulated, higher spatial ranks rejected, and some ONNX padding modes are restricted. |  |  |
| `BatchNormalization` | Direct WebNN | Direct WebNN primitive; inference mode only. |  |  |
| `Bernoulli` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `BitCast` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BitShift` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BitwiseAnd` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BitwiseNot` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BitwiseOr` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BitwiseXor` | Unsupported | WebNN has no corresponding bitwise/reinterpretation primitive. |  |  |
| `BlackmanWindow` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `Cast` | Direct WebNN | Direct WebNN cast where the target type supports it; numeric-to-Boolean casts normalize nonzero values through a WebNN comparison. Unsupported ONNX/string element types are rejected. |  |  |
| `CastLike` | Decomposed | Infers the second input's type at conversion time, then emits WebNN cast. | minimal |  |
| `Ceil` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Celu` | Decomposed | Elementwise decomposition; no native WebNN celu. | moderate |  |
| `CenterCropPad` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `Clip` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Col2Im` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `Compress` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `Concat` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ConcatFromSequence` | Unsupported | WebNN has no sequence operand type. |  |  |
| `Constant` | Folded/static | Creates an inline WebNN constant; no runtime ONNX operation remains. |  |  |
| `ConstantOfShape` | Folded/static | Materialized or expanded from statically resolvable shape/value data. |  |  |
| `Conv` | Direct WebNN | Direct WebNN conv2d; 1-D is emulated and 3-D is unsupported. |  |  |
| `ConvInteger` | Decomposed | Centers values in float, executes conv2d, then casts to int32; WebNN lacks integer accumulation, so large sums may lose exactness. | high | Yes |
| `ConvTranspose` | Direct WebNN | Direct WebNN convTranspose2d; 1-D is emulated and 3-D is unsupported. |  |  |
| `Cos` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Cosh` | Decomposed | Elementwise decomposition; WebNN has no native cosh. | moderate |  |
| `CumProd` | Decomposed | exp(cumulativeSum(log(x))); valid only for positive floating-point inputs. Zero and negative inputs are not generally correct. | moderate |  |
| `CumSum` | Direct WebNN | Direct WebNN cumulativeSum; axis must be constant. |  |  |
| `DFT` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `DeformConv` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `DepthToSpace` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `DequantizeLinear` | Direct WebNN | Direct WebNN primitive for supported scalar/effectively-scalar or full-rank block parameters; axis must be 1. |  |  |
| `Det` | Unsupported | No current executable lowering. |  |  |
| `Div` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Dropout` | Unsupported | No current inference lowering; exporters may remove applicable training artifacts. |  |  |
| `DynamicQuantizeLinear` | Decomposed | Reduction/arithmetic decomposition followed by WebNN quantizeLinear. | moderate |  |
| `Einsum` | Decomposed | Static reduce/transpose/reshape/matmul decomposition; at most two inputs, explicit-output ellipsis on one input, no repeated labels within a term. | not inferred |  |
| `Elu` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Equal` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Erf` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Exp` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Expand` | Direct WebNN | Direct WebNN expand; target shape must be build-time resolvable. |  |  |
| `EyeLike` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `Flatten` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Floor` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GRU` | Direct WebNN | Direct WebNN recurrent primitive with layout adaptation; sequence_lens and unsupported optional inputs are rejected. |  |  |
| `Gather` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GatherElements` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GatherND` | Direct WebNN | Direct WebNN gatherND; ONNX batch_dims is currently ignored. |  |  |
| `Gelu` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Gemm` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GlobalAveragePool` | Direct WebNN | Direct WebNN pool; supported for the implemented 1-D/2-D spatial forms. |  |  |
| `GlobalLpPool` | Unsupported | No current executable lowering. |  |  |
| `GlobalMaxPool` | Direct WebNN | Direct WebNN pool; supported for the implemented 1-D/2-D spatial forms. |  |  |
| `Greater` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GreaterOrEqual` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `GridSample` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `GroupNormalization` | Decomposed | Reduction and elementwise decomposition; relevant ranks and dimensions must be known. | moderate |  |
| `HammingWindow` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `HannWindow` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `HardSigmoid` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `HardSwish` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Hardmax` | Decomposed | Comparison/mask decomposition; requires known input shape. | moderate |  |
| `Identity` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `If` | Folded/static | Only constant or pinned-constant conditions; the selected branch is resolved and inlined during conversion because WebNN has no control flow. Runtime-selectable models therefore require a separate `.webnn`/Safetensors pair per branch. Shared weights are not deduplicated, so keeping both compiled graphs resident may duplicate them in RAM and VRAM. |  |  |
| `ImageDecoder` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `InstanceNormalization` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `IsInf` | Direct WebNN | Direct WebNN isInfinite; non-default ONNX detect_negative/detect_positive filtering is not implemented. |  |  |
| `IsNaN` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `LRN` | Unsupported | No current executable lowering. |  |  |
| `LSTM` | Direct WebNN | Direct WebNN recurrent primitive with layout adaptation; sequence_lens and peepholes are rejected. |  |  |
| `LayerNormalization` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `LeakyRelu` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Less` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `LessOrEqual` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Log` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `LogSoftmax` | Decomposed | Softmax/log decomposition; no native WebNN logSoftmax. | moderate |  |
| `Loop` | Unsupported | No WebNN runtime loop/control-flow representation. |  |  |
| `LpNormalization` | Unsupported | No current executable lowering. |  |  |
| `LpPool` | Direct WebNN | Direct WebNN l2Pool2d only when p=2; 1-D is emulated and higher spatial ranks rejected. |  |  |
| `MatMul` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `MatMulInteger` | Decomposed | Centers values in float, executes matmul, then casts to int32; WebNN lacks integer accumulation, so large sums may lose exactness. | high | Yes |
| `Max` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `MaxPool` | Direct WebNN | Direct WebNN pool; indices output is unsupported, 1-D is emulated, and higher spatial ranks are rejected. |  |  |
| `MaxRoiPool` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `MaxUnpool` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `Mean` | Decomposed | Variadic binary fold; operation order and floating-point rounding can differ. | not inferred |  |
| `MeanVarianceNormalization` | Unsupported | No current executable lowering. |  |  |
| `MelWeightMatrix` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `Min` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Mish` | Decomposed | Softplus/tanh/multiply decomposition; no native WebNN mish. | moderate |  |
| `Mod` | Decomposed | A - B * q(A/B) decomposition for both fmod modes; no WebNN remainder primitive. | moderate |  |
| `Mul` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Multinomial` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `Neg` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `NegativeLogLikelihoodLoss` | Unsupported | No current inference lowering; exporters may remove applicable training artifacts. |  |  |
| `NonMaxSuppression` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `NonZero` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `Not` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `OneHot` | Decomposed | Static comparison/mask decomposition; depth and values must be constant and indices shape known. | moderate |  |
| `Optional` | Unsupported | WebNN has no optional operand type. |  |  |
| `OptionalGetElement` | Unsupported | WebNN has no optional operand type. |  |  |
| `OptionalHasElement` | Unsupported | WebNN has no optional operand type. |  |  |
| `Or` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `PRelu` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Pad` | Direct WebNN | Direct WebNN pad; rank and non-negative pads must be known at build time. |  |  |
| `Pow` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `QLinearConv` | Unsupported | No fused WebNN quantized linear-algebra primitive; float decomposition is not implemented. |  |  |
| `QLinearMatMul` | Unsupported | No fused WebNN quantized linear-algebra primitive; float decomposition is not implemented. |  |  |
| `QuantizeLinear` | Direct WebNN | Direct WebNN primitive for supported scalar/effectively-scalar or full-rank block parameters; axis must be 1. |  |  |
| `RMSNormalization` | Decomposed | Reduction and elementwise decomposition; relevant ranks and dimensions must be known. | moderate |  |
| `RNN` | Unsupported | No current executable lowering. |  |  |
| `RandomNormal` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `RandomNormalLike` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `RandomUniform` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `RandomUniformLike` | Unsupported | WebNN has no graph random-number operators. |  |  |
| `Range` | Folded/static | Implemented for constant scalar or supported bounded-symbolic forms. |  |  |
| `Reciprocal` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceL1` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceL2` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceLogSum` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceLogSumExp` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceMax` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceMean` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceMin` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceProd` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceSum` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `ReduceSumSquare` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `RegexFullMatch` | Unsupported | WebNN has no string tensor/operation support. |  |  |
| `Relu` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Reshape` | Direct WebNN | Direct WebNN reshape; target shape must be build-time resolvable. |  |  |
| `Resize` | Direct WebNN | Maps to WebNN resample2d; nearest/linear supported forms only, with 1-D emulation. Cubic and unsupported coordinate/axis modes are rejected. |  |  |
| `ReverseSequence` | Direct WebNN | Partial mapping to WebNN reverse: sequence_lens and batch_axis semantics are ignored, so only full-axis reversal is equivalent. |  |  |
| `RoiAlign` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `RotaryEmbedding` | Decomposed | Gather/rotate/concat decomposition; concrete dimensions and compatible cache shapes required. | moderate |  |
| `Round` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `STFT` | Unsupported | WebNN has no corresponding signal-processing primitive. |  |  |
| `Scan` | Unsupported | No WebNN runtime loop/control-flow representation. |  |  |
| `ScatterElements` | Direct WebNN | Direct WebNN scatterElements; only reduction=none. |  |  |
| `ScatterND` | Direct WebNN | Direct WebNN scatterND; only reduction=none. |  |  |
| `Selu` | Decomposed | Elementwise decomposition; no native WebNN selu. | moderate |  |
| `SequenceAt` | Pattern-only | Only a constant index into the converter's SplitToSequence pseudo-sequence. |  |  |
| `SequenceConstruct` | Unsupported | WebNN has no sequence operand type. |  |  |
| `SequenceEmpty` | Unsupported | WebNN has no sequence operand type. |  |  |
| `SequenceErase` | Unsupported | WebNN has no sequence operand type. |  |  |
| `SequenceInsert` | Unsupported | WebNN has no sequence operand type. |  |  |
| `SequenceLength` | Unsupported | WebNN has no sequence operand type. |  |  |
| `SequenceMap` | Unsupported | WebNN has no sequence operand type. |  |  |
| `Shape` | Folded/static | Resolved from known graph shape metadata. |  |  |
| `Shrink` | Decomposed | Elementwise comparison/select decomposition; no native WebNN shrink. | moderate |  |
| `Sigmoid` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Sign` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Sin` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Sinh` | Decomposed | Elementwise decomposition; WebNN has no native sinh. | moderate |  |
| `Size` | Unsupported | No current executable lowering. |  |  |
| `Slice` | Direct WebNN | Direct WebNN slice/reverse; starts, sizes, axes, and steps must resolve at build time. Positive steps are preserved as strides; negative steps support only the full-axis reverse pattern. |  |  |
| `Softmax` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `SoftmaxCrossEntropyLoss` | Unsupported | No current inference lowering; exporters may remove applicable training artifacts. |  |  |
| `Softplus` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Softsign` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `SpaceToDepth` | Unsupported | No current lowering; the required vision/spatial primitive is absent or not implemented. |  |  |
| `Split` | Direct WebNN | Direct WebNN split; split sizes or the split dimension must be resolvable. |  |  |
| `SplitToSequence` | Pattern-only | Only the static split pattern consumed by constant-index SequenceAt; no sequence graph output. |  |  |
| `Sqrt` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Squeeze` | Direct WebNN | Direct WebNN squeeze; axes must be resolvable. |  |  |
| `StringConcat` | Unsupported | WebNN has no string tensor/operation support. |  |  |
| `StringNormalizer` | Unsupported | WebNN has no string tensor/operation support. |  |  |
| `StringSplit` | Unsupported | WebNN has no string tensor/operation support. |  |  |
| `Sub` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Sum` | Decomposed | Variadic binary fold; operation order and floating-point rounding can differ. | not inferred |  |
| `Swish` | Decomposed | Sigmoid/multiply decomposition; no native WebNN swish. | minimal |  |
| `Tan` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Tanh` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `TensorScatter` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `TfIdfVectorizer` | Unsupported | WebNN has no string tensor/operation support. |  |  |
| `ThresholdedRelu` | Decomposed | Comparison/select decomposition; no native WebNN thresholdedRelu. | minimal |  |
| `Tile` | Direct WebNN | Direct WebNN tile; repeats must be constant. |  |  |
| `TopK` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `Transpose` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Trilu` | Direct WebNN | Direct WebNN triangular; k must be resolvable. |  |  |
| `Unique` | Unsupported | No current lowering; missing primitive and/or data-dependent output shape. |  |  |
| `Unsqueeze` | Direct WebNN | Direct WebNN unsqueeze; axes must be resolvable. |  |  |
| `Where` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |
| `Xor` | Direct WebNN | Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply. |  |  |

## Additional contrib and fused operators

**Generated table. Source of truth:** `src/onnx/ops/catalog.rs`. **Regenerate:** `python3 scripts/generate_operator_catalog.py`.

| ONNX operator | Status | Execution scope and WebNN limitation | Estimated perf impact | Performance-limited |
|---------------|--------|--------------------------------------|-----------------------|---------------------|
| `GatherBlockQuantized` | Decomposed | Gathers packed rows/scales/zero points and dequantizes only the selected slice for supported 2-D axis-0 tables; other layouts fall back at conversion time. | moderate |  |
| `GroupQueryAttention` | Decomposed | Static attention/cache subgraph. WebNN has no fused attention; runtime sequence metadata is ignored and softcap/local-window modes are rejected. Fused backend kernels and the portable decomposition need not be bit-identical, so an immediately following dynamic quantizer can amplify legal rounding differences. | high | Yes |
| `MatMulBnb4` | Decomposed | Packed NF4/FP4 block path where representable; incompatible tails become dense constants, losing low-bit memory/performance benefits. | high | Yes |
| `MatMulNBits` | Decomposed | Supported 4/8-bit constant layouts dequantize into a matmul path; g_idx is rejected. WebNN has no low-bit matmul kernel. | high | Yes |
| `MoE` | Decomposed | All experts execute densely because WebNN has no TopK or sparse dispatch. Approximate compute overhead is num_experts/k; supported activation/fusion forms only. | high | Yes |
| `QMoE` | Decomposed | Supported 4/8-bit expert weights feed the same dense all-expert lowering; block-layout fallbacks may dequantize at conversion time. | high | Yes |
| `SimplifiedLayerNormalization` | Decomposed | RMS-style normalization decomposition; no matching WebNN fused primitive. | moderate |  |
| `SkipSimplifiedLayerNormalization` | Decomposed | Residual/bias plus simplified normalization decomposition; no matching WebNN fused primitive. | moderate |  |

### MoE interpretation

`MoE` and `QMoE` run every expert and blend outputs: functional support, but no sparse-dispatch
performance parity.

Numerical results: [model validation status and triage](model-validation-status-triage.md).

## References

- [W3C WebNN specification](https://www.w3.org/TR/webnn/)
- [ONNX operator schemas](https://github.com/onnx/onnx/blob/main/docs/Operators.md)
- [ONNX Runtime MoE/QMoE notes](https://github.com/microsoft/onnxruntime/blob/main/docs/contrib_ops/cuda/moe_qmoe.md)
