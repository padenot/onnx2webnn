//! Code-backed ONNX operator inventory. The generated documentation and Python data derive from this table.
use crate::protos::onnx::NodeProto;
use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, Serialize)]
pub struct OperatorCapability {
    pub name: &'static str,
    pub status: &'static str,
    pub scope: &'static str,
    pub perf_impact: &'static str,
    pub perf_limited: bool,
    pub domain: &'static str,
}

pub const OPERATOR_CAPABILITIES: &[OperatorCapability] = &[
    OperatorCapability { name: "Abs", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Acos", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Acosh", status: "Decomposed", scope: "Elementwise decomposition; WebNN has no native acosh.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Add", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "AffineGrid", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "And", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ArgMax", status: "Direct WebNN", scope: "Direct WebNN argMax; select_last_index is not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ArgMin", status: "Direct WebNN", scope: "Direct WebNN argMin; select_last_index is not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Asin", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Asinh", status: "Decomposed", scope: "Elementwise decomposition; WebNN has no native asinh.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Atan", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Atanh", status: "Decomposed", scope: "Elementwise decomposition; WebNN has no native atanh.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Attention", status: "Unsupported", scope: "No native WebNN attention; no standard Attention decomposition is implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "AveragePool", status: "Direct WebNN", scope: "Direct WebNN pool; 1-D is emulated, higher spatial ranks rejected, and some ONNX padding modes are restricted.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BatchNormalization", status: "Direct WebNN", scope: "Direct WebNN primitive; inference mode only.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Bernoulli", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitCast", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitShift", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitwiseAnd", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitwiseNot", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitwiseOr", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BitwiseXor", status: "Unsupported", scope: "WebNN has no corresponding bitwise/reinterpretation primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "BlackmanWindow", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Cast", status: "Direct WebNN", scope: "Direct WebNN cast where the target type supports it; numeric-to-Boolean casts normalize nonzero values through a WebNN comparison. Unsupported ONNX/string element types are rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "CastLike", status: "Decomposed", scope: "Infers the second input's type at conversion time, then emits WebNN cast.", perf_impact: "minimal", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Ceil", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Celu", status: "Decomposed", scope: "Elementwise decomposition; no native WebNN celu.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "CenterCropPad", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Clip", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Col2Im", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Compress", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Concat", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ConcatFromSequence", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Constant", status: "Folded/static", scope: "Creates an inline WebNN constant; no runtime ONNX operation remains.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ConstantOfShape", status: "Folded/static", scope: "Materialized or expanded from statically resolvable shape/value data.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Conv", status: "Direct WebNN", scope: "Direct WebNN conv2d; 1-D is emulated and 3-D is unsupported.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ConvInteger", status: "Decomposed", scope: "Centers values in float, executes conv2d, then casts to int32; WebNN lacks integer accumulation, so large sums may lose exactness.", perf_impact: "high", perf_limited: true, domain: "ai.onnx" },
    OperatorCapability { name: "ConvTranspose", status: "Direct WebNN", scope: "Direct WebNN convTranspose2d; 1-D is emulated and 3-D is unsupported.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Cos", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Cosh", status: "Decomposed", scope: "Elementwise decomposition; WebNN has no native cosh.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "CumProd", status: "Decomposed", scope: "exp(cumulativeSum(log(x))); valid only for positive floating-point inputs. Zero and negative inputs are not generally correct.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "CumSum", status: "Direct WebNN", scope: "Direct WebNN cumulativeSum; axis must be constant.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "DFT", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "DeformConv", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "DepthToSpace", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "DequantizeLinear", status: "Direct WebNN", scope: "Direct WebNN primitive for supported scalar/effectively-scalar or full-rank block parameters; axis must be 1.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Det", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Div", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Dropout", status: "Unsupported", scope: "No current inference lowering; exporters may remove applicable training artifacts.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "DynamicQuantizeLinear", status: "Decomposed", scope: "Reduction/arithmetic decomposition followed by WebNN quantizeLinear.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Einsum", status: "Decomposed", scope: "Static reduce/transpose/reshape/matmul decomposition; at most two inputs, explicit-output ellipsis on one input, no repeated labels within a term.", perf_impact: "not inferred", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Elu", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Equal", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Erf", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Exp", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Expand", status: "Direct WebNN", scope: "Direct WebNN expand; target shape must be build-time resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "EyeLike", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Flatten", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Floor", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GRU", status: "Direct WebNN", scope: "Direct WebNN recurrent primitive with layout adaptation; sequence_lens and unsupported optional inputs are rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Gather", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GatherElements", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GatherND", status: "Direct WebNN", scope: "Direct WebNN gatherND; ONNX batch_dims is currently ignored.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Gelu", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Gemm", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GlobalAveragePool", status: "Direct WebNN", scope: "Direct WebNN pool; supported for the implemented 1-D/2-D spatial forms.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GlobalLpPool", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GlobalMaxPool", status: "Direct WebNN", scope: "Direct WebNN pool; supported for the implemented 1-D/2-D spatial forms.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Greater", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GreaterOrEqual", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GridSample", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GroupNormalization", status: "Decomposed", scope: "Reduction and elementwise decomposition; relevant ranks and dimensions must be known.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "HammingWindow", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "HannWindow", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "HardSigmoid", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "HardSwish", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Hardmax", status: "Decomposed", scope: "Comparison/mask decomposition; requires known input shape.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Identity", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "If", status: "Folded/static", scope: "Only constant or pinned-constant conditions; the selected branch is resolved and inlined during conversion because WebNN has no control flow. Runtime-selectable models therefore require a separate `.webnn`/Safetensors pair per branch. Shared weights are not deduplicated, so keeping both compiled graphs resident may duplicate them in RAM and VRAM.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ImageDecoder", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "InstanceNormalization", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "IsInf", status: "Direct WebNN", scope: "Direct WebNN isInfinite; non-default ONNX detect_negative/detect_positive filtering is not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "IsNaN", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LRN", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LSTM", status: "Direct WebNN", scope: "Direct WebNN recurrent primitive with layout adaptation; sequence_lens and peepholes are rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LayerNormalization", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LeakyRelu", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Less", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LessOrEqual", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Log", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LogSoftmax", status: "Decomposed", scope: "Softmax/log decomposition; no native WebNN logSoftmax.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Loop", status: "Unsupported", scope: "No WebNN runtime loop/control-flow representation.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LpNormalization", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "LpPool", status: "Direct WebNN", scope: "Direct WebNN l2Pool2d only when p=2; 1-D is emulated and higher spatial ranks rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MatMul", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MatMulInteger", status: "Decomposed", scope: "Centers values in float, executes matmul, then casts to int32; WebNN lacks integer accumulation, so large sums may lose exactness.", perf_impact: "high", perf_limited: true, domain: "ai.onnx" },
    OperatorCapability { name: "Max", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MaxPool", status: "Direct WebNN", scope: "Direct WebNN pool; indices output is unsupported, 1-D is emulated, and higher spatial ranks are rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MaxRoiPool", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MaxUnpool", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Mean", status: "Decomposed", scope: "Variadic binary fold; operation order and floating-point rounding can differ.", perf_impact: "not inferred", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MeanVarianceNormalization", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "MelWeightMatrix", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Min", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Mish", status: "Decomposed", scope: "Softplus/tanh/multiply decomposition; no native WebNN mish.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Mod", status: "Decomposed", scope: "A - B * q(A/B) decomposition for both fmod modes; no WebNN remainder primitive.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Mul", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Multinomial", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Neg", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "NegativeLogLikelihoodLoss", status: "Unsupported", scope: "No current inference lowering; exporters may remove applicable training artifacts.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "NonMaxSuppression", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "NonZero", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Not", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "OneHot", status: "Decomposed", scope: "Static comparison/mask decomposition; depth and values must be constant and indices shape known.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Optional", status: "Unsupported", scope: "WebNN has no optional operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "OptionalGetElement", status: "Unsupported", scope: "WebNN has no optional operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "OptionalHasElement", status: "Unsupported", scope: "WebNN has no optional operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Or", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "PRelu", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Pad", status: "Direct WebNN", scope: "Direct WebNN pad; rank and non-negative pads must be known at build time.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Pow", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "QLinearConv", status: "Unsupported", scope: "No fused WebNN quantized linear-algebra primitive; float decomposition is not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "QLinearMatMul", status: "Unsupported", scope: "No fused WebNN quantized linear-algebra primitive; float decomposition is not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "QuantizeLinear", status: "Direct WebNN", scope: "Direct WebNN primitive for supported scalar/effectively-scalar or full-rank block parameters; axis must be 1.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RMSNormalization", status: "Decomposed", scope: "Reduction and elementwise decomposition; relevant ranks and dimensions must be known.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RNN", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RandomNormal", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RandomNormalLike", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RandomUniform", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RandomUniformLike", status: "Unsupported", scope: "WebNN has no graph random-number operators.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Range", status: "Folded/static", scope: "Implemented for constant scalar or supported bounded-symbolic forms.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Reciprocal", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceL1", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceL2", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceLogSum", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceLogSumExp", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceMax", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceMean", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceMin", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceProd", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceSum", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReduceSumSquare", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RegexFullMatch", status: "Unsupported", scope: "WebNN has no string tensor/operation support.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Relu", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Reshape", status: "Direct WebNN", scope: "Direct WebNN reshape; target shape must be build-time resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Resize", status: "Direct WebNN", scope: "Maps to WebNN resample2d; nearest/linear supported forms only, with 1-D emulation. Cubic and unsupported coordinate/axis modes are rejected.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ReverseSequence", status: "Direct WebNN", scope: "Partial mapping to WebNN reverse: sequence_lens and batch_axis semantics are ignored, so only full-axis reversal is equivalent.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RoiAlign", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "RotaryEmbedding", status: "Decomposed", scope: "Gather/rotate/concat decomposition; concrete dimensions and compatible cache shapes required.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Round", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "STFT", status: "Unsupported", scope: "WebNN has no corresponding signal-processing primitive.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Scan", status: "Unsupported", scope: "No WebNN runtime loop/control-flow representation.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ScatterElements", status: "Direct WebNN", scope: "Direct WebNN scatterElements; only reduction=none.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ScatterND", status: "Direct WebNN", scope: "Direct WebNN scatterND; only reduction=none.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Selu", status: "Decomposed", scope: "Elementwise decomposition; no native WebNN selu.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceAt", status: "Pattern-only", scope: "Only a constant index into the converter's SplitToSequence pseudo-sequence.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceConstruct", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceEmpty", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceErase", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceInsert", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceLength", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SequenceMap", status: "Unsupported", scope: "WebNN has no sequence operand type.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Shape", status: "Folded/static", scope: "Resolved from known graph shape metadata.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Shrink", status: "Decomposed", scope: "Elementwise comparison/select decomposition; no native WebNN shrink.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sigmoid", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sign", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sin", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sinh", status: "Decomposed", scope: "Elementwise decomposition; WebNN has no native sinh.", perf_impact: "moderate", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Size", status: "Unsupported", scope: "No current executable lowering.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Slice", status: "Direct WebNN", scope: "Direct WebNN slice/reverse; starts, sizes, axes, and steps must resolve at build time. Positive steps are preserved as strides; negative steps support only the full-axis reverse pattern.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Softmax", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SoftmaxCrossEntropyLoss", status: "Unsupported", scope: "No current inference lowering; exporters may remove applicable training artifacts.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Softplus", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Softsign", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SpaceToDepth", status: "Unsupported", scope: "No current lowering; the required vision/spatial primitive is absent or not implemented.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Split", status: "Direct WebNN", scope: "Direct WebNN split; split sizes or the split dimension must be resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "SplitToSequence", status: "Pattern-only", scope: "Only the static split pattern consumed by constant-index SequenceAt; no sequence graph output.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sqrt", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Squeeze", status: "Direct WebNN", scope: "Direct WebNN squeeze; axes must be resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "StringConcat", status: "Unsupported", scope: "WebNN has no string tensor/operation support.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "StringNormalizer", status: "Unsupported", scope: "WebNN has no string tensor/operation support.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "StringSplit", status: "Unsupported", scope: "WebNN has no string tensor/operation support.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sub", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Sum", status: "Decomposed", scope: "Variadic binary fold; operation order and floating-point rounding can differ.", perf_impact: "not inferred", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Swish", status: "Decomposed", scope: "Sigmoid/multiply decomposition; no native WebNN swish.", perf_impact: "minimal", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Tan", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Tanh", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "TensorScatter", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "TfIdfVectorizer", status: "Unsupported", scope: "WebNN has no string tensor/operation support.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "ThresholdedRelu", status: "Decomposed", scope: "Comparison/select decomposition; no native WebNN thresholdedRelu.", perf_impact: "minimal", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Tile", status: "Direct WebNN", scope: "Direct WebNN tile; repeats must be constant.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "TopK", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Transpose", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Trilu", status: "Direct WebNN", scope: "Direct WebNN triangular; k must be resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Unique", status: "Unsupported", scope: "No current lowering; missing primitive and/or data-dependent output shape.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Unsqueeze", status: "Direct WebNN", scope: "Direct WebNN unsqueeze; axes must be resolvable.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Where", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "Xor", status: "Direct WebNN", scope: "Direct standardized WebNN primitive; handler schema checks and WebNN operand-type limits apply.", perf_impact: "", perf_limited: false, domain: "ai.onnx" },
    OperatorCapability { name: "GatherBlockQuantized", status: "Decomposed", scope: "Gathers packed rows/scales/zero points and dequantizes only the selected slice for supported 2-D axis-0 tables; other layouts fall back at conversion time.", perf_impact: "moderate", perf_limited: false, domain: "com.microsoft" },
    OperatorCapability { name: "GroupQueryAttention", status: "Decomposed", scope: "Static attention/cache subgraph. WebNN has no fused attention; runtime sequence metadata is ignored and softcap/local-window modes are rejected. Fused backend kernels and the portable decomposition need not be bit-identical, so an immediately following dynamic quantizer can amplify legal rounding differences.", perf_impact: "high", perf_limited: true, domain: "com.microsoft" },
    OperatorCapability { name: "MatMulBnb4", status: "Decomposed", scope: "Packed NF4/FP4 block path where representable; incompatible tails become dense constants, losing low-bit memory/performance benefits.", perf_impact: "high", perf_limited: true, domain: "com.microsoft" },
    OperatorCapability { name: "MatMulNBits", status: "Decomposed", scope: "Supported 4/8-bit constant layouts dequantize into a matmul path; g_idx is rejected. WebNN has no low-bit matmul kernel.", perf_impact: "high", perf_limited: true, domain: "com.microsoft" },
    OperatorCapability { name: "MoE", status: "Decomposed", scope: "All experts execute densely because WebNN has no TopK or sparse dispatch. Approximate compute overhead is num_experts/k; supported activation/fusion forms only.", perf_impact: "high", perf_limited: true, domain: "com.microsoft" },
    OperatorCapability { name: "QMoE", status: "Decomposed", scope: "Supported 4/8-bit expert weights feed the same dense all-expert lowering; block-layout fallbacks may dequantize at conversion time.", perf_impact: "high", perf_limited: true, domain: "com.microsoft" },
    OperatorCapability { name: "SimplifiedLayerNormalization", status: "Decomposed", scope: "RMS-style normalization decomposition; no matching WebNN fused primitive.", perf_impact: "moderate", perf_limited: false, domain: "com.microsoft" },
    OperatorCapability { name: "SkipSimplifiedLayerNormalization", status: "Decomposed", scope: "Residual/bias plus simplified normalization decomposition; no matching WebNN fused primitive.", perf_impact: "moderate", perf_limited: false, domain: "com.microsoft" },
];

pub fn capability(op_type: &str) -> Option<&'static OperatorCapability> {
    OPERATOR_CAPABILITIES
        .iter()
        .find(|entry| entry.name == op_type)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoweringPath {
    Direct,
    Decomposed,
    Folded,
    PatternOnly,
}

pub fn catalog_path(op_type: &str) -> Option<LoweringPath> {
    match capability(op_type)?.status {
        "Direct WebNN" => Some(LoweringPath::Direct),
        "Decomposed" => Some(LoweringPath::Decomposed),
        "Folded/static" => Some(LoweringPath::Folded),
        "Pattern-only" => Some(LoweringPath::PatternOnly),
        _ => None,
    }
}

#[derive(Clone)]
struct NodeAssessment {
    domain: String,
    name: String,
    op: String,
    path: Option<LoweringPath>,
    reason: Option<String>,
    attempted: bool,
}

#[derive(Clone, Copy, Ord, PartialOrd, Eq, PartialEq)]
#[repr(usize)]
enum SupportClass {
    Fully,
    Partially,
    Unsupported,
}

impl SupportClass {
    fn from_path(path: Option<LoweringPath>) -> Self {
        match path {
            Some(LoweringPath::Direct) => Self::Fully,
            Some(_) => Self::Partially,
            None => Self::Unsupported,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Self::Fully => "fully",
            Self::Partially => "partially",
            Self::Unsupported => "unsupported",
        }
    }
}

struct ReportSummary {
    counts: [usize; 3],
    unique_counts: [usize; 3],
    unique_total: usize,
    perf_limited_nodes: usize,
    perf_limited_names: BTreeSet<(String, String)>,
}

/// Tracks the post-specialization graph and emits a complete assessment even on early failure.
pub struct OperatorReport {
    nodes: Vec<NodeAssessment>,
    enabled: bool,
}

impl OperatorReport {
    pub fn new(nodes: &[NodeProto]) -> Self {
        Self {
            nodes: nodes
                .iter()
                .enumerate()
                .map(|(index, node)| NodeAssessment {
                    domain: if node.domain.is_empty() {
                        "ai.onnx".into()
                    } else {
                        node.domain.clone()
                    },
                    name: if node.name.is_empty() {
                        format!("<unnamed#{index}>")
                    } else {
                        node.name.clone()
                    },
                    op: node.op_type.clone(),
                    path: None,
                    reason: None,
                    attempted: false,
                })
                .collect(),
            enabled: std::env::var_os("O2W_OPERATOR_REPORT").is_some(),
        }
    }

    pub fn converted(&mut self, index: usize, path: LoweringPath) {
        self.nodes[index].path = Some(path);
        self.nodes[index].attempted = true;
    }

    pub fn failed(&mut self, index: usize, reason: String) {
        self.nodes[index].reason = Some(reason);
        self.nodes[index].attempted = true;
    }

    fn summary(&self) -> ReportSummary {
        let mut counts = [0usize; 3];
        let mut unique: BTreeMap<&str, SupportClass> = BTreeMap::new();
        let mut perf_limited_nodes = 0;
        let mut perf_limited_names = BTreeSet::new();
        for node in &self.nodes {
            if node.path == Some(LoweringPath::Decomposed)
                && capability(&node.op).is_some_and(|entry| entry.perf_limited)
            {
                perf_limited_nodes += 1;
                perf_limited_names.insert((node.domain.clone(), node.op.clone()));
            }
            let class = match (node.path, node.reason.as_ref()) {
                (Some(path), _) => SupportClass::from_path(Some(path)),
                (None, Some(_)) => SupportClass::Unsupported,
                (None, None) => SupportClass::from_path(catalog_path(&node.op)),
            };
            counts[class as usize] += 1;
            unique
                .entry(&node.op)
                .and_modify(|previous| *previous = (*previous).max(class))
                .or_insert(class);
        }
        let mut unique_counts = [0usize; 3];
        for class in unique.values() {
            unique_counts[*class as usize] += 1;
        }
        ReportSummary {
            counts,
            unique_counts,
            unique_total: unique.len(),
            perf_limited_nodes,
            perf_limited_names,
        }
    }

    pub fn print(&self) {
        if !self.enabled {
            return;
        }
        let summary = self.summary();
        let total = self.nodes.len();
        let percent = |part: usize, whole: usize| {
            if whole == 0 {
                0.0
            } else {
                part as f64 * 100.0 / whole as f64
            }
        };
        println!();
        println!("Operator support (post-fold/specialization):");
        println!("  Legend: fully = direct; partially = decomposed/folded/pattern-only; unsupported = unavailable/rejected.");
        println!("  Performance-limited = known major runtime cost; a subset of partially supported, not a failure.");
        for class in [
            SupportClass::Fully,
            SupportClass::Partially,
            SupportClass::Unsupported,
        ] {
            let index = class as usize;
            let label = class.label();
            println!(
                "  {label}: {} nodes ({:.1}%), {} unique ops ({:.1}%)",
                summary.counts[index],
                percent(summary.counts[index], total),
                summary.unique_counts[index],
                percent(summary.unique_counts[index], summary.unique_total)
            );
        }
        println!(
            "  performance-limited (within partially): {} nodes ({:.1}% of total, {:.1}% of partial), {} unique ops",
            summary.perf_limited_nodes,
            percent(summary.perf_limited_nodes, total),
            percent(summary.perf_limited_nodes, summary.counts[SupportClass::Partially as usize]),
            summary.perf_limited_names.len()
        );
        for (domain, op) in &summary.perf_limited_names {
            if let Some(entry) = capability(op) {
                println!("    - {domain}::{op}: {}", entry.scope);
            }
        }
        let pending = self.nodes.iter().filter(|node| !node.attempted).count();
        if pending != 0 {
            println!("  not yet converted: {pending} nodes (catalog assessment)");
        }
        for node in &self.nodes {
            if let Some(reason) = &node.reason {
                println!(
                    "  rejected: {}::{} [{}] — {}",
                    node.domain, node.op, node.name, reason
                );
            } else if !node.attempted
                && capability(&node.op).is_none_or(|entry| entry.status == "Unsupported")
            {
                let reason = capability(&node.op).map_or("no catalog entry", |entry| entry.scope);
                println!(
                    "  unsupported: {}::{} [{}] — {}",
                    node.domain, node.op, node.name, reason
                );
            }
        }
        println!();
    }
}

impl Drop for OperatorReport {
    fn drop(&mut self) {
        self.print();
    }
}

pub fn report_eliminated_source_nodes(
    before: &[(String, String, String, Vec<String>)],
    after: &[NodeProto],
) {
    if std::env::var_os("O2W_OPERATOR_REPORT").is_none() {
        return;
    }
    let remaining: BTreeSet<_> = after
        .iter()
        .map(|node| {
            (
                node.name.as_str(),
                node.op_type.as_str(),
                node.domain.as_str(),
                node.output.as_slice(),
            )
        })
        .collect();
    let eliminated: Vec<_> = before
        .iter()
        .filter(|(name, op, domain, outputs)| {
            !remaining.contains(&(
                name.as_str(),
                op.as_str(),
                domain.as_str(),
                outputs.as_slice(),
            ))
        })
        .collect();
    println!(
        "Eliminated source nodes (folding/specialization/pruning): {}",
        eliminated.len()
    );
    let verbose = std::env::var_os("O2W_OPERATOR_REPORT_VERBOSE").is_some();
    let mut groups: BTreeMap<(&str, &str), (usize, &str)> = BTreeMap::new();
    for (name, op, domain, _) in &eliminated {
        let domain = if domain.is_empty() {
            "ai.onnx"
        } else {
            domain.as_str()
        };
        let name = if name.is_empty() {
            "<unnamed>"
        } else {
            name.as_str()
        };
        let group = groups.entry((domain, op.as_str())).or_insert((0, name));
        group.0 += 1;
        if verbose {
            println!("  eliminated: {domain}::{op} [{name}]");
        }
    }
    if !verbose {
        for ((domain, op), (count, example)) in groups {
            println!(
                "  {domain}::{op}: {count} eliminated (e.g. [{example}]); use --debug for names"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::ops::OpRegistry;

    #[test]
    fn catalog_covers_opset_26_and_matches_handlers() {
        let standard: Vec<_> = OPERATOR_CAPABILITIES
            .iter()
            .filter(|entry| entry.domain == "ai.onnx")
            .collect();
        assert_eq!(standard.len(), 198);
        let names: BTreeSet<_> = standard.iter().map(|entry| entry.name).collect();
        assert_eq!(names.len(), 198);
        assert_eq!(
            standard
                .iter()
                .filter(|entry| entry.status == "Direct WebNN")
                .count(),
            96
        );
        assert_eq!(
            standard
                .iter()
                .filter(|entry| entry.status != "Unsupported")
                .count(),
            129
        );
        assert_eq!(OPERATOR_CAPABILITIES.len(), 206);
        let registry = OpRegistry::new();
        let mut handled = BTreeSet::new();
        for handler in &registry.handlers {
            assert!(
                !handler.supported_ops().is_empty(),
                "registered handler has no operator list"
            );
            for &name in handler.supported_ops() {
                assert!(handled.insert(name), "multiple handlers claim {name}");
                assert!(
                    capability(name).is_some_and(|entry| entry.status != "Unsupported"),
                    "handler supports {name} without a supported catalog entry"
                );
            }
        }
        for entry in OPERATOR_CAPABILITIES {
            assert!(
                matches!(
                    entry.status,
                    "Direct WebNN"
                        | "Decomposed"
                        | "Folded/static"
                        | "Pattern-only"
                        | "Unsupported"
                ),
                "{}",
                entry.name
            );
            assert!(
                entry.status == "Decomposed" || entry.perf_impact.is_empty(),
                "{}",
                entry.name
            );
            if entry.status == "Decomposed" {
                assert!(matches!(
                    entry.perf_impact,
                    "minimal" | "moderate" | "high" | "not inferred"
                ));
            }
            if entry.perf_limited {
                assert_eq!(entry.status, "Decomposed", "{}", entry.name);
                assert_eq!(entry.perf_impact, "high", "{}", entry.name);
            }
            if entry.status != "Unsupported" && entry.name != "If" {
                assert!(
                    registry
                        .handlers
                        .iter()
                        .any(|handler| handler.supports(entry.name)),
                    "catalog advertises {} without handler",
                    entry.name
                );
            } else if entry.status == "Unsupported" {
                assert!(
                    !registry
                        .handlers
                        .iter()
                        .any(|handler| handler.supports(entry.name)),
                    "catalog marks handled operator {} unsupported",
                    entry.name
                );
            }
        }
    }

    #[test]
    fn performance_limited_inventory_is_explicit() {
        let names: BTreeSet<_> = OPERATOR_CAPABILITIES
            .iter()
            .filter(|entry| entry.perf_limited)
            .map(|entry| entry.name)
            .collect();
        assert_eq!(
            names,
            BTreeSet::from([
                "ConvInteger",
                "GroupQueryAttention",
                "MatMulBnb4",
                "MatMulInteger",
                "MatMulNBits",
                "MoE",
                "QMoE",
            ])
        );
    }

    #[test]
    fn performance_count_uses_converted_lowering_not_name_alone() {
        let node = |op: &str| NodeProto {
            op_type: op.into(),
            domain: "com.microsoft".into(),
            ..NodeProto::default()
        };
        let mut report = OperatorReport::new(&[
            node("QMoE"),
            node("QMoE"),
            node("MoE"),
            node("MatMulNBits"),
            node("Constant"),
        ]);
        report.converted(0, LoweringPath::Decomposed);
        report.converted(1, LoweringPath::Decomposed);
        report.converted(2, LoweringPath::Direct);
        report.converted(4, LoweringPath::Folded);
        let summary = report.summary();
        assert_eq!(summary.counts, [1, 4, 0]);
        assert_eq!(summary.perf_limited_nodes, 2);
        assert_eq!(summary.perf_limited_names.len(), 1);
    }
}
