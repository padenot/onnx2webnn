// Quantized inference operators for onnx2webnn.
//
// DynamicQuantizeLinear is implemented as a spec-compliant WebNN composite using
// existing ops (reduceMin, reduceMax, arithmetic, quantizeLinear).
//
// ConvInteger and MatMulInteger are mapped to new WebNN extension ops proposed in
// https://github.com/webmachinelearning/webnn/issues/623 ("WebNN should support NPU
// and QDQ operations"). These ops are NOT yet in the W3C WebNN spec but are proposed
// there. The rustnn ONNX converter emits them as native ONNX ConvInteger/MatMulInteger
// nodes, so ORT and TensorRT-RTX execute them with hardware int8 arithmetic.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::conv::{build_conv2d_options, parse_conv_attrs};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::operator_options::{MLClampOptions, MLOperatorOptions, MLReduceOptions};
use rustnn::DataType;
use serde_json;

pub struct QuantizationHandler;

impl OpHandler for QuantizationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type,
            "DynamicQuantizeLinear" | "ConvInteger" | "MatMulInteger"
            | "QuantizeLinear" | "DequantizeLinear"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        match node.op_type.as_str() {
            "DynamicQuantizeLinear" => convert_dynamic_quantize_linear(node, context, b),
            "ConvInteger"           => convert_conv_integer(node, context, b),
            "MatMulInteger"         => convert_matmul_integer(node, b),
            "QuantizeLinear"        => convert_quantize_linear(node, b),
            "DequantizeLinear"      => convert_dequantize_linear(node, b),
            _ => unreachable!(),
        }
    }
}

// DynamicQuantizeLinear: spec-compliant WebNN composite.
//
// ONNX semantics:
//   x: float32[...] → y: uint8[...], scale: float32[], zero_point: uint8[]
//   scale = (max(x) - min(x)) / 255  (clamped ≥ ε to avoid div-by-zero)
//   zero_point = clamp(round(-min(x) / scale), 0, 255) as uint8
//   y = clamp(round(x / scale) + zero_point, 0, 255) as uint8
//
// Implementation: all intermediate values stay float32 in WebNN (quantizeLinear
// returns the underlying quantized type). WebNN's quantizeLinear takes (input, scale,
// zero_point) and handles the rounding and clamping.
fn convert_dynamic_quantize_linear(
    node: &NodeProto,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    let outputs = node.output.as_slice();
    if inputs.is_empty() || outputs.len() < 3 {
        return Err(OnnxError::InvalidShape(
            "DynamicQuantizeLinear: need 1 input, 3 outputs".into(),
        ));
    }

    let node_name = node.output.first()
        .map(|s| crate::onnx::convert::sanitize_identifier(s))
        .unwrap_or_else(|| "dql".to_string());

    let x = b.resolve_operand(&inputs[0])?;

    // x_min = reduceMin(x)
    let min_opts = MLReduceOptions { label: format!("{node_name}_reduce_min"), axes: None, keep_dimensions: false };
    let x_min = b.builder.reduce_min_with_options(x, min_opts).map_err(map_op_error)?;

    // x_max = reduceMax(x)
    let max_opts = MLReduceOptions { label: format!("{node_name}_reduce_max"), axes: None, keep_dimensions: false };
    let x_max = b.builder.reduce_max_with_options(x, max_opts).map_err(map_op_error)?;

    // range = x_max - x_min
    let range_opts = labeled_opts(&format!("{node_name}_range"));
    let range = b.builder.sub_with_options(x_max, x_min, range_opts)
        .map_err(map_op_error)?;

    // scale = range / 255  (constant 255.0 as scalar)
    let c255_name = format!("{node_name}_c255");
    b.register_constant_from_bytes(&c255_name, DataType::Float32, &[], &255.0f32.to_le_bytes())?;
    let c255 = b.resolve_operand(&c255_name)?;
    let scale_opts = labeled_opts(&format!("{node_name}_scale"));
    let scale = b.builder.div_with_options(range, c255, scale_opts)
        .map_err(map_op_error)?;

    // zero_point = clamp(round(-x_min / scale), 0, 255)
    // neg_min = 0 - x_min = x_min * -1
    let c0_name = format!("{node_name}_c0");
    b.register_constant_from_bytes(&c0_name, DataType::Float32, &[], &0.0f32.to_le_bytes())?;
    let c0 = b.resolve_operand(&c0_name)?;
    let neg_min_opts = labeled_opts(&format!("{node_name}_neg_min"));
    let neg_min = b.builder.sub_with_options(c0, x_min, neg_min_opts)
        .map_err(map_op_error)?;

    let zp_raw_opts = labeled_opts(&format!("{node_name}_zp_raw"));
    let zp_raw = b.builder.div_with_options(neg_min, scale, zp_raw_opts)
        .map_err(map_op_error)?;

    let zp_rounded_opts = labeled_opts(&format!("{node_name}_zp_rounded"));
    let zp_rounded = b.builder.round_even_with_options(zp_raw, zp_rounded_opts)
        .map_err(map_op_error)?;

    use rustnn::operator_options::MLClampOptions;
    let clamp_opts = MLClampOptions {
        label: format!("{node_name}_zp_clamp"),
        min_value: Some(serde_json::json!(0.0)),
        max_value: Some(serde_json::json!(255.0)),
    };
    let zp_clamped = b.builder.clamp_with_options(zp_rounded, clamp_opts)
        .map_err(map_op_error)?;

    // Cast zero_point to Uint8 so quantizeLinear emits a Uint8 output.
    use rustnn::MLOperandDataType;
    let zp_cast_opts = labeled_opts(&format!("{node_name}_zp_u8"));
    let zero_point = b.builder.cast_with_options(zp_clamped, MLOperandDataType::Uint8, zp_cast_opts)
        .map_err(map_op_error)?;

    // y = quantizeLinear(x, scale, zero_point: uint8) → uint8
    let y = b.builder.quantize_linear_with_zeropoint(x, scale, zero_point)
        .map_err(map_op_error)?;

    // Record all three outputs.
    let y_label   = crate::onnx::convert::sanitize_identifier(&outputs[0]);
    let sc_label  = crate::onnx::convert::sanitize_identifier(&outputs[1]);
    let zp_label  = crate::onnx::convert::sanitize_identifier(&outputs[2]);

    record_node_output(b, &outputs[0], &y_label, y);
    // scale and zero_point are already MLOperand handles; alias them.
    b.record_operand(&[&outputs[1], &sc_label], scale);
    b.record_operand(&[&outputs[2], &zp_label], zero_point);

    let mut result = ConversionResult::default();
    result.output_types.insert(outputs[0].to_string(), DataType::Uint8);
    result.output_types.insert(outputs[1].to_string(), DataType::Float32);
    result.output_types.insert(outputs[2].to_string(), DataType::Uint8);
    Ok(result)
}

// ConvInteger → WebNN ConvInteger (proposed in webnn#623).
// ONNX inputs: [x_uint8, w_int8, x_zero_point?, w_zero_point?]
// Attributes: same as ONNX Conv (dilations, group, kernel_shape, pads, strides)
fn convert_conv_integer(
    node: &NodeProto,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidShape("ConvInteger: need ≥ 2 inputs".into()));
    }

    let node_name = node.output.first()
        .map(|s| crate::onnx::convert::sanitize_identifier(s))
        .unwrap_or_else(|| "conv_integer".to_string());

    let x = b.resolve_operand(&inputs[0])?;
    let w = b.resolve_operand(&inputs[1])?;
    let x_zp = inputs.get(2).filter(|s| !s.is_empty())
        .map(|n| b.resolve_operand(n)).transpose()?;
    let w_zp = inputs.get(3).filter(|s| !s.is_empty())
        .map(|n| b.resolve_operand(n)).transpose()?;

    let attrs = parse_conv_attrs(node);
    let opts = build_conv2d_options(&attrs, &node_name)?;

    let output_label_str = output_label(node, &node_name);
    let out = b.builder.conv_integer(x, w, x_zp, w_zp, opts)
        .map_err(map_op_error)?;

    if let Some(onnx_out) = node.output.first() {
        record_node_output(b, onnx_out, &output_label_str, out);
    }

    let mut result = ConversionResult::default();
    if let Some(out_name) = node.output.first() {
        result.output_types.insert(out_name.to_string(), DataType::Int32);
    }
    Ok(result)
}

// MatMulInteger → WebNN MatMulInteger (proposed in webnn#623).
// ONNX inputs: [a_uint8/int8, b_int8, a_zero_point?, b_zero_point?]
fn convert_matmul_integer(
    node: &NodeProto,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidShape("MatMulInteger: need ≥ 2 inputs".into()));
    }

    let node_name = node.output.first()
        .map(|s| crate::onnx::convert::sanitize_identifier(s))
        .unwrap_or_else(|| "matmul_integer".to_string());

    let a = b.resolve_operand(&inputs[0])?;
    let bw = b.resolve_operand(&inputs[1])?;
    let a_zp = inputs.get(2).filter(|s| !s.is_empty())
        .map(|n| b.resolve_operand(n)).transpose()?;
    let b_zp = inputs.get(3).filter(|s| !s.is_empty())
        .map(|n| b.resolve_operand(n)).transpose()?;

    let output_label_str = output_label(node, &node_name);
    let opts = MLOperatorOptions { label: output_label_str.clone(), ..Default::default() };
    let out = b.builder.matmul_integer(a, bw, a_zp, b_zp, opts)
        .map_err(map_op_error)?;

    if let Some(onnx_out) = node.output.first() {
        record_node_output(b, onnx_out, &output_label_str, out);
    }

    let mut result = ConversionResult::default();
    if let Some(out_name) = node.output.first() {
        result.output_types.insert(out_name.to_string(), DataType::Int32);
    }
    Ok(result)
}

// QuantizeLinear: static-scale quantization. In the QDQ model these wrap
// each op with fixed scale/zero_point computed during calibration.
// ONNX: QuantizeLinear(x, y_scale, y_zero_point?) → y (int8/uint8)
// WebNN: quantizeLinear(x, scale, zero_point) — direct 1:1 mapping.
fn convert_quantize_linear(
    node: &NodeProto,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidShape("QuantizeLinear: need ≥ 2 inputs".into()));
    }
    let node_name = node.output.first()
        .map(|s| crate::onnx::convert::sanitize_identifier(s))
        .unwrap_or_else(|| "quantize_linear".to_string());

    let x     = b.resolve_operand(&inputs[0])?;
    let scale = b.resolve_operand(&inputs[1])?;
    let output_label = crate::onnx::convert::sanitize_identifier(
        node.output.first().map(|s| s.as_str()).unwrap_or(&node_name)
    );

    let out = if inputs.len() >= 3 && !inputs[2].is_empty() {
        let zp = b.resolve_operand(&inputs[2])?;
        let opts = labeled_opts(&output_label);
        b.builder.quantize_linear_with_zeropoint(x, scale, zp).map_err(map_op_error)?
    } else {
        let opts = labeled_opts(&output_label);
        b.builder.quantize_linear_with_options(x, scale, None, opts).map_err(map_op_error)?
    };

    if let Some(onnx_out) = node.output.first() {
        crate::onnx::builder_helpers::record_node_output(b, onnx_out, &output_label, out);
    }
    // Output dtype = zero_point dtype (uint8 or int8); let the converter infer it.
    Ok(ConversionResult::default())
}

// DequantizeLinear: static-scale dequantization.
// ONNX: DequantizeLinear(x, x_scale, x_zero_point?) → y (float32)
// WebNN: dequantizeLinear(x, scale, zero_point) — direct 1:1 mapping.
fn convert_dequantize_linear(
    node: &NodeProto,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() < 2 {
        return Err(OnnxError::InvalidShape("DequantizeLinear: need ≥ 2 inputs".into()));
    }
    let node_name = node.output.first()
        .map(|s| crate::onnx::convert::sanitize_identifier(s))
        .unwrap_or_else(|| "dequantize_linear".to_string());

    let x     = b.resolve_operand(&inputs[0])?;
    let scale = b.resolve_operand(&inputs[1])?;
    let output_label = crate::onnx::convert::sanitize_identifier(
        node.output.first().map(|s| s.as_str()).unwrap_or(&node_name)
    );

    let out = if inputs.len() >= 3 && !inputs[2].is_empty() {
        let zp = b.resolve_operand(&inputs[2])?;
        b.builder.dequantize_linear_with_zeropoint(x, scale, zp).map_err(map_op_error)?
    } else {
        let opts = labeled_opts(&output_label);
        b.builder.dequantize_linear_with_options(x, scale, None, opts).map_err(map_op_error)?
    };

    if let Some(onnx_out) = node.output.first() {
        crate::onnx::builder_helpers::record_node_output(b, onnx_out, &output_label, out);
    }
    let mut result = ConversionResult::default();
    if let Some(out_name) = node.output.first() {
        result.output_types.insert(out_name.to_string(), DataType::Float32);
    }
    Ok(result)
}

fn labeled_opts(label: &str) -> MLOperatorOptions {
    MLOperatorOptions { label: label.to_string(), ..Default::default() }
}
