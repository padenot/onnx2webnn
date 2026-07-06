/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    map_op_result, record_node_output, reshape_with_shape, slice_with_params,
};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{MLDimension, MLSplitOptions, MLTransposeOptions};
use rustnn::DataType;

pub struct LstmHandler;

impl OpHandler for LstmHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "LSTM"
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        convert_lstm(node, context, b)
    }
}

fn convert_lstm(
    node: &NodeProto,
    context: &ConversionContext,
    b: &mut OnnxBuilder<'_, '_, '_>,
) -> Result<ConversionResult, OnnxError> {
    let node_name = if node.name.is_empty() {
        node.output
            .first()
            .map(|s| sanitize_identifier(s))
            .unwrap_or_else(|| "LSTM".to_string())
    } else {
        node.name.clone()
    };
    let label = sanitize_identifier(&node_name);
    let inputs = node.input.as_slice();
    if inputs.len() < 7 {
        return Err(OnnxError::InvalidShape(format!(
            "LSTM expects X, W, R, B, sequence_lens, initial_h and initial_c for this lowering, got {} inputs",
            inputs.len()
        )));
    }

    let x_name = required_input(inputs, 0, "X")?;
    let w_name = required_input(inputs, 1, "W")?;
    let r_name = required_input(inputs, 2, "R")?;
    let b_name = required_input(inputs, 3, "B")?;
    reject_present_input(inputs, 4, "sequence_lens")?;
    let h0_name = required_input(inputs, 5, "initial_h")?;
    let c0_name = required_input(inputs, 6, "initial_c")?;
    reject_present_input(inputs, 7, "P")?;

    let attrs = LstmAttrs::from_node(node)?;
    if attrs.direction != "forward" {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("{} uses direction '{}'", node_name, attrs.direction),
        });
    }
    if attrs.layout != 0 {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("{node_name} uses layout {}", attrs.layout),
        });
    }
    if attrs.input_forget != 0 {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("{node_name} uses input_forget"),
        });
    }
    if attrs.has_custom_activations || attrs.has_clip || attrs.has_activation_params {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("{node_name} uses non-default LSTM activations"),
        });
    }

    let x_shape = context
        .resolve_shape(x_name)
        .cloned()
        .ok_or_else(|| OnnxError::InvalidShape(format!("LSTM input '{x_name}' has no shape")))?;
    if x_shape.len() != 3 {
        return Err(OnnxError::InvalidShape(format!(
            "LSTM input X must be [seq_len, batch, input_size], got {x_shape:?}"
        )));
    }
    let seq_len = to_u32(x_shape[0], "seq_len")?;
    let batch = to_u32(x_shape[1], "batch")?;
    let input_size = to_u32(x_shape[2], "input_size")?;
    if seq_len != 1 {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("{node_name} has seq_len {seq_len}; this lowering handles one step"),
        });
    }

    let w_shape = initializer_shape(context, w_name)?;
    let r_shape = initializer_shape(context, r_name)?;
    let b_shape = initializer_shape(context, b_name)?;
    let hidden = attrs.hidden_size;
    let gates = hidden.checked_mul(4).ok_or_else(|| {
        OnnxError::InvalidShape("LSTM hidden size overflows gate count".to_string())
    })?;
    validate_initializer_shape(w_name, w_shape, &[1, gates, input_size])?;
    validate_initializer_shape(r_name, r_shape, &[1, gates, hidden])?;
    validate_initializer_shape(b_name, b_shape, &[1, gates * 2])?;

    validate_state_shape(context, h0_name, batch, hidden, "initial_h")?;
    validate_state_shape(context, c0_name, batch, hidden, "initial_c")?;

    let x = b.resolve_operand(x_name)?;
    let w = b.resolve_operand(w_name)?;
    let r = b.resolve_operand(r_name)?;
    let bias = b.resolve_operand(b_name)?;
    let h0 = b.resolve_operand(h0_name)?;
    let c0 = b.resolve_operand(c0_name)?;

    let w_dir = slice_then_reshape(
        b,
        w,
        &format!("{label}_W_forward"),
        &[0, 0, 0],
        &[1, gates, input_size],
        &[gates, input_size],
    )?;
    let r_dir = slice_then_reshape(
        b,
        r,
        &format!("{label}_R_forward"),
        &[0, 0, 0],
        &[1, gates, hidden],
        &[gates, hidden],
    )?;
    let b_dir = slice_then_reshape(
        b,
        bias,
        &format!("{label}_B_forward"),
        &[0, 0],
        &[1, gates * 2],
        &[gates * 2],
    )?;

    let w_t = transpose_2d(b, w_dir, &format!("{label}_W_transposed"))?;
    let r_t = transpose_2d(b, r_dir, &format!("{label}_R_transposed"))?;

    let x_gates = map_op_result(b.builder.matmul_with_options(
        x,
        w_t,
        OnnxBuilder::labeled_options(&format!("{label}_XW")),
    ))?;
    let h_gates = map_op_result(b.builder.matmul_with_options(
        h0,
        r_t,
        OnnxBuilder::labeled_options(&format!("{label}_HR")),
    ))?;
    let gates_without_bias = map_op_result(b.builder.add_with_options(
        x_gates,
        h_gates,
        OnnxBuilder::labeled_options(&format!("{label}_XW_plus_HR")),
    ))?;

    let bias_parts = map_op_result(b.builder.split_with_options(
        b_dir,
        &[gates, gates],
        MLSplitOptions {
            label: format!("{label}_split_bias"),
            axis: 0,
        },
    ))?;
    let bias_sum = map_op_result(b.builder.add_with_options(
        bias_parts[0],
        bias_parts[1],
        OnnxBuilder::labeled_options(&format!("{label}_bias_sum")),
    ))?;
    let bias_sum = reshape_with_shape(
        b,
        bias_sum,
        &format!("{label}_bias_broadcast"),
        dims(&[1, 1, gates]),
    )?;
    let all_gates = map_op_result(b.builder.add_with_options(
        gates_without_bias,
        bias_sum,
        OnnxBuilder::labeled_options(&format!("{label}_gates")),
    ))?;

    let gate_parts = map_op_result(b.builder.split_with_options(
        all_gates,
        &[hidden, hidden, hidden, hidden],
        MLSplitOptions {
            label: format!("{label}_split_gates"),
            axis: 2,
        },
    ))?;

    let input_gate = sigmoid(b, gate_parts[0], &format!("{label}_input_gate"))?;
    let output_gate = sigmoid(b, gate_parts[1], &format!("{label}_output_gate"))?;
    let forget_gate = sigmoid(b, gate_parts[2], &format!("{label}_forget_gate"))?;
    let cell_gate = tanh(b, gate_parts[3], &format!("{label}_cell_gate"))?;

    let forget_term = map_op_result(b.builder.mul_with_options(
        forget_gate,
        c0,
        OnnxBuilder::labeled_options(&format!("{label}_forget_term")),
    ))?;
    let input_term = map_op_result(b.builder.mul_with_options(
        input_gate,
        cell_gate,
        OnnxBuilder::labeled_options(&format!("{label}_input_term")),
    ))?;
    let c_new = map_op_result(b.builder.add_with_options(
        forget_term,
        input_term,
        OnnxBuilder::labeled_options(&format!("{label}_Y_c")),
    ))?;
    let c_new_tanh = tanh(b, c_new, &format!("{label}_Y_c_tanh"))?;
    let h_new = map_op_result(b.builder.mul_with_options(
        output_gate,
        c_new_tanh,
        OnnxBuilder::labeled_options(&format!("{label}_Y_h")),
    ))?;
    let y = reshape_with_shape(
        b,
        h_new,
        &format!("{label}_Y"),
        dims(&[seq_len, 1, batch, hidden]),
    )?;

    let mut result = ConversionResult::default();
    let output_type = value_type(context, x_name).unwrap_or(DataType::Float32);
    record_lstm_output(
        b,
        node,
        0,
        &format!("{label}_Y"),
        y,
        &mut result,
        output_type,
        &[seq_len, 1, batch, hidden],
    );
    record_lstm_output(
        b,
        node,
        1,
        &format!("{label}_Y_h"),
        h_new,
        &mut result,
        output_type,
        &[1, batch, hidden],
    );
    record_lstm_output(
        b,
        node,
        2,
        &format!("{label}_Y_c"),
        c_new,
        &mut result,
        output_type,
        &[1, batch, hidden],
    );

    Ok(result)
}

#[derive(Default)]
struct LstmAttrs {
    hidden_size: u32,
    direction: String,
    layout: i64,
    input_forget: i64,
    has_custom_activations: bool,
    has_activation_params: bool,
    has_clip: bool,
}

impl LstmAttrs {
    fn from_node(node: &NodeProto) -> Result<Self, OnnxError> {
        let mut attrs = LstmAttrs {
            direction: "forward".to_string(),
            ..Default::default()
        };
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "hidden_size" => attrs.hidden_size = to_u32(attr.i, "hidden_size")?,
                "direction" if !attr.s.is_empty() => {
                    attrs.direction = String::from_utf8_lossy(&attr.s).to_string();
                }
                "layout" => attrs.layout = attr.i,
                "input_forget" => attrs.input_forget = attr.i,
                "activations" => attrs.has_custom_activations = !attr.strings.is_empty(),
                "activation_alpha" | "activation_beta" => {
                    attrs.has_activation_params = !attr.floats.is_empty();
                }
                "clip" => attrs.has_clip = true,
                _ => {}
            }
        }
        if attrs.hidden_size == 0 {
            return Err(OnnxError::MissingAttribute {
                attr: "hidden_size".to_string(),
                op: "LSTM".to_string(),
            });
        }
        Ok(attrs)
    }
}

fn required_input<'a>(
    inputs: &'a [String],
    index: usize,
    name: &str,
) -> Result<&'a str, OnnxError> {
    inputs
        .get(index)
        .map(|s| s.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| OnnxError::InvalidShape(format!("LSTM missing required input {name}")))
}

fn reject_present_input(inputs: &[String], index: usize, name: &str) -> Result<(), OnnxError> {
    if inputs.get(index).map(|s| !s.is_empty()).unwrap_or(false) {
        return Err(OnnxError::UnsupportedOp {
            op: "LSTM".to_string(),
            node: format!("input {name} is not supported by primitive lowering"),
        });
    }
    Ok(())
}

fn initializer_shape<'a>(
    context: &'a ConversionContext<'_>,
    name: &str,
) -> Result<&'a [i64], OnnxError> {
    context
        .initializers
        .get(name)
        .or_else(|| context.initializers.get(&sanitize_identifier(name)))
        .map(|t: &&TensorProto| t.dims.as_slice())
        .ok_or_else(|| OnnxError::InvalidShape(format!("LSTM initializer '{name}' not found")))
}

fn validate_initializer_shape(
    name: &str,
    actual: &[i64],
    expected: &[u32],
) -> Result<(), OnnxError> {
    let expected_i64: Vec<i64> = expected.iter().map(|&v| v as i64).collect();
    if actual != expected_i64.as_slice() {
        return Err(OnnxError::InvalidShape(format!(
            "LSTM initializer '{name}' shape {actual:?} does not match expected {expected_i64:?}"
        )));
    }
    Ok(())
}

fn validate_state_shape(
    context: &ConversionContext<'_>,
    name: &str,
    batch: u32,
    hidden: u32,
    input_label: &str,
) -> Result<(), OnnxError> {
    let expected = vec![1, batch as i64, hidden as i64];
    if let Some(shape) = context.resolve_shape(name) {
        if shape != &expected {
            return Err(OnnxError::InvalidShape(format!(
                "LSTM {input_label} shape {shape:?} does not match expected {expected:?}"
            )));
        }
    }
    Ok(())
}

fn slice_then_reshape(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
    starts: &[u32],
    slice_sizes: &[u32],
    new_shape: &[u32],
) -> Result<MLOperand, OnnxError> {
    let sliced = slice_with_params(b, input, label, starts, &dims(slice_sizes))?;
    reshape_with_shape(b, sliced, &format!("{label}_reshape"), dims(new_shape))
}

fn transpose_2d(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    b.builder
        .transpose_with_options(
            input,
            MLTransposeOptions {
                label: label.to_string(),
                permutation: vec![1, 0],
            },
        )
        .map_err(map_op_error)
}

fn sigmoid(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    map_op_result(
        b.builder
            .sigmoid_with_options(input, OnnxBuilder::labeled_options(label)),
    )
}

fn tanh(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
) -> Result<MLOperand, OnnxError> {
    map_op_result(
        b.builder
            .tanh_with_options(input, OnnxBuilder::labeled_options(label)),
    )
}

fn record_lstm_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    node: &NodeProto,
    index: usize,
    label: &str,
    op: MLOperand,
    result: &mut ConversionResult,
    dtype: DataType,
    shape: &[u32],
) {
    let Some(output) = node.output.get(index) else {
        return;
    };
    if output.is_empty() {
        return;
    }
    record_node_output(b, output, label, op);
    result.output_types.insert(output.clone(), dtype);
    result
        .output_shapes
        .insert(output.clone(), shape.iter().map(|&d| d as i64).collect());
}

fn value_type(context: &ConversionContext<'_>, name: &str) -> Option<DataType> {
    context
        .value_types
        .get(name)
        .copied()
        .or_else(|| context.value_types.get(&sanitize_identifier(name)).copied())
}

fn to_u32(value: i64, label: &str) -> Result<u32, OnnxError> {
    u32::try_from(value).ok().filter(|v| *v > 0).ok_or_else(|| {
        OnnxError::InvalidShape(format!("LSTM {label} must be positive, got {value}"))
    })
}

fn dims(values: &[u32]) -> Vec<MLDimension> {
    values.iter().copied().map(MLDimension::Static).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx::ops::{convert_handler_with_context, empty_value_shape_dims};
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto, TensorProto_DataType};
    use std::collections::HashMap;

    fn lstm_node(inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: "LSTM".to_string(),
            name: "lstm".to_string(),
            input: inputs.into_iter().map(str::to_string).collect(),
            output: outputs.into_iter().map(str::to_string).collect(),
            attribute: vec![AttributeProto {
                name: "hidden_size".to_string(),
                i: 2,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn float_tensor(name: &str, dims: Vec<i64>) -> TensorProto {
        let numel = dims.iter().map(|&d| d as usize).product::<usize>();
        TensorProto {
            name: name.to_string(),
            data_type: TensorProto_DataType::Float as i32,
            dims,
            raw_data: vec![0u8; numel * std::mem::size_of::<f32>()],
            ..Default::default()
        }
    }

    fn conversion_error_for(node: &NodeProto) -> String {
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("X".to_string(), vec![1, 1, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let handler = LstmHandler;
        convert_handler_with_context(&handler, node, &context)
            .expect_err("unsupported LSTM form should be rejected")
            .to_string()
    }

    #[test]
    fn supports_lstm() {
        let handler = LstmHandler;
        assert!(handler.supports("LSTM"));
        assert!(!handler.supports("GRU"));
    }

    #[test]
    fn converts_one_step_forward_lstm_and_reports_shapes() {
        let handler = LstmHandler;
        let node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );

        let w = float_tensor("W", vec![1, 8, 3]);
        let r = float_tensor("R", vec![1, 8, 2]);
        let b = float_tensor("B", vec![1, 16]);
        let mut initializers = HashMap::new();
        initializers.insert("W".to_string(), &w);
        initializers.insert("R".to_string(), &r);
        initializers.insert("B".to_string(), &b);

        let mut value_shapes = HashMap::new();
        value_shapes.insert("X".to_string(), vec![1, 1, 3]);
        value_shapes.insert("H0".to_string(), vec![1, 1, 2]);
        value_shapes.insert("C0".to_string(), vec![1, 1, 2]);
        let mut value_types = HashMap::new();
        value_types.insert("X".to_string(), DataType::Float32);
        value_types.insert("H0".to_string(), DataType::Float32);
        value_types.insert("C0".to_string(), DataType::Float32);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result = convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.output_shapes.get("Y"), Some(&vec![1, 1, 1, 2]));
        assert_eq!(result.output_shapes.get("Y_h"), Some(&vec![1, 1, 2]));
        assert_eq!(result.output_shapes.get("Y_c"), Some(&vec![1, 1, 2]));
        assert_eq!(result.output_types.get("Y"), Some(&DataType::Float32));
    }

    #[test]
    fn rejects_multi_step_lstm() {
        let handler = LstmHandler;
        let node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("X".to_string(), vec![2, 1, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let err = convert_handler_with_context(&handler, &node, &context).unwrap_err();
        assert!(err.to_string().contains("handles one step"));
    }

    #[test]
    fn rejects_bidirectional_lstm() {
        let mut node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );
        node.attribute.push(AttributeProto {
            name: "direction".to_string(),
            s: b"bidirectional".to_vec(),
            ..Default::default()
        });

        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("X".to_string(), vec![1, 1, 3]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let value_types = HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let handler = LstmHandler;
        let err = convert_handler_with_context(&handler, &node, &context).unwrap_err();
        assert!(err.to_string().contains("bidirectional"));
    }

    #[test]
    fn rejects_layout_one_lstm() {
        let mut node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );
        node.attribute.push(AttributeProto {
            name: "layout".to_string(),
            i: 1,
            ..Default::default()
        });

        assert!(conversion_error_for(&node).contains("layout 1"));
    }

    #[test]
    fn rejects_lstm_with_peephole_input() {
        let node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0", "P"],
            vec!["Y", "Y_h", "Y_c"],
        );

        assert!(conversion_error_for(&node).contains("P"));
    }

    #[test]
    fn rejects_lstm_with_custom_activations() {
        let mut node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );
        node.attribute.push(AttributeProto {
            name: "activations".to_string(),
            strings: vec![b"Relu".to_vec(), b"Tanh".to_vec(), b"Sigmoid".to_vec()],
            ..Default::default()
        });

        assert!(conversion_error_for(&node).contains("non-default LSTM activations"));
    }

    #[test]
    fn rejects_lstm_with_clip() {
        let mut node = lstm_node(
            vec!["X", "W", "R", "B", "", "H0", "C0"],
            vec!["Y", "Y_h", "Y_c"],
        );
        node.attribute.push(AttributeProto {
            name: "clip".to_string(),
            f: 1.0,
            ..Default::default()
        });

        assert!(conversion_error_for(&node).contains("non-default LSTM activations"));
    }
}
