/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ONNX Pad -> WebNN pad
//
// ONNX pads layout: [b0, b1, ..., bk, e0, e1, ..., ek]
// WebNN uses separate beginningPadding and endingPadding vectors.

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{ml_number_from_tensor, output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::{NodeProto, TensorProto, TensorProto_DataType};
use rustnn::operator_options::MLPadOptions;
use serde_json::Value;
use std::collections::HashMap;

pub struct PadHandler;

impl OpHandler for PadHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "Pad"
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };
        self.convert_pad(node, &node_name, context, b)
    }
}

impl PadHandler {
    fn convert_pad(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Pad expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let rank = context
            .input_rank(&inputs[0])
            .ok_or_else(|| OnnxError::InvalidShape("Pad requires known input rank".to_string()))?;

        let onnx_pads = read_onnx_pads(node, context, rank)?;
        let (beginning_padding, ending_padding) = split_onnx_pads(&onnx_pads, rank)?;

        let mode = read_mode(node);
        let mut pad_opts = MLPadOptions {
            label: output_name.clone(),
            mode,
            value: None,
        };
        if pad_opts.mode == "constant" {
            if let Some(value) = read_constant_value(node, context, &inputs[0])? {
                pad_opts.value = Some(value);
            }
        }

        let out = b
            .builder
            .pad_with_options(input0, beginning_padding, ending_padding, pad_opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }

        let mut result = ConversionResult::default();
        if let Some(onnx_out) = node.output.first() {
            if let Some(dtype) = context.value_types.get(&inputs[0]) {
                result.output_types.insert(onnx_out.clone(), dtype.clone());
            }
        }
        Ok(result)
    }
}

pub fn read_onnx_pads(
    node: &NodeProto,
    context: &ConversionContext,
    rank: usize,
) -> Result<Vec<i64>, OnnxError> {
    read_onnx_pads_from_maps(
        node,
        context.initializers,
        context.const_values,
        rank,
    )
}

pub fn read_onnx_pads_from_maps(
    node: &NodeProto,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
    rank: usize,
) -> Result<Vec<i64>, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() >= 2 {
        if let Some(pads) = read_int64_tensor(&inputs[1], initializers, const_values) {
            return Ok(pads);
        }
    }

    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "pads" && !attr.ints.is_empty() {
            return Ok(attr.ints.clone());
        }
    }

    if rank == 0 {
        return Ok(Vec::new());
    }

    Err(OnnxError::InvalidShape(
        "Pad requires static pads (attribute or constant input)".to_string(),
    ))
}

pub fn split_onnx_pads(pads: &[i64], rank: usize) -> Result<(Vec<u32>, Vec<u32>), OnnxError> {
    if pads.len() != 2 * rank {
        return Err(OnnxError::InvalidShape(format!(
            "Pad pads length {} does not match 2 * rank {}",
            pads.len(),
            rank
        )));
    }

    let mut beginning_padding = Vec::with_capacity(rank);
    let mut ending_padding = Vec::with_capacity(rank);
    for i in 0..rank {
        let begin = pads[i];
        let end = pads[i + rank];
        if begin < 0 || end < 0 {
            return Err(OnnxError::InvalidShape(format!(
                "Pad padding must be non-negative, got begin={begin} end={end}"
            )));
        }
        beginning_padding.push(begin as u32);
        ending_padding.push(end as u32);
    }
    Ok((beginning_padding, ending_padding))
}

pub fn infer_pad_output_shape(input_shape: &[i64], pads: &[i64]) -> Option<Vec<i64>> {
    let rank = input_shape.len();
    if pads.len() != 2 * rank {
        return None;
    }
    let mut out = input_shape.to_vec();
    for i in 0..rank {
        out[i] = out[i].saturating_add(pads[i]).saturating_add(pads[i + rank]);
    }
    Some(out)
}

fn read_mode(node: &NodeProto) -> String {
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "mode" && !attr.s.is_empty() {
            let mode = String::from_utf8_lossy(&attr.s).to_ascii_lowercase();
            return match mode.as_str() {
                "reflect" => "reflection".to_string(),
                "edge" => "edge".to_string(),
                _ => "constant".to_string(),
            };
        }
    }
    "constant".to_string()
}

fn read_constant_value(
    node: &NodeProto,
    context: &ConversionContext,
    data_input: &str,
) -> Result<Option<Value>, OnnxError> {
    let inputs = node.input.as_slice();
    if inputs.len() >= 3 {
        if let Some(value) = read_scalar_value(&inputs[2], context) {
            return Ok(Some(value));
        }
    }

    for attr in node.attribute.as_slice() {
        if attr.name.as_str() == "value" {
            if let Some(t) = attr.t.as_ref() {
                return Ok(Some(ml_number_from_tensor(t)?));
            }
        }
    }

    // Default fill for constant mode when no explicit value is provided.
    let _ = data_input;
    Ok(Some(serde_json::json!(0)))
}

fn read_scalar_value(name: &str, context: &ConversionContext) -> Option<Value> {
    context
        .initializers
        .get(name)
        .and_then(|t| ml_number_from_tensor(t).ok())
}

fn read_int64_tensor(
    name: &str,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    if let Some(vals) = const_values.get(name) {
        return Some(vals.clone());
    }
    if let Some(t) = initializers.get(name) {
        return read_int64_tensor_proto(t);
    }
    None
}

fn read_int64_tensor_proto(t: &TensorProto) -> Option<Vec<i64>> {
    if !t.raw_data.is_empty() {
        if t.data_type == TensorProto_DataType::Int32 as i32 {
            return Some(
                t.raw_data
                    .chunks_exact(4)
                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                    .collect(),
            );
        }
        return Some(
            t.raw_data
                .chunks_exact(8)
                .map(|c| {
                    i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                })
                .collect(),
        );
    }
    if !t.int64_data.is_empty() {
        return Some(t.int64_data.clone());
    }
    if !t.int32_data.is_empty() {
        return Some(t.int32_data.iter().map(|&v| v as i64).collect());
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;

    fn create_test_node(inputs: Vec<&str>, outputs: Vec<&str>, attrs: Vec<(&str, Vec<i64>)>) -> NodeProto {
        NodeProto {
            op_type: "Pad".to_string(),
            name: "test_pad".to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            attribute: attrs
                .into_iter()
                .map(|(name, ints)| crate::protos::onnx::AttributeProto {
                    name: name.to_string(),
                    ints,
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn split_onnx_pads_reorders() {
        let (begin, end) = split_onnx_pads(&[0, 0, 0, 0, 0, 0, 1, 1], 4).unwrap();
        assert_eq!(begin, vec![0, 0, 0, 0]);
        assert_eq!(end, vec![0, 0, 1, 1]);
    }

    #[test]
    fn infer_pad_output_shape_adds_padding() {
        let out = infer_pad_output_shape(&[1, 3, 260, 260], &[0, 0, 0, 0, 0, 0, 1, 1]).unwrap();
        assert_eq!(out, vec![1, 3, 261, 261]);
    }

    #[test]
    fn convert_pad_opset9_attribute_pads() {
        let handler = PadHandler;
        let node = create_test_node(vec!["x"], vec!["y"], vec![("pads", vec![0, 0, 1, 1])]);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 3]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }
}
