/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Normalization operators: LayerNormalization, Softmax

use crate::onnx::builder::{map_op_error, operand_index, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{
    normalize_axis_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use rustnn::operator_options::MLLayerNormalizationOptions;

pub struct NormalizationHandler;

impl OpHandler for NormalizationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "LayerNormalization" | "Softmax")
    }

    fn convert(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.clone()
        } else {
            "unnamed".to_string()
        };

        match op_type {
            "LayerNormalization" => self.convert_layer_norm(node, &node_name, context, b),
            "Softmax" => self.convert_softmax(node, &node_name, context, b),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl NormalizationHandler {
    fn convert_layer_norm(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "LayerNormalization expects at least 1 input".to_string(),
            ));
        }

        let mut epsilon = 1e-5f64;
        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "epsilon" if attr.f != 0.0 => epsilon = attr.f as f64,
                "axis" if attr.i != 0 => axis = attr.i,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;

        let axes = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            Some(vec![normalize_axis_best_effort(axis, rank) as u32])
        } else if axis != -1 {
            Some(vec![axis as u32])
        } else {
            None
        };

        let scale = inputs
            .get(1)
            .map(|n| b.resolve_operand(n))
            .transpose()?
            .map(operand_index);
        let bias = inputs
            .get(2)
            .map(|n| b.resolve_operand(n))
            .transpose()?
            .map(operand_index);

        let opts = MLLayerNormalizationOptions {
            label: output_name.clone(),
            scale,
            bias,
            axes,
            epsilon,
        };
        let out = b
            .builder
            .layer_normalization_with_options(input, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_softmax(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Softmax expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut axis = -1i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;
        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank) as u32
        } else {
            axis as u32
        };
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .softmax_with_options(input, axis, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::{AttributeProto, NodeProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_float_attribute(node: &mut NodeProto, name: &str, value: f32) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            f: value,
            ..Default::default()
        });
    }

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        });
    }

    #[test]
    fn test_normalization_handler_supports() {
        let handler = NormalizationHandler;
        assert!(handler.supports("LayerNormalization"));
        assert!(handler.supports("Softmax"));
    }

    #[test]
    fn test_convert_layer_norm() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("LayerNormalization", vec!["x", "scale", "bias"], vec!["y"]);
        add_float_attribute(&mut node, "epsilon", 1e-5);
        add_int_attribute(&mut node, "axis", -1);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_softmax() {
        let handler = NormalizationHandler;
        let mut node = create_test_node("Softmax", vec!["x"], vec!["y"]);
        add_int_attribute(&mut node, "axis", 1);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
