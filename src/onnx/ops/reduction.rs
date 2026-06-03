/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Reduction operators: ReduceMean, ReduceSum, ReduceMax, ReduceMin

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{
    normalize_axes_best_effort, ConversionContext, ConversionResult, OpHandler,
};
use crate::protos::onnx::NodeProto;
use rustnn::operator_options::MLReduceOptions;

pub struct ReductionHandler;

impl OpHandler for ReductionHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin"
        )
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
            "ReduceMean" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_mean_with_options(i, o)
            }),
            "ReduceSum" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_sum_with_options(i, o)
            }),
            "ReduceMax" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_max_with_options(i, o)
            }),
            "ReduceMin" => self.convert_reduce(node, &node_name, context, b, |g, i, o| {
                g.reduce_min_with_options(i, o)
            }),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReductionHandler {
    fn convert_reduce(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
        emit: impl FnOnce(
            &mut rustnn::mlgraphbuilder::MLGraphBuilder<'_, '_>,
            rustnn::mlcontext::MLOperand,
            MLReduceOptions,
        ) -> Result<rustnn::mlcontext::MLOperand, rustnn::error::GraphBuilderError>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects at least 1 input",
                node.op_type
            )));
        }

        let mut axes: Option<Vec<i64>> = None;
        let mut keepdims = 1i64;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axes" if !attr.ints.is_empty() => axes = Some(attr.ints.clone()),
                "keepdims" if attr.i != 0 => keepdims = attr.i,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let input = b.resolve_operand(&inputs[0])?;

        let axes_u32 = axes.map(|axes_values| {
            if let Some(rank) = context.input_rank(inputs[0].as_str()) {
                normalize_axes_best_effort(&axes_values, rank)
            } else {
                axes_values
            }
            .into_iter()
            .map(|a| a as u32)
            .collect::<Vec<_>>()
        });

        let opts = MLReduceOptions {
            label: output_name.clone(),
            axes: axes_u32,
            keep_dimensions: keepdims != 0,
        };
        let out = emit(b.builder, input, opts).map_err(map_op_error)?;

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

    fn add_int_attribute(node: &mut NodeProto, name: &str, value: i64) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        });
    }

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        });
    }

    #[test]
    fn test_reduction_handler_supports() {
        let handler = ReductionHandler;
        assert!(handler.supports("ReduceMean"));
        assert!(handler.supports("ReduceSum"));
    }

    #[test]
    fn test_convert_reduce_mean() {
        let handler = ReductionHandler;
        let mut node = create_test_node("ReduceMean", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1, 2]);
        add_int_attribute(&mut node, "keepdims", 1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![1, 2, 3, 4]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = crate::onnx::ops::ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
    }

    #[test]
    fn test_convert_reduce_sum() {
        let handler = ReductionHandler;
        let node = create_test_node("ReduceSum", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
