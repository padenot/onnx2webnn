/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Conditional operators: Where

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;

pub struct ConditionalHandler;

impl OpHandler for ConditionalHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "Where")
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

        let inputs = node.input.as_slice();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "Where expects 3 inputs (condition, x, y), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, &node_name);
        let cond = b.resolve_operand(&inputs[0])?;
        let t = b.resolve_operand(&inputs[1])?;
        let f = b.resolve_operand(&inputs[2])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .where_with_options(cond, t, f, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }

        let mut result = ConversionResult::default();
        if let Some(onnx_out) = node.output.first() {
            if let Some(dtype) = context.value_types.get(&inputs[1]) {
                result.output_types.insert(onnx_out.clone(), dtype.clone());
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustnn::DataType;
    use crate::protos::onnx::NodeProto;
    use std::collections::HashMap;

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    #[test]
    fn test_conditional_handler_supports() {
        let handler = ConditionalHandler;
        assert!(handler.supports("Where"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_where_conversion() {
        let handler = ConditionalHandler;
        let node = create_test_node("Where", vec!["condition", "x", "y"], vec!["output"]);
        let initializers = HashMap::new();
        let mut value_shapes = HashMap::new();
        value_shapes.insert("condition".to_string(), vec![2, 2]);
        value_shapes.insert("x".to_string(), vec![2, 2]);
        value_shapes.insert("y".to_string(), vec![2, 2]);
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("condition".to_string(), DataType::Uint8);
        value_types.insert("x".to_string(), DataType::Float32);
        value_types.insert("y".to_string(), DataType::Float32);
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

    #[test]
    fn test_where_invalid_inputs() {
        let handler = ConditionalHandler;
        let node = create_test_node("Where", vec!["condition", "x"], vec!["output"]);
        let initializers = HashMap::new();
        let value_shapes = HashMap::new();
        let const_values = HashMap::new();
        let value_ids = HashMap::new();
        let mut value_types = HashMap::new();
        value_types.insert("x".to_string(), DataType::Float32);
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };
        let err =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap_err();
        assert!(err.to_string().contains("expects 3 inputs"));
    }
}
