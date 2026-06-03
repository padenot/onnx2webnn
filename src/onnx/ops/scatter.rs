/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use rustnn::DataType;
use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;

pub struct ScatterHandler;

impl ScatterHandler {
    fn get_string_attr(node: &NodeProto, name: &str) -> Option<String> {
        for a in node.attribute.as_slice() {
            if a.name.as_str() != name {
                continue;
            }
            let raw = a.s.clone();
            if raw.is_empty() {
                return None;
            }
            return String::from_utf8(raw).ok();
        }
        None
    }
}

impl OpHandler for ScatterHandler {
    fn supports(&self, op_type: &str) -> bool {
        op_type == "ScatterND"
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let reduction =
            Self::get_string_attr(node, "reduction").unwrap_or_else(|| "none".to_string());
        if reduction != "none" {
            let node_name = node.name.clone();
            return Err(OnnxError::UnsupportedOp {
                op: format!("ScatterND(reduction={reduction})"),
                node: node_name,
            });
        }

        let inputs = node.input.as_slice();
        if inputs.len() != 3 {
            return Err(OnnxError::InvalidShape(format!(
                "ScatterND expects 3 inputs (data, indices, updates), got {}",
                inputs.len()
            )));
        }

        let outputs = node.output.as_slice();
        if outputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "ScatterND expects 1 output, got {}",
                outputs.len()
            )));
        }

        let node_name = if node.name.is_empty() {
            outputs[0].clone()
        } else {
            node.name.clone()
        };
        let output_name = output_label(node, &node_name);
        let data = b.resolve_operand(&inputs[0])?;
        let indices = b.resolve_operand(&inputs[1])?;
        let updates = b.resolve_operand(&inputs[2])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .scatter_nd_with_options(data, indices, updates, opts)
            .map_err(map_op_error)?;

        record_node_output(b, &outputs[0], &output_name, out);

        let mut result = ConversionResult::default();
        if let Some(dtype) = context.value_types.get(&inputs[0]) {
            result
                .output_types
                .insert(outputs[0].clone(), dtype.clone());
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustnn::DataType;
    use crate::protos::onnx::{AttributeProto, NodeProto, TensorProto};

    fn create_test_node(op_type: &str, inputs: Vec<&str>, outputs: Vec<&str>) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: format!("test_{}", op_type.to_lowercase()),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..Default::default()
        }
    }

    fn add_string_attr(node: &mut NodeProto, name: &str, value: &str) {
        node.attribute.push(AttributeProto {
            name: name.to_string(),
            s: value.as_bytes().to_vec(),
            ..Default::default()
        });
    }

    struct TestContext {
        initializers: std::collections::HashMap<String, &'static TensorProto>,
        value_shapes: std::collections::HashMap<String, Vec<i64>>,
        const_values: std::collections::HashMap<String, Vec<i64>>,
        value_ids: std::collections::HashMap<String, String>,
        value_types: std::collections::HashMap<String, DataType>,
    }

    impl TestContext {
        fn new() -> Self {
            Self {
                initializers: std::collections::HashMap::new(),
                value_shapes: std::collections::HashMap::new(),
                const_values: std::collections::HashMap::new(),
                value_ids: std::collections::HashMap::new(),
                value_types: std::collections::HashMap::new(),
            }
        }

        fn ctx(&self) -> ConversionContext<'_> {
            ConversionContext {
                initializers: &self.initializers,
                value_shapes: &self.value_shapes,
                value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
                const_values: &self.const_values,
                value_ids: &self.value_ids,
                value_types: &self.value_types,
            }
        }
    }

    #[test]
    fn test_scatter_handler_supports() {
        let handler = ScatterHandler;
        assert!(handler.supports("ScatterND"));
    }

    #[test]
    fn test_convert_scatter_nd() {
        let handler = ScatterHandler;
        let node = create_test_node("ScatterND", vec!["data", "indices", "updates"], vec!["y"]);
        let mut tc = TestContext::new();
        tc.value_shapes.insert("data".to_string(), vec![2, 3]);
        tc.value_shapes
            .insert("indices".to_string(), vec![2, 1]);
        tc.value_shapes.insert("updates".to_string(), vec![2, 3]);
        tc.value_types.insert("data".to_string(), DataType::Float32);
        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &tc.ctx()).unwrap();
        assert_eq!(result.output_types.get("y"), Some(&DataType::Float32));
    }

    #[test]
    fn test_convert_scatter_nd_reduction_unsupported() {
        let handler = ScatterHandler;
        let mut node = create_test_node("ScatterND", vec!["data", "indices", "updates"], vec!["y"]);
        add_string_attr(&mut node, "reduction", "add");
        let tc = TestContext::new();
        let context = tc.ctx();
        match crate::onnx::ops::convert_handler_with_context(&handler, &node, &context) {
            Err(OnnxError::UnsupportedOp { op, .. }) => assert!(op.contains("reduction=add")),
            other => panic!("expected UnsupportedOp, got {other:?}"),
        }
    }

    #[test]
    fn test_convert_scatter_nd_invalid_input_count() {
        let handler = ScatterHandler;
        let node = create_test_node("ScatterND", vec!["data", "indices"], vec!["y"]);
        let tc = TestContext::new();
        let err =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &tc.ctx()).unwrap_err();
        assert!(err.to_string().contains("expects 3 inputs"));
    }

    #[test]
    fn test_convert_scatter_nd_invalid_output_count() {
        let handler = ScatterHandler;
        let node = create_test_node(
            "ScatterND",
            vec!["data", "indices", "updates"],
            vec!["y0", "y1"],
        );
        let tc = TestContext::new();
        let err =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &tc.ctx()).unwrap_err();
        assert!(err.to_string().contains("expects 1 output"));
    }
}
