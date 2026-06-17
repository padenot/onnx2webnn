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

// Elementwise binary operators: Add, Sub, Mul, Div, Pow

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;

pub struct ElementwiseHandler;

impl OpHandler for ElementwiseHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max"
        )
    }

    fn convert(
        &self,
        node: &NodeProto,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            node.output.first()
                .map(|s| crate::onnx::convert::sanitize_identifier(s))
                .unwrap_or_else(|| node.op_type.to_string())
        };

        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 2 inputs, got {}",
                op_type,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = b.resolve_operand(&inputs[0])?;
        let input1 = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = emit_binary(op_type, b, input0, input1, opts, &node_name)?;

        if let Some(output) = node.output.as_slice().first() {
            b.record_operand(&[output.as_str(), &output_name], out);
        } else {
            b.record_operand(&[&output_name], out);
        }

        Ok(ConversionResult::default())
    }
}

fn emit_binary(
    op_type: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    a: MLOperand,
    b_in: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match op_type {
        "Add" => b
            .builder
            .add_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Sub" => b
            .builder
            .sub_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Mul" => b
            .builder
            .mul_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Div" => b
            .builder
            .div_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Pow" => b
            .builder
            .pow_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Min" => b
            .builder
            .min_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        "Max" => b
            .builder
            .max_with_options(a, b_in, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name.to_string(),
            })
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protos::onnx::NodeProto;

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
    fn test_elementwise_handler_supports() {
        let handler = ElementwiseHandler;
        assert!(handler.supports("Add"));
        assert!(handler.supports("Sub"));
        assert!(handler.supports("Mul"));
        assert!(handler.supports("Div"));
        assert!(handler.supports("Pow"));
        assert!(handler.supports("Min"));
        assert!(handler.supports("Max"));
        assert!(!handler.supports("MatMul"));
    }

    #[test]
    fn test_convert_add() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Add", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_mul() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Mul", vec!["x", "y"], vec!["z"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_div() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Div", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_min() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Min", vec!["x", "y"], vec!["z"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_max() {
        let handler = ElementwiseHandler;
        let node = create_test_node("Max", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
