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

// Activation and unary math operators: Relu, Gelu, Tanh, Sigmoid, Sqrt, Exp, Log, Abs, Neg, Erf

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::mlcontext::MLOperand;

pub struct ActivationHandler;

impl OpHandler for ActivationHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Relu"
                | "Gelu"
                | "Tanh"
                | "Sigmoid"
                | "Sqrt"
                | "Exp"
                | "Log"
                | "Abs"
                | "Neg"
                | "Erf"
                | "Cos"
                | "Sin"
                | "Floor"
                | "Ceil"
                | "Round"
                | "Identity"
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
            node.name.as_str().to_string()
        } else {
            node.output.first()
                .map(|s| crate::onnx::convert::sanitize_identifier(s))
                .unwrap_or_else(|| node.op_type.to_string())
        };

        let webnn_op = match op_type {
            "Relu" => "relu",
            "Gelu" => "gelu",
            "Tanh" => "tanh",
            "Sigmoid" => "sigmoid",
            "Sqrt" => "sqrt",
            "Exp" => "exp",
            "Log" => "log",
            "Abs" => "abs",
            "Neg" => "neg",
            "Erf" => "erf",
            "Cos" => "cos",
            "Sin" => "sin",
            "Floor" => "floor",
            "Ceil" => "ceil",
            "Round" => "round",
            "Identity" => "identity",
            _ => {
                return Err(OnnxError::UnsupportedOp {
                    op: op_type.to_string(),
                    node: node_name,
                })
            }
        };

        self.convert_unary(node, &node_name, webnn_op, context, b)
    }
}

impl ActivationHandler {
    fn convert_unary(
        &self,
        node: &NodeProto,
        node_name: &str,
        webnn_op: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "{} expects 1 input, got {}",
                webnn_op,
                inputs.len()
            )));
        }

        let output_name = if node.output.as_slice().is_empty() {
            format!("{}_output", node_name)
        } else {
            sanitize_identifier(&node.output.as_slice()[0].to_string())
        };

        let input0 = b.resolve_operand(&inputs[0])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = emit_unary(webnn_op, b, input0, opts, node_name)?;

        if let Some(output) = node.output.as_slice().first() {
            b.record_operand(&[output.as_str(), &output_name], out);
        } else {
            b.record_operand(&[&output_name], out);
        }

        Ok(ConversionResult::default())
    }
}

fn emit_unary(
    webnn_op: &str,
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    opts: rustnn::operator_options::MLOperatorOptions,
    node_name: &str,
) -> Result<MLOperand, OnnxError> {
    Ok(match webnn_op {
        "relu" => b.builder.relu_with_options(input, opts).map_err(map_op_error)?,
        "gelu" => b.builder.gelu_with_options(input, opts).map_err(map_op_error)?,
        "tanh" => b.builder.tanh_with_options(input, opts).map_err(map_op_error)?,
        "sigmoid" => b
            .builder
            .sigmoid_with_options(input, opts)
            .map_err(map_op_error)?,
        "sqrt" => b.builder.sqrt_with_options(input, opts).map_err(map_op_error)?,
        "exp" => b.builder.exp_with_options(input, opts).map_err(map_op_error)?,
        "log" => b.builder.log_with_options(input, opts).map_err(map_op_error)?,
        "abs" => b.builder.abs_with_options(input, opts).map_err(map_op_error)?,
        "neg" => b.builder.neg_with_options(input, opts).map_err(map_op_error)?,
        "erf" => b.builder.erf_with_options(input, opts).map_err(map_op_error)?,
        "cos" => b.builder.cos_with_options(input, opts).map_err(map_op_error)?,
        "sin" => b.builder.sin_with_options(input, opts).map_err(map_op_error)?,
        "floor" => b.builder.floor_with_options(input, opts).map_err(map_op_error)?,
        "ceil" => b.builder.ceil_with_options(input, opts).map_err(map_op_error)?,
        "round" => b.builder.round_even_with_options(input, opts).map_err(map_op_error)?,
        "identity" => b
            .builder
            .identity_with_options(input, opts)
            .map_err(map_op_error)?,
        _ => {
            return Err(OnnxError::UnsupportedOp {
                op: webnn_op.to_string(),
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
    fn test_activation_handler_supports() {
        let handler = ActivationHandler;
        assert!(handler.supports("Relu"));
        assert!(handler.supports("Gelu"));
        assert!(handler.supports("Tanh"));
        assert!(handler.supports("Sigmoid"));
        assert!(handler.supports("Sqrt"));
        assert!(handler.supports("Exp"));
        assert!(handler.supports("Log"));
        assert!(handler.supports("Abs"));
        assert!(handler.supports("Neg"));
        assert!(handler.supports("Erf"));
        assert!(handler.supports("Cos"));
        assert!(handler.supports("Sin"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_relu() {
        let handler = ActivationHandler;
        let node = create_test_node("Relu", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_sqrt() {
        let handler = ActivationHandler;
        let node = create_test_node("Sqrt", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_gelu() {
        let handler = ActivationHandler;
        let node = create_test_node("Gelu", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_cos() {
        let handler = ActivationHandler;
        let node = create_test_node("Cos", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_sin() {
        let handler = ActivationHandler;
        let node = create_test_node("Sin", vec!["x"], vec!["y"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
