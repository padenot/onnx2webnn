/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// MatMul and Gemm operator handlers

use crate::onnx::builder::{map_op_error, operand_index, OnnxBuilder};
use crate::onnx::builder_helpers::{output_label, record_node_output};
use crate::onnx::convert::OnnxError;
use crate::onnx::ops::{ConversionContext, ConversionResult, OpHandler};
use crate::protos::onnx::NodeProto;
use rustnn::operator_options::MLGemmOptions;

pub struct MatMulHandler;

impl OpHandler for MatMulHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(op_type, "MatMul" | "Gemm")
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
            "MatMul" => self.convert_matmul(node, &node_name, b),
            "Gemm" => self.convert_gemm(node, &node_name, context, b),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl MatMulHandler {
    fn convert_matmul(
        &self,
        node: &NodeProto,
        node_name: &str,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "MatMul expects 2 inputs, got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let opts = OnnxBuilder::labeled_options(&output_name);
        let out = b
            .builder
            .matmul_with_options(a, b_in, opts)
            .map_err(map_op_error)?;

        if let Some(onnx_out) = node.output.first() {
            record_node_output(b, onnx_out, &output_name, out);
        } else {
            b.record_operand(&[&output_name], out);
        }
        Ok(ConversionResult::default())
    }

    fn convert_gemm(
        &self,
        node: &NodeProto,
        node_name: &str,
        _context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Gemm expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        let mut alpha = 1.0f64;
        let mut beta = 1.0f64;
        let mut trans_a = false;
        let mut trans_b = false;
        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "alpha" if attr.f != 0.0 => alpha = attr.f as f64,
                "beta" if attr.f != 0.0 => beta = attr.f as f64,
                "transA" if attr.i != 0 => trans_a = true,
                "transB" if attr.i != 0 => trans_b = true,
                _ => {}
            }
        }

        let output_name = output_label(node, node_name);
        let a = b.resolve_operand(&inputs[0])?;
        let b_in = b.resolve_operand(&inputs[1])?;
        let c = inputs
            .get(2)
            .map(|name| b.resolve_operand(name))
            .transpose()?;

        let opts = MLGemmOptions {
            label: output_name.clone(),
            alpha,
            beta,
            a_transpose: trans_a,
            b_transpose: trans_b,
            c: c.map(operand_index),
        };
        let out = b
            .builder
            .gemm_with_options(a, b_in, opts)
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
    fn test_matmul_handler_supports() {
        let handler = MatMulHandler;
        assert!(handler.supports("MatMul"));
        assert!(handler.supports("Gemm"));
    }

    #[test]
    fn test_convert_matmul() {
        let handler = MatMulHandler;
        let node = create_test_node("MatMul", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }

    #[test]
    fn test_convert_gemm_simple() {
        let handler = MatMulHandler;
        let node = create_test_node("Gemm", vec!["a", "b"], vec!["c"]);
        crate::onnx::ops::convert_with_test_builder(&handler, &node).unwrap();
    }
}
