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

// Reshape operators: Reshape, Transpose, Concat, Split, Unsqueeze, Squeeze

use crate::onnx::builder::{map_op_error, OnnxBuilder};
use crate::onnx::builder_helpers::{
    ast_dims_to_mldim, expand_with_shape, merge_dims_with_i64_values,
    merge_dims_with_static_values, output_label, record_node_output, reshape_with_shape,
    u32_slice_to_mldim,
};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::onnx::ops::{
    normalize_axes_best_effort, normalize_axis_best_effort, ConversionContext, ConversionResult,
    OpHandler,
};
use crate::onnx::shape_inference::{static_shape_from_dims, value_shape_dims_for};
use crate::protos::onnx::{NodeProto, TensorProto_DataType};
use rustnn::operator_options::{
    MLDimension, MLSplitOptions, MLSqueezeOptions, MLTransposeOptions, MLUnsqueezeOptions,
};

pub struct ReshapeHandler;

impl OpHandler for ReshapeHandler {
    fn supports(&self, op_type: &str) -> bool {
        matches!(
            op_type,
            "Reshape"
                | "Transpose"
                | "Concat"
                | "Split"
                | "Unsqueeze"
                | "Squeeze"
                | "Tile"
                | "Expand"
                | "Flatten"
        )
    }

    fn convert<'a>(
        &self,
        node: &NodeProto,
        context: &ConversionContext<'a>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let op_type = node.op_type.as_str();
        let node_name = if !node.name.is_empty() {
            node.name.as_str().to_string()
        } else {
            node.output
                .first()
                .map(|s| crate::onnx::convert::sanitize_identifier(s))
                .unwrap_or_else(|| node.op_type.to_string())
        };

        match op_type {
            "Reshape" => self.convert_reshape(node, &node_name, context, b),
            "Transpose" => self.convert_transpose(node, &node_name, context, b),
            "Concat" => self.convert_concat(node, &node_name, context, b),
            "Split" => self.convert_split(node, &node_name, context, b),
            "Unsqueeze" => self.convert_unsqueeze(node, &node_name, context, b),
            "Squeeze" => self.convert_squeeze(node, &node_name, context, b),
            "Tile" => self.convert_tile(node, &node_name, context, b),
            "Expand" => self.convert_expand(node, &node_name, context, b),
            "Flatten" => self.convert_flatten(node, &node_name, context, b),
            _ => Err(OnnxError::UnsupportedOp {
                op: op_type.to_string(),
                node: node_name,
            }),
        }
    }
}

impl ReshapeHandler {
    fn initializer_i64_values(context: &ConversionContext<'_>, name: &str) -> Option<Vec<i64>> {
        let initializer = context.initializers.get(name)?;
        let raw_data = initializer.raw_data.as_slice();
        if !raw_data.is_empty() {
            return Some(match initializer.data_type {
                x if x == TensorProto_DataType::Int32 as i32 => raw_data
                    .chunks_exact(4)
                    .map(|chunk| {
                        i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64
                    })
                    .collect(),
                _ => raw_data
                    .chunks_exact(8)
                    .map(|chunk| {
                        i64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                            chunk[7],
                        ])
                    })
                    .collect(),
            });
        }
        if !initializer.int64_data.as_slice().is_empty() {
            return Some(initializer.int64_data.as_slice().to_vec());
        }
        if !initializer.int32_data.as_slice().is_empty() {
            return Some(
                initializer
                    .int32_data
                    .as_slice()
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            );
        }
        None
    }

    fn const_i64_values(context: &ConversionContext<'_>, name: &str) -> Option<Vec<i64>> {
        context
            .const_values
            .get(name)
            .or_else(|| {
                let sanitized = sanitize_identifier(name);
                context.const_values.get(&sanitized)
            })
            .or_else(|| context.const_values.get(name.trim_start_matches('/')))
            .cloned()
            .or_else(|| Self::initializer_i64_values(context, name))
    }

    fn transpose_const_values(
        values: &[i64],
        input_shape: &[i64],
        permutation: &[usize],
    ) -> Option<Vec<i64>> {
        if input_shape.len() != permutation.len()
            || input_shape.iter().any(|&dim| dim < 0)
            || permutation.iter().any(|&axis| axis >= input_shape.len())
        {
            return None;
        }
        let input_numel = input_shape.iter().try_fold(1usize, |acc, &dim| {
            usize::try_from(dim).ok().map(|d| acc.saturating_mul(d))
        })?;
        if input_numel != values.len() {
            return None;
        }

        let output_shape: Vec<usize> = permutation
            .iter()
            .map(|&axis| usize::try_from(input_shape[axis]).ok())
            .collect::<Option<_>>()?;
        let mut input_strides = vec![1usize; input_shape.len()];
        for axis in (0..input_shape.len().saturating_sub(1)).rev() {
            input_strides[axis] = input_strides[axis + 1]
                .saturating_mul(usize::try_from(input_shape[axis + 1]).ok()?);
        }
        let mut output_strides = vec![1usize; output_shape.len()];
        for axis in (0..output_shape.len().saturating_sub(1)).rev() {
            output_strides[axis] = output_strides[axis + 1].saturating_mul(output_shape[axis + 1]);
        }

        let mut output = vec![0i64; values.len()];
        for (out_index, out_value) in output.iter_mut().enumerate() {
            let mut remaining = out_index;
            let mut input_index = 0usize;
            for axis in 0..output_shape.len() {
                let coord = if output_strides[axis] == 0 {
                    0
                } else {
                    remaining / output_strides[axis]
                };
                remaining %= output_strides[axis].max(1);
                input_index = input_index.saturating_add(coord * input_strides[permutation[axis]]);
            }
            *out_value = values[input_index];
        }
        Some(output)
    }

    fn repair_stale_hidden_split_shape(
        input_shape: &[i64],
        target_shape: &[i64],
        infer_index: usize,
    ) -> Option<Vec<u32>> {
        if input_shape.len() < 2 || target_shape.len() != input_shape.len() + 1 || infer_index != 1
        {
            return None;
        }
        let hidden = *input_shape.last()?;
        if target_shape.first().copied()? != hidden {
            return None;
        }
        let split_product =
            target_shape[infer_index + 1..]
                .iter()
                .try_fold(
                    1i64,
                    |acc, &dim| {
                        if dim <= 0 {
                            None
                        } else {
                            acc.checked_mul(dim)
                        }
                    },
                )?;
        if split_product != hidden {
            return None;
        }

        let mut repaired = Vec::with_capacity(target_shape.len());
        for &dim in &input_shape[..input_shape.len() - 1] {
            repaired.push(u32::try_from(dim).ok()?);
        }
        for &dim in &target_shape[infer_index + 1..] {
            repaired.push(u32::try_from(dim).ok()?);
        }
        Some(repaired)
    }

    fn record_output(
        b: &mut OnnxBuilder<'_, '_, '_>,
        node: &NodeProto,
        label: &str,
        op: rustnn::mlcontext::MLOperand,
        context: &ConversionContext,
        dtype_input: Option<&str>,
    ) -> Result<ConversionResult, OnnxError> {
        if let Some(onnx_out) = node.output.first() {
            let output_shape = b
                .builder
                .rustnn_operand_shape(op)
                .ok()
                .map(|shape| shape.into_iter().map(|dim| dim as i64).collect());
            record_node_output(b, onnx_out, label, op);
            let mut result = ConversionResult::default();
            if let Some(inp) = dtype_input {
                if let Some(dtype) = context.value_types.get(inp) {
                    result.output_types.insert(onnx_out.clone(), dtype.clone());
                }
            }
            if let Some(shape) = output_shape {
                result.output_shapes.insert(onnx_out.clone(), shape);
            }
            Ok(result)
        } else {
            b.record_operand(&[label], op);
            Ok(ConversionResult::default())
        }
    }

    fn emit_reshape_with_shape(
        b: &mut OnnxBuilder<'_, '_, '_>,
        node: &NodeProto,
        output_name: &str,
        input: rustnn::mlcontext::MLOperand,
        new_shape: Vec<MLDimension>,
        context: &ConversionContext,
        dtype_input: &str,
    ) -> Result<ConversionResult, OnnxError> {
        let out = reshape_with_shape(b, input, output_name, new_shape)?;
        Self::record_output(b, node, output_name, out, context, Some(dtype_input))
    }

    fn emit_expand_with_shape(
        b: &mut OnnxBuilder<'_, '_, '_>,
        node: &NodeProto,
        output_name: &str,
        input: rustnn::mlcontext::MLOperand,
        new_shape: Vec<MLDimension>,
        context: &ConversionContext,
        dtype_input: &str,
    ) -> Result<ConversionResult, OnnxError> {
        let out = expand_with_shape(b, input, output_name, new_shape)?;
        Self::record_output(b, node, output_name, out, context, Some(dtype_input))
    }

    fn normalize_unsqueeze_axes_best_effort(&self, axes: &[i64], input_rank: usize) -> Vec<i64> {
        // ONNX Unsqueeze interprets negative axes against the output rank.
        let output_rank = input_rank.saturating_add(axes.len());
        let output_rank_i64 = output_rank as i64;
        axes.iter()
            .map(|&axis| {
                let normalized = if axis < 0 {
                    axis + output_rank_i64
                } else {
                    axis
                };
                if normalized < 0 || normalized >= output_rank_i64 {
                    axis
                } else {
                    normalized
                }
            })
            .collect()
    }

    fn read_axes_from_attr_or_const(
        &self,
        node: &NodeProto,
        context: &ConversionContext,
    ) -> Result<Vec<i64>, OnnxError> {
        if let Some(attr_axes) = node
            .attribute
            .as_slice()
            .iter()
            .find(|a| a.name.as_str() == "axes")
            .map(|a| a.ints.clone())
        {
            return Ok(if attr_axes.is_empty() {
                vec![0]
            } else {
                attr_axes
            });
        }

        if node.input.as_slice().len() >= 2 {
            let name = node.input.as_slice()[1].to_string();
            if let Some(vals) = context.const_values.get(&name) {
                return Ok(if vals.is_empty() {
                    vec![0]
                } else {
                    vals.clone()
                });
            }
            if let Some(t) = context.initializers.get(&name) {
                let raw = t.raw_data.as_slice();
                if !raw.is_empty() {
                    let mut axes: Vec<i64> = raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                } else if !t.int64_data.as_slice().is_empty() {
                    let mut axes = t.int64_data.as_slice().to_vec();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                } else if !t.int32_data.as_slice().is_empty() {
                    let mut axes: Vec<i64> =
                        t.int32_data.as_slice().iter().map(|&v| v as i64).collect();
                    if axes.is_empty() {
                        axes.push(0);
                    }
                    return Ok(axes);
                }
            }
            return Ok(vec![0]);
        }

        Ok(vec![0])
    }

    /// Ensure Unsqueeze axes are available even if missing in the ONNX node by
    /// defaulting to a new leading dimension. This guards against malformed
    /// exports where the axes input was stripped.
    /// Convert ONNX Reshape to WebNN reshape
    /// ONNX Reshape takes shape as a second input (constant tensor)
    /// WebNN reshape takes newShape as a static array option
    fn convert_reshape<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Reshape expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let data_input_raw = inputs[0].to_string();
        let shape_input_raw = inputs[1].to_string();
        let data_input = b.resolve_operand(&data_input_raw)?;

        // Resolve shape from const-folded values or initializers
        let mut shape_values: Vec<i64> = if let Some(values) =
            context.const_values.get(&shape_input_raw)
        {
            values.clone()
        } else if let Some(values) = Self::initializer_i64_values(context, shape_input_raw.as_str())
        {
            values
        } else {
            Vec::new()
        };
        let shape_from_const = !shape_values.is_empty();

        if shape_values.is_empty() {
            if let Some(values) = Self::initializer_i64_values(context, shape_input_raw.as_str()) {
                shape_values = values;
            }
        }

        // Fallback: derive a candidate target shape from known output metadata when the
        // shape tensor itself is not const. Do not emit immediately from this metadata:
        // later reconciliation needs a chance to compare it with the real input element
        // count and repair stale ONNX value_info.
        if shape_values.is_empty() {
            if let Some(out) = node.output.as_slice().first() {
                if let Some(output_dims) =
                    value_shape_dims_for(out.as_str(), &context.value_shape_dims)
                {
                    if let Some(output_shape) = static_shape_from_dims(output_dims) {
                        if !output_shape.is_empty() {
                            shape_values = output_shape;
                        }
                    }
                }
            }

            if shape_values.is_empty() {
                if let Some(out_name) = node.output.as_slice().first() {
                    let out_s = out_name.to_string();
                    let known_output_shape = context
                        .value_shapes
                        .get(&out_s)
                        .or_else(|| context.value_shapes.get(&sanitize_identifier(&out_s)))
                        .or_else(|| context.value_shapes.get(out_s.trim_start_matches('/')))
                        .cloned();
                    if let Some(out_shape) = known_output_shape {
                        if !out_shape.is_empty() && out_shape.iter().all(|&d| d > 0) {
                            shape_values = out_shape;
                        }
                    }
                }
            }
        }

        if shape_values.is_empty() {
            if let Some(ds) = context
                .value_shapes
                .get(data_input_raw.as_str())
                .or_else(|| context.value_shapes.get(&data_input_raw))
            {
                // Do not collapse higher-rank inputs to rank-3; that breaks reshape
                // targets built from Concat shape vectors (e.g. SmolLM attention).
                shape_values = ds.clone();
            } else {
                return Err(OnnxError::InvalidShape(format!(
                    "Reshape shape input '{}' must be a constant (initializer/constant-folded) or input shape must be known. \
                     data input='{}'.",
                    shape_input_raw, data_input_raw
                )));
            }
        }

        // Handle -1 (dimension inference marker) - compute the inferred dimension
        // WebNN requires all dimensions to be explicit, so we need to resolve -1 values
        let input_shape_opt = b
            .builder
            .rustnn_operand_shape(data_input)
            .ok()
            .map(|shape| shape.into_iter().map(|dim| dim as i64).collect())
            .or_else(|| {
                let trimmed = data_input_raw.trim_start_matches('/');
                context
                    .value_shapes
                    .get(data_input_raw.as_str())
                    .or_else(|| context.value_shapes.get(&data_input_raw))
                    .or_else(|| context.value_shapes.get(trimmed))
                    .cloned()
            });

        let shape_values: Vec<u32> = if shape_values.contains(&-1) {
            // Prefer strict inference when we know the input shape; otherwise fall back to
            // best-effort by replacing -1 with 1 so conversion can proceed for fixed-step
            // decoder exports (batch=1, seq=1) even when upstream shape info is missing.
            if let Some(input_shape) = input_shape_opt.clone() {
                // Need to infer the -1 dimension based on input tensor shape
                // Validate all input dimensions are positive (WebNN requirement)
                if input_shape.iter().any(|&d| d <= 0) {
                    return Err(OnnxError::InvalidShape(format!(
                        "Cannot infer reshape dimension: input '{}' has dynamic/unknown dimensions {:?}. \
                        WebNN requires all dimensions to be statically known (> 0). \
                        Please ensure onnx-simplifier fully resolved all dimensions.",
                        data_input_raw, input_shape
                    )));
                }

                // Calculate total elements in input tensor
                let total_elements: i64 = input_shape.iter().product();

                // Calculate product of known dimensions and infer the -1 dimension
                let mut inferred_shape = Vec::new();
                let mut known_product: i64 = 1;
                let mut infer_index = None;

                for (i, &dim) in shape_values.iter().enumerate() {
                    if dim == -1 {
                        if infer_index.is_some() {
                            return Err(OnnxError::InvalidShape(
                                "Reshape cannot have multiple -1 dimensions".to_string(),
                            ));
                        }
                        infer_index = Some(i);
                        inferred_shape.push(0); // Placeholder
                    } else {
                        known_product *= dim;
                        inferred_shape.push(dim as u32);
                    }
                }

                // Compute inferred dimension
                if let Some(idx) = infer_index {
                    let inferred_dim = total_elements / known_product;
                    if inferred_dim <= 0 || total_elements % known_product != 0 {
                        if let Some(repaired_shape) =
                            Self::repair_stale_hidden_split_shape(&input_shape, &shape_values, idx)
                        {
                            crate::debug_println!(
                                "[reshape] repaired stale hidden split shape: {} input_shape={:?} target_shape={:?} repaired_shape={:?}",
                                output_name,
                                input_shape,
                                shape_values,
                                repaired_shape
                            );
                            return Self::emit_reshape_with_shape(
                                b,
                                node,
                                &output_name,
                                data_input,
                                u32_slice_to_mldim(&repaired_shape),
                                context,
                                &data_input_raw,
                            );
                        }
                        // Some decoder models (e.g. GPT exports with KV-cache inputs) can carry
                        // partially-known intermediate shapes even after override resolution.
                        // In those cases, prefer the existing best-effort fallback instead of
                        // failing conversion outright.
                        if total_elements > 0 {
                            crate::debug_println!(
                                "[reshape] cannot infer -1 for {} from input {:?} and target {:?}; replacing -1 with 1",
                                data_input_raw,
                                input_shape,
                                shape_values
                            );
                            let fallback_shape: Vec<u32> = shape_values
                                .iter()
                                .map(|&v| if v == -1 { 1 } else { v as u32 })
                                .collect();
                            return Self::emit_reshape_with_shape(
                                b,
                                node,
                                &output_name,
                                data_input,
                                u32_slice_to_mldim(&fallback_shape),
                                context,
                                &data_input_raw,
                            );
                        }

                        return Err(OnnxError::InvalidShape(format!(
                            "Cannot infer reshape dimension: {} elements cannot be reshaped to {:?}",
                            total_elements, shape_values
                        )));
                    }
                    inferred_shape[idx] = inferred_dim as u32;
                }

                inferred_shape
            } else {
                crate::debug_println!(
                    "[reshape] missing input shape for {}, shape {:?}; replacing -1 with 1",
                    data_input_raw,
                    shape_values
                );
                shape_values
                    .iter()
                    .map(|&v| if v == -1 { 1 } else { v as u32 })
                    .collect()
            }
        } else {
            // All dimensions are positive, use const shape if available; otherwise repair if needed.
            let input_shape = input_shape_opt.unwrap_or_default();
            let total_input: i64 = input_shape.iter().product();
            let total_target: i64 = shape_values.iter().product();
            let mut candidate: Vec<i64> = shape_values.clone();

            // If element counts don't match, repair using available hints.
            if total_input > 0 && total_target > 0 && total_input != total_target {
                let mut one_dim_repairs = Vec::new();
                if !shape_from_const {
                    for idx in 0..shape_values.len() {
                        let other_product: i64 = shape_values
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != idx)
                            .map(|(_, &dim)| dim)
                            .product();
                        if other_product <= 0 || total_input % other_product != 0 {
                            continue;
                        }
                        let repaired_dim = total_input / other_product;
                        if repaired_dim > 0 && repaired_dim != shape_values[idx] {
                            let mut repaired = shape_values.clone();
                            repaired[idx] = repaired_dim;
                            one_dim_repairs.push(repaired);
                        }
                    }
                }

                if one_dim_repairs.len() == 1 {
                    candidate = one_dim_repairs.pop().unwrap();
                    crate::debug_println!(
                        "[reshape] repair via one-dim solve: {} input_shape={:?} target_shape={:?} repaired_shape={:?}",
                        output_name,
                        input_shape,
                        shape_values,
                        candidate
                    );
                } else if !shape_from_const && input_shape.len() == 4 && shape_values.len() == 3 {
                    let tail: i64 = input_shape[2..].iter().product();
                    candidate = vec![input_shape[0], input_shape[1], tail];
                    crate::debug_println!(
                        "[reshape] repair via input flatten: {} input_shape={:?} target_shape={:?} repaired_shape={:?}",
                        output_name,
                        input_shape,
                        shape_values,
                        candidate
                    );
                } else {
                    // Rebuild using batch/seq hints (from known inputs) and hidden from target.
                    let mut batch_hint = input_shape.first().copied().unwrap_or(1);
                    let mut seq_hint = input_shape.get(1).copied().unwrap_or(1);
                    for (name, shape) in context.value_shapes.iter() {
                        if shape.len() >= 2 && !context.initializers.contains_key(name) {
                            if shape[0] > batch_hint {
                                batch_hint = shape[0];
                            }
                            if shape[1] > seq_hint {
                                seq_hint = shape[1];
                            }
                        }
                    }

                    let hidden = shape_values.last().copied().unwrap_or(1);
                    crate::debug_println!(
                        "[reshape] repair via fallback hints: {} input_shape={:?} target_shape={:?} batch_hint={} seq_hint={} hidden={}",
                        output_name,
                        input_shape,
                        shape_values,
                        batch_hint,
                        seq_hint,
                        hidden
                    );
                    candidate = vec![batch_hint, seq_hint, hidden];
                }
            } else if !shape_from_const && !input_shape.is_empty() {
                // If the target is rank-3 and the input is rank-4, flatten the last two dims.
                if input_shape.len() == 4 && shape_values.len() == 3 {
                    let tail: i64 = input_shape[2..].iter().product();
                    candidate = vec![input_shape[0], input_shape[1], tail];
                }
            }
            candidate.iter().map(|&v| v as u32).collect()
        };

        let new_shape = node
            .output
            .as_slice()
            .first()
            .and_then(|out| {
                let out_s = out.to_string();
                context
                    .value_shape_dims
                    .get(&out_s)
                    .or_else(|| context.value_shape_dims.get(&sanitize_identifier(&out_s)))
                    .or_else(|| context.value_shape_dims.get(out_s.trim_start_matches('/')))
                    .and_then(|dims| merge_dims_with_static_values(dims, &shape_values))
            })
            .or_else(|| {
                let shape_dims_key = shape_input_raw.clone();
                context
                    .value_shape_dims
                    .get(&shape_dims_key)
                    .or_else(|| {
                        context
                            .value_shape_dims
                            .get(&sanitize_identifier(&shape_dims_key))
                    })
                    .and_then(|dims| merge_dims_with_static_values(dims, &shape_values))
            })
            .unwrap_or_else(|| u32_slice_to_mldim(&shape_values));

        let mut result = Self::emit_reshape_with_shape(
            b,
            node,
            &output_name,
            data_input,
            new_shape,
            context,
            &data_input_raw,
        )?;
        if let Some(values) = Self::const_i64_values(context, data_input_raw.as_str()) {
            if let Some(output) = node.output.as_slice().first() {
                result.const_values.insert(output.to_string(), values);
            }
        }
        Ok(result)
    }

    /// Convert ONNX Expand to WebNN expand (broadcast to target shape)
    fn convert_expand<'a>(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &crate::onnx::ops::ConversionContext<'a>,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Expand expects 2 inputs (data, shape), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let data_input_raw = inputs[0].to_string();
        let shape_input_raw = inputs[1].to_string();
        let data_input = b.resolve_operand(&data_input_raw)?;

        let shape_key_sanitized = sanitize_identifier(&shape_input_raw);
        let shape_key_trimmed = shape_input_raw.trim_start_matches('/').to_string();

        let mut shape_values: Vec<i64> = if let Some(values) = context
            .const_values
            .get(&shape_input_raw)
            .or_else(|| context.const_values.get(&shape_key_sanitized))
            .or_else(|| context.const_values.get(&shape_key_trimmed))
        {
            values.clone()
        } else if let Some(initializer) = context.initializers.get(shape_input_raw.as_str()) {
            let raw_data = initializer.raw_data.as_slice();
            if !raw_data.is_empty() {
                match initializer.data_type {
                    x if x == TensorProto_DataType::Int32 as i32 => raw_data
                        .chunks_exact(4)
                        .map(|chunk| {
                            i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as i64
                        })
                        .collect(),
                    _ => raw_data
                        .chunks_exact(8)
                        .map(|chunk| {
                            i64::from_le_bytes([
                                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5],
                                chunk[6], chunk[7],
                            ])
                        })
                        .collect(),
                }
            } else if !initializer.int64_data.as_slice().is_empty() {
                initializer.int64_data.as_slice().to_vec()
            } else if !initializer.int32_data.as_slice().is_empty() {
                initializer
                    .int32_data
                    .as_slice()
                    .iter()
                    .map(|&v| v as i64)
                    .collect()
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let all_ones_placeholder = !shape_values.is_empty() && shape_values.iter().all(|&v| v == 1);
        if all_ones_placeholder {
            // All-ones placeholders from Where(ConstantOfShape, …) are not reliable target shapes.
            shape_values.clear();
        }

        let input_shape_for_expand = context.resolve_shape(&data_input_raw).cloned();
        if shape_values.iter().any(|&v| v < 0) {
            let Some(input_shape) = input_shape_for_expand.as_ref() else {
                return Err(OnnxError::InvalidShape(format!(
                    "Expand {} has negative target dimensions {:?} but input shape is unknown",
                    node_name, shape_values
                )));
            };
            let target_rank = shape_values.len();
            let input_rank = input_shape.len();
            let rank_offset = target_rank.saturating_sub(input_rank);
            for (axis, dim) in shape_values.iter_mut().enumerate() {
                if *dim == -1 {
                    if axis < rank_offset {
                        return Err(OnnxError::InvalidShape(format!(
                            "Expand {} cannot copy leading -1 dimension from input shape {:?}",
                            node_name, input_shape
                        )));
                    }
                    *dim = input_shape[axis - rank_offset];
                } else if *dim < -1 {
                    return Err(OnnxError::InvalidShape(format!(
                        "Expand {} has invalid negative target dimension {}",
                        node_name, dim
                    )));
                }
            }
        }

        let shape_input_dim_shape = context
            .value_shape_dims
            .get(&shape_input_raw)
            .or_else(|| context.value_shape_dims.get(&shape_key_sanitized))
            .or_else(|| context.value_shape_dims.get(&shape_key_trimmed))
            .cloned();

        let output_dim_shape = node
            .output
            .as_slice()
            .first()
            .and_then(|out| {
                let out_s = out.to_string();
                context
                    .value_shape_dims
                    .get(&out_s)
                    .or_else(|| context.value_shape_dims.get(&sanitize_identifier(&out_s)))
                    .or_else(|| context.value_shape_dims.get(out_s.trim_start_matches('/')))
            })
            .cloned();

        let mut dynamic_new_shape: Option<Vec<MLDimension>> =
            output_dim_shape.as_ref().and_then(|dims| {
                merge_dims_with_i64_values(dims, &shape_values).or_else(|| {
                    if shape_values.is_empty()
                        && dims
                            .iter()
                            .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
                    {
                        Some(ast_dims_to_mldim(dims))
                    } else {
                        None
                    }
                })
            });
        // If shape_values has fewer elements than the input rank, it can't be a valid
        // target shape for broadcasting (output rank must be >= input rank). Treat as
        // invalid and let the fallback use the ONNX-inferred output shape instead.
        let input_rank_for_check = input_shape_for_expand
            .as_ref()
            .map(|s| s.len())
            .unwrap_or(0);
        let shape_values_invalid =
            !shape_values.is_empty() && shape_values.len() < input_rank_for_check;

        let shape_values: Vec<i64> = if shape_values.is_empty() || shape_values_invalid {
            let output_shape_opt = node
                .output
                .as_slice()
                .first()
                .and_then(|out| {
                    let out_s = out.to_string();
                    context
                        .value_shapes
                        .get(&out_s)
                        .or_else(|| context.value_shapes.get(&sanitize_identifier(&out_s)))
                        .or_else(|| context.value_shapes.get(out_s.trim_start_matches('/')))
                })
                .cloned();

            if let Some(output_shape) = output_shape_opt {
                let has_dynamic_output_dim = output_dim_shape.as_ref().is_some_and(|dims| {
                    dims.iter()
                        .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
                });
                if output_shape.iter().all(|&d| d > 0) && !has_dynamic_output_dim {
                    crate::debug_println!(
                        "[expand] using inferred output shape for {}: {:?}",
                        node_name,
                        output_shape
                    );
                    output_shape
                } else {
                    Vec::new()
                }
            } else if input_shape_for_expand.is_some() {
                input_shape_for_expand.clone().unwrap_or_default()
            } else {
                Vec::new()
            }
        } else {
            shape_values
        };

        if dynamic_new_shape.is_none() && shape_values.is_empty() {
            if let Some(dims) = output_dim_shape.as_ref().or(shape_input_dim_shape.as_ref()) {
                if dims
                    .iter()
                    .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
                {
                    dynamic_new_shape = Some(ast_dims_to_mldim(dims));
                }
            }
        }

        let shape_u32: Vec<u32> = shape_values
            .iter()
            .map(|&v| {
                u32::try_from(v).map_err(|_| {
                    OnnxError::InvalidShape(format!(
                        "Expand {} target dimension {} cannot be represented as u32",
                        node_name, v
                    ))
                })
            })
            .collect::<Result<_, _>>()?;

        // Determine if this is a broadcast (WebNN expand) or reshape operation
        // by checking if shapes are broadcast-compatible (ONNX Expand rules)
        let input_shape = input_shape_for_expand.as_ref();
        let op_type = if dynamic_new_shape.is_some() {
            "expand"
        } else if let Some(input_shape) = input_shape {
            // Check broadcast compatibility: align from right, dimensions must be equal or one must be 1
            let mut is_broadcast_compatible = true;
            let input_rank = input_shape.len();
            let target_rank = shape_values.len();

            for i in 0..input_rank.min(target_rank) {
                let input_dim = input_shape[input_rank - 1 - i];
                let target_dim = shape_values[target_rank - 1 - i];

                // Dimensions are compatible if they're equal or either is 1
                if input_dim != target_dim && input_dim != 1 && target_dim != 1 {
                    is_broadcast_compatible = false;
                    break;
                }
            }

            if is_broadcast_compatible {
                "expand"
            } else {
                // Not broadcast-compatible, use reshape instead
                "reshape"
            }
        } else {
            // No shape info available, assume expand (broadcasting)
            "expand"
        };

        let new_shape = dynamic_new_shape
            .or_else(|| {
                context
                    .value_shape_dims
                    .get(&shape_input_raw)
                    .or_else(|| {
                        context
                            .value_shape_dims
                            .get(&sanitize_identifier(&shape_input_raw))
                    })
                    .and_then(|dims| merge_dims_with_i64_values(dims, &shape_values))
            })
            .unwrap_or_else(|| u32_slice_to_mldim(&shape_u32));

        if op_type == "reshape" {
            return Self::emit_reshape_with_shape(
                b,
                node,
                &output_name,
                data_input,
                new_shape,
                context,
                &data_input_raw,
            );
        }
        Self::emit_expand_with_shape(
            b,
            node,
            &output_name,
            data_input,
            new_shape,
            context,
            &data_input_raw,
        )
    }

    /// Convert ONNX Transpose to WebNN transpose
    fn convert_transpose(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Transpose expects 1 input, got {}",
                inputs.len()
            )));
        }

        // Extract perm attribute (permutation)
        let mut perm: Option<Vec<i64>> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "perm" {
                perm = Some(attr.ints.clone());
            }
        }

        let output_name = output_label(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;
        let input_shape = context.resolve_shape(inputs[0].as_str()).cloned();
        let permutation: Vec<usize> = perm
            .clone()
            .unwrap_or_else(|| {
                input_shape
                    .as_ref()
                    .map(|shape| (0..shape.len() as i64).rev().collect())
                    .unwrap_or_default()
            })
            .into_iter()
            .map(|axis| axis as usize)
            .collect();
        let opts = MLTransposeOptions {
            label: output_name.clone(),
            permutation: permutation.iter().map(|&axis| axis as u32).collect(),
        };
        let out = b
            .builder
            .transpose_with_options(input0, opts)
            .map_err(map_op_error)?;
        let mut result = Self::record_output(b, node, &output_name, out, context, None)?;
        if let (Some(values), Some(shape), Some(output)) = (
            Self::const_i64_values(context, inputs[0].as_str()),
            input_shape.as_ref(),
            node.output.as_slice().first(),
        ) {
            if let Some(transposed) = Self::transpose_const_values(&values, shape, &permutation) {
                result.const_values.insert(output.to_string(), transposed);
            }
        }
        Ok(result)
    }

    /// Convert ONNX Concat to WebNN concat
    fn convert_concat(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() < 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Concat expects at least 2 inputs, got {}",
                inputs.len()
            )));
        }

        // Extract axis attribute (required in ONNX)
        let mut axis = 0i64;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" && attr.i != 0 {
                axis = attr.i;
            }
        }

        let output_name = output_label(node, node_name);
        let input_ranks: Vec<Option<usize>> = inputs
            .iter()
            .map(|input| context.input_rank(input.as_str()))
            .collect();
        let target_rank = input_ranks.iter().flatten().copied().max();
        let axis = if let Some(rank) = target_rank {
            normalize_axis_best_effort(axis, rank)
        } else {
            // Rank unknown: pass axis as-is; negative values will be u32-wrapped and
            // the shape_inference layer will normalize them against the actual rank.
            axis
        };
        let mut operands = Vec::with_capacity(inputs.len());
        for (index, input) in inputs.iter().enumerate() {
            let mut operand = b.resolve_operand(input)?;
            if target_rank == Some(1) && axis == 0 && input_ranks[index] == Some(0) {
                operand = reshape_with_shape(
                    b,
                    operand,
                    &format!("{output_name}_scalar_{index}"),
                    vec![MLDimension::Static(1)],
                )?;
            }
            operands.push(operand);
        }
        let out = b
            .builder
            .concat_with_options(
                &operands,
                axis as u32,
                OnnxBuilder::labeled_options(&output_name),
            )
            .map_err(map_op_error)?;
        let mut result = Self::record_output(b, node, &output_name, out, context, None)?;
        if target_rank == Some(1) && axis == 0 {
            let mut values = Vec::new();
            let mut all_inputs_const = true;
            for input in inputs {
                if let Some(input_values) = Self::const_i64_values(context, input.as_str()) {
                    values.extend(input_values);
                } else {
                    all_inputs_const = false;
                    break;
                }
            }
            if all_inputs_const {
                if let Some(output) = node.output.as_slice().first() {
                    result.const_values.insert(output.to_string(), values);
                }
            }
        }
        Ok(result)
    }

    /// Convert ONNX Split to WebNN split
    fn convert_split(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 input".to_string(),
            ));
        }

        // Extract axis attribute
        let mut axis = 0i64;
        let mut splits: Option<Vec<i64>> = None;

        for attr in node.attribute.as_slice() {
            match attr.name.as_str() {
                "axis" if attr.i != 0 => {
                    axis = attr.i;
                }
                "split" => {
                    splits = Some(attr.ints.clone());
                }
                _ => {}
            }
        }

        let outputs = node.output.as_slice();
        if outputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Split expects at least 1 output".to_string(),
            ));
        }

        let input0 = b.resolve_operand(&inputs[0])?;
        let sanitized_outputs: Vec<String> = outputs
            .iter()
            .map(|s| sanitize_identifier(&s.to_string()))
            .collect();

        let axis = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axis_best_effort(axis, rank)
        } else {
            axis
        };

        let split_sizes: Vec<u32> = if let Some(split_values) = splits {
            split_values
                .iter()
                .map(|&v| {
                    u32::try_from(v)
                        .map_err(|_| OnnxError::InvalidShape(format!("invalid split size: {v}")))
                })
                .collect::<Result<_, _>>()?
        } else {
            let axis_usize = axis as usize;
            let dim = context
                .value_shapes
                .get(&inputs[0])
                .and_then(|s| s.get(axis_usize).copied())
                .ok_or_else(|| {
                    OnnxError::InvalidShape(
                        "Split without splits attribute requires known axis dimension".into(),
                    )
                })?;
            let n = outputs.len() as i64;
            if dim % n != 0 {
                return Err(OnnxError::InvalidShape(format!(
                    "Split axis dimension {dim} not divisible by output count {n}"
                )));
            }
            vec![(dim / n) as u32; outputs.len()]
        };

        let split_label = sanitize_identifier(&format!("{node_name}_split"));
        let split_opts = MLSplitOptions {
            label: split_label,
            axis: axis as u32,
        };
        let outs = b
            .builder
            .split_with_options(input0, &split_sizes, split_opts)
            .map_err(map_op_error)?;

        if outs.len() != sanitized_outputs.len() {
            return Err(OnnxError::InvalidShape(format!(
                "split produced {} outputs, expected {}",
                outs.len(),
                sanitized_outputs.len()
            )));
        }

        for (onnx_out, (webnn_out, op)) in outputs
            .iter()
            .zip(sanitized_outputs.iter().zip(outs.into_iter()))
        {
            record_node_output(b, onnx_out, webnn_out, op);
        }
        Ok(ConversionResult::default())
    }

    /// Convert ONNX Unsqueeze to WebNN expand
    /// ONNX Unsqueeze adds dimensions at specified axes
    /// In opset >= 13, axes is a second input; in earlier opsets, it's an attribute
    fn convert_unsqueeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Unsqueeze expects at least 1 input".to_string(),
            ));
        }

        let output_name = output_label(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;

        // Extract axes attribute (opset < 13) or use second input (opset >= 13).
        // If missing/empty, default to [0] to ensure the emitted unsqueeze is valid.
        let axes_values = {
            let mut axes: Option<Vec<i64>> = None;
            for attr in node.attribute.as_slice() {
                if attr.name.as_str() == "axes" {
                    axes = Some(attr.ints.clone());
                }
            }

            if let Some(a) = axes {
                if a.is_empty() {
                    vec![0]
                } else {
                    a
                }
            } else {
                let mut from_const = self.read_axes_from_attr_or_const(node, context)?;
                if from_const.is_empty() {
                    from_const.push(0);
                }
                from_const
            }
        };
        let axes_values = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            self.normalize_unsqueeze_axes_best_effort(&axes_values, rank)
        } else {
            axes_values
        };

        // Negative axes that weren't normalized (unknown rank) are invalid as u32.
        // Re-normalize against best-effort output rank (axes.len() + known_rank or 0).
        let known_rank = context.input_rank(inputs[0].as_str()).unwrap_or(0);
        let output_rank = (known_rank + axes_values.len()) as i64;
        let axes_u32: Vec<u32> = axes_values
            .into_iter()
            .map(|a| {
                let norm = if a < 0 && output_rank > 0 {
                    a + output_rank
                } else {
                    a
                };
                norm.max(0) as u32
            })
            .collect();

        let opts = MLUnsqueezeOptions {
            label: output_name.clone(),
            axes: axes_u32,
        };
        let out = b
            .builder
            .unsqueeze_with_options(input0, opts)
            .map_err(map_op_error)?;
        let mut result = Self::record_output(b, node, &output_name, out, context, None)?;
        if let Some(values) = Self::const_i64_values(context, inputs[0].as_str()) {
            if let Some(output) = node.output.as_slice().first() {
                result.const_values.insert(output.to_string(), values);
            }
        }
        Ok(result)
    }

    /// Convert ONNX Squeeze to WebNN squeeze (emulation)
    /// ONNX Squeeze removes dimensions of size 1
    fn convert_squeeze(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.is_empty() {
            return Err(OnnxError::InvalidShape(
                "Squeeze expects at least 1 input".to_string(),
            ));
        }

        // Extract axes attribute (opset < 13) or use second input (opset >= 13)
        let mut axes: Option<Vec<i64>> = None;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axes" {
                axes = Some(attr.ints.to_vec());
            }
        }

        let output_name = output_label(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;

        let axes_values = if let Some(a) = axes {
            a
        } else {
            self.read_axes_from_attr_or_const(node, context)?
        };
        let axes_values = if let Some(rank) = context.input_rank(inputs[0].as_str()) {
            normalize_axes_best_effort(&axes_values, rank)
        } else {
            axes_values
        };

        let opts = MLSqueezeOptions {
            label: output_name.clone(),
            axes: axes_values.into_iter().map(|a| a as u32).collect(),
        };
        let out = b
            .builder
            .squeeze_with_options(input0, opts)
            .map_err(map_op_error)?;
        let mut result = Self::record_output(b, node, &output_name, out, context, None)?;
        if let Some(values) = Self::const_i64_values(context, inputs[0].as_str()) {
            if let Some(output) = node.output.as_slice().first() {
                result.const_values.insert(output.to_string(), values);
            }
        }
        Ok(result)
    }

    /// Convert ONNX Tile to WebNN tile
    /// Repeats the input tensor along each dimension according to the repeats input
    fn convert_tile(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 2 {
            return Err(OnnxError::InvalidShape(format!(
                "Tile expects 2 inputs (input, repeats), got {}",
                inputs.len()
            )));
        }

        let output_name = output_label(node, node_name);
        let input0 = b.resolve_operand(&inputs[0])?;

        // The repeats input must be a constant for WebNN
        let repeats_name = inputs[1].as_str();

        // Try to read repeats from const_values or initializers
        let repeats = if let Some(vals) = Self::const_i64_values(context, repeats_name) {
            vals
        } else if let Some(tensor) = context.initializers.get(repeats_name) {
            // Read from initializer
            let raw = tensor.raw_data.as_slice();
            if !raw.is_empty() {
                match tensor.data_type {
                    x if x == TensorProto_DataType::Int64 as i32 => raw
                        .chunks_exact(8)
                        .map(|c| {
                            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                        })
                        .collect(),
                    x if x == TensorProto_DataType::Int32 as i32 => raw
                        .chunks_exact(4)
                        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                        .collect(),
                    _ => {
                        return Err(OnnxError::InvalidShape(
                            "Tile repeats must be int32 or int64".to_string(),
                        ))
                    }
                }
            } else if !tensor.int64_data.as_slice().is_empty() {
                tensor.int64_data.as_slice().to_vec()
            } else if !tensor.int32_data.as_slice().is_empty() {
                tensor
                    .int32_data
                    .as_slice()
                    .iter()
                    .map(|&v| v as i64)
                    .collect()
            } else {
                return Err(OnnxError::InvalidShape(
                    "Tile repeats tensor has no data".to_string(),
                ));
            }
        } else {
            return Err(OnnxError::InvalidShape(
                "Tile repeats must be constant for WebNN".to_string(),
            ));
        };

        let reps_u32: Vec<u32> = repeats
            .iter()
            .map(|&v| {
                u32::try_from(v)
                    .map_err(|_| OnnxError::InvalidShape(format!("invalid tile repetition: {v}")))
            })
            .collect::<Result<_, _>>()?;
        let out = b
            .builder
            .tile_with_options(input0, reps_u32, OnnxBuilder::labeled_options(&output_name))
            .map_err(map_op_error)?;
        Self::record_output(b, node, &output_name, out, context, Some(&inputs[0]))
    }

    /// Convert ONNX Flatten to WebNN reshape.
    ///
    /// ONNX Flatten reshapes the input to a 2-D matrix `(d_0 * ... * d_{axis-1},
    /// d_axis * ... * d_{n-1})` where `axis` defaults to 1 and may be negative.
    fn convert_flatten(
        &self,
        node: &NodeProto,
        node_name: &str,
        context: &ConversionContext,
        b: &mut OnnxBuilder<'_, '_, '_>,
    ) -> Result<ConversionResult, OnnxError> {
        let inputs = node.input.as_slice();
        if inputs.len() != 1 {
            return Err(OnnxError::InvalidShape(format!(
                "Flatten expects 1 input, got {}",
                inputs.len()
            )));
        }

        let mut axis: i64 = 1;
        for attr in node.attribute.as_slice() {
            if attr.name.as_str() == "axis" {
                axis = attr.i;
                break;
            }
        }

        let output_name = output_label(node, node_name);
        let input_raw = inputs[0].to_string();
        let input_id = b.resolve_operand(&input_raw)?;
        let input_shape = context
            .value_shapes
            .get(&input_raw)
            .or_else(|| context.value_shapes.get(&sanitize_identifier(&input_raw)))
            .cloned()
            .ok_or_else(|| {
                OnnxError::InvalidShape(format!("Flatten: input '{}' shape is unknown", input_raw))
            })?;

        let rank = input_shape.len() as i64;
        let normalized_axis = if axis < 0 { axis + rank } else { axis };
        if normalized_axis < 0 || normalized_axis > rank {
            return Err(OnnxError::InvalidShape(format!(
                "Flatten axis {} out of range for input rank {}",
                axis, rank
            )));
        }
        let axis_usize = normalized_axis as usize;

        // ONNX semantics: axis == 0 means output [1, prod(shape)]; axis == rank means [prod(shape), 1].
        let outer: i64 = if axis_usize == 0 {
            1
        } else {
            input_shape[..axis_usize].iter().product()
        };
        let inner: i64 = if axis_usize == input_shape.len() {
            1
        } else {
            input_shape[axis_usize..].iter().product()
        };

        let new_shape = u32_slice_to_mldim(&[outer as u32, inner as u32]);
        Self::emit_reshape_with_shape(
            b,
            node,
            &output_name,
            input_id,
            new_shape,
            context,
            &input_raw,
        )
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
        let attr = AttributeProto {
            name: name.to_string(),
            i: value,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    fn add_ints_attribute(node: &mut NodeProto, name: &str, values: Vec<i64>) {
        let attr = AttributeProto {
            name: name.to_string(),
            ints: values,
            ..Default::default()
        };
        node.attribute.push(attr);
    }

    #[test]
    fn test_reshape_handler_supports() {
        let handler = ReshapeHandler;
        assert!(handler.supports("Reshape"));
        assert!(handler.supports("Transpose"));
        assert!(handler.supports("Concat"));
        assert!(handler.supports("Split"));
        assert!(handler.supports("Unsqueeze"));
        assert!(handler.supports("Squeeze"));
        assert!(handler.supports("Tile"));
        assert!(!handler.supports("Add"));
    }

    #[test]
    fn test_convert_reshape() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape"], vec!["reshaped"]);

        // Create a mock shape initializer [1, 2, 3, 4]
        let shape_tensor = crate::protos::onnx::TensorProto {
            name: "shape".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![4],
            int64_data: vec![1, 2, 3, 4],
            ..Default::default()
        };

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("shape".to_string(), &shape_tensor);

        // Add input shape for inference (24 elements = 1*2*3*4)
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![2, 3, 4]); // 24 elements

        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_reshape_fallback_when_inference_diverges() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape"], vec!["reshaped"]);

        // Target shape contains -1 and hidden size (common transformer pattern)
        let shape_tensor = crate::protos::onnx::TensorProto {
            name: "shape".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![-1, 768],
            ..Default::default()
        };

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("shape".to_string(), &shape_tensor);

        // Deliberately incomplete/partial shape info (1 element), which cannot satisfy [-1, 768]
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![1]);

        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_reshape_errors_when_shape_non_const_and_input_unknown() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape_dyn"], vec!["reshaped"]);

        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let err = crate::onnx::ops::convert_handler_with_context(&handler, &node, &context)
            .expect_err("expected reshape error");
        let msg = err.to_string();
        assert!(msg.contains("shape input"));
        assert!(msg.contains("must be a constant"));
    }

    #[test]
    fn test_convert_reshape_uses_known_output_shape_when_shape_input_non_const() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape_dyn"], vec!["reshaped"]);

        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("reshaped".to_string(), vec![1, 128, 384]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context)
            .expect("reshape should convert");
    }

    #[test]
    fn test_convert_reshape_prefers_dynamic_output_dims_over_static_shape_tensor() {
        let handler = ReshapeHandler;
        let node = create_test_node("Reshape", vec!["data", "shape_const"], vec!["reshaped"]);

        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![4096]);
        value_shapes.insert("reshaped".to_string(), vec![4096, 1]);
        let mut value_shape_dims = std::collections::HashMap::new();
        value_shape_dims.insert(
            "reshaped".to_string(),
            vec![
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "sequence_length".to_string(),
                    max_size: 4096,
                }),
                rustnn::graph::Dimension::Static(1),
            ],
        );
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("shape_const".to_string(), vec![4096, 1]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: &value_shape_dims,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context)
            .expect("reshape should convert");
    }

    #[test]
    fn test_convert_transpose() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Transpose", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "perm", vec![1, 0, 2]);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 3, 4]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_transpose_propagates_const_values() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Transpose", vec!["shape_values"], vec!["transposed"]);
        add_ints_attribute(&mut node, "perm", vec![1, 0]);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("shape_values".to_string(), vec![2, 2]);
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("shape_values".to_string(), vec![1, 2, 3, 4]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(
            result.const_values.get("transposed"),
            Some(&vec![1, 3, 2, 4])
        );
    }

    #[test]
    fn test_convert_expand_uses_output_shape_when_shape_input_non_const() {
        let handler = ReshapeHandler;
        let node = create_test_node("Expand", vec!["data", "shape_dyn"], vec!["expanded"]);

        let initializers = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();

        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![1, 1, 768]);
        value_shapes.insert("expanded".to_string(), vec![1, 1, 768]);

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
    fn test_convert_expand_prefers_dynamic_output_dims_over_static_shape_tensor() {
        let handler = ReshapeHandler;
        let node = create_test_node("Expand", vec!["data", "shape_const"], vec!["expanded"]);

        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("data".to_string(), vec![1, 1, 4096, 1]);
        value_shapes.insert("expanded".to_string(), vec![1, 1, 4096, 4096]);
        let mut value_shape_dims = std::collections::HashMap::new();
        value_shape_dims.insert(
            "expanded".to_string(),
            vec![
                rustnn::graph::Dimension::Static(1),
                rustnn::graph::Dimension::Static(1),
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "sequence_length".to_string(),
                    max_size: 4096,
                }),
                rustnn::graph::Dimension::Dynamic(rustnn::graph::DynamicDimension {
                    name: "past_sequence_length + 1".to_string(),
                    max_size: 4096,
                }),
            ],
        );
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("shape_const".to_string(), vec![1, 1, 1, 1]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: &value_shape_dims,
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        crate::onnx::ops::convert_handler_with_context(&handler, &node, &context)
            .expect("expand should convert");
    }

    #[test]
    fn test_convert_concat() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Concat", vec!["a", "b", "c"], vec!["result"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("a".to_string(), vec![1, 2, 3]);
        value_shapes.insert("b".to_string(), vec![1, 2, 3]);
        value_shapes.insert("c".to_string(), vec![1, 2, 3]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_concat_propagates_const_values() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Concat", vec!["a", "b"], vec!["shape"]);
        add_int_attribute(&mut node, "axis", 0);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("a".to_string(), vec![1]);
        value_shapes.insert("b".to_string(), vec![1]);
        let mut const_values = std::collections::HashMap::new();
        const_values.insert("a".to_string(), vec![2]);
        const_values.insert("b".to_string(), vec![3]);
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
        let context = ConversionContext {
            initializers: &initializers,
            value_shapes: &value_shapes,
            value_shape_dims: crate::onnx::ops::empty_value_shape_dims(),
            const_values: &const_values,
            value_ids: &value_ids,
            value_types: &value_types,
        };

        let result =
            crate::onnx::ops::convert_handler_with_context(&handler, &node, &context).unwrap();
        assert_eq!(result.const_values.get("shape"), Some(&vec![2, 3]));
    }

    #[test]
    fn test_convert_split() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Split", vec!["x"], vec!["y1", "y2"]);
        add_int_attribute(&mut node, "axis", -1);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 4]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_unsqueeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Unsqueeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![0, 2]);
        let initializers = std::collections::HashMap::new();
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_unsqueeze_opset13_with_input_axes() {
        // Test ONNX opset 13+ where axes are provided as a second input tensor
        let handler = ReshapeHandler;
        let node = create_test_node("Unsqueeze", vec!["x", "axes_tensor"], vec!["y"]);

        // Create a mock axes tensor [1, 3]
        let axes_tensor = crate::protos::onnx::TensorProto {
            name: "axes_tensor".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![1, 3],
            ..Default::default()
        };

        let leaked_axes: &'static crate::protos::onnx::TensorProto =
            Box::leak(Box::new(axes_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("axes_tensor".to_string(), leaked_axes);
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_unsqueeze_opset13_normalizes_negative_axis_against_output_rank() {
        let handler = ReshapeHandler;
        let node = create_test_node("Unsqueeze", vec!["x", "axes_tensor"], vec!["y"]);

        let axes_tensor = crate::protos::onnx::TensorProto {
            name: "axes_tensor".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![1],
            int64_data: vec![-1],
            ..Default::default()
        };

        let leaked_axes: &'static crate::protos::onnx::TensorProto =
            Box::leak(Box::new(axes_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("axes_tensor".to_string(), leaked_axes);
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 3, 4, 5]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_squeeze() {
        let handler = ReshapeHandler;
        let mut node = create_test_node("Squeeze", vec!["x"], vec!["y"]);
        add_ints_attribute(&mut node, "axes", vec![1]);
        let initializers = std::collections::HashMap::new();
        let mut value_shapes = std::collections::HashMap::new();
        value_shapes.insert("x".to_string(), vec![2, 1]);
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
    fn test_convert_tile() {
        let handler = ReshapeHandler;
        let node = create_test_node("Tile", vec!["input", "repeats"], vec!["output"]);

        // Create a mock repeats tensor [2, 3]
        let repeats_tensor = crate::protos::onnx::TensorProto {
            name: "repeats".to_string(),
            data_type: crate::protos::onnx::TensorProto_DataType::Int64.into(),
            dims: vec![2],
            int64_data: vec![2, 3],
            ..Default::default()
        };

        let leaked_repeats: &'static crate::protos::onnx::TensorProto =
            Box::leak(Box::new(repeats_tensor));

        let mut initializers = std::collections::HashMap::new();
        initializers.insert("repeats".to_string(), leaked_repeats);
        let value_shapes = std::collections::HashMap::new();
        let const_values = std::collections::HashMap::new();
        let value_ids = std::collections::HashMap::new();
        let value_types = std::collections::HashMap::new();
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
