/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Shared helpers for direct [`MLGraphBuilder`] lowering (no Node IR).

use crate::onnx::builder::{map_op_error, operand_index, OnnxBuilder};
use crate::onnx::convert::{sanitize_identifier, OnnxError};
use crate::protos::onnx::{TensorProto, TensorProto_DataType};
use rustnn::graph::{Dimension, DynamicDimension};
use rustnn::mlcontext::MLOperand;
use rustnn::operator_options::{MLDimension, MLDynamicDimension, MLSliceOptions};
use serde_json::{json, Value};

pub fn record_node_output(
    b: &mut OnnxBuilder<'_, '_, '_>,
    onnx_output: &str,
    label: &str,
    op: MLOperand,
) {
    b.record_operand(&[onnx_output, label], op);
}

pub fn output_label(node: &crate::protos::onnx::NodeProto, node_name: &str) -> String {
    if let Some(out) = node.output.first() {
        sanitize_identifier(out)
    } else {
        format!("{node_name}_output")
    }
}

pub fn ast_dims_to_mldim(dims: &[Dimension]) -> Vec<MLDimension> {
    dims.iter()
        .map(|d| match d {
            Dimension::Static(v) => MLDimension::Static(*v),
            Dimension::Dynamic(dd) => MLDimension::Dynamic(MLDynamicDimension {
                name: dd.name.clone(),
                max_size: dd.max_size,
            }),
        })
        .collect()
}

/// Merge ONNX dim_param metadata with computed static shape values for reshape/expand.
pub fn merge_dims_with_static_values(
    dims: &[Dimension],
    static_values: &[u32],
) -> Option<Vec<MLDimension>> {
    if !dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))) {
        return None;
    }
    if dims.len() != static_values.len() {
        return None;
    }
    Some(
        dims.iter()
            .zip(static_values.iter())
            .map(|(d, &sv)| match d {
                Dimension::Dynamic(dd) => MLDimension::Dynamic(MLDynamicDimension {
                    name: dd.name.clone(),
                    max_size: dd.max_size,
                }),
                Dimension::Static(_) => MLDimension::Static(sv),
            })
            .collect(),
    )
}

/// Like [`merge_dims_with_static_values`] but accepts i64 shape values from ONNX inference.
pub fn merge_dims_with_i64_values(
    dims: &[Dimension],
    static_values: &[i64],
) -> Option<Vec<MLDimension>> {
    if !dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))) {
        return None;
    }
    if !static_values.is_empty() && dims.len() != static_values.len() {
        return None;
    }
    Some(
        dims.iter()
            .map(|d| match d {
                Dimension::Static(v) => MLDimension::Static(*v),
                Dimension::Dynamic(dd) => MLDimension::Dynamic(MLDynamicDimension {
                    name: dd.name.clone(),
                    max_size: dd.max_size,
                }),
            })
            .collect(),
    )
}

pub fn u32_slice_to_mldim(dims: &[u32]) -> Vec<MLDimension> {
    dims.iter().copied().map(MLDimension::Static).collect()
}

pub fn i64_slice_to_mldim(dims: &[i64]) -> Result<Vec<MLDimension>, OnnxError> {
    dims.iter()
        .map(|&d| {
            if d < 0 {
                Err(OnnxError::InvalidShape(format!(
                    "negative dimension {d} is not valid for static MLDimension"
                )))
            } else {
                Ok(MLDimension::Static(d as u32))
            }
        })
        .collect()
}

pub fn i64_starts_as_u32(starts: &[i64]) -> Result<Vec<u32>, OnnxError> {
    starts
        .iter()
        .map(|&s| {
            u32::try_from(s).map_err(|_| {
                OnnxError::InvalidShape(format!("slice start {s} is negative or too large"))
            })
        })
        .collect()
}

pub fn slice_sizes_from_i64(
    sizes: &[i64],
    dynamic: &[Option<DynamicDimension>],
) -> Result<Vec<MLDimension>, OnnxError> {
    if sizes.len() != dynamic.len() {
        return Err(OnnxError::InvalidShape(
            "slice sizes and dynamic metadata length mismatch".to_string(),
        ));
    }
    sizes
        .iter()
        .zip(dynamic.iter())
        .map(|(&sz, dyn_info)| {
            if let Some(dd) = dyn_info {
                Ok(MLDimension::Dynamic(MLDynamicDimension {
                    name: dd.name.clone(),
                    max_size: dd.max_size,
                }))
            } else {
                if sz < 0 {
                    return Err(OnnxError::InvalidShape(format!(
                        "slice size {sz} is negative"
                    )));
                }
                Ok(MLDimension::Static(sz as u32))
            }
        })
        .collect()
}

pub fn optional_operand_index(
    b: &OnnxBuilder<'_, '_, '_>,
    name: Option<&str>,
) -> Result<Option<u32>, OnnxError> {
    match name {
        Some(n) => Ok(Some(operand_index(b.resolve_operand(n)?))),
        None => Ok(None),
    }
}

pub fn map_op_result<T>(
    result: Result<T, rustnn::error::GraphBuilderError>,
) -> Result<T, OnnxError> {
    result.map_err(map_op_error)
}

pub fn reshape_with_shape(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
    new_shape: Vec<MLDimension>,
) -> Result<MLOperand, OnnxError> {
    map_op_result(b.builder.reshape_with_options(
        input,
        new_shape,
        OnnxBuilder::labeled_options(label),
    ))
}

pub fn expand_with_shape(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
    new_shape: Vec<MLDimension>,
) -> Result<MLOperand, OnnxError> {
    map_op_result(b.builder.expand_with_options(
        input,
        new_shape,
        OnnxBuilder::labeled_options(label),
    ))
}

pub fn slice_with_params(
    b: &mut OnnxBuilder<'_, '_, '_>,
    input: MLOperand,
    label: &str,
    starts: &[u32],
    sizes: &[MLDimension],
) -> Result<MLOperand, OnnxError> {
    let opts = MLSliceOptions {
        label: label.to_string(),
        ..Default::default()
    };
    map_op_result(b.builder.slice_with_options(input, starts, sizes, opts))
}

/// First scalar element of an ONNX tensor as WebNN `MLNumber` (`serde_json::Value`).
pub fn ml_number_from_tensor(t: &TensorProto) -> Result<Value, OnnxError> {
    match t.data_type {
        x if x == TensorProto_DataType::Float as i32 => {
            if !t.float_data.is_empty() {
                return Ok(json!(t.float_data[0]));
            }
            if t.raw_data.len() >= 4 {
                let bits = u32::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                ]);
                return Ok(json!(f32::from_bits(bits)));
            }
            Ok(json!(0.0))
        }
        x if x == TensorProto_DataType::Float16 as i32 => {
            if t.raw_data.len() >= 2 {
                let bits = u16::from_le_bytes([t.raw_data[0], t.raw_data[1]]);
                return Ok(json!(half::f16::from_bits(bits).to_f32()));
            }
            Ok(json!(0.0))
        }
        x if x == TensorProto_DataType::Int64 as i32 => {
            if !t.int64_data.is_empty() {
                return Ok(json!(t.int64_data[0]));
            }
            if t.raw_data.len() >= 8 {
                let v = i64::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                    t.raw_data[4],
                    t.raw_data[5],
                    t.raw_data[6],
                    t.raw_data[7],
                ]);
                return Ok(json!(v));
            }
            Ok(json!(0))
        }
        x if x == TensorProto_DataType::Int32 as i32 => {
            if !t.int32_data.is_empty() {
                return Ok(json!(t.int32_data[0]));
            }
            if t.raw_data.len() >= 4 {
                let v = i32::from_le_bytes([
                    t.raw_data[0],
                    t.raw_data[1],
                    t.raw_data[2],
                    t.raw_data[3],
                ]);
                return Ok(json!(v));
            }
            Ok(json!(0))
        }
        _ => Err(OnnxError::InvalidShape(format!(
            "unsupported scalar tensor type {}",
            t.data_type
        ))),
    }
}
