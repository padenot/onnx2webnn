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

// Static shape/type inference scaffold for ONNX graphs.
// Conservative: records only fully-static shapes and folds small integer constants
// to unblock reshape/axes/starts/ends calculations. Dynamic dims cause errors so
// callers can ask users to run onnx-simplifier or provide overrides.
use crate::onnx::convert::{map_onnx_data_type, sanitize_identifier};
use rustnn::DataType;
use rustnn::graph::{Dimension, DynamicDimension};
use std::collections::BTreeMap;
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    GraphProto, ModelProto, NodeProto, TensorProto, TensorProto_DataType,
};
use std::collections::{HashMap, HashSet};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ShapeInferenceError {
    #[error("input '{0}' is missing shape information")]
    MissingInputShape(String),
    #[error("input '{input}' has dynamic dimension '{dim}', please provide an override")]
    DynamicDim { input: String, dim: String },
    #[error("unsupported ONNX data type: {0}")]
    UnsupportedDataType(i32),
    #[error("could not infer shape for op '{op}'")]
    CannotInfer { op: String },
}

#[derive(Debug, Default)]
pub struct InferenceResult {
    pub value_shapes: HashMap<String, Vec<i64>>,
    pub value_types: HashMap<String, DataType>,
    pub const_values: HashMap<String, Vec<i64>>,
}

/// Run a lightweight static shape/type inference pass.
/// Returns only fully-known shapes; dynamic dimensions trigger an error.
pub fn infer_static_shapes(
    model: &ModelProto,
    overrides: &HashMap<String, u32>,
) -> Result<InferenceResult, ShapeInferenceError> {
    let mut result = InferenceResult::default();

    if model.graph.is_none() {
        return Ok(result);
    }

    let graph = model.graph.as_ref().unwrap();
    let initializer_names: HashSet<String> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|i| i.name.as_str().to_string())
        .collect();
    let initializers: HashMap<String, &TensorProto> = graph
        .initializer
        .as_slice()
        .iter()
        .map(|init| (init.name.clone(), init))
        .collect();

    seed_inputs(graph, overrides, &initializer_names, &mut result)?;
    seed_initializers(graph, &mut result)?;
    seed_constant_nodes(graph, &mut result)?;

    propagate_node_shapes(graph, &initializers, &mut result)?;

    Ok(result)
}

fn seed_inputs(
    graph: &GraphProto,
    overrides: &HashMap<String, u32>,
    initializer_names: &HashSet<String>,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for input in graph.input.as_slice() {
        let name = input.name.as_str().to_string();

        if initializer_names.contains(&name) {
            continue;
        }

        let type_proto = input
            .r#type
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let tensor_type = match &type_proto.value {
            Some(TypeProtoValue::TensorType(tt)) => tt,
            _ => return Err(ShapeInferenceError::MissingInputShape(name.clone())),
        };

        let dtype = if tensor_type.elem_type != 0 {
            map_onnx_data_type(tensor_type.elem_type)
                .map_err(|_| ShapeInferenceError::UnsupportedDataType(tensor_type.elem_type))?
        } else {
            return Err(ShapeInferenceError::UnsupportedDataType(0));
        };

        let shape = tensor_type
            .shape
            .as_ref()
            .ok_or_else(|| ShapeInferenceError::MissingInputShape(name.clone()))?;

        let mut shape_dims = Vec::new();
        for dim in shape.dim.as_slice() {
            if let Some(value) = &dim.value {
                match value {
                    DimensionValue::DimValue(v) => {
                        shape_dims.push(*v);
                    }
                    DimensionValue::DimParam(key) => {
                        if let Some(v) = overrides.get(key.as_str()) {
                            shape_dims.push(*v as i64);
                        } else {
                            return Err(ShapeInferenceError::DynamicDim {
                                input: name.clone(),
                                dim: key.clone(),
                            });
                        }
                    }
                }
            } else {
                return Err(ShapeInferenceError::MissingInputShape(name.clone()));
            }
        }

        result.value_types.insert(name.clone(), dtype);
        result.value_shapes.insert(name, shape_dims);
    }
    Ok(())
}

fn seed_initializers(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for init in graph.initializer.as_slice() {
        let name = init.name.as_str().to_string();

        // Skip initializers with unsupported data types (e.g. fp16) — we still
        // record their shape so downstream shape inference can proceed.
        let dtype = match map_onnx_data_type(init.data_type) {
            Ok(dt) => dt,
            Err(_) => {
                // Store shape-only for unsupported types (no type, no const values)
                let shape: Vec<i64> = init.dims.as_slice().to_vec();
                result.value_shapes.insert(name, shape);
                continue;
            }
        };
        let shape: Vec<i64> = init.dims.as_slice().to_vec();
        result.value_types.insert(name.clone(), dtype.clone());
        result.value_shapes.insert(name.clone(), shape);

        if matches!(
            dtype,
            DataType::Int32 | DataType::Int64 | DataType::Uint32 | DataType::Uint64
        ) {
            let values = read_int_tensor(init);
            if !values.is_empty() {
                result.const_values.insert(name, values);
            }
        }
    }
    Ok(())
}

fn seed_constant_nodes(
    graph: &GraphProto,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    for node in graph.node.as_slice() {
        if node.op_type.as_str() != "Constant" {
            continue;
        }

        if let Some(out) = node.output.as_slice().first() {
            let out_name = out.to_string();

            if let Some(attr) = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "value" && a.t.is_some())
            {
                let t = attr.t.as_ref().unwrap();
                let dtype = match map_onnx_data_type(t.data_type) {
                    Ok(dt) => dt,
                    Err(_) => {
                        // Skip Constant nodes with unsupported types (e.g. fp16); record shape only.
                        let shape: Vec<i64> = t.dims.as_slice().to_vec();
                        result.value_shapes.insert(out_name, shape);
                        continue;
                    }
                };
                result.value_types.insert(out_name.clone(), dtype);

                let vals = read_int_tensor(t);
                if !vals.is_empty() {
                    result.const_values.insert(out_name.clone(), vals.clone());
                    let shape: Vec<i64> = if vals.len() == 1 {
                        Vec::new()
                    } else {
                        vec![vals.len() as i64]
                    };
                    result.value_shapes.insert(out_name.clone(), shape);
                }
            }
        }
    }
    Ok(())
}

fn propagate_node_shapes(
    graph: &GraphProto,
    initializers: &HashMap<String, &TensorProto>,
    result: &mut InferenceResult,
) -> Result<(), ShapeInferenceError> {
    let mut progress = true;
    let max_iters = 8;
    let mut iter = 0;

    while progress && iter < max_iters {
        progress = false;
        iter += 1;

        for node in graph.node.as_slice() {
            let outputs = node.output.as_slice();
            if outputs.is_empty() {
                continue;
            }
            if outputs
                .iter()
                .all(|o| result.value_shapes.contains_key(o.as_str()))
            {
                continue;
            }

            if let Some(shape) = infer_node_output_shape(
                node,
                &result.value_shapes,
                initializers,
                &result.const_values,
            ) {
                let out_name = outputs[0].to_string();
                result.value_shapes.entry(out_name.clone()).or_insert(shape);

                // Propagate dtype from first input if available.
                if let Some(first_in) = node.input.as_slice().first() {
                    if let Some(dtype) = result.value_types.get(first_in).cloned() {
                        result.value_types.entry(out_name.clone()).or_insert(dtype);
                    }
                }

                progress = true;
            }
        }

        // Opportunistic const folding for integer tensors to unlock more shapes.
        progress |= fold_integer_constants(graph, result);
    }

    Ok(())
}

#[allow(dead_code)]

pub fn infer_node_output_shape(
    node: &crate::protos::onnx::NodeProto,
    value_shapes: &HashMap<String, Vec<i64>>,
    initializers: &HashMap<String, &TensorProto>,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let op = node.op_type.as_str();

    match op {
        // Unary operations that preserve shape
        "Cast" | "Relu" | "Tanh" | "Sigmoid" | "Erf" | "Softmax" | "Gelu" | "Exp" | "Log"
        | "Abs" | "Neg" | "Sqrt" | "Floor" | "Ceil" | "Round" | "Not"
        | "LayerNormalization" | "Trilu" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            value_shapes.get(ins[0].as_str()).cloned()
        }

        // Binary operations with NumPy-style broadcasting semantics.
        "Add" | "Sub" | "Mul" | "Div" | "Pow" | "And" | "Or" | "Xor" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let shape_a = value_shapes.get(ins[0].as_str());
            let shape_b = value_shapes.get(ins[1].as_str());

            match (shape_a, shape_b) {
                (Some(a), Some(b)) => {
                    let rank = a.len().max(b.len());
                    let mut out_rev = Vec::with_capacity(rank);
                    for i in 0..rank {
                        let da = a.get(a.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        let db = b.get(b.len().wrapping_sub(1 + i)).copied().unwrap_or(1);
                        if da == db || da == 1 {
                            out_rev.push(db);
                        } else if db == 1 {
                            out_rev.push(da);
                        } else {
                            return None;
                        }
                    }
                    out_rev.reverse();
                    Some(out_rev)
                }
                (Some(a), None) => Some(a.clone()),
                (None, Some(b)) => Some(b.clone()),
                (None, None) => None,
            }
        }

        // MatMul (2D matrix multiplication)
        "MatMul" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            // Handle 2D case: [M, K] @ [K, N] -> [M, N]
            if a_shape.len() >= 2 && b_shape.len() >= 2 {
                let m = a_shape[a_shape.len() - 2];
                let n = b_shape[b_shape.len() - 1];

                // For higher-dim inputs, preserve batch dimensions
                if a_shape.len() == 2 && b_shape.len() == 2 {
                    return Some(vec![m, n]);
                } else if a_shape.len() > 2 {
                    let mut result = a_shape[..a_shape.len() - 2].to_vec();
                    result.push(m);
                    result.push(n);
                    return Some(result);
                }
            }
            None
        }

        // Transpose preserves shape with permuted dimensions
        "Transpose" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Get perm attribute
            let perm: Vec<usize> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "perm")
                .map(|a| a.ints.iter().map(|&i| i as usize).collect::<Vec<usize>>())
                .unwrap_or_else(|| (0..input_shape.len()).rev().collect());

            // Apply permutation — return None if any index is out of bounds
            if perm.iter().any(|&i| i >= input_shape.len()) {
                return None;
            }
            Some(perm.iter().map(|&i| input_shape[i]).collect())
        }

        // Reduce operations
        "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;

            // Check keepdims attribute (default is 1/true)
            let keepdims = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "keepdims")
                .map(|a| a.i != 0)
                .unwrap_or(true);

            // Get axes — from attribute (opset ≤ 12) or second input (opset 13+).
            let mut axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();
            if axes.is_empty() && ins.len() >= 2 && !ins[1].is_empty() {
                if let Some(vals) = const_values.get(ins[1].as_str()) {
                    if !vals.is_empty() {
                        axes = vals.iter().map(|&v| v).collect();
                    }
                }
            }

            if axes.is_empty() {
                // Reduce all dimensions
                if keepdims {
                    Some(vec![1; input_shape.len()])
                } else {
                    Some(vec![])
                }
            } else {
                // Reduce specific axes
                let mut output_shape = input_shape.clone();
                for &axis in &axes {
                    let idx = if axis < 0 {
                        (input_shape.len() as i64 + axis) as usize
                    } else {
                        axis as usize
                    };
                    if idx < output_shape.len() {
                        if keepdims {
                            output_shape[idx] = 1;
                        } else {
                            output_shape[idx] = -1; // Mark for removal
                        }
                    }
                }
                if !keepdims {
                    output_shape.retain(|&d| d != -1);
                }
                Some(output_shape)
            }
        }

        // Gemm (generalized matrix multiplication)
        "Gemm" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let a_shape = value_shapes.get(ins[0].as_str())?;
            let b_shape = value_shapes.get(ins[1].as_str())?;

            if a_shape.len() != 2 || b_shape.len() != 2 {
                return None;
            }

            // Check transA and transB attributes
            let trans_a = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transA")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let trans_b = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "transB")
                .and_then(|a| if a.i != 0 { Some(a.i != 0) } else { None })
                .unwrap_or(false);

            let m = if trans_a { a_shape[1] } else { a_shape[0] };
            let n = if trans_b { b_shape[0] } else { b_shape[1] };

            Some(vec![m, n])
        }

        "Gather" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let data_shape = value_shapes.get(ins[0].as_str())?;
            let indices_shape = value_shapes.get(ins[1].as_str())?;

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += data_shape.len() as i64;
            }

            let axis_usize = axis as usize;
            if axis_usize > data_shape.len() {
                return None;
            }

            let mut output = Vec::new();
            output.extend_from_slice(&data_shape[..axis_usize]);
            output.extend(indices_shape.iter().cloned());
            if axis_usize < data_shape.len() {
                output.extend_from_slice(&data_shape[axis_usize + 1..]);
            }
            Some(output)
        }

        "Unsqueeze" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?.clone();
            let mut axes: Vec<i64> = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axes")
                .map(|a| a.ints.clone())
                .unwrap_or_default();

            if axes.is_empty() {
                return None;
            }

            axes.sort();
            let mut output_shape = input_shape;
            for axis in axes {
                let idx = if axis < 0 {
                    (output_shape.len() as i64 + axis + 1) as usize
                } else {
                    axis as usize
                };
                if idx <= output_shape.len() {
                    output_shape.insert(idx, 1);
                }
            }
            Some(output_shape)
        }

        "Concat" => {
            let mut shapes = Vec::new();
            for inp in node.input.as_slice() {
                let shape = value_shapes.get(inp.as_str())?;
                shapes.push(shape.clone());
            }

            if shapes.is_empty() {
                return None;
            }

            let mut axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .and_then(|a| if a.i != 0 { Some(a.i) } else { None })
                .unwrap_or(0);

            if axis < 0 {
                axis += shapes[0].len() as i64;
            }
            let axis_usize = axis as usize;

            let mut output = shapes[0].clone();
            for shape in shapes.iter().skip(1) {
                if shape.len() != output.len() || axis_usize >= shape.len() {
                    return None;
                }
                output[axis_usize] += shape[axis_usize];
            }
            Some(output)
        }

        "Pad" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let input_shape = value_shapes.get(ins[0].as_str())?;
            let rank = input_shape.len();
            let pads = crate::onnx::ops::pad::read_onnx_pads_from_maps(
                node,
                initializers,
                const_values,
                rank,
            )
            .ok()?;
            crate::onnx::ops::pad::infer_pad_output_shape(input_shape, &pads)
        }

        "Reshape" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;
            let shape_input = ins[1].as_str();
            let mut target: Vec<i64> = if let Some(values) = const_values.get(shape_input) {
                values.clone()
            } else if let Some(shape_tensor) = initializers.get(shape_input) {
                if !shape_tensor.raw_data.as_slice().is_empty() {
                    if shape_tensor.data_type == TensorProto_DataType::Int32 as i32 {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        shape_tensor
                            .raw_data
                            .as_slice()
                            .chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
                    }
                } else if !shape_tensor.int64_data.as_slice().is_empty() {
                    shape_tensor.int64_data.as_slice().to_vec()
                } else if !shape_tensor.int32_data.as_slice().is_empty() {
                    shape_tensor
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

            if target.is_empty() {
                return None;
            }

            if target.contains(&-1) {
                let total_input: i64 = input_shape.iter().product();
                let known: i64 = target.iter().filter(|&&d| d != -1).product();
                if known == 0 || total_input % known != 0 {
                    return None;
                }
                if let Some(idx) = target.iter().position(|&d| d == -1) {
                    target[idx] = total_input / known;
                }
            }

            Some(target)
        }

        // Pooling: maxPool / averagePool / global variants.  Only handles fully-static inputs.
        "MaxPool" | "AveragePool" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            if x_shape.len() < 3 {
                return None;
            }
            let spatial_rank = x_shape.len() - 2;

            let mut auto_pad = String::from("NOTSET");
            let mut strides: Vec<i64> = vec![1; spatial_rank];
            let mut dilations: Vec<i64> = vec![1; spatial_rank];
            let mut pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let mut kernel_shape: Vec<i64> = Vec::new();
            let mut ceil_mode = false;
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "auto_pad" => {
                        if let Ok(s) = String::from_utf8(attr.s.clone()) {
                            if !s.is_empty() {
                                auto_pad = s;
                            }
                        }
                    }
                    "kernel_shape" if !attr.ints.is_empty() => kernel_shape = attr.ints.clone(),
                    "strides" if !attr.ints.is_empty() => strides = attr.ints.clone(),
                    "dilations" if !attr.ints.is_empty() => dilations = attr.ints.clone(),
                    "pads" if !attr.ints.is_empty() => pads = attr.ints.clone(),
                    "ceil_mode" => ceil_mode = attr.i != 0,
                    _ => {}
                }
            }
            if kernel_shape.len() != spatial_rank
                || strides.len() != spatial_rank
                || dilations.len() != spatial_rank
                || pads.len() != 2 * spatial_rank
            {
                return None;
            }

            let mut out_spatial = Vec::with_capacity(spatial_rank);
            for i in 0..spatial_rank {
                let in_dim = x_shape[2 + i];
                let k = kernel_shape[i];
                let s = strides[i];
                let d = dilations[i];
                let dilated_k = d * (k - 1) + 1;
                let out_dim = match auto_pad.as_str() {
                    "SAME_UPPER" | "SAME_LOWER" => (in_dim + s - 1) / s,
                    "VALID" => (in_dim - dilated_k) / s + 1,
                    _ => {
                        let pad_begin = pads[i];
                        let pad_end = pads[i + spatial_rank];
                        let numerator = in_dim + pad_begin + pad_end - dilated_k;
                        if ceil_mode {
                            (numerator + s - 1) / s + 1
                        } else {
                            numerator / s + 1
                        }
                    }
                };
                if out_dim < 0 {
                    return None;
                }
                out_spatial.push(out_dim);
            }

            let mut out = vec![x_shape[0], x_shape[1]];
            out.extend(out_spatial);
            Some(out)
        }

        "GlobalMaxPool" | "GlobalAveragePool" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            if x_shape.len() < 3 {
                return None;
            }
            let mut out = vec![x_shape[0], x_shape[1]];
            out.extend(std::iter::repeat_n(1i64, x_shape.len() - 2));
            Some(out)
        }

        "Flatten" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            let axis = node
                .attribute
                .as_slice()
                .iter()
                .find(|a| a.name.as_str() == "axis")
                .map(|a| a.i)
                .unwrap_or(1);
            let rank = x_shape.len() as i64;
            let norm = if axis < 0 { axis + rank } else { axis };
            if norm < 0 || norm > rank {
                return None;
            }
            let norm = norm as usize;
            let outer: i64 = if norm == 0 {
                1
            } else {
                x_shape[..norm].iter().product()
            };
            let inner: i64 = if norm == x_shape.len() {
                1
            } else {
                x_shape[norm..].iter().product()
            };
            Some(vec![outer, inner])
        }

        // Convolution / transposed convolution: derive output spatial dims.
        // Only handles fully-static inputs.  Higher-rank cases fall through to None.
        "Conv" | "ConvTranspose" => {
            let ins = node.input.as_slice();
            if ins.len() < 2 {
                return None;
            }
            let x_shape = value_shapes.get(ins[0].as_str())?.clone();
            let w_shape = value_shapes.get(ins[1].as_str()).cloned().or_else(|| {
                initializers
                    .get(ins[1].as_str())
                    .map(|t| t.dims.as_slice().to_vec())
            })?;
            if x_shape.len() < 3 || w_shape.len() < 3 {
                return None;
            }
            let spatial_rank = x_shape.len() - 2;
            if w_shape.len() != x_shape.len() {
                return None;
            }

            // Read attributes.
            let mut auto_pad = String::from("NOTSET");
            let mut strides: Vec<i64> = vec![1; spatial_rank];
            let mut dilations: Vec<i64> = vec![1; spatial_rank];
            let mut pads: Vec<i64> = vec![0; 2 * spatial_rank];
            let mut kernel_shape: Vec<i64> = w_shape[2..].to_vec();
            let mut group: i64 = 1;
            let mut output_padding: Vec<i64> = vec![0; spatial_rank];
            let mut output_shape_attr: Vec<i64> = Vec::new();
            for attr in node.attribute.as_slice() {
                match attr.name.as_str() {
                    "auto_pad" => {
                        if let Ok(s) = String::from_utf8(attr.s.clone()) {
                            if !s.is_empty() {
                                auto_pad = s;
                            }
                        }
                    }
                    "strides" if !attr.ints.is_empty() => strides = attr.ints.clone(),
                    "dilations" if !attr.ints.is_empty() => dilations = attr.ints.clone(),
                    "pads" if !attr.ints.is_empty() => pads = attr.ints.clone(),
                    "kernel_shape" if !attr.ints.is_empty() => kernel_shape = attr.ints.clone(),
                    "group" if attr.i > 0 => group = attr.i,
                    "output_padding" if !attr.ints.is_empty() => output_padding = attr.ints.clone(),
                    "output_shape" if !attr.ints.is_empty() => {
                        output_shape_attr = attr.ints.clone()
                    }
                    _ => {}
                }
            }
            if strides.len() != spatial_rank
                || dilations.len() != spatial_rank
                || kernel_shape.len() != spatial_rank
                || pads.len() != 2 * spatial_rank
                || output_padding.len() != spatial_rank
            {
                return None;
            }
            let _ = group; // not needed for shape inference

            let transpose = op == "ConvTranspose";
            // Output channel count.
            let m = if transpose {
                // Filter layout for ConvTranspose: (C_in, M/group, kSpatial...).
                // M = w_shape[1] * group, but with default group=1 we just use w_shape[1].
                w_shape[1] * group
            } else {
                w_shape[0]
            };

            // If output_shape attr is provided (ConvTranspose), it directly tells us H/W.
            if transpose && !output_shape_attr.is_empty() {
                let sizes = if output_shape_attr.len() == spatial_rank {
                    output_shape_attr.clone()
                } else if output_shape_attr.len() == x_shape.len() {
                    output_shape_attr[2..].to_vec()
                } else {
                    return None;
                };
                let mut out = vec![x_shape[0], m];
                out.extend(sizes);
                return Some(out);
            }

            let mut out_spatial = Vec::with_capacity(spatial_rank);
            for i in 0..spatial_rank {
                let in_dim = x_shape[2 + i];
                let k = kernel_shape[i];
                let s = strides[i];
                let d = dilations[i];
                let dilated_k = d * (k - 1) + 1;

                let out_dim = match auto_pad.as_str() {
                    "SAME_UPPER" | "SAME_LOWER" if !transpose => {
                        // Standard "SAME": out = ceil(in / stride)
                        (in_dim + s - 1) / s
                    }
                    "SAME_UPPER" | "SAME_LOWER" if transpose => {
                        // For transpose: out = in * stride
                        in_dim * s
                    }
                    "VALID" if !transpose => (in_dim - dilated_k) / s + 1,
                    "VALID" if transpose => (in_dim - 1) * s + dilated_k,
                    _ => {
                        // explicit pads (NOTSET) ΓÇö pads layout: [b1, b2, ..., bk, e1, e2, ..., ek]
                        let pad_begin = pads[i];
                        let pad_end = pads[i + spatial_rank];
                        if transpose {
                            (in_dim - 1) * s - pad_begin - pad_end + dilated_k + output_padding[i]
                        } else {
                            (in_dim + pad_begin + pad_end - dilated_k) / s + 1
                        }
                    }
                };
                if out_dim < 0 {
                    return None;
                }
                out_spatial.push(out_dim);
            }

            let mut out = vec![x_shape[0], m];
            out.extend(out_spatial);
            Some(out)
        }

        "Slice" => {
            let ins = node.input.as_slice();
            if ins.is_empty() {
                return None;
            }

            let input_shape = value_shapes.get(ins[0].as_str())?;

            let read_ints = |name: Option<&String>| -> Option<Vec<i64>> {
                if let Some(n) = name {
                    if let Some(v) = const_values.get(n) {
                        return Some(v.clone());
                    }
                    if let Some(t) = initializers.get(n) {
                        let raw = t.raw_data.as_slice();
                        if !raw.is_empty() {
                            if t.data_type == TensorProto_DataType::Int32 as i32 {
                                return Some(
                                    raw.chunks_exact(4)
                                        .map(|c| {
                                            i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64
                                        })
                                        .collect(),
                                );
                            } else {
                                return Some(
                                    raw.chunks_exact(8)
                                        .map(|c| {
                                            i64::from_le_bytes([
                                                c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                            ])
                                        })
                                        .collect(),
                                );
                            }
                        } else if !t.int64_data.as_slice().is_empty() {
                            return Some(t.int64_data.as_slice().to_vec());
                        } else if !t.int32_data.as_slice().is_empty() {
                            return Some(
                                t.int32_data.as_slice().iter().map(|&v| v as i64).collect(),
                            );
                        }
                    }
                }
                None
            };

            let starts = read_ints(ins.get(1))?;
            let ends = read_ints(ins.get(2))?;
            let axes =
                read_ints(ins.get(3)).unwrap_or_else(|| (0..input_shape.len() as i64).collect());
            let steps = read_ints(ins.get(4)).unwrap_or_else(|| vec![1; axes.len()]);

            if axes.len() != starts.len() || axes.len() != ends.len() || axes.len() != steps.len() {
                return None;
            }

            let mut output = input_shape.clone();
            for i in 0..axes.len() {
                let axis = if axes[i] < 0 {
                    (input_shape.len() as i64 + axes[i]) as usize
                } else {
                    axes[i] as usize
                };
                if axis >= output.len() {
                    return None;
                }

                let step = steps[i];
                if step != 1 {
                    return None;
                }

                let dim = input_shape[axis];
                let mut start = starts[i];
                let mut end = ends[i];

                if start < 0 {
                    start += dim;
                }
                if end < 0 {
                    end += dim;
                }

                start = start.max(0);
                end = end.min(dim);

                if end < start {
                    output[axis] = 0;
                } else {
                    output[axis] = end - start;
                }
            }

            Some(output)
        }

        _ => None,
    }
}

fn shape_numel(shape: &[i64]) -> Option<usize> {
    shape.iter().try_fold(1usize, |acc, &d| {
        if d < 0 {
            return None;
        }
        usize::try_from(d).ok().map(|v| acc.saturating_mul(v))
    })
}

fn const_shape_for_folding(
    name: &str,
    values: &[i64],
    value_shapes: &HashMap<String, Vec<i64>>,
) -> Vec<i64> {
    if let Some(shape) = value_shapes.get(name) {
        if shape_numel(shape) == Some(values.len()) {
            return shape.clone();
        }
    }

    if values.len() == 1 {
        Vec::new()
    } else {
        vec![values.len() as i64]
    }
}

fn broadcast_shape(shape_a: &[i64], shape_b: &[i64]) -> Option<Vec<i64>> {
    let rank = shape_a.len().max(shape_b.len());
    let mut out_rev = Vec::with_capacity(rank);
    for i in 0..rank {
        let da = shape_a
            .get(shape_a.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        let db = shape_b
            .get(shape_b.len().wrapping_sub(1 + i))
            .copied()
            .unwrap_or(1);
        if da <= 0 || db <= 0 {
            return None;
        }
        if da == db || da == 1 {
            out_rev.push(db);
        } else if db == 1 {
            out_rev.push(da);
        } else {
            return None;
        }
    }
    out_rev.reverse();
    Some(out_rev)
}

fn linear_index_for_broadcast_operand(
    out_linear_idx: usize,
    out_shape: &[i64],
    in_shape: &[i64],
) -> Option<usize> {
    if in_shape.is_empty() {
        return Some(0);
    }

    let in_rank = in_shape.len();
    let out_rank = out_shape.len();
    if in_rank > out_rank {
        return None;
    }

    let mut in_linear_idx = 0usize;
    let mut in_stride = 1usize;
    let mut rem = out_linear_idx;

    for out_axis_rev in 0..out_rank {
        let out_axis = out_rank - 1 - out_axis_rev;
        let out_dim = usize::try_from(out_shape[out_axis]).ok()?;
        if out_dim == 0 {
            return None;
        }
        let out_coord = rem % out_dim;
        rem /= out_dim;

        if out_axis_rev < in_rank {
            let in_axis = in_rank - 1 - out_axis_rev;
            let in_dim = usize::try_from(in_shape[in_axis]).ok()?;
            if in_dim == 0 {
                return None;
            }
            let in_coord = if in_dim == 1 { 0 } else { out_coord };
            in_linear_idx = in_linear_idx.saturating_add(in_coord.saturating_mul(in_stride));
            in_stride = in_stride.saturating_mul(in_dim);
        }
    }

    Some(in_linear_idx)
}

fn fold_binary_const_i64(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
) -> Option<(Vec<i64>, Vec<i64>)> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;

    let mut out_values = Vec::with_capacity(out_numel);
    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let v = match op_type {
            "Add" => av + bv,
            "Sub" => av - bv,
            "Mul" => av * bv,
            "Div" => {
                if bv == 0 {
                    return None;
                }
                av / bv
            }
            "Equal" => {
                if av == bv {
                    1
                } else {
                    0
                }
            }
            _ => return None,
        };
        out_values.push(v);
    }

    Some((out_values, out_shape))
}

pub(crate) fn value_shape_dims_for<'a>(
    name: &str,
    value_shape_dims: &'a HashMap<String, Vec<Dimension>>,
) -> Option<&'a [Dimension]> {
    let sanitized = sanitize_identifier(name);
    let trimmed = name.trim_start_matches('/');
    value_shape_dims
        .get(name)
        .or_else(|| value_shape_dims.get(&sanitized))
        .or_else(|| value_shape_dims.get(trimmed))
        .map(Vec::as_slice)
}

fn dims_contain_dynamic(dims: &[Dimension]) -> bool {
    dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
}

/// Propagate `value_shape_dims` through shape-preserving ONNX ops so downstream
/// Shape/Gather/Concat/Reshape chains retain dynamic batch/sequence metadata.
fn propagate_dynamic_dims_metadata(
    graph: &GraphProto,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
) {
    const PRESERVE_INPUT_SHAPE: &[&str] = &[
        "Abs",
        "Add",
        "Cast",
        "Div",
        "Equal",
        "Greater",
        "GreaterOrEqual",
        "LayerNormalization",
        "Less",
        "LessOrEqual",
        "Mul",
        "Neg",
        "Not",
        "Relu",
        "Sigmoid",
        "Sin",
        "Cos",
        "Sqrt",
        "Sub",
        "Softmax",
        "Tanh",
        "Where",
    ];

    for _ in 0..graph.node.as_slice().len().max(1) {
        let mut changed = false;
        for node in graph.node.as_slice() {
            let Some(out) = node.output.as_slice().first() else {
                continue;
            };
            if value_shape_dims.contains_key(out.as_str()) {
                continue;
            }

            let op_type = node.op_type.as_str();
            let input_dims = if op_type == "MatMul" || op_type == "Gemm" {
                node.input
                    .as_slice()
                    .first()
                    .and_then(|inp| value_shape_dims_for(inp.as_str(), value_shape_dims))
            } else if PRESERVE_INPUT_SHAPE.contains(&op_type) {
                node.input
                    .as_slice()
                    .first()
                    .and_then(|inp| value_shape_dims_for(inp.as_str(), value_shape_dims))
            } else {
                None
            };

            if let Some(dims) = input_dims {
                if !dims.is_empty() {
                    value_shape_dims.insert(out.to_string(), dims.to_vec());
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }
}

pub(crate) fn parse_dynamic_dim_expr(dim_name: &str) -> (String, i64) {
    let s = dim_name.trim();
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), offset);
        }
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        if let Ok(offset) = rhs.trim().parse::<i64>() {
            return (lhs.trim().to_string(), -offset);
        }
    }
    (s.to_string(), 0)
}

pub(crate) fn format_dynamic_dim_expr(base: &str, offset: i64) -> String {
    if offset > 0 {
        format!("{base} + {offset}")
    } else if offset < 0 {
        format!("{base} - {}", offset.abs())
    } else {
        base.to_string()
    }
}

fn parse_additive_dynamic_dim_expr(dim_name: &str) -> Option<(BTreeMap<String, i64>, i64)> {
    let expr = dim_name.trim();
    if expr.is_empty() {
        return None;
    }

    let normalized = expr.replace('+', " + ").replace('-', " - ");
    let mut terms = BTreeMap::new();
    let mut constant = 0i64;
    let mut sign = 1i64;
    let mut saw_term = false;

    for token in normalized.split_whitespace() {
        match token {
            "+" => sign = 1,
            "-" => sign = -1,
            _ => {
                saw_term = true;
                if let Ok(value) = token.parse::<i64>() {
                    constant += sign * value;
                } else {
                    *terms.entry(token.to_string()).or_insert(0) += sign;
                }
                sign = 1;
            }
        }
    }

    if !saw_term {
        return None;
    }

    terms.retain(|_, coeff| *coeff != 0);
    Some((terms, constant))
}

fn format_additive_dynamic_dim_expr(
    terms: &BTreeMap<String, i64>,
    constant: i64,
) -> Option<String> {
    if terms.is_empty() && constant == 0 {
        return None;
    }

    let mut out = String::new();
    for (name, coeff) in terms {
        for _ in 0..coeff.abs() {
            if out.is_empty() {
                if *coeff < 0 {
                    out.push_str("- ");
                }
                out.push_str(name);
            } else if *coeff < 0 {
                out.push_str(" - ");
                out.push_str(name);
            } else {
                out.push_str(" + ");
                out.push_str(name);
            }
        }
    }

    if constant != 0 {
        if out.is_empty() {
            out.push_str(&constant.to_string());
        } else if constant < 0 {
            out.push_str(" - ");
            out.push_str(&constant.abs().to_string());
        } else {
            out.push_str(" + ");
            out.push_str(&constant.to_string());
        }
    }

    Some(out)
}

fn is_runtime_resolvable_dynamic_dim_expr(dim_name: &str) -> bool {
    let s = dim_name.trim();
    if s.is_empty() || s.contains('*') || s.contains('/') {
        return false;
    }
    if let Some((lhs, rhs)) = s.rsplit_once('+') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    if let Some((lhs, rhs)) = s.rsplit_once('-') {
        return !lhs.trim().is_empty() && rhs.trim().parse::<i64>().is_ok();
    }
    true
}

fn shift_dynamic_dimension(dim: &DynamicDimension, delta: i64) -> Option<DynamicDimension> {
    let (base, offset) = parse_dynamic_dim_expr(&dim.name);
    let name = format_dynamic_dim_expr(&base, offset.checked_add(delta)?);
    let shifted_max = (dim.max_size as i64).checked_add(delta)?.max(0);
    let max_size = u32::try_from(shifted_max).ok()?;
    Some(DynamicDimension { name, max_size })
}

pub(crate) fn dynamic_scalar_dimension_for_value(
    name: &str,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<DynamicDimension> {
    let dims = value_shape_dims_for(name, value_shape_dims)?;
    if dims.len() != 1 {
        return None;
    }
    match &dims[0] {
        Dimension::Dynamic(dim) => Some(dim.clone()),
        Dimension::Static(_) => None,
    }
}

fn dimension_vector_for_value(
    name: &str,
    const_values: &HashMap<String, Vec<i64>>,
    value_shape_dims: &HashMap<String, Vec<Dimension>>,
) -> Option<Vec<Dimension>> {
    if let Some(dims) = value_shape_dims_for(name, value_shape_dims) {
        return Some(dims.to_vec());
    }
    let values = const_values.get(name)?;
    values
        .iter()
        .map(|&v| u32::try_from(v).ok().map(Dimension::Static))
        .collect()
}

fn is_trivial_static_dimension_vector(dims: &[Dimension]) -> bool {
    !dims.is_empty() && dims.iter().all(|d| matches!(d, Dimension::Static(1)))
}

fn is_all_ones_shape_vector(values: &[i64]) -> bool {
    !values.is_empty() && values.iter().all(|&v| v == 1)
}

fn combine_binary_dimension(
    op_type: &str,
    dynamic: &DynamicDimension,
    static_value: i64,
    dynamic_on_lhs: bool,
) -> Option<Dimension> {
    match op_type {
        "Add" => shift_dynamic_dimension(dynamic, static_value).map(Dimension::Dynamic),
        "Sub" if dynamic_on_lhs => {
            shift_dynamic_dimension(dynamic, -static_value).map(Dimension::Dynamic)
        }
        "Mul" if static_value == 0 => Some(Dimension::Static(0)),
        "Mul" if static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Mul" if static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: if dynamic_on_lhs {
                format!("{} * {}", dynamic.name, static_value)
            } else {
                format!("{} * {}", static_value, dynamic.name)
            },
            max_size: dynamic.max_size.saturating_mul(static_value as u32),
        })),
        "Div" if dynamic_on_lhs && static_value == 1 => Some(Dimension::Dynamic(dynamic.clone())),
        "Div" if dynamic_on_lhs && static_value > 1 => Some(Dimension::Dynamic(DynamicDimension {
            name: format!("{} / {}", dynamic.name, static_value),
            max_size: dynamic.max_size / (static_value as u32),
        })),
        _ => None,
    }
}

fn combine_dynamic_dimensions(
    op_type: &str,
    lhs: &DynamicDimension,
    rhs: &DynamicDimension,
    lhs_value: i64,
    rhs_value: i64,
) -> Option<Dimension> {
    match op_type {
        "Add" | "Sub" => {
            let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&lhs.name)?;
            let (rhs_terms, rhs_constant) = parse_additive_dynamic_dim_expr(&rhs.name)?;
            let rhs_sign = if op_type == "Add" { 1 } else { -1 };

            for (name, coeff) in rhs_terms {
                *terms.entry(name).or_insert(0) += rhs_sign * coeff;
            }
            constant += rhs_sign * rhs_constant;
            terms.retain(|_, coeff| *coeff != 0);

            let value = if op_type == "Add" {
                lhs_value.checked_add(rhs_value)?
            } else {
                lhs_value.checked_sub(rhs_value)?
            };
            if terms.is_empty() {
                return u32::try_from(value).ok().map(Dimension::Static);
            }

            let name = format_additive_dynamic_dim_expr(&terms, constant)?;
            let max_size = u32::try_from(value).ok()?;
            Some(Dimension::Dynamic(DynamicDimension { name, max_size }))
        }
        _ => None,
    }
}

fn fold_binary_dynamic_dims(
    op_type: &str,
    a_values: &[i64],
    b_values: &[i64],
    a_shape: &[i64],
    b_shape: &[i64],
    a_dims: Option<&[Dimension]>,
    b_dims: Option<&[Dimension]>,
) -> Option<Vec<Dimension>> {
    let out_shape = broadcast_shape(a_shape, b_shape)?;
    let out_numel = shape_numel(&out_shape)?;
    let mut out_dims = Vec::with_capacity(out_numel);
    let mut has_dynamic = false;

    for out_idx in 0..out_numel {
        let a_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, a_shape)?;
        let b_idx = linear_index_for_broadcast_operand(out_idx, &out_shape, b_shape)?;
        let av = *a_values.get(a_idx)?;
        let bv = *b_values.get(b_idx)?;
        let a_dim = a_dims.and_then(|dims| dims.get(a_idx));
        let b_dim = b_dims.and_then(|dims| dims.get(b_idx));

        let out_dim = match (a_dim, b_dim) {
            (Some(Dimension::Dynamic(dynamic)), Some(Dimension::Static(_)))
            | (Some(Dimension::Dynamic(dynamic)), None) => {
                let dim = combine_binary_dimension(op_type, dynamic, bv, true)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Static(_)), Some(Dimension::Dynamic(dynamic)))
            | (None, Some(Dimension::Dynamic(dynamic))) => {
                let dim = combine_binary_dimension(op_type, dynamic, av, false)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            (Some(Dimension::Dynamic(a_dynamic)), Some(Dimension::Dynamic(b_dynamic))) => {
                let dim = combine_dynamic_dimensions(op_type, a_dynamic, b_dynamic, av, bv)?;
                has_dynamic |= matches!(dim, Dimension::Dynamic(_));
                dim
            }
            _ => {
                let value = match op_type {
                    "Add" => av + bv,
                    "Sub" => av - bv,
                    "Mul" => av * bv,
                    "Div" => {
                        if bv == 0 {
                            return None;
                        }
                        av / bv
                    }
                    _ => return None,
                };
                Dimension::Static(u32::try_from(value).ok()?)
            }
        };

        out_dims.push(out_dim);
    }

    has_dynamic.then_some(out_dims)
}

pub(crate) fn dynamic_range_length_dimension(
    start: i64,
    delta: i64,
    start_dim: Option<&DynamicDimension>,
    limit: &DynamicDimension,
) -> Option<DynamicDimension> {
    if delta != 1 {
        return None;
    }

    let (mut terms, mut constant) = parse_additive_dynamic_dim_expr(&limit.name)?;
    if let Some(start_dim) = start_dim {
        let (start_terms, start_constant) = parse_additive_dynamic_dim_expr(&start_dim.name)?;
        for (name, coeff) in start_terms {
            *terms.entry(name).or_insert(0) -= coeff;
        }
        constant -= start_constant;
    } else {
        constant -= start;
    }
    terms.retain(|_, coeff| *coeff != 0);
    if terms.is_empty() {
        return None;
    }

    let name = format_additive_dynamic_dim_expr(&terms, constant)?;
    if !is_runtime_resolvable_dynamic_dim_expr(&name) {
        return None;
    }

    let max_size = u32::try_from((limit.max_size as i64).checked_sub(start)?).ok()?;
    Some(DynamicDimension { name, max_size })
}
/// Options controlling shape-subgraph constant folding.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FoldShapeConstantsOptions {
    pub require_positive_dims: bool,
    pub experimental_dynamic_inputs: bool,
    pub fold_where_values: bool,
    pub fold_reshape: bool,
    pub fold_unsqueeze_axes: bool,
}

impl FoldShapeConstantsOptions {
    fn early_pass() -> Self {
        Self {
            require_positive_dims: false,
            experimental_dynamic_inputs: false,
            fold_where_values: true,
            fold_reshape: true,
            fold_unsqueeze_axes: true,
        }
    }

    fn from_propagate(opts: &PropagateOptions) -> Self {
        Self {
            require_positive_dims: true,
            experimental_dynamic_inputs: opts.experimental_dynamic_inputs,
            fold_where_values: false,
            fold_reshape: false,
            fold_unsqueeze_axes: false,
        }
    }
}

fn producer_of<'a>(graph: &'a GraphProto, output: &str) -> Option<&'a NodeProto> {
    graph
        .node
        .as_slice()
        .iter()
        .find(|n| n.output.as_slice().first().map(|s| s.as_str()) == Some(output))
}

/// Resolve scalar shape values through Unsqueeze wrappers (common in shape-vector Concat).
fn const_values_for_input(
    graph: &GraphProto,
    name: &str,
    const_values: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    if let Some(v) = const_values.get(name) {
        return Some(v.clone());
    }
    if let Some(node) = producer_of(graph, name) {
        if node.op_type.as_str() == "Unsqueeze" {
            if let Some(inp) = node.input.as_slice().first() {
                return const_values.get(inp.as_str()).cloned();
            }
        }
    }
    None
}

fn value_shape_dims_for_input<'a>(
    graph: &GraphProto,
    name: &str,
    value_shape_dims: &'a HashMap<String, Vec<Dimension>>,
) -> Option<&'a [Dimension]> {
    if let Some(dims) = value_shape_dims_for(name, value_shape_dims) {
        return Some(dims);
    }
    if let Some(node) = producer_of(graph, name) {
        if node.op_type.as_str() == "Unsqueeze" {
            if let Some(inp) = node.input.as_slice().first() {
                return value_shape_dims_for(inp, value_shape_dims);
            }
        }
    }
    None
}

/// Parse ConstantOfShape `value` attribute (default: int64 zero).
fn constant_of_shape_fill(node: &NodeProto) -> (DataType, i64) {
    let mut fill_value: i64 = 0;
    let mut dtype = DataType::Int64;
    for attr in node.attribute.as_slice() {
        if attr.name.as_str() != "value" {
            continue;
        }
        let Some(t) = attr.t.as_ref() else {
            continue;
        };
        match t.data_type {
            x if x == TensorProto_DataType::Float as i32 => {
                dtype = DataType::Float32;
                if !t.float_data.as_slice().is_empty() {
                    fill_value = t.float_data.as_slice()[0].to_bits() as i64;
                } else if t.raw_data.as_slice().len() >= 4 {
                    let raw = &t.raw_data.as_slice()[..4];
                    fill_value =
                        u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]) as i64;
                } else {
                    fill_value = 0f32.to_bits() as i64;
                }
            }
            x if x == TensorProto_DataType::Int64 as i32 => {
                dtype = DataType::Int64;
                if !t.int64_data.as_slice().is_empty() {
                    fill_value = t.int64_data.as_slice()[0];
                } else if t.raw_data.as_slice().len() >= 8 {
                    let raw = &t.raw_data.as_slice()[..8];
                    fill_value = i64::from_le_bytes([
                        raw[0], raw[1], raw[2], raw[3], raw[4], raw[5], raw[6], raw[7],
                    ]);
                }
            }
            _ => {}
        }
    }
    (dtype, fill_value)
}

/// Fold integer shape subgraphs (Shape/Gather/Concat/Range/Where/…).
fn fold_shape_constants(
    graph: &GraphProto,
    value_shapes: &mut HashMap<String, Vec<i64>>,
    value_types: &mut HashMap<String, DataType>,
    const_values: &mut HashMap<String, Vec<i64>>,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
    options: &FoldShapeConstantsOptions,
) -> bool {
    let consts_before = const_values.len();

    for node in graph.node.as_slice() {
        let op_type = node.op_type.as_str();
        let outputs = node.output.as_slice();
        if outputs.is_empty() {
            continue;
        }
        if const_values.contains_key(outputs[0].as_str()) {
            continue;
        }

        if options.fold_reshape && op_type == "Reshape" {
            let inputs = node.input.as_slice();
            if inputs.len() >= 2 {
                if let (Some(data), Some(mut target)) = (
                    const_values.get(inputs[0].as_str()).cloned(),
                    const_values.get(inputs[1].as_str()).cloned(),
                ) {
                    if target.contains(&-1) {
                        let total: i64 = if data.is_empty() { 1 } else { data.len() as i64 };
                        let known: i64 = target.iter().filter(|&&d| d != -1).product();
                        if known != 0 {
                            if let Some(idx) = target.iter().position(|&d| d == -1) {
                                target[idx] = total / known;
                            }
                        }
                    }
                    let out_name = outputs[0].to_string();
                    const_values.insert(out_name.clone(), data);
                    value_shapes.insert(out_name, target);
                }
            }
            continue;
        }

        if options.fold_unsqueeze_axes && op_type == "Unsqueeze" {
            let inputs = node.input.as_slice();
            if let Some(data) = inputs.first().and_then(|i| const_values.get(i.as_str()).cloned()) {
                let mut axes: Vec<i64> = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "axes")
                    .map(|a| a.ints.clone())
                    .unwrap_or_default();
                if axes.is_empty() && inputs.len() > 1 {
                    axes = const_values
                        .get(inputs[1].as_str())
                        .cloned()
                        .unwrap_or_default();
                }
                if !axes.is_empty() {
                    let mut sorted_axes = axes;
                    sorted_axes.sort();
                    let mut out_shape = value_shapes
                        .get(inputs[0].as_str())
                        .cloned()
                        .unwrap_or_else(|| {
                            if data.len() <= 1 {
                                Vec::new()
                            } else {
                                vec![data.len() as i64]
                            }
                        });
                    for axis in sorted_axes {
                        let idx = if axis < 0 {
                            (out_shape.len() as i64 + axis + 1) as usize
                        } else {
                            axis as usize
                        };
                        if idx <= out_shape.len() {
                            out_shape.insert(idx, 1);
                        }
                    }
                    let out_name = outputs[0].to_string();
                    const_values.insert(out_name.clone(), data);
                    value_shapes.insert(out_name, out_shape);
                    continue;
                }
            }
        }

        if op_type == "Where" && options.fold_where_values {
            let inputs = node.input.as_slice();
                            if inputs.len() < 3 {
                                continue;
                            }

                            // Debug: always log Where operations that involve rotary
                            if inputs.iter().any(|i| i.contains("rotary")) {
                                crate::debug_println!("[WHERE DEBUG] Processing Where node");
                                crate::debug_println!("  inputs: {:?}", inputs);
                                crate::debug_println!("  outputs: {:?}", outputs);
                            }

                            let cond = const_values.get(inputs[0].as_str()).cloned();
                            let a = const_values.get(inputs[1].as_str()).cloned();
                            let b = const_values.get(inputs[2].as_str()).cloned();
                            let cond_is_const = cond.is_some();

                            if inputs.iter().any(|i| i.contains("rotary")) {
                                crate::debug_println!("  cond const: {}", cond.is_some());
                                crate::debug_println!("  a const: {}", a.is_some());
                                crate::debug_println!("  b const: {}", b.is_some());
                            }

                            // Case 1: All inputs are constant - evaluate fully
                            if let (Some(cond), Some(a), Some(b)) = (cond, a, b) {
                                if cond.len() != a.len() || a.len() != b.len() {
                                    continue;
                                }

                                // HEURISTIC: If one branch is a trivial all-ones placeholder and the other is not,
                                // prefer the non-trivial one regardless of condition value.
                                // This handles rotary/Expand patterns like Where(cond, [1,1,1], [1,32,1]).
                                let is_trivial =
                                    |vals: &[i64]| -> bool { is_all_ones_shape_vector(vals) };

                                let mut out = if is_trivial(&a) && !is_trivial(&b) {
                                    if inputs.iter().any(|i| i.contains("rotary")) {
                                        crate::debug_println!("[WHERE SMART EVAL] Preferring non-trivial branch b={:?} over trivial a={:?}", b, a);
                                    }
                                    b
                                } else if is_trivial(&b) && !is_trivial(&a) {
                                    if inputs.iter().any(|i| i.contains("rotary")) {
                                        crate::debug_println!("[WHERE SMART EVAL] Preferring non-trivial branch a={:?} over trivial b={:?}", a, b);
                                    }
                                    a
                                } else {
                                    // Normal element-wise evaluation
                                    let mut result = Vec::with_capacity(a.len());
                                    for i in 0..a.len() {
                                        result.push(if cond[i] != 0 { a[i] } else { b[i] });
                                    }
                                    result
                                };

                                // HEURISTIC: If the output contains -1 (reshape placeholder), try to resolve it
                                // For rotary embedding patterns, check if this feeds into an Expand operation
                                if out.contains(&-1) && !outputs.is_empty() {
                                    let output_name = outputs[0].as_str();
                                    // Look for Expand nodes that use this Where output as their shape input
                                    for node in graph.node.as_slice() {
                                        if node.op_type.as_str() == "Expand"
                                            && node.input.len() >= 2
                                            && node.input[1].as_str() == output_name
                                        {
                                            // Found the Expand - check its data input shape
                                            let data_input = node.input[0].as_str();
                                            if let Some(data_shape) = value_shapes.get(data_input) {
                                                // Resolve -1 based on data shape
                                                if out.len() == data_shape.len() {
                                                    for i in 0..out.len() {
                                                        if out[i] == -1 {
                                                            out[i] = data_shape[i];
                                                            if inputs.iter().any(|inp| inp.contains("rotary")) {
                                                                crate::debug_println!("[WHERE RESOLVE] Resolved -1 at position {} to {} from data shape {:?}", i, data_shape[i], data_shape);
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }

                                let out_name = outputs[0].to_string();
                                let shape = if out.len() == 1 {
                                    Vec::new()
                                } else {
                                    vec![out.len() as i64]
                                };
                                if inputs.iter().any(|i| i.contains("rotary")) {
                                    crate::debug_println!("[WHERE STORE] Storing {} = {:?}", out_name, out);
                                }
                                const_values.insert(out_name.clone(), out);
                                value_shapes.insert(out_name, shape);
                            } else {
                                // Case 2: Some inputs are dynamic - use shape inference heuristics
                                // This handles the common pattern: Where(dynamic_condition, trivial_constant, dynamic_value)
                                // Prefer the more specific/larger shape over trivial shapes like [1,1,1]

                                let a_const = const_values.get(inputs[1].as_str());
                                let b_const = const_values.get(inputs[2].as_str());
                                let a_shape = value_shapes.get(inputs[1].as_str());
                                let b_shape = value_shapes.get(inputs[2].as_str());

                                // Heuristic: If one branch is a trivial all-ones placeholder and the other has
                                // shape info, use the other. When the condition is dynamic, never bake the
                                // placeholder branch into const_values.
                                let is_trivial_constant =
                                    |vals: &[i64]| -> bool { is_all_ones_shape_vector(vals) };

                                let preferred_values = if cond_is_const {
                                    if let (Some(a_vals), None) = (a_const, b_const) {
                                        if is_trivial_constant(a_vals) && b_shape.is_some() {
                                            crate::debug_println!("[WHERE HEURISTIC] Preferring dynamic input {} (shape {:?}) over trivial constant {:?}", inputs[2], b_shape, a_vals);
                                            b_shape.cloned()
                                        } else {
                                            Some(a_vals.clone())
                                        }
                                    } else if let (None, Some(b_vals)) = (a_const, b_const) {
                                        if is_trivial_constant(b_vals) && a_shape.is_some() {
                                            crate::debug_println!("[WHERE HEURISTIC] Preferring dynamic input {} (shape {:?}) over trivial constant {:?}", inputs[1], a_shape, b_vals);
                                            a_shape.cloned()
                                        } else {
                                            Some(b_vals.clone())
                                        }
                                    } else {
                                        None
                                    }
                                } else if let (Some(a_vals), None) = (a_const, b_const) {
                                    if is_trivial_constant(a_vals) {
                                        b_shape.cloned()
                                    } else {
                                        None
                                    }
                                } else if let (None, Some(b_vals)) = (a_const, b_const) {
                                    if is_trivial_constant(b_vals) {
                                        a_shape.cloned()
                                    } else {
                                        None
                                    }
                                } else {
                                    None
                                };

                                // Set both const_values and value_shapes for the output
                                if let Some(values) = preferred_values {
                                    let out_name = outputs[0].to_string();
                                    let shape = if values.len() == 1 {
                                        Vec::new()
                                    } else {
                                        vec![values.len() as i64]
                                    };
                                    const_values.insert(out_name.clone(), values);
                                    value_shapes.insert(out_name, shape);
                                }
                            }
            continue;
        }

        if op_type == "Shape" {
            if let (Some(inp), Some(out)) = (
                node.input.as_slice().first(),
                node.output.as_slice().first(),
            ) {
                let out = out.to_string();
                if let Some(shape) = value_shapes.get(inp).cloned() {
                if !options.require_positive_dims || shape.iter().all(|d| *d > 0) {
                        // Propagate dynamic dim metadata: Shape output is a 1-D
                        // tensor whose elements correspond to input dimensions.
                        if options.experimental_dynamic_inputs {
                            let inp_s = inp.to_string();
                            if let Some(dims) = value_shape_dims.get(&inp_s).or_else(|| {
                                value_shape_dims.get(&sanitize_identifier(&inp_s))
                            }) {
                                // Each element of the Shape output corresponds to one
                                // input dimension.  Build a 1-D dim vector where
                                // dynamic input dims become Dynamic elements.
                                let out_dims: Vec<rustnn::graph::Dimension> = dims
                                    .iter()
                                    .map(|d| match d {
                                        rustnn::graph::Dimension::Dynamic(dd) => {
                                            rustnn::graph::Dimension::Dynamic(dd.clone())
                                        }
                                        rustnn::graph::Dimension::Static(v) => {
                                            rustnn::graph::Dimension::Static(*v)
                                        }
                                    })
                                    .collect();
                                value_shape_dims.insert(out.clone(), out_dims);
                            }
                        }
                        const_values.insert(out.clone(), shape.clone());
                        let inferred_shape = vec![shape.len() as i64];
                        // Force the correct shape - Shape operation computes exact output shape
                        value_shapes.insert(out.clone(), inferred_shape.clone());
                        value_shapes.insert(sanitize_identifier(&out), inferred_shape);
                        value_types.insert(out, DataType::Int64);
                    }
                }
            }
        } else if op_type == "Gather" {
            if let (Some(data_name), Some(indices_name), Some(out)) = (
                node.input.as_slice().first(),
                node.input.as_slice().get(1),
                node.output.as_slice().first(),
            ) {
                if let (Some(data), Some(indices)) =
                    (const_values.get(data_name), const_values.get(indices_name))
                {
                    let axis = node
                        .attribute
                        .as_slice()
                        .iter()
                        .find(|a| a.name.as_str() == "axis" && a.i != 0)
                        .map(|a| a.i)
                        .unwrap_or(0);

                    if axis == 0 {
                        let mut gathered = Vec::new();
                        let mut gathered_dims = Vec::new();
                        let data_dims = if options.experimental_dynamic_inputs {
                            value_shape_dims
                                .get(data_name)
                                .or_else(|| {
                                    value_shape_dims.get(&sanitize_identifier(data_name))
                                })
                                .cloned()
                        } else {
                            None
                        };
                        for &idx in indices {
                            let i = if idx < 0 {
                                (data.len() as i64 + idx) as usize
                            } else {
                                idx as usize
                            };
                            if let Some(v) = data.get(i) {
                                gathered.push(*v);
                                if let Some(ref dd) = data_dims {
                                    if let Some(dim) = dd.get(i) {
                                        gathered_dims.push(dim.clone());
                                    }
                                }
                            }
                        }
                        if !gathered.is_empty() {
                            if options.experimental_dynamic_inputs
                                && gathered_dims.len() == gathered.len()
                                && gathered_dims
                                    .iter()
                                    .any(|d| matches!(d, rustnn::graph::Dimension::Dynamic(_)))
                            {
                                value_shape_dims.insert(out.to_string(), gathered_dims);
                            }
                            const_values.insert(out.to_string(), gathered.clone());
                            let out_shape = if gathered.len() == 1 {
                                Vec::new()
                            } else {
                                vec![gathered.len() as i64]
                            };
                            // Force the correct shape - Gather operation computes exact output shape
                            value_shapes.insert(out.to_string(), out_shape.clone());
                            value_shapes.insert(sanitize_identifier(out), out_shape);
                            value_types.insert(out.to_string(), DataType::Int64);
                        }
                    }
                }
            }
        } else if matches!(op_type, "Add" | "Sub" | "Mul" | "Div") {
            if node.input.as_slice().len() >= 2 {
                if let (Some(a_name), Some(b_name), Some(out)) = (
                    node.input.as_slice().first(),
                    node.input.as_slice().get(1),
                    node.output.as_slice().first(),
                ) {
                    let a = const_values.get(a_name);
                    let b = const_values.get(b_name);
                    if let (Some(a), Some(b)) = (a, b) {
                        let a_shape = const_shape_for_folding(a_name, a, &value_shapes);
                        let b_shape = const_shape_for_folding(b_name, b, &value_shapes);
                        if let Some((result_vals, out_shape)) =
                            fold_binary_const_i64(op_type, a, b, &a_shape, &b_shape)
                        {
                            if options.experimental_dynamic_inputs {
                                let a_dims =
                                    value_shape_dims_for(a_name, &value_shape_dims);
                                let b_dims =
                                    value_shape_dims_for(b_name, &value_shape_dims);
                                if let Some(out_dims) = fold_binary_dynamic_dims(
                                    op_type, a, b, &a_shape, &b_shape, a_dims, b_dims,
                                ) {
                                    value_shape_dims.insert(out.to_string(), out_dims);
                                }
                            }
                            const_values.insert(out.to_string(), result_vals.clone());
                            // Force the correct shape - Binary operations compute exact output shape
                            value_shapes.insert(out.to_string(), out_shape.clone());
                            value_shapes.insert(sanitize_identifier(out), out_shape);
                            if let Some(dtype) = node
                                .input
                                .as_slice()
                                .iter()
                                .find_map(|i| value_types.get(i).cloned())
                            {
                                value_types.insert(out.to_string(), dtype);
                            }
                        }
                    }
                }
            }
        } else if op_type == "Unsqueeze" || op_type == "Squeeze" {
            if let (Some(inp), Some(out)) = (
                node.input.as_slice().first(),
                node.output.as_slice().first(),
            ) {
                if let Some(vals) = const_values.get(inp).cloned() {
                    // Propagate dynamic dim metadata
                    if options.experimental_dynamic_inputs {
                        if let Some(dims) = value_shape_dims
                            .get(inp)
                            .or_else(|| value_shape_dims.get(&sanitize_identifier(inp)))
                            .cloned()
                        {
                            value_shape_dims.insert(out.to_string(), dims);
                        }
                    }
                    const_values.insert(out.to_string(), vals.clone());
                    let out_shape = if vals.len() == 1 {
                        Vec::new()
                    } else {
                        vec![vals.len() as i64]
                    };
                    value_shapes.insert(out.to_string(), out_shape);
                    if let Some(dtype) = value_types.get(inp).cloned() {
                        value_types.insert(out.to_string(), dtype);
                    }
                }
            }
        } else if op_type == "Range" {
            if node.input.as_slice().len() == 3 {
                if let (Some(start_name), Some(limit_name), Some(delta_name)) = (
                    node.input.as_slice().first(),
                    node.input.as_slice().get(1),
                    node.input.as_slice().get(2),
                ) {
                    if options.experimental_dynamic_inputs {
                        let start_dim = dynamic_scalar_dimension_for_value(
                            start_name,
                            &value_shape_dims,
                        );
                        if let Some(limit_dim) = dynamic_scalar_dimension_for_value(
                            limit_name,
                            &value_shape_dims,
                        ) {
                            if let (Some(start_vals), Some(delta_vals), Some(out)) = (
                                const_values.get(start_name),
                                const_values.get(delta_name),
                                node.output.as_slice().first(),
                            ) {
                                if !start_vals.is_empty() && !delta_vals.is_empty() {
                                    let start = start_vals[0];
                                    let delta = delta_vals[0];
                                    if let Some(range_dim) = dynamic_range_length_dimension(
                                        start,
                                        delta,
                                        start_dim.as_ref(),
                                        &limit_dim,
                                    ) {
                                        let out_shape = vec![range_dim.max_size as i64];
                                        value_shape_dims.insert(
                                            out.to_string(),
                                            vec![Dimension::Dynamic(range_dim.clone())],
                                        );
                                        value_shapes
                                            .insert(out.to_string(), out_shape.clone());
                                        value_shapes
                                            .insert(sanitize_identifier(out), out_shape);
                                        value_types
                                            .insert(out.to_string(), DataType::Int64);
                                    }
                                }
                            }
                            continue;
                        }
                    }

                    // Range(start, limit, delta) -> [start, start+delta, start+2*delta, ...]
                    if let (Some(start_vals), Some(limit_vals), Some(delta_vals)) = (
                        const_values.get(start_name),
                        const_values.get(limit_name),
                        const_values.get(delta_name),
                    ) {
                        if !start_vals.is_empty()
                            && !limit_vals.is_empty()
                            && !delta_vals.is_empty()
                        {
                            let start = start_vals[0];
                            let limit = limit_vals[0];
                            let delta = delta_vals[0];

                            let mut range_vals = Vec::new();
                            if delta > 0 {
                                let mut current = start;
                                while current < limit {
                                    range_vals.push(current);
                                    current += delta;
                                }
                            } else if delta < 0 {
                                let mut current = start;
                                while current > limit {
                                    range_vals.push(current);
                                    current += delta;
                                }
                            }

                            if let Some(out) = node.output.as_slice().first() {
                                const_values.insert(out.to_string(), range_vals.clone());
                                let out_shape = vec![range_vals.len() as i64];
                                // Force the correct shape - Range computes exact output shape
                                value_shapes.insert(out.to_string(), out_shape.clone());
                                value_shapes.insert(sanitize_identifier(out), out_shape);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            }
        } else if op_type == "Unsqueeze" && options.experimental_dynamic_inputs {
            if let (Some(inp), Some(out)) = (
                node.input.as_slice().first(),
                node.output.as_slice().first(),
            ) {
                if let Some(dims) = value_shape_dims_for_input(graph, inp, value_shape_dims) {
                    if !dims.is_empty() {
                        value_shape_dims.insert(out.to_string(), dims.to_vec());
                    }
                }
            }
        } else if op_type == "Reshape" && options.experimental_dynamic_inputs {
            let inputs = node.input.as_slice();
            if inputs.len() >= 2 {
                if let Some(shape_dims) =
                    value_shape_dims_for_input(graph, inputs[1].as_str(), value_shape_dims)
                {
                    if !shape_dims.is_empty() {
                        value_shape_dims.insert(outputs[0].to_string(), shape_dims.to_vec());
                    }
                }
            }
        } else if op_type == "Concat" {
            // Concatenate constant inputs (often used to build shape tensors)
            if let Some(out) = node.output.as_slice().first() {
                let mut concatenated: Vec<i64> = Vec::new();
                let mut all_const = true;
                for inp in node.input.as_slice() {
                    if let Some(vals) = const_values_for_input(graph, inp.as_str(), const_values) {
                        concatenated.extend_from_slice(&vals);
                    } else {
                        all_const = false;
                        break;
                    }
                }

                // Handle axis=0 or axis=-1 (common for shape building)
                let axis = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "axis" && a.i != 0)
                    .map(|a| a.i)
                    .unwrap_or(0);

                if all_const && (axis == 0 || axis == -1) {
                    if out.contains("rotary") && out.contains("Where") {
                        crate::debug_println!(
                            "[CONCAT WRITE] Writing {} = {:?}",
                            out,
                            concatenated
                        );
                    }
                    // Propagate dynamic dim metadata through concat
                    if options.experimental_dynamic_inputs {
                        let mut concat_dims: Vec<rustnn::graph::Dimension> = Vec::new();
                        let mut has_dynamic = false;
                        for inp in node.input.as_slice() {
                            if let Some(dims) =
                                value_shape_dims_for_input(graph, inp.as_str(), value_shape_dims)
                            {
                                for d in dims {
                                    if matches!(d, rustnn::graph::Dimension::Dynamic(_)) {
                                        has_dynamic = true;
                                    }
                                    concat_dims.push(d.clone());
                                }
                            } else if let Some(vals) =
                                const_values_for_input(graph, inp.as_str(), const_values)
                            {
                                for v in vals {
                                    concat_dims
                                        .push(rustnn::graph::Dimension::Static(v as u32));
                                }
                            }
                        }
                        if has_dynamic && concat_dims.len() == concatenated.len() {
                            value_shape_dims.insert(out.to_string(), concat_dims);
                        }
                    }
                    const_values.insert(out.to_string(), concatenated.clone());
                    let out_shape = vec![concatenated.len() as i64];
                    // Force the correct shape - Concat computes exact output shape
                    value_shapes.insert(out.to_string(), out_shape.clone());
                    value_shapes.insert(sanitize_identifier(out), out_shape);
                    value_types.insert(out.to_string(), DataType::Int64);
                }
            }
        } else if op_type == "ConstantOfShape" {
            // ConstantOfShape(shape) -> tensor filled with constant value
            if let Some(shape_name) = node.input.as_slice().first() {
                let dynamic_output_dims = if options.experimental_dynamic_inputs {
                    value_shape_dims_for(shape_name, &value_shape_dims)
                        .map(|dims| dims.to_vec())
                        .filter(|dims| dims_contain_dynamic(dims))
                } else {
                    None
                };

                if let (Some(out), Some(dims)) =
                    (node.output.as_slice().first(), dynamic_output_dims.as_ref())
                {
                    value_shape_dims.insert(out.to_string(), dims.to_vec());
                    const_values.remove(out.as_str());
                }

                if let Some(shape_vals) = const_values.get(shape_name).cloned() {
                    let (fill_dtype, fill_value) = constant_of_shape_fill(node);

                    // Calculate number of elements
                    let numel = if shape_vals.is_empty() {
                        1
                    } else {
                        shape_vals.iter().product::<i64>()
                    };

                    if numel > 0 && numel < 1_000_000 {
                        if let Some(out) = node.output.as_slice().first() {
                            let should_keep_const = dynamic_output_dims
                                .as_ref()
                                .is_none_or(|dims| !dims_contain_dynamic(dims));
                            // Only fold integer fills into const_values; float tensors must
                            // lower through convert_constant_of_shape (expand + correct dtype).
                            if should_keep_const && fill_dtype == DataType::Int64 {
                                let filled_tensor = vec![fill_value; numel as usize];
                                const_values.insert(out.to_string(), filled_tensor);
                            } else {
                                const_values.remove(out.as_str());
                            }
                            // Force the correct shape - ConstantOfShape creates exact output shape
                            value_shapes.insert(out.to_string(), shape_vals.clone());
                            value_shapes
                                .insert(sanitize_identifier(out), shape_vals.clone());
                            value_types.insert(out.to_string(), fill_dtype);
                        }
                    }
                }
            }
        } else if op_type == "Equal" {
            // Equal(a, b) -> boolean tensor (represented as i64: 1 for true, 0 for false)
            if node.input.as_slice().len() >= 2 {
                if let (Some(a_name), Some(b_name), Some(out)) = (
                    node.input.as_slice().first(),
                    node.input.as_slice().get(1),
                    node.output.as_slice().first(),
                ) {
                    let a = const_values.get(a_name);
                    let b = const_values.get(b_name);
                    if let (Some(a), Some(b)) = (a, b) {
                        let a_shape = const_shape_for_folding(a_name, a, &value_shapes);
                        let b_shape = const_shape_for_folding(b_name, b, &value_shapes);
                        if let Some((result_vals, out_shape)) =
                            fold_binary_const_i64("Equal", a, b, &a_shape, &b_shape)
                        {
                            const_values.insert(out.to_string(), result_vals.clone());
                            // Force the correct shape - Equal operation computes exact output shape
                            value_shapes.insert(out.to_string(), out_shape.clone());
                            value_shapes.insert(sanitize_identifier(out), out_shape);
                            value_types.insert(out.to_string(), DataType::Uint8);
                        }
                    }
                }
            }
        } else if op_type == "Where" {
            if options.experimental_dynamic_inputs && node.input.as_slice().len() >= 3 {
                if let Some(out) = node.output.as_slice().first() {
                    let cond = const_values.get(node.input.as_slice()[0].as_str());
                    let a_dims = dimension_vector_for_value(
                        node.input.as_slice()[1].as_str(),
                        &const_values,
                        &value_shape_dims,
                    );
                    let b_dims = dimension_vector_for_value(
                        node.input.as_slice()[2].as_str(),
                        &const_values,
                        &value_shape_dims,
                    );
                    let out_dims = if let (Some(cond), Some(a_dims), Some(b_dims)) =
                        (cond, a_dims.as_ref(), b_dims.as_ref())
                    {
                        if cond.len() == 1 && a_dims.len() == b_dims.len() {
                            Some(if cond[0] != 0 {
                                a_dims.clone()
                            } else {
                                b_dims.clone()
                            })
                        } else if cond.len() == a_dims.len() && cond.len() == b_dims.len() {
                            Some(
                                cond.iter()
                                    .enumerate()
                                    .map(|(idx, c)| {
                                        if *c != 0 {
                                            a_dims[idx].clone()
                                        } else {
                                            b_dims[idx].clone()
                                        }
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    } else if let (Some(a_dims), Some(b_dims)) =
                        (a_dims.as_ref(), b_dims.as_ref())
                    {
                        let a_has_dynamic =
                            a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                        let b_has_dynamic =
                            b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)));
                        if a_has_dynamic && !b_has_dynamic {
                            Some(a_dims.clone())
                        } else if b_has_dynamic && !a_has_dynamic {
                            Some(b_dims.clone())
                        } else if a_has_dynamic
                            && b_has_dynamic
                            && a_dims.len() == b_dims.len()
                        {
                            Some(
                                a_dims
                                    .iter()
                                    .zip(b_dims.iter())
                                    .map(|(a_dim, b_dim)| match (a_dim, b_dim) {
                                        (Dimension::Dynamic(dim), _) => {
                                            Dimension::Dynamic(dim.clone())
                                        }
                                        (_, Dimension::Dynamic(dim)) => {
                                            Dimension::Dynamic(dim.clone())
                                        }
                                        (Dimension::Static(v), _) => Dimension::Static(*v),
                                    })
                                    .collect(),
                            )
                        } else {
                            None
                        }
                    } else if let Some(a_dims) = a_dims.as_ref() {
                        if a_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                            && !is_trivial_static_dimension_vector(a_dims)
                        {
                            Some(a_dims.clone())
                        } else {
                            None
                        }
                    } else if let Some(b_dims) = b_dims.as_ref() {
                        if b_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_)))
                            && !is_trivial_static_dimension_vector(b_dims)
                        {
                            Some(b_dims.clone())
                        } else {
                            None
                        }
                    } else {
                        None
                    };

                    if let Some(out_dims) = out_dims {
                        if out_dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))) {
                            value_shape_dims.insert(out.to_string(), out_dims);
                        }
                    }
                }
            }
            // Keep Where dynamic to avoid baking shape-driving expressions
            // (e.g., past_sequence_length + 1) into fixed constants.
            continue;
        }
    }
    const_values.len() != consts_before
}

fn fold_integer_constants(graph: &GraphProto, ctx: &mut InferenceResult) -> bool {
    let mut value_shape_dims = HashMap::new();
    fold_shape_constants(
        graph,
        &mut ctx.value_shapes,
        &mut ctx.value_types,
        &mut ctx.const_values,
        &mut value_shape_dims,
        &FoldShapeConstantsOptions::early_pass(),
    )
}

fn read_int_tensor(tensor: &TensorProto) -> Vec<i64> {
    let raw = tensor.raw_data.as_slice();
    if !raw.is_empty() {
        match tensor.data_type {
            x if x == TensorProto_DataType::Int32 as i32 => raw
                .chunks_exact(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                .collect(),
            _ => raw
                .chunks_exact(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
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
        Vec::new()
    }
}


/// Options for extended shape propagation during ONNX lowering.
#[derive(Debug, Clone, Copy)]
pub struct PropagateOptions {
    pub optimize: bool,
    pub experimental_dynamic_inputs: bool,
}

/// Propagate ONNX value shapes and fold integer shape subgraphs.
pub fn propagate_shapes_and_fold_constants(
    graph: &GraphProto,
    initializers: &HashMap<String, &TensorProto>,
    value_shapes: &mut HashMap<String, Vec<i64>>,
    value_types: &mut HashMap<String, DataType>,
    const_values: &mut HashMap<String, Vec<i64>>,
    value_shape_dims: &mut HashMap<String, Vec<Dimension>>,
    options: &PropagateOptions,
) {
// Propagate shapes and fold constant shape expressions in a few passes
for _ in 0..3 {
    if options.optimize {
        let max_iterations = 10;
        for iteration in 0..max_iterations {
            let initial_count = value_shapes.len();

            for onnx_node in graph.node.as_slice() {
                let all_outputs_known = onnx_node
                    .output
                    .as_slice()
                    .iter()
                    .all(|out| value_shapes.contains_key(out.as_str()));
                if all_outputs_known {
                    continue;
                }

                if let Some(inferred) =
                    infer_node_output_shape(onnx_node, &value_shapes, &initializers, &const_values)
                {
                    if let Some(output_name) = onnx_node.output.as_slice().first() {
                        // Debug: track shape changes for layer 15 operations
                        if output_name.contains("layers_15_self_attn")
                            && (output_name.contains("Reshape")
                                || output_name.contains("Transpose"))
                        {
                            crate::debug_println!(
                                "[SHAPE DEBUG] {} {} -> {:?}",
                                onnx_node.op_type.as_str(),
                                output_name,
                                inferred
                            );
                        }
                        // Force the correct shape - shape inference computes exact output shape
                        value_shapes.insert(output_name.to_string(), inferred);
                    }
                }
            }

            if value_shapes.len() == initial_count {
                break;
            }

            if iteration == max_iterations - 1 {
                crate::debug_println!(
                    "Warning: Shape propagation reached max iterations ({}/{})",
                    value_shapes.len(),
                    graph.node.as_slice().len()
                );
            }
        }
    }

    // If we know the input_ids shape (batch, seq), upgrade any lone hidden-dim
    // tensors (length-1 shapes) to [batch, seq, hidden] to unblock downstream
    // matmul/reshape resolution in decoder graphs that lost batch/seq dims.
    if let Some(ids_shape) = value_shapes.get("input_ids") {
        if ids_shape.len() == 2 {
            let (batch, seq) = (ids_shape[0], ids_shape[1]);
            let upgrades: Vec<(String, Vec<i64>)> = value_shapes
                .iter()
                .filter_map(|(k, v)| {
                    if v.len() == 1 && v[0] > 1 {
                        Some((k.clone(), vec![batch, seq, v[0]]))
                    } else {
                        None
                    }
                })
                .collect();
            for (k, v) in upgrades {
                value_shapes.insert(k, v);
            }
        }
    }

    crate::debug_println!(
        "[debug] layer_norm shape {:?}",
        value_shapes.get("/decoder/block.0/layer.0/layer_norm/Mul_1_output_0")
    );
    crate::debug_println!(
        "[debug] matmul q shape {:?}",
        value_shapes.get("/decoder/block.0/layer.0/SelfAttention/q/MatMul_output_0")
    );
    crate::debug_println!(
        "[debug] input_ids shape {:?}",
        value_shapes.get("input_ids")
    );
    crate::debug_println!(
        "[debug] ln div shape {:?}",
        value_shapes.get("/decoder/block.0/layer.0/layer_norm/Div_output_0")
    );

    let consts_before = const_values.len();

    // DEBUG: Check value before propagation
    if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
        crate::debug_println!("[PROP BEFORE] /model/rotary_emb/Where_output_0 = {:?}", val);
    }

    let fold_opts = FoldShapeConstantsOptions::from_propagate(options);
    if options.experimental_dynamic_inputs {
        propagate_dynamic_dims_metadata(graph, value_shape_dims);
    }
    if fold_shape_constants(
        graph,
        value_shapes,
        value_types,
        const_values,
        value_shape_dims,
        &fold_opts,
    ) {
        // at least one node folded this pass
    }

    if const_values.len() == consts_before {
        break;
    }

    // DEBUG: Check value after propagation pass
    if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
        crate::debug_println!("[PROP AFTER] /model/rotary_emb/Where_output_0 = {:?}", val);
    }
}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dynamic_dim_requires_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let res = infer_static_shapes(&model, &HashMap::new());
        assert!(matches!(
            res,
            Err(ShapeInferenceError::DynamicDim { dim, .. }) if dim == "batch"
        ));
    }

    #[test]
    fn override_allows_static_shape() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};

        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch".to_string(),
            )),
            denotation: String::new(),
        };
        let shape = crate::protos::onnx::TensorShapeProto { dim: vec![dim] };

        let tensor_type = type_proto::Tensor {
            elem_type: crate::protos::onnx::TensorProto_DataType::Float.into(),
            shape: Some(shape),
        };

        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let vi = crate::protos::onnx::ValueInfoProto {
            name: "input".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let graph = crate::protos::onnx::GraphProto {
            input: vec![vi],
            ..Default::default()
        };

        let model = crate::protos::onnx::ModelProto {
            graph: Some(graph),
            ..Default::default()
        };

        let mut overrides = HashMap::new();
        overrides.insert("batch".to_string(), 1);
        let res = infer_static_shapes(&model, &overrides).unwrap();
        assert_eq!(res.value_shapes.get("input"), Some(&vec![1]));
    }
}
