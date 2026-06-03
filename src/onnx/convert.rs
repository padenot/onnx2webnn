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

// Main ONNX to WebNN conversion logic

use rustnn::DataType; use rustnn::graph::{Dimension, DynamicDimension};
use crate::onnx::builder::{map_rustnn_error, tensor_proto_to_bytes, OnnxBuilder};
use crate::protos::onnx::{
    tensor_shape_proto::dimension::Value as DimensionValue, type_proto::Value as TypeProtoValue,
    ModelProto, TensorProto_DataType,
};
use prost::Message;
use rustnn::mlcontext::{
    MLContext, MLContextOptions, MLGraph, MLGraphBuilder, MLOperand, MLPowerPreference,
};
use serde_json::Value as JsonValue;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use thiserror::Error;
use webnn_onnx_utils::{data_types as utils_data_types, identifiers};

/// ONNX model lowered and validated via rustnn ORT `build()`.
pub struct ValidatedGraph<'ctx> {
    pub context: MLContext<'ctx>,
    pub graph: MLGraph<'ctx>,
}

const MIN_SUPPORTED_OPSET: i64 = 11;
const MAX_SUPPORTED_OPSET: i64 = 18;

/// ONNX ops that lower to WebNN element-wise logical ops and must emit `uint8` outputs.
/// Do not inline-fold them as integer constants (e.g. i64), since `where()` requires uint8 conditions.
fn is_element_wise_logical_onnx_op(op_type: &str) -> bool {
    matches!(
        op_type,
        "Equal" | "Greater" | "Less" | "GreaterOrEqual" | "LessOrEqual"
    )
}

#[derive(Debug, Error)]
pub enum OnnxError {
    #[error("failed to read ONNX file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("failed to parse ONNX protobuf: {0}")]
    ProtobufError(String),

    #[error("unsupported ONNX opset version {version} for domain '{domain}'")]
    UnsupportedOpset { domain: String, version: i64 },

    #[error("unsupported operator: {op} (node: {node})")]
    UnsupportedOp { op: String, node: String },

    #[error("missing required attribute: {attr} in {op}")]
    MissingAttribute { attr: String, op: String },

    #[error("invalid tensor shape: {0}")]
    InvalidShape(String),

    #[error("type conversion error: {0}")]
    TypeConversion(#[from] webnn_onnx_utils::error::ConversionError),

    #[error("shape inference failed for node: {0}")]
    ShapeInference(String),
}

/// Sanitize ONNX identifiers for WebNN DSL compatibility
/// Replaces problematic characters that would confuse the parser, and prefixes
/// digit-leading names (e.g. anonymous ONNX outputs like "495") with `_` so they
/// remain parseable in the .webnn text format.
pub fn sanitize_identifier(name: &str) -> String {
    let base = identifiers::sanitize_for_webnn(name);
    match base.chars().next() {
        Some(c) if c.is_ascii_digit() => format!("_{}", base),
        _ => base,
    }
}

/// Convert ONNX data type code to WebNN DataType using shared utilities
pub(crate) fn map_onnx_data_type(onnx_type: i32) -> Result<DataType, OnnxError> {
    if onnx_type == TensorProto_DataType::Bool as i32 {
        return Ok(DataType::Uint8);
    }

    let utils_dtype = utils_data_types::onnx_to_webnn(onnx_type)?;
    Ok(match utils_dtype {
        utils_data_types::DataType::Float32 => DataType::Float32,
        utils_data_types::DataType::Float16 => DataType::Float16,
        utils_data_types::DataType::Int32 => DataType::Int32,
        utils_data_types::DataType::Uint32 => DataType::Uint32,
        utils_data_types::DataType::Int64 => DataType::Int64,
        utils_data_types::DataType::Uint64 => DataType::Uint64,
        utils_data_types::DataType::Int8 => DataType::Int8,
        utils_data_types::DataType::Uint8 => DataType::Uint8,
    })
}

/// Infer output shape for an ONNX node based on its operation type and inputs

/// Conversion options for ONNX → MLGraphBuilder lowering + ORT validation.
#[derive(Debug, Clone)]
pub struct ConvertOptions {
    /// Override dynamic dimension values (e.g., batch_size=1, sequence_length=128)
    pub free_dim_overrides: HashMap<String, u32>,
    /// Enable constant folding and shape propagation optimizations
    pub optimize: bool,
    /// Experimental: preserve unresolved dynamic input dimensions in graph metadata
    pub experimental_dynamic_inputs: bool,
}

impl Default for ConvertOptions {
    fn default() -> Self {
        Self {
            free_dim_overrides: HashMap::new(),
            optimize: false,
            experimental_dynamic_inputs: false,
        }
    }
}

struct TensorInfo {
    _data_type: DataType,
    _shape: Vec<i64>,
}

/// Main converter structure
pub struct OnnxConverter {
    model: ModelProto,
    _value_info: HashMap<String, TensorInfo>,
}

impl OnnxConverter {
    /// Create a new converter from an ONNX model
    pub fn new(model: ModelProto) -> Result<Self, OnnxError> {
        Ok(Self {
            model,
            _value_info: HashMap::new(),
        })
    }

    /// Extract metadata from ONNX model
    pub fn extract_metadata(&self) -> Result<(), OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        let graph = self.model.graph.as_ref().unwrap();
        let graph_name = if graph.name.is_empty() {
            "graph"
        } else {
            graph.name.as_str()
        };

        // Print basic info
        println!("Model name: {graph_name}");
        println!("Inputs: {}", graph.input.as_slice().len());
        println!("Outputs: {}", graph.output.as_slice().len());
        println!("Nodes: {}", graph.node.as_slice().len());
        println!("Initializers: {}", graph.initializer.as_slice().len());

        Ok(())
    }

    /// Lower ONNX into an [`OnnxBuilder`] (MLGraphBuilder + operand map).
    pub fn convert_with_builder(
        &self,
        b: &mut OnnxBuilder<'_, '_, '_>,
        options: &ConvertOptions,
    ) -> Result<(), OnnxError> {
        if self.model.graph.is_none() {
            return Err(OnnxError::ProtobufError(
                "Missing graph in model".to_string(),
            ));
        }

        // Validate opset imports
        for import in self.model.opset_import.as_slice() {
            let domain = import.domain.as_str();
            let version = import.version;
            let domain_name = if domain.is_empty() {
                "ai.onnx".to_string()
            } else {
                domain.to_string()
            };

            if (domain.is_empty() || domain == "ai.onnx")
                && !(MIN_SUPPORTED_OPSET..=MAX_SUPPORTED_OPSET).contains(&version)
            {
                return Err(OnnxError::UnsupportedOpset {
                    domain: domain_name,
                    version,
                });
            }
        }

        let onnx_graph = self.model.graph.as_ref().unwrap();
        let mut value_name_map: HashMap<String, String> = HashMap::new();
        let mut effective_overrides = options.free_dim_overrides.clone();
        let mut inference_overrides = effective_overrides.clone();
        let mut value_types: HashMap<String, DataType> = HashMap::new();

        // Merge overrides from model metadata if present
        for meta in self.model.metadata_props.as_slice() {
            if meta
                .key
                .as_str()
                .eq_ignore_ascii_case("freedimensionoverrides")
            {
                if let Ok(json) = serde_json::from_str::<JsonValue>(meta.value.as_str()) {
                    let obj = json
                        .get("freeDimensionOverrides")
                        .unwrap_or(&json)
                        .as_object()
                        .cloned();
                    if let Some(map) = obj {
                        for (name, value) in map {
                            if let Some(v) = value.as_u64() {
                                effective_overrides.entry(name.clone()).or_insert(v as u32);
                            }
                        }
                    }
                }
            }
        }

        // Process inputs (exclude initializers)
        let initializer_names: HashSet<String> = onnx_graph
            .initializer
            .as_slice()
            .iter()
            .map(|init| init.name.as_str().to_string())
            .collect();

        let default_dynamic_max_size: u32 = 65_535;
        let default_inference_dim_values: HashMap<&str, u32> =
            HashMap::from([("batch_size", 1), ("batch", 1), ("n", 1), ("b", 1)]);
        let dynamic_max_for_dim = |name: &str| -> u32 {
            let lower = name.to_ascii_lowercase();
            if lower.contains("past")
                || lower.contains("seq")
                || lower.contains("length")
                || lower == "s"
                || lower == "t"
            {
                4096
            } else if lower.contains("batch") || lower == "b" || lower == "n" {
                8
            } else {
                default_dynamic_max_size
            }
        };
        let resolve_dim_override =
            |dim_param: &str, overrides: &HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = overrides.get(dim_param) {
                    return Some(*v);
                }

                let lower = dim_param.to_ascii_lowercase();
                overrides.get(&lower).copied()
            };
        let dimension_for_param =
            |dim_param: &str, overrides: &HashMap<String, u32>| -> Dimension {
                if let Some(v) = resolve_dim_override(dim_param, overrides) {
                    Dimension::Static(v)
                } else {
                    Dimension::Dynamic(DynamicDimension {
                        name: dim_param.to_string(),
                        max_size: dynamic_max_for_dim(dim_param),
                    })
                }
            };

        let resolve_dim_for_inference =
            |dim_param: &str, overrides: &mut HashMap<String, u32>| -> Option<u32> {
                if let Some(v) = resolve_dim_override(dim_param, overrides) {
                    return Some(v);
                }
                let lower = dim_param.to_ascii_lowercase();
                if let Some(v) = default_inference_dim_values.get(lower.as_str()) {
                    overrides.insert(dim_param.to_string(), *v);
                    return Some(*v);
                }
                None
            };

        for input in onnx_graph.input.as_slice() {
            let raw_name = input.name.as_str().to_string();
            let name = sanitize_identifier(&raw_name);

            // Skip if this is an initializer (constant)
            if initializer_names.contains(&raw_name) {
                continue;
            }

            // Get type info
            if let Some(type_proto) = &input.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    let data_type = if tensor_type.elem_type != 0 {
                        let onnx_type = tensor_type.elem_type;
                        map_onnx_data_type(onnx_type)?
                    } else {
                        DataType::Float32 // Default
                    };

                    let shape = if let Some(shape_proto) = &tensor_type.shape {
                        let mut resolved: Vec<Dimension> = Vec::new();
                        for (idx, dim) in shape_proto.dim.iter().enumerate() {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            resolved.push(Dimension::Static(*v as u32));
                                        } else if options.experimental_dynamic_inputs {
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: format!("{}_dim{}", name, idx),
                                                max_size: default_dynamic_max_size,
                                            }));
                                        } else {
                                            let dim_hint = format!("{}_dim{}", name, idx);
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has non-positive dim value ({}) at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name,
                                                v,
                                                idx,
                                                dim_hint
                                            )));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if let Some(v) = resolve_dim_override(
                                            dim_param,
                                            &effective_overrides,
                                        ) {
                                            resolved.push(Dimension::Static(v));
                                        } else if options.experimental_dynamic_inputs {
                                            let max_size = dynamic_max_for_dim(dim_param);
                                            resolved.push(Dimension::Dynamic(DynamicDimension {
                                                name: dim_param.to_string(),
                                                max_size,
                                            }));
                                        } else if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            effective_overrides
                                                .entry(dim_param.clone())
                                                .or_insert(v);
                                            resolved.push(Dimension::Static(v));
                                        } else {
                                            return Err(OnnxError::InvalidShape(format!(
                                                "Input '{}' has unresolved dynamic dimension '{}'. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                                raw_name, dim_param, dim_param
                                            )));
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                resolved.push(Dimension::Dynamic(DynamicDimension {
                                    name: format!("{}_dim{}", name, idx),
                                    max_size: default_dynamic_max_size,
                                }));
                            } else {
                                let dim_hint = format!("{}_dim{}", name, idx);
                                return Err(OnnxError::InvalidShape(format!(
                                    "Input '{}' has unknown dimension at index {}. \
Provide --override-dim {}=<value> or enable --experimental-dynamic-inputs.",
                                    raw_name, idx, dim_hint
                                )));
                            }
                        }
                        resolved
                    } else {
                        return Err(OnnxError::InvalidShape(format!(
                            "Input '{}' is missing shape information",
                            raw_name
                        )));
                    };

                    if shape.is_empty() {
                        continue;
                    }

                    b.register_input(&raw_name, data_type.clone(), &shape)?;

                    value_name_map.insert(raw_name.clone(), name.clone());
                    value_name_map.insert(name.clone(), name.clone());
                    value_types.insert(raw_name.clone(), data_type.clone());
                    value_types.insert(name.clone(), data_type);
                }
            }
        }

        // Process initializers (constants/weights)
        for initializer in onnx_graph.initializer.as_slice() {
            let name = sanitize_identifier(initializer.name.as_str());
            let raw_data = initializer.raw_data.as_slice();

            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !raw_data.is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                crate::debug_println!("Warning: Skipping initializer '{}' with no data", name);
                continue;
            }

            let onnx_type = initializer.data_type;
            let data_type = map_onnx_data_type(onnx_type)?;
            let shape: Vec<u32> = initializer
                .dims
                .as_slice()
                .iter()
                .map(|d| *d as u32)
                .collect();

            let bytes = tensor_proto_to_bytes(initializer)?;
            b.register_constant_from_bytes(initializer.name.as_str(), data_type.clone(), &shape, &bytes)?;

            value_name_map.insert(initializer.name.as_str().to_string(), name.clone());
            value_name_map.insert(name.clone(), name.clone());
            value_types.insert(initializer.name.as_str().to_string(), data_type.clone());
            value_types.insert(name, data_type);
        }

        // Process nodes using OpRegistry
        let registry = crate::onnx::ops::OpRegistry::new();

        // Build initializers map for resolving constant shapes
        let mut initializers_map = std::collections::HashMap::new();
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            initializers_map.insert(initializer.name.as_str().to_string(), initializer);
        }

        // Build value_shapes map from value_info and inputs for shape inference
        let mut value_shapes = std::collections::HashMap::new();
        let mut value_shape_dims = std::collections::HashMap::new();

        // Add input shapes (already validated)
        for (raw_name, mapped_name) in value_name_map.clone() {
            if initializer_names.contains(&raw_name) {
                continue;
            }
            if let Some(input) = onnx_graph
                .input
                .as_slice()
                .iter()
                .find(|i| i.name.as_str() == raw_name)
            {
                if let Some(type_proto) = &input.r#type {
                    if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                        if let Some(shape_proto) = &tensor_type.shape {
                            let mut shape: Vec<i64> = Vec::new();
                            let mut unknown = false;
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                shape.push(*v);
                                            } else if options.experimental_dynamic_inputs {
                                                shape.push(default_dynamic_max_size as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            if options.experimental_dynamic_inputs {
                                                shape.push(
                                                    resolve_dim_override(
                                                        dim_param,
                                                        &inference_overrides,
                                                    )
                                                    .unwrap_or_else(|| dynamic_max_for_dim(dim_param))
                                                    as i64,
                                                );
                                            } else if let Some(v) = resolve_dim_for_inference(
                                                dim_param,
                                                &mut inference_overrides,
                                            ) {
                                                shape.push(v as i64);
                                            } else {
                                                unknown = true;
                                                break;
                                            }
                                        }
                                    }
                                } else if options.experimental_dynamic_inputs {
                                    shape.push(default_dynamic_max_size as i64);
                                } else {
                                    unknown = true;
                                    break;
                                }
                            }
                            if !unknown && !shape.is_empty() {
                                value_shapes.insert(raw_name.clone(), shape.clone());
                                value_shapes.insert(mapped_name.clone(), shape);
                            }
                            let mut dims = Vec::new();
                            for dim in &shape_proto.dim {
                                if let Some(dim_value) = &dim.value {
                                    match dim_value {
                                        DimensionValue::DimValue(v) => {
                                            if *v > 0 {
                                                dims.push(rustnn::graph::Dimension::Static(*v as u32));
                                            }
                                        }
                                        DimensionValue::DimParam(dim_param) => {
                                            dims.push(dimension_for_param(dim_param, &inference_overrides));
                                        }
                                    }
                                }
                            }
                            if !dims.is_empty() {
                                value_shape_dims.insert(raw_name.clone(), dims.clone());
                                value_shape_dims.insert(mapped_name.clone(), dims);
                            }
                        }
                    }
                }
            }
        }

        // Add initializer shapes
        for initializer in onnx_graph.initializer.as_slice() {
            // Skip initializers with no data (check both raw_data and typed data fields)
            let has_data = !initializer.raw_data.as_slice().is_empty()
                || !initializer.float_data.as_slice().is_empty()
                || !initializer.int32_data.as_slice().is_empty()
                || !initializer.int64_data.as_slice().is_empty()
                || !initializer.double_data.as_slice().is_empty();

            if !has_data {
                continue;
            }
            let shape: Vec<i64> = initializer.dims.as_slice().to_vec();
            value_shapes.insert(initializer.name.as_str().to_string(), shape);
            let dims: Vec<rustnn::graph::Dimension> = initializer
                .dims
                .iter()
                .copied()
                .filter(|d| *d > 0)
                .map(|d| rustnn::graph::Dimension::Static(d as u32))
                .collect();
            if !dims.is_empty() {
                value_shape_dims.insert(initializer.name.as_str().to_string(), dims);
            }
        }

        // Add value_info shapes (intermediate tensors from shape inference)
        // Try to resolve dynamic dimensions using overrides
        for value_info in onnx_graph.value_info.as_slice() {
            if let Some(type_proto) = &value_info.r#type {
                if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                    if let Some(shape_proto) = &tensor_type.shape {
                        let mut shape: Vec<i64> = Vec::new();
                        let mut unknown = false;

                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            shape.push(*v);
                                        } else if options.experimental_dynamic_inputs {
                                            shape.push(default_dynamic_max_size as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        if options.experimental_dynamic_inputs {
                                            shape.push(
                                                resolve_dim_override(dim_param, &inference_overrides)
                                                    .unwrap_or_else(|| dynamic_max_for_dim(dim_param))
                                                    as i64,
                                            );
                                        } else if let Some(v) = resolve_dim_for_inference(
                                            dim_param,
                                            &mut inference_overrides,
                                        ) {
                                            shape.push(v as i64);
                                        } else {
                                            unknown = true;
                                            break;
                                        }
                                    }
                                }
                            } else if options.experimental_dynamic_inputs {
                                shape.push(default_dynamic_max_size as i64);
                            } else {
                                unknown = true;
                                break;
                            }
                        }

                        if !unknown && !shape.is_empty() && shape.iter().all(|&d| d > 0) {
                            value_shapes.insert(value_info.name.as_str().to_string(), shape);
                        }
                        let mut dims = Vec::new();
                        for dim in &shape_proto.dim {
                            if let Some(dim_value) = &dim.value {
                                match dim_value {
                                    DimensionValue::DimValue(v) => {
                                        if *v > 0 {
                                            dims.push(rustnn::graph::Dimension::Static(*v as u32));
                                        }
                                    }
                                    DimensionValue::DimParam(dim_param) => {
                                        dims.push(dimension_for_param(dim_param, &inference_overrides));
                                    }
                                }
                            }
                        }
                        if !dims.is_empty() {
                            value_shape_dims.insert(value_info.name.as_str().to_string(), dims);
                        }
                    }
                }
            }
        }

        // Seed const values with integer initializers and Constant nodes
        let mut const_values: HashMap<String, Vec<i64>> = HashMap::new();
        for (name, initializer) in &initializers_map {
            if initializer.data_type == TensorProto_DataType::Int64 as i32
                || initializer.data_type == TensorProto_DataType::Int32 as i32
            {
                let raw = initializer.raw_data.as_slice();
                let values = if !raw.is_empty() {
                    if initializer.data_type == TensorProto_DataType::Int32 as i32 {
                        raw.chunks_exact(4)
                            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                            .collect()
                    } else {
                        raw.chunks_exact(8)
                            .map(|c| {
                                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                            })
                            .collect()
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
                };

                if !values.is_empty() {
                    const_values.insert(name.clone(), values);
                }
            }
        }

        for node in onnx_graph.node.as_slice() {
            if node.op_type.as_str() == "Constant" {
                if let Some(attr) = node
                    .attribute
                    .as_slice()
                    .iter()
                    .find(|a| a.name.as_str() == "value" && a.t.is_some())
                {
                    let tensor = attr.t.as_ref().unwrap();
                    if tensor.data_type == TensorProto_DataType::Int64 as i32
                        || tensor.data_type == TensorProto_DataType::Int32 as i32
                    {
                        let raw = tensor.raw_data.as_slice();
                        let values = if !raw.is_empty() {
                            if tensor.data_type == TensorProto_DataType::Int32 as i32 {
                                raw.chunks_exact(4)
                                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
                                    .collect()
                            } else {
                                raw.chunks_exact(8)
                                    .map(|c| {
                                        i64::from_le_bytes([
                                            c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                                        ])
                                    })
                                    .collect()
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
                        };

                        if let Some(out) = node.output.as_slice().first() {
                            if !values.is_empty() {
                                const_values.insert(out.to_string(), values);
                                value_types.insert(out.to_string(), DataType::Int64);
                            }
                        }
                    }
                }
            }
        }

        // Run the static shape/type inference scaffold to seed shapes/types/constants
        // before lowering. Errors surface early if dynamic dims remain.
        if options.experimental_dynamic_inputs {
            for input in onnx_graph.input.as_slice() {
                if initializer_names.contains(&input.name) {
                    continue;
                }
                if let Some(type_proto) = &input.r#type {
                    if let Some(TypeProtoValue::TensorType(tensor_type)) = &type_proto.value {
                        if let Some(shape_proto) = &tensor_type.shape {
                            for dim in &shape_proto.dim {
                                if let Some(DimensionValue::DimParam(dim_param)) = &dim.value {
                                    inference_overrides
                                        .entry(dim_param.clone())
                                        .or_insert_with(|| dynamic_max_for_dim(dim_param));
                                }
                            }
                        }
                    }
                }
            }
        }

        let mut dynamic_inference_attempts: HashSet<String> = HashSet::new();
        loop {
            match crate::onnx::shape_inference::infer_static_shapes(
                &self.model,
                &inference_overrides,
            ) {
                Ok(inferred) => {
                    // Initial seeding: use or_insert since these are the first values
                    // (no prior shapes to override)
                    for (k, v) in inferred.value_shapes {
                        value_shapes.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.value_types {
                        value_types.entry(k).or_insert(v);
                    }
                    for (k, v) in inferred.const_values {
                        // Use insert() instead of or_insert() to allow shape inference to correct
                        // earlier wrong values (e.g., Where operation heuristics)
                        if k.contains("rotary") && k.contains("Where") {
                            if let Some(old_val) = const_values.get(&k) {
                                crate::debug_println!(
                                    "[CONVERT] Overwriting {} from {:?} to {:?}",
                                    k,
                                    old_val,
                                    v
                                );
                            } else {
                                crate::debug_println!("[CONVERT] Inserting new {} = {:?}", k, v);
                            }
                        }
                        const_values.insert(k, v);
                    }
                    break;
                }
                Err(crate::onnx::shape_inference::ShapeInferenceError::DynamicDim {
                    input,
                    dim,
                }) => {
                    if options.experimental_dynamic_inputs
                        && !dynamic_inference_attempts.contains(dim.as_str())
                    {
                        let fallback = resolve_dim_override(&dim, &inference_overrides)
                            .unwrap_or_else(|| dynamic_max_for_dim(&dim));
                        inference_overrides.insert(dim.clone(), fallback);
                        dynamic_inference_attempts.insert(dim.clone());
                        crate::debug_println!(
                            "[CONVERT] Retrying static shape inference with inferred override {}={} \
                             (required by input '{}')",
                            dim,
                            fallback,
                            input
                        );
                        continue;
                    }
                    crate::debug_println!(
                        "[CONVERT] Skipping static shape inference due to unresolved dynamic dim '{}' on input '{}'",
                        dim,
                        input
                    );
                    break;
                }
                Err(e) => return Err(OnnxError::ShapeInference(e.to_string())),
            }
        }

        crate::onnx::shape_inference::propagate_shapes_and_fold_constants(
            onnx_graph,
            &initializers_map,
            &mut value_shapes,
            &mut value_types,
            &mut const_values,
            &mut value_shape_dims,
            &crate::onnx::shape_inference::PropagateOptions {
                optimize: options.optimize,
                experimental_dynamic_inputs: options.experimental_dynamic_inputs,
            },
        );

        // DEBUG: Check value before node conversion
        if let Some(val) = const_values.get("/model/rotary_emb/Where_output_0") {
            crate::debug_println!("[NODE CONV] /model/rotary_emb/Where_output_0 = {:?}", val);
        }
        for onnx_node in onnx_graph.node.as_slice() {
            // If all outputs are compile-time constants, emit them directly and skip conversion
            let outputs = onnx_node.output.as_slice();
            let has_dynamic_output_metadata = outputs.iter().any(|o| {
                crate::onnx::shape_inference::value_shape_dims_for(o.as_str(), &value_shape_dims)
                    .map(|dims| dims.iter().any(|d| matches!(d, Dimension::Dynamic(_))))
                    .unwrap_or(false)
            });
            if !outputs.is_empty()
                && !has_dynamic_output_metadata
                && onnx_node.op_type.as_str() != "Cast"
                && onnx_node.op_type.as_str() != "ConstantOfShape"
                && !is_element_wise_logical_onnx_op(onnx_node.op_type.as_str())
                && outputs
                    .iter()
                    .all(|o| const_values.contains_key(o.as_str()))
            {
                // Check if outputs are true scalars (rank 0), not just single-element tensors
                let all_scalar = outputs.iter().all(|o| {
                    value_shapes
                        .get(o.as_str())
                        .map(|s| s.is_empty()) // True scalar has empty shape
                        .unwrap_or_else(|| {
                            // Fallback: check if data length is 1
                            const_values
                                .get(o.as_str())
                                .map(|v| v.len() == 1)
                                .unwrap_or(false)
                        })
                });

                // Handle scalar constants by emitting them inline
                if all_scalar {
                    for out in outputs {
                        if let Some(values) = const_values.get(out) {
                            let const_name = sanitize_identifier(out);
                            // Use the intended shape from value_shapes, not just empty for single-element
                            let shape = value_shapes
                                .get(out.as_str())
                                .map(|s| s.iter().map(|&d| d as u32).collect())
                                .unwrap_or_else(Vec::new);

                            let bytes = values[0].to_le_bytes().to_vec();
                            b.register_constant_from_bytes(
                                &const_name,
                                DataType::Int64,
                                &shape,
                                &bytes,
                            )?;

                            value_name_map.insert(out.to_string(), const_name.clone());
                            value_name_map.insert(const_name.clone(), const_name.clone());
                            value_types.insert(out.to_string(), DataType::Int64);
                            value_types.insert(const_name, DataType::Int64);
                        }
                    }
                }
                // For non-scalar constants (like Range output), emit inline consts so downstream
                // nodes have a defined producer.
                for out in outputs {
                    if let Some(values) = const_values.get(out) {
                        let const_name = sanitize_identifier(out);
                        let mut shape = value_shapes
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or_else(|| vec![values.len() as i64]);
                        let declared_numel = shape
                            .iter()
                            .try_fold(1usize, |acc, d| usize::try_from(*d).ok().map(|v| acc * v));
                        if declared_numel != Some(values.len()) {
                            // Some folded constants are broadcast candidates where value_shapes
                            // carries the post-broadcast shape but const_values stores the compact payload.
                            // Keep shape/data internally consistent by using the compact shape.
                            shape = vec![values.len() as i64];
                        }
                        let dtype = value_types
                            .get(out.as_str())
                            .cloned()
                            .unwrap_or(DataType::Int64);

                        // Flatten i64 values into little-endian bytes
                        let mut bytes = Vec::with_capacity(values.len() * 8);
                        for v in values {
                            bytes.extend_from_slice(&v.to_le_bytes());
                        }

                        let shape_u32: Vec<u32> = shape.iter().map(|d| *d as u32).collect();
                        b.register_constant_from_bytes(&const_name, dtype.clone(), &shape_u32, &bytes)?;

                        value_name_map.insert(out.to_string(), const_name.clone());
                        value_name_map.insert(const_name.clone(), const_name.clone());
                        value_types.insert(out.to_string(), dtype.clone());
                        value_types.insert(const_name, dtype);
                    }
                }
                continue;
            }

            let context = crate::onnx::ops::ConversionContext {
                initializers: &initializers_map,
                value_shapes: &value_shapes,
                value_shape_dims: &value_shape_dims,
                const_values: &const_values,
                value_ids: &value_name_map,
                value_types: &value_types,
            };

            let converted = registry.convert_node(onnx_node, &context, b)?;

            for (onnx_out, dtype) in converted.output_types {
                let webnn_id = sanitize_identifier(&onnx_out);
                value_name_map.insert(onnx_out.clone(), webnn_id.clone());
                value_types.insert(webnn_id, dtype);
            }

            // Track output shapes after conversion to prevent shape inflation
            // Use .insert() to force correct shapes (not .or_insert() which preserves old shapes)
            if let Some(inferred_shape) =
                crate::onnx::shape_inference::infer_node_output_shape(
                    onnx_node,
                    &value_shapes,
                    &initializers_map,
                    &const_values,
                )
            {
                for output_name in onnx_node.output.as_slice() {
                    // Insert shape for both raw and sanitized names
                    value_shapes.insert(output_name.to_string(), inferred_shape.clone());
                    value_shapes.insert(sanitize_identifier(output_name), inferred_shape.clone());
                }
            }

        }

        Ok(())
    }
}

/// Convert an ONNX file and validate via rustnn ORT `MLGraphBuilder::build()`.
pub fn convert_onnx<P: AsRef<Path>>(
    onnx_path: P,
    mut options: ConvertOptions,
) -> Result<ValidatedGraph<'static>, OnnxError> {
    // Read ONNX file
    let onnx_path_ref = onnx_path.as_ref();
    let onnx_bytes = fs::read(onnx_path_ref)?;

    // Parse protobuf
    let mut model: ModelProto =
        ModelProto::decode(&onnx_bytes[..]).map_err(|e| OnnxError::ProtobufError(e.to_string()))?;

    // Apply constant folding if optimize flag is set
    if options.optimize {
        crate::debug_println!("Running constant folding...");
        let evaluators = crate::onnx::constant_folding::evaluators::get_evaluators();
        let nodes_folded =
            crate::onnx::constant_folding::fold_constants_in_model(&mut model, &evaluators)?;
        crate::debug_println!("Constant folding: {} nodes folded", nodes_folded);
    }

    // Merge overrides from sidecar dims file if provided implicitly and not already set
    if options.free_dim_overrides.is_empty() {
        let mut sidecar = onnx_path_ref.to_path_buf();
        sidecar.set_extension("dims.json");
        if sidecar.exists() {
            let content = fs::read_to_string(&sidecar)?;
            if let Ok(json) = serde_json::from_str::<JsonValue>(&content) {
                if let Some(obj) = json
                    .get("freeDimensionOverrides")
                    .unwrap_or(&json)
                    .as_object()
                {
                    for (name, value) in obj {
                        if let Some(v) = value.as_u64() {
                            options
                                .free_dim_overrides
                                .entry(name.clone())
                                .or_insert(v as u32);
                        }
                    }
                }
            }
        }
    }

    convert_model(model, &options)
}

/// Lower ONNX to [`MLGraphBuilder`] and validate with ORT `build()`.
pub(crate) fn convert_model(
    model: ModelProto,
    options: &ConvertOptions,
) -> Result<ValidatedGraph<'static>, OnnxError> {
    let converter = OnnxConverter::new(model.clone())?;
    converter.extract_metadata()?;

    let mut context = MLContext::create(&MLContextOptions::new(
        MLPowerPreference::Default,
        false,
    ))
    .map_err(|e| OnnxError::ShapeInference(format!("MLContext::create failed: {e}")))?;

    let mut ml_builder = MLGraphBuilder::new(&mut context).map_err(map_rustnn_error)?;
    let mut onnx_builder = OnnxBuilder::new(&mut ml_builder);

    converter.convert_with_builder(&mut onnx_builder, options)?;

    let onnx_graph = model
        .graph
        .as_ref()
        .ok_or_else(|| OnnxError::ProtobufError("Missing graph in model".to_string()))?;

    let mut outputs: HashMap<String, MLOperand> = HashMap::new();
    for output in onnx_graph.output.as_slice() {
        let op = onnx_builder.output_operand(output.name.as_str())?;
        let output_key = onnx_builder.build_output_key(output.name.as_str());
        outputs.insert(output_key, op);
    }
    let output_refs: HashMap<&str, MLOperand> =
        outputs.iter().map(|(k, v)| (k.as_str(), *v)).collect();

    let graph = onnx_builder.finish_build(output_refs)?;

    Ok(ValidatedGraph { context, graph })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_options_default() {
        let options = ConvertOptions::default();
        assert!(!options.optimize);
        assert!(options.free_dim_overrides.is_empty());
    }

    #[test]
    fn test_sanitize_identifier_replaces_colons() {
        assert_eq!(sanitize_identifier("foo::bar"), "foo__bar");
        assert_eq!(sanitize_identifier("foo:bar"), "foo_bar");
    }

    #[test]
    fn test_sanitize_identifier_replaces_dots() {
        assert_eq!(sanitize_identifier("encoder.block.0"), "encoder_block_0");
        assert_eq!(
            sanitize_identifier("model.layer.weight"),
            "model_layer_weight"
        );
        assert_eq!(sanitize_identifier("a.b.c"), "a_b_c");
    }

    #[test]
    fn test_sanitize_identifier_replaces_combined() {
        // Test combinations of :: : and .
        assert_eq!(
            sanitize_identifier("module::class:method.field"),
            "module__class_method_field"
        );
        assert_eq!(
            sanitize_identifier("encoder.attention::output:dense"),
            "encoder_attention__output_dense"
        );
    }

    #[test]
    fn test_sanitize_identifier_no_change() {
        // Identifiers that don't need sanitization
        assert_eq!(sanitize_identifier("simple_name"), "simple_name");
        assert_eq!(sanitize_identifier("CamelCase"), "CamelCase");
        assert_eq!(sanitize_identifier("name123"), "name123");
    }

    #[test]
    fn test_inline_bytes_encoding_for_i64_values() {
        // Test the inline bytes encoding logic used for non-scalar constants
        // This simulates what happens when Range or similar ops produce constant arrays
        let values: Vec<i64> = vec![0, 1, 2, 3, 4];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        // Verify byte length
        assert_eq!(bytes.len(), 40); // 5 values * 8 bytes each

        // Verify first value (0)
        let first_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(first_bytes), 0);

        // Verify last value (4)
        let last_bytes: [u8; 8] = bytes[32..40].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(last_bytes), 4);
    }

    #[test]
    fn test_inline_bytes_encoding_single_value() {
        // Test single value encoding
        let values: Vec<i64> = vec![42];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 8);
        let decoded: [u8; 8] = bytes.try_into().unwrap();
        assert_eq!(i64::from_le_bytes(decoded), 42);
    }

    #[test]
    fn test_inline_bytes_encoding_negative_values() {
        // Test with negative values (important for Range with negative delta)
        let values: Vec<i64> = vec![5, 4, 3, 2, 1, 0, -1, -2];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 64); // 8 values * 8 bytes each

        // Verify a negative value
        let neg_bytes: [u8; 8] = bytes[56..64].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(neg_bytes), -2);
    }

    #[test]
    fn test_inline_bytes_encoding_large_values() {
        // Test with large i64 values
        let values: Vec<i64> = vec![i64::MAX, i64::MIN, 0];
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }

        assert_eq!(bytes.len(), 24);

        // Verify MAX value
        let max_bytes: [u8; 8] = bytes[0..8].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(max_bytes), i64::MAX);

        // Verify MIN value
        let min_bytes: [u8; 8] = bytes[8..16].try_into().unwrap();
        assert_eq!(i64::from_le_bytes(min_bytes), i64::MIN);
    }

    #[test]
    fn test_convert_preserves_dynamic_input_dim_without_override() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "batch_size".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        convert_model(
            model,
            &ConvertOptions {
                experimental_dynamic_inputs: true,
                ..ConvertOptions::default()
            },
        )
        .expect("ORT build should succeed for experimental dynamic inputs");
    }

    #[test]
    fn test_convert_rejects_dynamic_input_dim_without_flag() {
        use crate::protos::onnx::{tensor_shape_proto, type_proto};
        use crate::protos::onnx::{GraphProto, ModelProto, TensorShapeProto, ValueInfoProto};

        let dim_batch = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimParam(
                "unknown_dim".to_string(),
            )),
            denotation: String::new(),
        };
        let dim_seq = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(1)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto {
            dim: vec![dim_batch, dim_seq],
        };

        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Int64.into(),
            shape: Some(shape),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type)),
            denotation: String::new(),
        };

        let input_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let output_vi = ValueInfoProto {
            name: "input_ids".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![input_vi],
                output: vec![output_vi],
                ..Default::default()
            }),
            ..Default::default()
        };

        let msg = match convert_model(model, &ConvertOptions::default()) {
            Err(err) => err.to_string(),
            Ok(_) => panic!("should require overrides or flag"),
        };
        assert!(msg.contains("override-dim"));
        assert!(msg.contains("experimental-dynamic-inputs"));
    }
}
