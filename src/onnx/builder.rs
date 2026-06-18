/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! ONNX → [`MLGraphBuilder`] bridge (operand map, naming, rustnn error mapping).

use rustnn::DataType; use rustnn::graph::Dimension;
use crate::protos::onnx::TensorProto;
use crate::onnx::convert::{map_onnx_data_type, sanitize_identifier, OnnxError};
use rustnn::error::{Error as RustnnError, GraphBuilderError};
use rustnn::mlcontext::{MLGraph, MLGraphBuilder, MLOperand};
use rustnn::operator_enums::MLOperandDataType;
use rustnn::operator_options::MLOperatorOptions;
use rustnn::mlcontext::MLOperandDescriptor;
use std::collections::{HashMap, HashSet};

pub struct OnnxBuilder<'a, 'ctx, 'bld> {
    pub builder: &'a mut MLGraphBuilder<'ctx, 'bld>,
    operands: HashMap<String, MLOperand>,
    /// Operand ids registered via `input()` — cannot be passed directly to `build()`.
    input_operands: HashSet<u32>,
    /// Sanitized + raw ONNX names registered as graph inputs.
    input_names: HashSet<String>,
}

/// Operand index inside the builder graph (`MLOperand::id` is `pub(crate)` in rustnn).
pub(crate) fn operand_index(op: MLOperand) -> u32 {
    #[repr(C)]
    struct Layout {
        id: usize,
    }
    // Safety: `MLOperand` is `{ id: usize }` only (rustnn `mlcontext.rs`).
    let layout: Layout = unsafe { std::mem::transmute(op) };
    layout.id as u32
}

impl<'a, 'ctx, 'bld> OnnxBuilder<'a, 'ctx, 'bld> {
    pub fn new(builder: &'a mut MLGraphBuilder<'ctx, 'bld>) -> Self {
        Self {
            builder,
            operands: HashMap::new(),
            input_operands: HashSet::new(),
            input_names: HashSet::new(),
        }
    }

    pub fn webnn_id(onnx_name: &str) -> String {
        sanitize_identifier(onnx_name)
    }

    pub fn record_operand(&mut self, keys: &[&str], op: MLOperand) {
        for key in keys {
            if key.is_empty() {
                continue;
            }
            self.operands.insert(key.to_string(), op);
            let sanitized = sanitize_identifier(key);
            self.operands.insert(sanitized, op);
        }
    }

    pub fn resolve_operand(&self, name: &str) -> Result<MLOperand, OnnxError> {
        if let Some(&op) = self.operands.get(name) {
            return Ok(op);
        }
        let sanitized = sanitize_identifier(name);
        if let Some(&op) = self.operands.get(&sanitized) {
            return Ok(op);
        }
        let trimmed = name.trim_start_matches('/');
        if let Some(&op) = self.operands.get(trimmed) {
            return Ok(op);
        }
        Err(OnnxError::InvalidShape(format!(
            "no MLOperand registered for ONNX value '{name}' (sanitized: '{sanitized}')"
        )))
    }

    pub fn register_input(
        &mut self,
        name: &str,
        data_type: DataType,
        shape: &[Dimension],
    ) -> Result<(), OnnxError> {
        let id = Self::webnn_id(name);
        let desc = descriptor_from_parts(data_type, shape)?;
        let op = self
            .builder
            .input(&id, &desc)
            .map_err(map_rustnn_error)?;
        self.input_operands.insert(operand_index(op));
        self.input_names.insert(name.to_string());
        self.input_names.insert(id.clone());
        self.record_operand(&[name, &id], op);
        Ok(())
    }

    /// Resolve an ONNX graph output for `build()`.
    ///
    /// MLGraphBuilder rejects using an input operand directly as a graph output; insert
    /// `identity` when the value is still the original graph input (e.g. no-op ONNX graphs).
    pub fn output_operand(&mut self, name: &str) -> Result<MLOperand, OnnxError> {
        let op = self.resolve_operand(name)?;
        if self.input_operands.contains(&operand_index(op)) {
            let label = format!("{}__output", Self::webnn_id(name));
            let opts = Self::labeled_options(&label);
            return Ok(self
                .builder
                .identity_with_options(op, opts)
                .map_err(map_op_error)?);
        }
        Ok(op)
    }

    /// Build-time output key; disambiguate when ONNX reuses an input name as output.
    pub fn build_output_key(&self, output_name: &str) -> String {
        let sanitized = Self::webnn_id(output_name);
        if self.input_names.contains(&sanitized) || self.input_names.contains(output_name) {
            format!("{sanitized}__output")
        } else {
            sanitized
        }
    }

    pub fn register_constant_from_bytes(
        &mut self,
        name: &str,
        data_type: DataType,
        shape: &[u32],
        bytes: &[u8],
    ) -> Result<(), OnnxError> {
        let id = Self::webnn_id(name);
        let desc = descriptor_static(data_type.clone(), shape)?;
        let op = match data_type {
            DataType::Float32 => self.builder.constant_from_slice(
                &desc,
                bytemuck::try_cast_slice::<_, f32>(bytes)
                    .map_err(|e| OnnxError::InvalidShape(e.to_string()))?,
            ),
            DataType::Float16 => self.builder.constant_from_slice(
                &desc,
                bytemuck::try_cast_slice::<_, u16>(bytes)
                    .map_err(|e| OnnxError::InvalidShape(e.to_string()))?,
            ),
            DataType::Int32 => self.builder.constant_from_slice(
                &desc,
                bytemuck::try_cast_slice::<_, i32>(bytes)
                    .map_err(|e| OnnxError::InvalidShape(e.to_string()))?,
            ),
            DataType::Int64 => self.builder.constant_from_slice(
                &desc,
                bytemuck::try_cast_slice::<_, i64>(bytes)
                    .map_err(|e| OnnxError::InvalidShape(e.to_string()))?,
            ),
            DataType::Uint8 => self.builder.constant_from_slice(
                &desc,
                bytes,
            ),
            DataType::Int8 => self.builder.constant_from_slice(
                &desc,
                bytes,
            ),
            other => {
                return Err(OnnxError::InvalidShape(format!(
                    "unsupported constant data type for builder: {other:?}"
                )));
            }
        }
        .map_err(map_rustnn_error)?;
        self.record_operand(&[name, &id], op);
        Ok(())
    }

    pub fn labeled_options(label: &str) -> MLOperatorOptions {
        MLOperatorOptions {
            label: label.to_string(),
            ..Default::default()
        }
    }

    pub fn finish_build(
        &mut self,
        outputs: HashMap<&str, MLOperand>,
    ) -> Result<MLGraph<'ctx>, OnnxError> {
        self.builder.build(&outputs).map_err(map_rustnn_error)
    }
}

pub fn map_rustnn_error(err: RustnnError) -> OnnxError {
    OnnxError::ShapeInference(err.to_string())
}

pub fn map_op_error(err: GraphBuilderError) -> OnnxError {
    OnnxError::ShapeInference(err.to_string())
}

pub fn descriptor_static(data_type: DataType, shape: &[u32]) -> Result<MLOperandDescriptor, OnnxError> {
    let dt = map_ast_data_type(data_type)?;
    Ok(MLOperandDescriptor::new(
        dt,
        shape.iter().map(|&d| d as u64).collect(),
    ))
}

pub fn descriptor_from_parts(
    data_type: DataType,
    shape: &[Dimension],
) -> Result<MLOperandDescriptor, OnnxError> {
    let dt = map_ast_data_type(data_type)?;
    let mut dims = Vec::with_capacity(shape.len());
    for dim in shape {
        match dim {
            Dimension::Static(v) => dims.push(*v as u64),
            Dimension::Dynamic(d) => dims.push(d.max_size as u64),
        }
    }
    Ok(MLOperandDescriptor::new(dt, dims))
}

pub fn map_ast_data_type(dt: DataType) -> Result<MLOperandDataType, OnnxError> {
    Ok(match dt {
        DataType::Float32 => MLOperandDataType::Float32,
        DataType::Float16 => MLOperandDataType::Float16,
        DataType::Int32 => MLOperandDataType::Int32,
        DataType::Uint32 => MLOperandDataType::Uint32,
        DataType::Int64 => MLOperandDataType::Int64,
        DataType::Uint64 => MLOperandDataType::Uint64,
        DataType::Int8 => MLOperandDataType::Int8,
        DataType::Uint8 => MLOperandDataType::Uint8,
        DataType::Int4 | DataType::Uint4 => {
            return Err(OnnxError::InvalidShape(
                "int4/uint4 not supported on MLGraphBuilder path".to_string(),
            ));
        }
    })
}

pub fn map_onnx_tensor_type(onnx_type: i32) -> Result<MLOperandDataType, OnnxError> {
    map_ast_data_type(map_onnx_data_type(onnx_type)?)
}

/// Extract initializer / constant tensor bytes for `constant_from_slice`.
pub fn tensor_proto_to_bytes(tensor: &TensorProto) -> Result<Vec<u8>, OnnxError> {
    if !tensor.raw_data.is_empty() {
        return Ok(tensor.raw_data.clone());
    }
    if !tensor.float_data.is_empty() {
        return Ok(tensor
            .float_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    if !tensor.int32_data.is_empty() {
        // ONNX stores int8/uint8 values in int32_data (one i32 per element, low byte only).
        // Detect by data_type and truncate to 1 byte per element.
        use crate::protos::onnx::TensorProto_DataType;
        let dt = tensor.data_type;
        if dt == TensorProto_DataType::Int8 as i32 {
            return Ok(tensor.int32_data.iter().map(|&v| v as i8 as u8).collect());
        }
        if dt == TensorProto_DataType::Uint8 as i32 {
            return Ok(tensor.int32_data.iter().map(|&v| v as u8).collect());
        }
        if dt == TensorProto_DataType::Bool as i32 {
            return Ok(tensor.int32_data.iter().map(|&v| v as u8).collect());
        }
        return Ok(tensor
            .int32_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    if !tensor.int64_data.is_empty() {
        return Ok(tensor
            .int64_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    if !tensor.double_data.is_empty() {
        return Ok(tensor
            .double_data
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect());
    }
    Err(OnnxError::InvalidShape(format!(
        "tensor '{}' has no payload",
        tensor.name
    )))
}

#[cfg(test)]
mod tests {
    use crate::onnx::convert::{convert_model, ConvertOptions};
    use crate::protos::onnx::{
        GraphProto, ModelProto, NodeProto, TensorProto_DataType, ValueInfoProto,
        tensor_shape_proto, type_proto, TensorShapeProto,
    };

    #[test]
    fn test_add_ort_build_succeeds() {
        let dim = tensor_shape_proto::Dimension {
            value: Some(tensor_shape_proto::dimension::Value::DimValue(2)),
            denotation: String::new(),
        };
        let shape = TensorShapeProto { dim: vec![dim.clone(), dim] };
        let tensor_type = type_proto::Tensor {
            elem_type: TensorProto_DataType::Float.into(),
            shape: Some(shape.clone()),
        };
        let type_proto = crate::protos::onnx::TypeProto {
            value: Some(type_proto::Value::TensorType(tensor_type.clone())),
            denotation: String::new(),
        };

        let a_input = ValueInfoProto {
            name: "a".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let b_input = ValueInfoProto {
            name: "b".to_string(),
            r#type: Some(type_proto.clone()),
            ..Default::default()
        };
        let out = ValueInfoProto {
            name: "c".to_string(),
            r#type: Some(type_proto),
            ..Default::default()
        };

        let add = NodeProto {
            op_type: "Add".to_string(),
            input: vec!["a".to_string(), "b".to_string()],
            output: vec!["c".to_string()],
            ..Default::default()
        };

        let model = ModelProto {
            graph: Some(GraphProto {
                input: vec![a_input, b_input],
                output: vec![out],
                node: vec![add],
                ..Default::default()
            }),
            ..Default::default()
        };

        convert_model(model, &ConvertOptions::default()).expect("Add graph should build on ORT");
    }
}
