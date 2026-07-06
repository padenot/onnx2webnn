use std::fs;
use std::path::Path;
use std::process::Command;

use onnx2webnn::protos::onnx::{
    tensor_shape_proto, type_proto, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto_DataType, TensorShapeProto, TypeProto, ValueInfoProto,
};
use onnx2webnn::{convert_onnx_save_webnn, ConvertOptions};
use prost::Message;
use tempfile::TempDir;
use webnn_graph::ast::{ConstInit, GraphJson};
use webnn_graph::weights::WeightsManifest;

fn onnx2webnn_bin() -> String {
    std::env::var("CARGO_BIN_EXE_onnx2webnn").expect("onnx2webnn binary path")
}

fn tensor_value_info(name: &str, elem_type: TensorProto_DataType, dims: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: name.to_string(),
        r#type: Some(TypeProto {
            value: Some(type_proto::Value::TensorType(type_proto::Tensor {
                elem_type: elem_type.into(),
                shape: Some(TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|dim| tensor_shape_proto::Dimension {
                            value: Some(tensor_shape_proto::dimension::Value::DimValue(*dim)),
                            denotation: String::new(),
                        })
                        .collect(),
                }),
            })),
            denotation: String::new(),
        }),
        ..Default::default()
    }
}

fn write_relu_model(path: &Path) {
    let model = ModelProto {
        ir_version: 7,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 13,
        }],
        graph: Some(GraphProto {
            name: "tiny_relu".to_string(),
            input: vec![tensor_value_info(
                "input",
                TensorProto_DataType::Float,
                &[1, 2],
            )],
            output: vec![tensor_value_info(
                "output",
                TensorProto_DataType::Float,
                &[1, 2],
            )],
            node: vec![NodeProto {
                name: "relu".to_string(),
                op_type: "Relu".to_string(),
                input: vec!["input".to_string()],
                output: vec!["output".to_string()],
                ..Default::default()
            }],
            ..Default::default()
        }),
        ..Default::default()
    };
    fs::write(path, model.encode_to_vec()).unwrap();
}

#[test]
fn save_api_converts_tiny_onnx_model_and_writes_sidecars() {
    let temp_dir = TempDir::new().unwrap();
    let onnx_path = temp_dir.path().join("tiny.onnx");
    let graph_path = temp_dir.path().join("tiny.webnn.json");
    let weights_path = temp_dir.path().join("tiny.weights");
    let manifest_path = temp_dir.path().join("tiny.manifest.json");
    write_relu_model(&onnx_path);

    convert_onnx_save_webnn(
        &onnx_path,
        ConvertOptions::default(),
        graph_path.to_str().unwrap(),
        weights_path.to_str().unwrap(),
        manifest_path.to_str().unwrap(),
    )
    .expect("save tiny ONNX as WebNN");

    let graph: GraphJson = serde_json::from_slice(&fs::read(graph_path).unwrap()).unwrap();
    assert!(graph.inputs.contains_key("input"));
    assert_eq!(graph.outputs["output"], "output");
    assert!(graph
        .consts
        .values()
        .all(|decl| { !matches!(decl.init, ConstInit::InlineBytes { .. }) }));

    let weights = fs::read(weights_path).unwrap();
    assert_eq!(&weights[0..4], b"WGWT");
    let manifest: WeightsManifest =
        serde_json::from_slice(&fs::read(manifest_path).unwrap()).unwrap();
    assert!(manifest.tensors.is_empty());
}

#[test]
fn save_cli_writes_graph_weights_and_manifest() {
    let temp_dir = TempDir::new().unwrap();
    let onnx_path = temp_dir.path().join("tiny.onnx");
    let graph_path = temp_dir.path().join("tiny.webnn.json");
    let weights_path = temp_dir.path().join("tiny.weights");
    let manifest_path = temp_dir.path().join("tiny.manifest.json");
    write_relu_model(&onnx_path);

    let status = Command::new(onnx2webnn_bin())
        .args([
            "save",
            "--input",
            onnx_path.to_str().unwrap(),
            "--output",
            graph_path.to_str().unwrap(),
            "--weights",
            weights_path.to_str().unwrap(),
            "--manifest",
            manifest_path.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());

    assert!(graph_path.exists());
    assert!(weights_path.exists());
    assert!(manifest_path.exists());
}

#[test]
fn save_cli_rejects_incomplete_reuse_flags() {
    let temp_dir = TempDir::new().unwrap();
    let output_path = temp_dir.path().join("out.json");
    let reuse_weights_path = temp_dir.path().join("source.weights");

    let output = Command::new(onnx2webnn_bin())
        .args([
            "save",
            "--input",
            "unused.onnx",
            "--output",
            output_path.to_str().unwrap(),
            "--reuse-weights",
            reuse_weights_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("--reuse-weights and --reuse-manifest must be passed together"));
}
