/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Tarek Ziadé <tarek@ziade.org>
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

use clap::{Parser, Subcommand};
use std::collections::HashMap;
use std::path::Path;

use onnx2webnn::{convert_onnx, convert_onnx_save_webnn};
use onnx2webnn::ConvertOptions;

#[derive(Parser)]
#[command(name = "onnx2webnn")]
#[command(about = "Convert ONNX models to WebNN via MLGraphBuilder (ORT validation)")]
struct Cli {
    #[arg(long, global = true)]
    debug: bool,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Lower ONNX to MLGraphBuilder and validate with rustnn ORT
    Convert {
        #[arg(long)]
        input: String,
        #[arg(long = "override-dim")]
        override_dims: Vec<String>,
        #[arg(long = "override-dims-file")]
        override_dims_file: Option<String>,
        #[arg(long)]
        optimize: bool,
        #[arg(long)]
        experimental_dynamic_inputs: bool,
    },

    /// Convert ONNX to WebNN graph files (.webnn + .weights + .manifest.json) for AOT loading
    Save {
        #[arg(long)]
        input: String,
        /// Output .webnn graph file
        #[arg(long)]
        output: String,
        /// Output .weights binary (defaults to <output>.weights)
        #[arg(long)]
        weights: Option<String>,
        /// Output .manifest.json (defaults to <output>.manifest.json)
        #[arg(long)]
        manifest: Option<String>,
        /// Also save ORT cache ONNX for fast loading (e.g. model.ort_cache.onnx)
        #[arg(long = "ort-cache")]
        ort_cache: Option<String>,
        #[arg(long = "override-dim")]
        override_dims: Vec<String>,
        #[arg(long)]
        optimize: bool,
    },
}

fn parse_override_dims(override_dims: Vec<String>, override_dims_file: Option<String>) -> anyhow::Result<HashMap<String, u32>> {
    let mut map = if let Some(path) = override_dims_file {
        let content = std::fs::read_to_string(&path)?;
        let json: serde_json::Value = serde_json::from_str(&content)?;
        let obj = json.get("freeDimensionOverrides").unwrap_or(&json)
            .as_object()
            .ok_or_else(|| anyhow::anyhow!("override-dims-file must be a JSON object"))?;
        obj.iter()
            .map(|(k, v)| {
                v.as_u64().map(|n| (k.clone(), n as u32))
                    .ok_or_else(|| anyhow::anyhow!("override value for '{}' must be integer", k))
            })
            .collect::<Result<HashMap<_, _>, _>>()?
    } else {
        HashMap::new()
    };

    for s in override_dims {
        let (k, v) = s.split_once('=')
            .ok_or_else(|| anyhow::anyhow!("expected NAME=VALUE, got {s:?}"))?;
        map.insert(k.trim().to_string(), v.trim().parse()
            .map_err(|_| anyhow::anyhow!("invalid value in {s:?}"))?);
    }
    Ok(map)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    if cli.debug {
        onnx2webnn::debug::enable();
    }

    match cli.cmd {
        Command::Convert { input, override_dims, override_dims_file, optimize, experimental_dynamic_inputs } => {
            let free_dim_overrides = parse_override_dims(override_dims, override_dims_file)?;
            let options = ConvertOptions { free_dim_overrides, optimize, experimental_dynamic_inputs, ..Default::default() };
            convert_onnx(Path::new(&input).to_str().unwrap(), options)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            eprintln!("✓ ORT graph build succeeded for {input}");
        }

        Command::Save { input, output, weights, manifest, ort_cache, override_dims, optimize } => {
            let weights_path = weights.unwrap_or_else(|| {
                let p = Path::new(&output);
                p.with_extension("weights").to_string_lossy().to_string()
            });
            let manifest_path = manifest.unwrap_or_else(|| {
                let p = Path::new(&output);
                format!("{}.manifest.json", p.with_extension("").display())
            });

            let free_dim_overrides = parse_override_dims(override_dims, None)?;
            let options = ConvertOptions { free_dim_overrides, optimize, ..Default::default() };

            // Ensure the output file uses .json extension (load_graph_from_path requires .webnn or .json)
            let output_path = if !output.ends_with(".json") && !output.ends_with(".webnn") {
                format!("{output}.json")
            } else {
                output.clone()
            };

            convert_onnx_save_webnn(Path::new(&input).to_str().unwrap(), options.clone(), &output_path, &weights_path, &manifest_path)
                .map_err(|e| anyhow::anyhow!("{e}"))?;

            eprintln!("✓ WebNN graph: {output_path}");
            eprintln!("✓ Weights:     {weights_path}");
            eprintln!("✓ Manifest:    {manifest_path}");

            if let Some(cache_path) = ort_cache {
                // Build the full ORT session to get the ONNX bytes for the cache.
                eprintln!("Building ORT session to produce cache...");
                let validated = convert_onnx(Path::new(&input).to_str().unwrap(), options)
                    .map_err(|e| anyhow::anyhow!("ORT build: {e}"))?;
                rustnn::save_ort_graph_cache(&validated.graph, Path::new(&cache_path))
                    .map_err(|e| anyhow::anyhow!("save ORT cache: {e}"))?;
                eprintln!("✓ ORT cache:   {cache_path}");
            }
        }
    }

    Ok(())
}
