# onnx2webnn

ONNX → WebNN lowering crate extracted from [webnn-graph](../webnn-graph). ONNX operators lower
directly to [rustnn](../rustnn) `MLGraphBuilder`; full-graph validation runs via ORT CPU
`build()` (`onnx-runtime` feature). There is no JSON IR and no on-disk graph export — success
means `builder.build()` returns `Ok(MLGraph)`.

Supported ONNX opset range: **11–18** (see `MIN_SUPPORTED_OPSET` / `MAX_SUPPORTED_OPSET` in
`src/onnx/convert.rs`).

## Build

```powershell
cargo build
# or
cargo build --release
```

`make build`, `make test`, `make fmt`, and `make check` are defined in the repo `Makefile`.

## Convert

```powershell
cargo run -- convert --input model.onnx --optimize --override-dim batch_size=1
```

Dynamic ONNX inputs (unresolved symbolic dims kept as WebNN dynamic metadata):

```powershell
cargo run -- convert --input model.onnx `
  --experimental-dynamic-inputs `
  --override-dim batch_size=1 `
  --override-dim sequence_length=1
```

If `model.dims.json` sits beside the ONNX file and no overrides were passed on the CLI, dimension
bindings are loaded from that sidecar (`freeDimensionOverrides` or a flat JSON object).

| Flag | Purpose |
|------|---------|
| `--input` | Input `.onnx` path (required) |
| `--optimize` | Constant folding and shape propagation |
| `--override-dim NAME=VALUE` | Bind a symbolic dim (repeatable) |
| `--override-dims-file` | JSON overrides (`freeDimensionOverrides` or flat object) |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata |
| `--debug` | Verbose conversion logging (global) |

On success the CLI prints `✓ ORT graph build succeeded for …`.

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

## Layout

| Path | Purpose |
|------|---------|
| `src/onnx/convert.rs` | ONNX load, optional folding, lowering, ORT `build()` |
| `src/onnx/builder.rs` | `OnnxBuilder` — operand map and `MLGraphBuilder` bridge |
| `src/onnx/builder_helpers.rs` | Shared lowering helpers |
| `src/onnx/shape_inference.rs` | Static shape/type propagation |
| `src/onnx/constant_folding.rs` | Constant folding driver (with `--optimize`) |
| `src/onnx/constant_folding/evaluators/` | Per-op fold evaluators |
| `src/onnx/ops/` | ONNX op handlers (activation, conv, pool, reshape, …) |
| `src/protos.rs` | ONNX protobuf types |
| `src/debug.rs` | Debug logging toggle |

## Dependencies

- **rustnn** (`../rustnn`, `onnx-runtime`) — `MLGraphBuilder`, shape inference, ORT `build()` validation
- **webnn-onnx-utils** — ONNX protos, op names, data types

## Related

- [webnn-graph](../webnn-graph) — DSL parser, validator, JS/HTML emit (source of the extracted lowering code)
