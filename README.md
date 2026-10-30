# onnx2webnn

onnx2webnn lowers ONNX opsets **1–26** through [RustNN](../rustnn)'s WebNN builder. `convert`
builds the graph with CPU ORT by default; `--backend coreml` or `--backend trtx` selects an
available alternative.
`--no-backend-graph-build` exports a reloadable `.webnn`/Safetensors pair without building it.
`--build-and-export` saves the pair and builds it without numerical comparison; `--validate`
also reloads the pair and compares execution with native CPU ORT. These modes are exclusive. See
[operator coverage](docs/operator-conversion-status.md) and
[model results](docs/model-validation-status-triage.md).

## Build

### Prerequisites

- A stable Rust toolchain.
- `protoc` (Debian/Ubuntu: `protobuf-compiler`; macOS: Homebrew `protobuf`).
- A sibling `../rustnn` checkout, as required by `Cargo.toml`.
- ONNX Runtime 1.27+ for ORT graph builds and validation. Export-only conversion does not load it.
  RustNN can download the pinned version:

```powershell
make -C ../rustnn onnxruntime-download
```

Set `ORT_DYLIB_PATH` to the downloaded shared library before ORT use; see
[CI setup](.github/workflows/ci.yml) for platform paths.

```powershell
cargo build
# or
cargo build --release
```

The `Makefile` also provides `build`, `test`, `fmt`, and `check` targets.

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

Without CLI overrides, `model.dims.json` can supply dimension bindings. Downloads use the
standard Hugging Face cache; WebNN exports overwrite the OS cache's `onnx2webnn/webnn` directory.
Set `O2W_CACHE_DIR`, `O2W_ONNX_CACHE`, or `O2W_WEBNN_CACHE` to override locations.
`--validate-cached` reloads existing artifacts without reconversion and is exclusive with
`--validate`, `--build-and-export`, and `--no-backend-graph-build`.

Single-input ONNX `Concat` is forwarded directly. One-dimensional `ConvTranspose` uses the
builder's inferred output dimensions when removing the emulation's trailing unit dimension.
Static `Einsum` supports an explicit-output ellipsis on one input, including outer products
such as `..., f -> ... f`; ellipses shared between inputs remain unsupported. Floating-point
arithmetic stays out of the integer shape-constant evaluator, including casts of integral
values followed by division.

WebNN lacks fused `MatMulNBits`; conversion dequantizes packed q4/q8 weights before floating-point
matmul. ORT `accuracy_level=4` can instead use Int8 activations, so numerical parity is not assured.

Merged decoders (optimum's `decoder_model_merged*.onnx`) branch at runtime on `use_cache_branch`.
WebNN has no runtime `If`, so pin the input and convert each branch separately:

```powershell
cargo run -- convert --input decoder_model_merged.onnx --optimize `
  --pin-input use_cache_branch=false `
  --override-dim batch_size=1 --override-dim decoder_sequence_length=4 `
  --override-dim past_decoder_sequence_length=0 ...
```

Each pinned branch gets its own `.webnn`/Safetensors pair. Shared weights are not deduplicated;
loading both graphs may duplicate RAM/VRAM use. Proper growing-KV-cache execution still needs
dynamic shapes or further specialization.

| Flag | Purpose |
|------|---------|
| `--input` | Input `.onnx` path (required) |
| `--optimize` | Constant folding and shape propagation |
| `--override-dim NAME=VALUE` | Bind a symbolic dim (repeatable); unnamed zero dims are addressed as `<input>_dim<axis>` |
| `--override-dims-file` | JSON overrides (`freeDimensionOverrides` or flat object) |
| `--pin-input NAME=VALUE` | Freeze a graph input to `true`/`false`/an integer (repeatable) |
| `--allow-missing-external-data` | Zero-fill external tensors whose data file is absent (weight-stripped skeleton models) |
| `--experimental-dynamic-inputs` | Preserve unresolved symbolic dims as dynamic metadata |
| `--debug` | Verbose conversion logging (global) |
| `--backend ort\|coreml\|trtx` | Select the graph-build/validation backend (global; default `ort`); CoreML and TensorRT require their Cargo features |
| `--no-backend-graph-build` | Overwrite cache-backed `.onnx`, `.webnn`, and Safetensors artifacts without compiling a backend graph |
| `--build-and-export` | Overwrite cache artifacts and compile the selected backend graph without numerical comparison |
| `--validate` | Overwrite and reload the cache artifacts, then numerically compare against native ORT |
| `--validate-cached` | Validate existing cache artifacts without reconverting |

Export-only mode rejects a non-ORT `--backend`, since it builds no backend graph.

`validate-files` accepts local models without a manifest. Pair each `--model` with a complete
`--inputs` SafeTensors archive, or omit all `--inputs` for deterministic feeds. Names, types, and
shapes must match feedable ONNX inputs. Pins and dimension overrides apply to every model in that
invocation; ONNX external-data files stay beside their source.

```bash
cargo run -- validate-files --model first.onnx --inputs first.safetensors \
  --model second.onnx --inputs second.safetensors
```

Operator statistics are code-backed; “fully supported” is a lowering classification, not measured
hardware parity. Performance-limited paths remain functional. Cargo features make CoreML/TensorRT
available, while `--backend` selects one; native comparison remains CPU ORT.
Deterministic fallback feeds are reproducible. Scale-aware quantization feeds use fixed model
scale/zero-point values; ambiguous quantization patterns warn and use the generic deterministic
feed, while malformed quantization parameters still fail validation.

## Model sweep

[`manifest.json`](tests/models/manifest.json) is a generated Transformers.js selection, not a
pinned CI gate. The skeleton sweep fetches graph structure and small constants with HTTP ranges,
zero-fills missing large weights, and builds each graph in ORT or CI-selected CoreML. It checks
construction, not numerical correctness.

### Numerical-validation CI

PR CI numerically validates the curated
[`ci-validation.json`](tests/models/ci-validation.json): Linux/ORT blocks; macOS/CoreML is
experimental; Windows does not run this check. Each entry pins a Hugging Face commit and ONNX
SHA-256, verified on download and reuse. To add one, test both backends and record its file,
revision, digest, dimensions, and pinned inputs in the curated file—not the generated manifest.
Downloads use the standard Hugging Face cache unless `O2W_ONNX_CACHE` or `O2W_CACHE_DIR` overrides
it.

```bash
target/release/onnx2webnn validate-models \
  --manifest tests/models/ci-validation.json \
  --selection all --weights real --jobs 1
```

The manual full-manifest real-weight workflow needs a dedicated Linux runner (≥100 GiB disk,
~32 GiB RAM). Model failures are diagnostic; setup failures are fatal. Logs persist for 14 days.

For the skeleton test, `O2W_MODELS=hub|dir=<path>|strip=<path>` selects source models; unset
outside CI skips it. `O2W_MODEL_FETCH_JOBS`, `O2W_MODEL_TEST_JOBS`, and
`O2W_MODEL_TEST_SKIP_HEAVY` control concurrency and large cases.

```powershell
$env:O2W_MODELS = "dir=..\transformers_js_experiments\models"; cargo test --release --test model_skeletons
```

Library API:

```rust
use onnx2webnn::{convert_onnx, ConvertOptions};

let graph = convert_onnx("model.onnx", ConvertOptions::default())?;
```

Operator handlers live in `src/onnx/ops/`; graph conversion is in `src/onnx/convert.rs`.
See [webnn-graph](../webnn-graph) for the `.webnn` format.
