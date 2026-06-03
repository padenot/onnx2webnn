.PHONY: build test fmt check clean help

build:
	cargo build

test:
	cargo test

fmt:
	cargo fmt

check:
	cargo check

clean:
	cargo clean

help:
	@echo "onnx2webnn - ONNX to WebNN converter"
	@echo "  make build  - build binary"
	@echo "  make test   - run tests"
	@echo "  make fmt    - format Rust code"
