# PyTorch 2.15 compatibility

This branch preserves the existing `tch 0.26.0` and `torch-sys 0.26.0` Rust APIs while targeting **PyTorch/libtorch 2.15.0**. It adopts C++20 and substitutes the four removed C++ Cholesky/QR entry points with the equivalent `torch::linalg_*` calls. The Rust API signatures are unchanged.

The first port was validated against PyTorch 2.15 RC on macOS ARM64 CPU by the isolated `timpra/rew-room-correction` canary. This fork's CI exercises CPU compilation, autograd and the affected linear algebra wrappers on Linux x86_64 and macOS ARM64.

## Local build with the release-candidate wheel

```sh
python3 -m pip install --pre --index-url https://download.pytorch.org/whl/test/cpu torch==2.15.0
export LIBTORCH_USE_PYTORCH=1
cargo test --test pytorch_215_smoke
```

Do not set `LIBTORCH_BYPASS_VERSION_CHECK`. The fork explicitly expects version 2.15.0.

## Use from a Cargo application

```toml
[dependencies]
tch = { git = "https://github.com/sudoDeity/tch-rs", branch = "compat/pytorch-2.15" }

[patch.crates-io]
torch-sys = { git = "https://github.com/sudoDeity/tch-rs", branch = "compat/pytorch-2.15" }
```

For reproducible builds replace the branch with a pinned commit SHA. CUDA 13.2, Windows and other API surfaces are **not yet validated** by this CPU smoke matrix; CUDA 13.4 is a prototype target. PyTorch 2.15 does not distribute CUDA 12.x or CUDA 13.0 binaries.
