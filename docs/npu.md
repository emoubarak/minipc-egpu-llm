# The NPU (AMD XDNA 1)

The Ryzen 7 8845HS has a first-generation XDNA NPU (`npu1`, AIE2, 16 TOPS, PCI ID `1022:1502`). Status in
October 2026: **no way to run an LLM on it under Linux.**

- The `amdxdna` driver is in mainline Linux (6.14+) and loads fine on Fedora 44.
- XRT (the userspace runtime) is being packaged for Fedora; until then the COPR `abn/amd-npu` provides it.
- FastFlowLM, Lemonade's NPU backend and Ryzen AI Software for Linux all target **XDNA 2** (Strix Point and
  newer). XDNA 1 is not supported.
- The remaining route is IRON / MLIR-AIE, which means writing kernels: research demos, not an inference engine.

What did work in May 2026 (XRT 2.23, NPU firmware 1.5.5.391):

| Test | Result |
|---|---|
| `xrt-smi validate -r latency` | 86 µs average |
| `xrt-smi validate -r throughput` | 26,059 ops/s |
| pyxrt, no-op kernel, sequential | 208 µs, 4,813 ops/s |
| pyxrt, no-op kernel, 4 in flight | 122 µs, 8,169 ops/s |

The older firmware (1.5.2.380) aborted the validation kernels (`ERT_CMD_STATE_ABORT`); 1.5.5.391 fixed it.
