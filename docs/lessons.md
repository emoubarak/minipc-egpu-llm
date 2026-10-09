# Lessons

What held up after five months of measuring on this box. Numbers link back to [results.md](results.md).

## 1. A 35B MoE runs fine on 8 GB, if you place the experts yourself

Qwen3.6-35B-A3B at Q4 is 21-22 GB and only ~3B parameters are active per token. With `-ngl 99 -ncmoe N`, all
attention weights and the KV cache stay on the GPU and the experts of N layers stay in system RAM, where the CPU
computes them. That gave 45.6 t/s without speculation and 59 t/s with MTP on the RTX 3060 Ti (Gen4 link).

- **Sweep `-ncmoe`, do not trust a formula.** On this card, UD-Q4_K_M loaded from 29 or 30 depending on the
  build, UD-IQ4_XS from 26 to 28. Lower is usually faster, until the compute buffers no longer fit. MTP needs ~500 MB more: the best MTP point was 33.
- **Try `-cmoe` too** (every expert on the CPU). On the slower Gen3 link it beat `-ncmoe 30` (37.0 against 33.0
  t/s) while using 2.6 GB of VRAM instead of 7.4, which leaves room for context. It reads long prompts ~19 %
  slower.
- **Use the CUDA build for this on NVIDIA.** Same model, same `-ncmoe 30`: 45.6 t/s with CUDA against 33.8 with
  Vulkan (Oct 2026). For a small dense model fully in VRAM, Vulkan was the faster one in May (64.3 t/s on
  Qwen3-8B).
- Ollama, at the time, ran the same model at 11.2 t/s on the same card: it had no equivalent of `-ncmoe`.

## 2. MTP is the best speed-up; tune it, and pin the build

Qwen3.6 and Gemma 4 ship a multi-token prediction head (Qwen inside the GGUF, Gemma 4 QAT as a separate
`mtp-*.gguf` drafter). `--spec-type draft-mtp --spec-draft-n-max 2 -np 1` gave:

- Qwen3.6-35B-A3B, eGPU: +30 % (45.6 → 59.4 t/s), acceptance 75 %.
- Gemma 4 26B QAT, eGPU: +13.5 % (47.8 → 54.2 t/s).
- Qwen3.6-35B-A3B, iGPU: +20 % (23.4 → 28.2 t/s).

**n=2 beat n=3** in every recent run: each extra draft token costs a verification, and acceptance drops fast
(75 % → 60 % on Qwen, 56 % → 40 % on Gemma, which lost speed at n=3). The build matters as much as the flags:
master `de7fa0a` dropped acceptance from 0.75 to 0.63 against `abeada3` on the same model. Keep the binary that
worked.

`llama-bench` cannot measure any of this; use `llama-server` (or [`bin/serve-bench`](../bin/serve-bench)).

## 3. The eGPU link is a real bottleneck for MoE

With experts in system RAM, large batches (prompt processing) copy expert weights to the GPU over the link.
At PCIe 3.0 x4 (~1.1 GB/s measured) this hurts:

- The **iGPU read a 5,780-token prompt 2.5x faster** than the eGPU (390 against 156 t/s), because it shares the
  RAM and copies nothing. For a 30k-token prompt with a 500-token answer: about 3 min 25 s on the eGPU against
  1 min 35 s on the iGPU (estimated from these speeds).
- `GGML_OP_OFFLOAD_MIN_BATCH=1024` keeps batches under 1024 tokens on the CPU instead of copying experts. Prompts
  of 150-700 tokens went from 10-16 s to 2.4-6 s before the first token; a 37k-token prompt was unchanged
  (280 → 288 t/s).
- The MoE expert GPU cache (`--moe-cache-mib`) was a **net loss**: every miss crosses the link, and 8 GB only
  holds a small share of the experts. 35 t/s without it, 1-8 t/s with it.
- Generation also fell after the link dropped from Gen4 to Gen3 (59 → 41 t/s with MTP, same build and flags).
  That one is not explained yet: small batches should not copy experts.

If the cable throws errors at Gen4 (`dmesg` AER messages, link training down), replace it before tuning flags.

## 4. The Radeon 780M is a serious second device

The iGPU has no 8 GB wall: it uses the 16 GB carve-out plus GTT from system RAM. It ran the 22 GB model at
26-28 t/s with MTP, about two thirds of the eGPU speed, and processed long prompts faster. What made the difference:

- **Vulkan, not ROCm**, for generation: 15.75 against 11.3 t/s on Qwen3-8B. ROCm prompt processing was faster
  (372 against 273), and on HIP, flash attention only takes the fast path when K and V caches have the same type.
- **`-ctv q8_0` for models over ~10 GB**: Gemma 4 26B went from 9.9 to 19.0 t/s with a q8_0 V cache and without
  `nogttspill` (both changed together). Memory bandwidth (~89 GB/s shared) is the limit, so a smaller V cache
  helps. On the eGPU it did nothing.
- **`RADV_PERFTEST=nogttspill` only when the model fits in the carve-out.** It keeps allocations in UMA; with a
  model larger than UMA it forces a slow overflow (28.0 → 20.5 t/s on the 22 GB model).
- `-ub 1024` for prompt processing (+15 %).
- Recent Vulkan builds matter here: `abeada3` was 10-25 % faster than b11433 on the 780M.

## 5. Splitting one model across both GPUs rarely pays

Layer split (`-dev Vulkan0,Vulkan1 -ts ...`) runs at the speed of the slowest device plus the synchronisation:

- Qwen3-8B: 23.8 t/s split, against 64.3 on the eGPU alone.
- Qwen3.6-35B-A3B: 7.3 t/s split, against 28-33 on either GPU alone.
- Experts on the iGPU, the rest on the eGPU: 20-21 t/s, slower than the iGPU alone.
- The one win: Qwen3.6-27B dense (16 GB, too big for the card) read prompts 43 % faster split 3/2 than on the
  iGPU alone, at the same 4.6 t/s generation.

Two **separate** servers do work: an iGPU server lost almost nothing while an eGPU server ran next to it, and the
eGPU one lost ~30 % (they share CPU and RAM bandwidth). Good for an embedding model or a small helper model next
to the main one, which is what [`examples/llama-swap.yaml`](../examples/llama-swap.yaml) does.

## 6. What did not help (here)

- **DFlash** (block-diffusion drafter, `--spec-type draft-dflash`): best case 40.6-40.9 t/s at n=3, below MTP
  n=2 (40.9-45.0) in the same session. With experts in RAM, verifying more tokens at once touches more distinct experts, and
  acceptance falls with block size (0.45 at n=3, 0.12-0.26 at n=15). It shines when the whole model fits in VRAM.
- **TurboQuant KV cache** (fork, not upstream): same speed as q8_0/f16 within noise, and the standard cache
  already reached 128k context on this hybrid-attention model.
- **A small separate draft model** across the two GPUs: about 3x slower than no speculation, from cross-device
  synchronisation.
- **Ollama** (spring 2026): about 3x slower than llama.cpp in generation on the MoE model (11.2 against 33.3 t/s),
  no MTP on CUDA, no `-ncmoe`.
- **The NPU** (XDNA 1): no LLM runtime on Linux, see [npu.md](npu.md).

## 7. Benchmarking hygiene that saved bad conclusions

- Give every test server its own port and log file. Two servers once wrote to the same log and produced numbers
  that had to be thrown away.
- Write down the build hash, the link speed and the exact flags next to each number. Several apparent
  regressions found here were a different build, a different `-ncmoe` or a different measuring method.
- Run each config at least three times and report the range. MTP runs varied by up to 11 % between runs.
- Measure the way you use it: a 200-token poem, a short chat and a long chat ranked the configs differently.
