# Results

Every number below comes with its date, llama.cpp build and the flags that produced it. They were taken in
three sessions on the same machine, with different builds and a different eGPU link speed, so **compare rows
within a session, not across sessions**. The raw values are in [`data/results.csv`](../data/results.csv).

| Session | Dates | llama.cpp | Software | eGPU link |
|---|---|---|---|---|
| May | 2026-05-05 to 05-09 | b9013 (`e48034dfc`), MTP from PR #22673 (`5d5f1b46e`) | kernel 6.19, NVIDIA 595.71, Mesa 26.0.5, CUDA 13.2 | PCIe 4.0 x4 |
| Oct, Gen4 | 2026-10-06 to 10-07 | master `abeada3` (CUDA), b11433 `50569eb87` (Vulkan) | kernel 7.2, NVIDIA 615.71, Mesa 26.2, CUDA 13.4.92 | PCIe 4.0 x4 |
| Oct, Gen3 | 2026-10-09 | master `abeada3` and `de7fa0a` | same | **PCIe 3.0 x4** (cable errors at 4.0, ~1.1 GB/s host-to-device measured with `cudaMemcpy`) |

Method: `llama-bench -p 512 -n 200 -r 3` where it says llama-bench. Everything with MTP or another kind of
speculative decoding comes from `llama-server` (llama-bench cannot run it), with the prompt
"Write a short poem about GPU programming.", `temperature 0`, 200 (or 150) generated tokens, read from the
server's timings. [`bin/serve-bench`](../bin/serve-bench) reproduces that method. "pp" is prompt processing,
"tg" is token generation, both in tokens per second.

## Qwen3.6-35B-A3B (MoE, 256 experts, 8 active, ~21-22 GB at Q4)

The model is 2.5x larger than the card's VRAM. On the eGPU, attention and some expert layers live in VRAM and
the remaining experts stay in system RAM (`-ncmoe N` = number of layers whose experts stay on the CPU).

### Oct, Gen4 link: CUDA master `abeada3`, Unsloth `Qwen3.6-35B-A3B-MTP` UD-Q4_K_M (21.1 GiB)

Common flags: `-ngl 99 -fa 1 -ctk q8_0 -ctv f16 -ub 512`. Server runs: `-c 2048 -np 1`, tg200.

| Config | pp512 | tg | MTP acceptance |
|---|---:|---:|---:|
| llama-bench, `-ncmoe 30`, no MTP | 298.8 | 45.61 | |
| llama-server, `-ncmoe 30`, no MTP (2 runs, 2 days) | | 45.55-45.98 | |
| llama-server, `-ncmoe 33`, MTP n=2 (4 runs + 1 replay) | | **58.78-59.37** | 74.8 % |
| llama-server, `-ncmoe 32`, MTP n=2 (3 runs) | | 51.68-57.34 | 68.9 % |
| llama-server, `-ncmoe 33`, MTP n=3 (4 runs) | | 54.22-54.49 | 59.6 % |
| llama-server, `-ncmoe 30`, MTP n=2 | | OOM (draft context needs 98 MiB more) | |
| Vulkan b11433, llama-bench, `-ncmoe 30` | 217.42 | 33.76 | |

`-ncmoe` sweep, Vulkan b11433 (tg200): 28 OOM, 29 33.76, **30 33.76**, 31 33.20, 32 31.01, 33 31.97, 34 31.20,
36 29.77, 40 27.31.

### Oct, Gen3 link: one model, four ways to run it

`Huihui-Qwen3.6-35B-A3B-abliterated` Q4_K (a finetune with the same architecture and MTP head), MTP n=2,
`-ctk q8_0 -ctv q8_0`, `--jinja`, thinking off. Three workloads: the completion prompt above, the same text as a
short chat, and a long chat (5,780-token prompt).

- eGPU: CUDA `abeada3`, `-ncmoe 33 -ub 2048 -b 2048 -c 65536`, `GGML_OP_OFFLOAD_MIN_BATCH=1024`
- iGPU: Vulkan, `-dev Vulkan0 -ub 1024 -c 32768`

| Config | Completion tg | Short chat tg | Long chat pp / tg |
|---|---:|---:|---:|
| eGPU, CUDA | 41.5-43.2 | 31-37 | 155-158 / 38-43 |
| iGPU, Vulkan `abeada3` | 25.6-27.9 | 20-21 | **390-394** / 28.5-29.7 |
| iGPU, Vulkan b11433 | 23.6-25 | 16-19 | 373-375 / 22-23.5 |
| Both GPUs, experts on the iGPU (`-ot ffn_.*_exps=Vulkan0`) | 20-21 | 16-17 | 175-310 / 19.6 |
| Two servers at once: the eGPU one | 28-29 | 21-26 | 155 / 24-38 |
| Two servers at once: the iGPU one | 27-29.5 | 20.5-21 | 379-397 / 25-28 |

**The Gen4 numbers did not come back.** On 2026-10-09, `abeada3` with the exact 2026-10-06 flags measured
41 t/s with MTP (59 before) and 35.5 without (46 before), at the same acceptance. The only known change is the
link dropping to PCIe 3.0. That explains the slower prompt processing (experts copied to the GPU for big
batches); it is **not proven** for generation, where small batches should not copy experts. To be re-measured
with a new cable.

### Oct, Gen3 link: where to put the experts (master `de7fa0a`)

Unsloth UD-Q4_K_M, `-ctk q8_0 -ctv q8_0`, llama-server tg200, short prompt.

| Config | tg | VRAM used |
|---|---:|---:|
| `-cmoe` (all experts on CPU), no MTP | 37.0 | 2.6 GB |
| `-ncmoe 30`, no MTP | 33.0 | 7.4 GB |
| `-cmoe`, MTP n=2 | **45.6** | 3.1 GB |
| `-ncmoe 36`, MTP n=2 | 44.8 | |
| `-ncmoe 33`, MTP n=2 | 39.9 | |
| Long prompt (5,780 tok, `-ub 2048 -c 16384`), `-ncmoe 33` | pp 141 / tg 35.7 | |
| Long prompt, `-cmoe` | pp 114 / tg 40.6 | |

`-cmoe` generates ~14 % faster and reads prompts ~19 % slower than `-ncmoe 33` on this link.

`master de7fa0a` against `abeada3`, MTP n=2, `-ncmoe 33`: acceptance 0.748 → 0.634 and 41.1 → 37.8 t/s on the
completion prompt. The production config stayed on `abeada3`.

### Speculative decoding alternatives (Oct, Gen3 link, `de7fa0a`)

DFlash drafter `z-lab/Qwen3.6-35B-A3B-DFlash` converted to GGUF bf16 (772 MB), `-ngl 99 -fa 1 -ctk q8_0 -ctv f16
-ub 512 -np 1 -c 4096`, tg200, three runs each.

| Config | tg (3 runs) | Acceptance | VRAM |
|---|---|---:|---:|
| MTP n=2, `-cmoe` | 45.0 / 40.9 / 44.5 | 0.67 | 3.1 GB |
| no speculation, `-cmoe` | 43.0 / 39.4 / 38.7 | | 2.7 GB |
| DFlash n=3, `-ncmoe 36` | 40.9 / 40.6 / 40.7 | 0.45 | 6.6 GB |
| DFlash n=3, `-cmoe` | 38.3 / 39.1 / 38.2 | 0.45 | 4.7 GB |
| MTP n=2, `-ncmoe 33` | 37.8 / 37.8 / 38.0 | 0.63 | 6.7 GB |
| DFlash n=4, `-cmoe` | 36.1 / 36.6 / 37.0 | 0.40 | 4.7 GB |
| DFlash n=6, `-cmoe` | 22.4 / 22.7 / 22.3 | 0.28 | 4.9 GB |

MoE expert GPU cache (`--moe-cache-mib`, PR #29887, `de7fa0a`): `-ncmoe 30` without cache 35.0 t/s;
`-ncmoe 32` + 1000 MiB cache 1-4.5 t/s (53 % hit rate); `-cmoe` + 4000 MiB 7.9 t/s; `-cmoe` + 4500 MiB 8.2 t/s
(81 % hit rate); `-ncmoe 33` + 2000 MiB OOM. Every miss crosses the slow link.

### May: Vulkan b9013 and the MTP pull request

`-ngl 99 -fa 1 -ctk q8_0 -r 3`, `Qwen3.6-35B-A3B` Q4_K_M (22.28 GiB).

| Config | pp512 | tg200 |
|---|---:|---:|
| eGPU Vulkan, `-ncmoe 32` | 191 | 33.3 |
| iGPU Vulkan | 309 | 28.0 |
| iGPU Vulkan + `RADV_PERFTEST=nogttspill` | 333 | 20.5 |
| Both GPUs, `-ts 4/1` | 310.8 | 7.3 |
| Ollama (CUDA), same model | 71.7 | 11.2 |

MTP, PR #22673 build `5d5f1b46e`, `localweights/Qwen3.6-35B-A3B-MTP-Q4_K_M` (20.22 GiB), llama-server tg150:

| Device | Config | tg | Acceptance |
|---|---|---:|---:|
| eGPU CUDA | `-ncmoe 30`, no MTP | 39.5 | |
| eGPU CUDA | `-ncmoe 33`, MTP n=3 | 50.0 | 65.3 % |
| eGPU CUDA | `-ncmoe 35`, MTP n=3 | 44.6 | 58.0 % |
| eGPU CUDA | `-ncmoe 37`, MTP n=3 | 40.1 | 50.3 % |
| iGPU Vulkan `-ub 1024` | no MTP | 23.4 | |
| iGPU Vulkan `-ub 1024` | MTP n=2 | 28.2 | 73.3 % |
| iGPU Vulkan `-ub 1024` | MTP n=3 | 27.2 | 59.8 % |

## Gemma 4 26B-A4B (MoE, 128 experts)

Oct, Gen4, CUDA `abeada3`, `unsloth/gemma-4-26B-A4B-it-qat` UD-Q4_K_XL with the separate MTP drafter
`mtp-gemma-4-26b-a4b-it.gguf` placed on CUDA0 (`-devd CUDA0 -otd token_embd.weight=CUDA0`), `-ncmoe 17
-ctk q8_0 -ctv f16 -ub 512 -c 4096`:

| Config | tg200 | Acceptance |
|---|---:|---:|
| llama-bench | 49.00 (pp512 516.82) | |
| llama-server, no MTP | 47.78 | |
| llama-server, MTP n=2 | **54.24** | 55.6 % |
| llama-server, MTP n=3 | 46.64 | 40.2 % |

May, b9013, UD-Q4_K_M (15.78 GiB):

| Config | pp512 | tg200 |
|---|---:|---:|
| iGPU Vulkan, `-ctv q8_0 -ub 1024` | 334 | 19.0 |
| iGPU Vulkan, f16 V cache + `nogttspill` | 123 | 9.9 |
| eGPU Vulkan, `-ncmoe 22` | 282 | 29.0 |

## Dense models

May, b9013, `-ngl 99 -fa 1 -ctk q8_0`.

| Model | Device | pp512 | tg200 |
|---|---|---:|---:|
| Qwen3-8B Q4_K_M (4.86 GiB) | eGPU Vulkan | 2324 | 64.3 |
| Qwen3-8B Q4_K_M | iGPU Vulkan (`nogttspill`) | 272 | 15.65 |
| Qwen3-8B Q4_K_M | both, `-ts 1/3` | 640 | 23.8 |
| Qwen3-8B Q4_K_M | eGPU, Ollama CUDA | 2317.8 | 71.5 |
| Qwen3.6-27B Q4_K_M (16.21 GiB) | iGPU Vulkan | 67.7 | 4.56 |
| Qwen3.6-27B Q4_K_M | both, `-ts 3/2` | 98.5 | 4.6 |
| Llama-3-8B-Instruct EXL2 4.0bpw | eGPU, ExLlamaV2 0.3.2 | 2900 | 73.3 |

The ExLlamaV2 row uses a different model and is only a framework comparison. ExLlamaV2 has since been archived
in favour of ExLlamaV3.

### Radeon 780M: Vulkan against ROCm (Qwen3-8B Q4_K_M, May)

ROCm build: `GGML_HIP=ON`, `AMDGPU_TARGETS=gfx1103`, `HSA_OVERRIDE_GFX_VERSION=11.0.3`. ROCm hung with
`-ngl 99` (offloading the output layer); `-ngl 36` (the layer count) works.

| Backend | Config | pp512 | tg200 |
|---|---|---:|---:|
| Vulkan | `-ngl 99 -fa 1` | 273 | 15.75 |
| ROCm | `-ngl 36 -fa 1` | 372 | 11.3 |
| ROCm | `-ngl 36 -fa 1 -ctk q8_0` (V cache f16) | 247 | 10.33 |
| ROCm | `-ngl 36 -fa 1 -ctk q8_0 -ctv q8_0` | 349 | 10.95 |

The first two rows come from one comparison and the last two from another run the same day; the KV cache types
of the first comparison were not recorded. On HIP, mixed K/V types fall off the fused flash-attention path
(llama.cpp discussion #22411), hence 247 against 349.

### Context that fits on the 8 GB card

- Qwen3-8B Q4_K_M, `-ctk q8_0` (May): 32,768 loads, 40,960 runs out of memory.
- Qwen3.6-35B-A3B UD-IQ4_XS, `-ncmoe 32 -ub 512 -np 1` (Oct): 32k, 64k and 128k all load with q8_0/f16 KV
  (only checked with `/health`, not with a prompt that long). This hybrid model has only 10 full-attention
  layers out of 40, so its KV cache is small.
- Production config: Qwen3.6-35B-A3B Q4_K, `-ncmoe 33 -ub 2048 -c 65536`, q8_0/q8_0. `-ncmoe 30`, `-ub 4096` and
  `-c 131072` each ran out of memory with MTP on.

## KV cache compression fork (TurboQuant)

Not in upstream llama.cpp. Tested from the [TheTom/llama-cpp-turboquant](https://github.com/TheTom/llama-cpp-turboquant)
fork (`bcb85fc`), Oct, Gen4, UD-IQ4_XS:

| Cache K/V | `-ncmoe` | pp512 | tg200 (llama-bench) | tg (server) |
|---|---:|---:|---:|---:|
| q8_0 / f16 | 28 | 382.84 | 44.37 | |
| turbo3 / turbo3 | 28 | 375.85 | 44.91 | |
| turbo4 / turbo3 | 28 | 375.29 | 44.45 | |
| q8_0 / f16 | 32 | | 42.89 | 40.70 |
| turbo4 / turbo3 | 32 | 344.09 | 41.23 | 37.55 |
| iGPU Vulkan q8_0 / f16 | | 355.50 | 24.02 | 24.35 |
| iGPU Vulkan turbo | | 346.44 | 23.95 | 24.22 |

No speed gain, and no context gain here since the standard cache already reached 128k. One main run per config.
