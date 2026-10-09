# Setup

What was installed on the machine, as of October 2026. Commands are for Fedora 44; the llama.cpp part works on
any distribution.

## Hardware

| Part | Detail |
|---|---|
| CPU | Ryzen 7 8845HS, 8 cores / 16 threads, Zen 4 (AVX-512, VNNI, BF16) |
| iGPU | Radeon 780M, RDNA3, `gfx1103`. 16 GB UMA carve-out set in the BIOS, plus GTT from system RAM |
| eGPU | RTX 3060 Ti 8 GB GDDR6 (448 GB/s), OCuLink dock, PCIe x4 |
| RAM | 64 GB DDR5 (about 47 GB left to Linux with the 16 GB carve-out) |
| NPU | AMD XDNA 1 (`npu1`), 16 TOPS: not usable for LLMs on Linux, see [npu.md](npu.md) |
| OS | Fedora 44, kernel 7.2, NVIDIA 615.71 (RPM Fusion), Mesa 26.2, GCC 16.2 |

OCuLink is plain PCIe on a cable, so there is no Thunderbolt controller in the way. It still is only four lanes,
and the cable matters: this one started to throw errors at PCIe 4.0 and the link was forced down to 3.0, about
1.1 GB/s measured host to device. Check what the link can do and what it trained to:

```bash
nvidia-smi --query-gpu=pcie.link.gen.max,pcie.link.gen.gpumax,pcie.link.width.max --format=csv
# The current generation drops at idle to save power: read it while the GPU is busy.
nvidia-smi --query-gpu=pcie.link.gen.current,pcie.link.width.current --format=csv -l 1
sudo dmesg | grep -i aer      # corrected/uncorrected PCIe errors point at the cable or dock
```

## Drivers

### NVIDIA and CUDA

```bash
# Enable RPM Fusion (free and nonfree), then install the driver from it
sudo dnf install -y \
  https://mirrors.rpmfusion.org/free/fedora/rpmfusion-free-release-$(rpm -E %fedora).noarch.rpm \
  https://mirrors.rpmfusion.org/nonfree/fedora/rpmfusion-nonfree-release-$(rpm -E %fedora).noarch.rpm
sudo dnf install -y akmod-nvidia xorg-x11-drv-nvidia-cuda
# Toolkit from NVIDIA's own repository (CUDA 13.4 supports GCC 16 and Fedora 44)
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora44/x86_64/cuda-fedora44.repo
sudo dnf install -y cuda-toolkit
export PATH="/usr/local/cuda/bin:$PATH"
```

### Vulkan for the iGPU (and the eGPU)

```bash
sudo dnf install -y vulkan-headers vulkan-loader-devel glslc spirv-headers-devel
```

Mesa's RADV driver is the one to use on the 780M. ROCm works too (Fedora ships ROCm 7.1 built for `gfx1103`)
but it generated 28 % slower than Vulkan here (11.3 against 15.75 t/s): see [results.md](results.md#radeon-780m-vulkan-against-rocm-qwen3-8b-q4_k_m-may).

## llama.cpp

Two builds: Vulkan (sees both GPUs) and CUDA (sees the NVIDIA card only). MTP is in master since b9180, no
branch needed.

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
cmake -B build -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j"$(nproc)" --target llama-server llama-bench
```

Builds move fast and not always forward for a given setup: master `de7fa0a` predicted worse than `abeada3` with
MTP (acceptance 0.75 → 0.63). Keep a copy of a build that works before pulling. A plain `cp` is not enough: the
binaries have an absolute RUNPATH to `build-cuda/bin` and would load the libraries of the next build. Point the
copy at itself with `patchelf` (`sudo dnf install patchelf`):

```bash
mkdir -p ~/llama-builds
snap=~/llama-builds/cuda-$(git -C ~/llama.cpp rev-parse --short HEAD)
cp -a ~/llama.cpp/build-cuda/bin "$snap"
find "$snap" -maxdepth 1 -type f \( -name 'llama-*' -o -name 'lib*.so*' \) -exec patchelf --set-rpath '$ORIGIN' {} \;
ldd "$snap/llama-server" | grep -E 'ggml|llama'   # every library must resolve inside $snap
```

Check the device names of the Vulkan build, `ai-auto` assumes `Vulkan0` = iGPU and `Vulkan1` = eGPU:

```bash
~/llama.cpp/build/bin/llama-server --list-devices
```

Flags renamed since spring 2026: `--spec-type mtp` → `--spec-type draft-mtp`; `--draft-max/--draft-min` →
`--spec-draft-n-max/--spec-draft-n-min`; `-cd` removed (one draft context); `--no-mmap/--mlock` →
`-lm/--load-mode`.

## Models

```bash
curl -LsSf https://hf.co/cli/install.sh | bash      # the `hf` CLI
hf download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --local-dir models/qwen3.6-35b-a3b-mtp
hf download unsloth/gemma-4-26B-A4B-it-qat-GGUF gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf mtp-gemma-4-26B-A4B-it.gguf \
  --local-dir models/gemma-4-26b-qat
```

Unsloth's MTP files have the same name as the non-MTP ones, so keep them in their own folder. `ai-auto` reads
`nextn_predict_layers` from the header, it does not rely on the file name.

## Tools in this repo

```bash
export LLAMA_DIR=~/llama.cpp
bin/ai-auto info models/qwen3.6-35b-a3b-mtp/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf   # print the commands
bin/ai-auto serve models/.../model.gguf                                     # run llama-server
bin/serve-bench --server "$LLAMA_DIR/build-cuda/bin/llama-server" \
  -- -m model.gguf -ngl 99 -fa on -ncmoe 33 --spec-type draft-mtp --spec-draft-n-max 2 -np 1
```

## A chat UI

Any OpenAI-compatible client works with `llama-server`. Open WebUI with rootless Podman, with llama-server on
`127.0.0.1:8080`:

```bash
podman run -d --name open-webui \
  --network pasta:-T,8081:8080 -p 127.0.0.1:3000:8080 \
  -v open-webui:/app/backend/data \
  -e ENABLE_OLLAMA_API=false \
  -e OPENAI_API_BASE_URL=http://127.0.0.1:8081/v1 \
  -e OPENAI_API_KEY=none \
  ghcr.io/open-webui/open-webui:main
```

`host.containers.internal` does **not** reach a server bound to the host's `127.0.0.1` (Podman 5.8 with pasta:
connection refused, checked). `-T,8081:8080` makes pasta forward the container's own `127.0.0.1:8081` to the
host's loopback port 8080 (8081 because Open WebUI itself listens on 8080 inside the container). Publish the UI
on `127.0.0.1` explicitly.

To keep several models behind one endpoint and load them on demand, see
[`examples/llama-swap.yaml`](../examples/llama-swap.yaml).
