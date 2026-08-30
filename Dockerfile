# Unsloth for AMD GPUs — ROCm + Vulkan
#
# Training/fine-tuning runs on PyTorch ROCm. Inference (Unsloth Studio /
# llama.cpp) uses ROCm by default on AMD; set UNSLOTH_FORCE_VULKAN=1 at
# runtime to use the Vulkan backend instead (useful for RDNA2 / iGPUs /
# anything ROCm doesn't cover). Both userspace stacks are included:
#   - ROCm 7.1 (HIP, hipBLAS, rocBLAS, amd-smi) from the base image
#   - Vulkan loader + Mesa RADV ICD + glslc (so llama.cpp Vulkan builds work)
#
# Build:
#   docker build -t unsloth-amd .
#
# Run (Studio web UI on http://localhost:8000):
#   docker run -it --rm \
#     --device=/dev/kfd --device=/dev/dri \
#     --security-opt seccomp=unconfined \
#     -p 8000:8000 -v $PWD/work:/workspace \
#     unsloth-amd
#
# See README.md for GPU-specific env vars (HSA_OVERRIDE_GFX_VERSION, etc.)

ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04
ARG ROCM_TAG=7.1-complete
FROM ${ROCM_IMAGE}:${ROCM_TAG}

# PyTorch ROCm wheel index must match the base image's ROCm major.minor
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/rocm7.1
# Unsloth's documented pin for AMD (https://unsloth.ai/docs/get-started/install/amd)
ARG TORCH_SPEC="torch>=2.4,<2.11.0"
# ROCm-compatible bitsandbytes preview wheel (multi-backend), per Unsloth AMD docs
ARG BNB_WHEEL=https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_main/bitsandbytes-1.33.7.preview-py3-none-manylinux_2_24_x86_64.whl

ENV DEBIAN_FRONTEND=noninteractive

# Python, build toolchain (llama.cpp source-build fallback needs cmake/ninja
# and, for the Vulkan backend, glslc + libvulkan-dev), and the Vulkan runtime:
# libvulkan1 (loader), mesa-vulkan-drivers (RADV ICD for AMD), vulkan-tools
# (vulkaninfo, for debugging GPU visibility in the container).
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip python3-dev \
        git curl ca-certificates \
        build-essential cmake ninja-build pkg-config \
        libcurl4-openssl-dev \
        libvulkan1 libvulkan-dev vulkan-tools mesa-vulkan-drivers glslc \
    && rm -rf /var/lib/apt/lists/*

# Ubuntu 24.04 is PEP 668 externally-managed — use a venv.
# Path matters: the `unsloth studio` CLI only runs in-process when
# sys.prefix == $UNSLOTH_STUDIO_HOME/unsloth_studio (otherwise it demands the
# curl-installer's separate venv, which would duplicate the whole torch stack).
ENV UNSLOTH_STUDIO_HOME=/opt/studio
ENV VIRTUAL_ENV=${UNSLOTH_STUDIO_HOME}/unsloth_studio
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# PyTorch built for ROCm
RUN pip install --no-cache-dir "${TORCH_SPEC}" torchvision torchaudio \
        --index-url "${TORCH_INDEX_URL}"

# Unsloth with AMD extras + Studio web UI dependencies
RUN pip install --no-cache-dir "unsloth[amd,studio]"

# ROCm-compatible bitsandbytes (enables 4-bit QLoRA on AMD).
# --no-deps + --force-reinstall: replace whatever CUDA build unsloth pulled in
# without touching the rest of the resolved dependency tree.
RUN pip install --no-cache-dir --force-reinstall --no-deps "${BNB_WHEEL}"

# Finish Unsloth Studio setup at build time (isolated Node, frontend assets,
# install manifest) so the container starts instantly instead of demanding
# install.sh. --no-verify: the damage check flags triton-rocm's unversioned
# bundled libs (libelf.so vs libelf.so.1 etc.) — the symlinks below satisfy
# both the manifest and the dynamic loader, then the final verify must pass.
RUN unsloth studio update --no-verify \
    && cd "${VIRTUAL_ENV}"/lib/python3*/site-packages/triton/backends/amd/lib \
    && ln -sf libelf.so libelf.so.1 \
    && ln -sf libnuma.so libnuma.so.1 \
    && ln -sf libtinfo.so libtinfo.so.6 \
    && unsloth studio verify-install

# Fixes Hugging Face Hub downloads on AMD setups (per Unsloth AMD docs)
ENV HF_HUB_DISABLE_XET=1
# Unsloth's torch._grouped_mm runtime probe segfaults natively on RDNA
# consumer GPUs / APUs (verified on Strix Halo gfx1151, hipBLASLt grouped-GEMM
# unsupported). "native_torch" skips the probe entirely. Only affects MoE-model
# performance — on Instinct (MI200+) you can override with
# -e UNSLOTH_MOE_BACKEND=grouped_mm to re-enable the fast path.
ENV UNSLOTH_MOE_BACKEND=native_torch
# Keep models/datasets on the mounted volume so they survive container removal
ENV HF_HOME=/workspace/.cache/huggingface
# Runtime toggles you may want (documented, not forced):
#   HSA_OVERRIDE_GFX_VERSION  — spoof gfx arch for ROCm on consumer GPUs
#   UNSLOTH_FORCE_VULKAN=1    — make Studio use the Vulkan llama.cpp build

# Studio's persistent state (auth db + admin password, studio.db, download
# cache, datasets, outputs, runs, llama.cpp builds) lives here, symlinked from
# UNSLOTH_STUDIO_HOME by the entrypoint. Mount a volume on THIS path — not on
# /opt/studio, which also holds the Python venv and would freeze it across
# image upgrades. The entrypoint also makes UNSLOTH_STUDIO_PASSWORD idempotent
# (Studio itself exits if it is set once a password exists).
ENV UNSLOTH_STUDIO_DATA=/opt/studio-data
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p "${UNSLOTH_STUDIO_DATA}"
VOLUME ["/workspace", "/opt/studio-data"]

WORKDIR /workspace

EXPOSE 8000

ENTRYPOINT ["docker-entrypoint.sh"]
# Bind to 0.0.0.0 so the published port is reachable from the host.
# Override CMD to run training scripts instead, e.g.:
#   docker run ... unsloth-amd python my_finetune.py
CMD ["unsloth", "studio", "-H", "0.0.0.0", "-p", "8000"]
