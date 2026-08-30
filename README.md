# Unsloth Docker image for AMD GPUs (ROCm + Vulkan)

[Unsloth](https://github.com/unslothai/unsloth) fully supports AMD GPUs, but doesn't ship an AMD Docker image. This repo builds one, with **both GPU stacks** inside:

- **ROCm 7.1** — used for training/fine-tuning (PyTorch ROCm, Triton, ROCm-compatible bitsandbytes for 4-bit QLoRA) and as the default llama.cpp inference backend in Unsloth Studio.
- **Vulkan** (Mesa RADV) — alternative llama.cpp inference backend, useful for GPUs with weak or missing ROCm support (RDNA2, iGPUs/APUs) or when ROCm misbehaves. Enabled at runtime with `UNSLOTH_FORCE_VULKAN=1`.

The image also carries the full build toolchain (`cmake`, `ninja`, `hipcc` from the base image, `glslc`, `libvulkan-dev`), so Unsloth's llama.cpp *source-build fallback* works for both backends when no prebuilt matches your gfx architecture.

## Requirements

- Linux host with the `amdgpu` kernel driver (for ROCm also the KFD interface — installed with ROCm or `amdgpu-dkms`; only the *kernel* part is needed on the host, userspace lives in the image).
- Docker (or Podman with equivalent flags).

Supported hardware follows [Unsloth's AMD matrix](https://unsloth.ai/docs/basics/amd): RDNA3/3.5/4 and Instinct MI200–MI350 fully, RDNA2 nearly (Vulkan recommended there).

## Get the image

Prebuilt by [GitHub Actions](.github/workflows/build.yml) on every Dockerfile change, tag, and weekly (to pick up new Unsloth releases). Every build is a clean build (~15–20 min): a registry layer cache on GHCR turned out to be unusable — GHCR throttles the multi-GB cached layers with `429 Too Many Requests` and BuildKit does not retry, so cached builds failed reliably.

```bash
docker pull ghcr.io/selfref/unsloth-docker-amd:latest
```

Or build locally:

```bash
docker build -t unsloth-amd .
```

Build args if you need different versions:

| Arg | Default | Purpose |
|---|---|---|
| `ROCM_TAG` | `7.1-complete` | ROCm base image tag (`rocm/dev-ubuntu-24.04`) |
| `TORCH_INDEX_URL` | `.../whl/rocm7.1` | PyTorch ROCm wheel index — keep in sync with `ROCM_TAG` |
| `TORCH_SPEC` | `torch>=2.4,<2.11.0` | Unsloth's documented torch pin for AMD |
| `BNB_WHEEL` | bitsandbytes `1.33.7.preview` | ROCm-compatible bitsandbytes wheel URL |
| `UNSLOTH_VERSION` | *(empty = latest pip resolves)* | Pin the Unsloth release. CI passes the newest PyPI version and publishes it as the `unsloth-<version>` image tag and the `dev.selfref.unsloth.version` label. |

## Run

### Unsloth Studio (web UI), default:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --ipc=host \
  -p 8000:8000 \
  -v "$PWD/work:/workspace" \
  -v "$PWD/huggingface:/workspace/.cache/huggingface" \
  -v unsloth-studio:/opt/studio-data \
  unsloth-amd
```

Open http://localhost:8000. Or use compose: `docker compose up`.

### Training scripts instead of the UI:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$PWD/work:/workspace" \
  unsloth-amd python my_finetune.py
```

## Choosing ROCm vs Vulkan

| | |
|---|---|
| Training / fine-tuning | Always ROCm (PyTorch). Vulkan is inference-only. |
| Studio inference, default | ROCm llama.cpp build matched to your gfx arch |
| Studio inference, Vulkan | add `-e UNSLOTH_FORCE_VULKAN=1` |

Vulkan-only usage (e.g. RDNA2 without ROCm on the host) needs only `--device=/dev/dri`; `/dev/kfd` can be omitted.

## GPU-specific environment variables

ROCm on consumer GPUs often needs the gfx override:

| GPU | Env var |
|---|---|
| RX 7000 (gfx1100/1101/1102) | `-e HSA_OVERRIDE_GFX_VERSION=11.0.0` |
| RX 6000 (gfx1030, RDNA2) | `-e HSA_OVERRIDE_GFX_VERSION=10.3.0` |
| Ryzen AI MAX / Strix Halo (gfx1151) | `-e HSA_OVERRIDE_GFX_VERSION=11.5.1` |
| Instinct MI300X | usually none needed (`9.4.2` if detection fails) |

Multi-GPU selection: `-e HIP_VISIBLE_DEVICES=0`.

### MoE backend

The image sets `UNSLOTH_MOE_BACKEND=native_torch` by default: Unsloth's `torch._grouped_mm` runtime probe segfaults (uncatchable native crash on import) on RDNA consumer GPUs/APUs — verified on Strix Halo (gfx1151). This only affects MoE-model training speed. On Instinct datacenter GPUs (MI200+), re-enable the fast path with `-e UNSLOTH_MOE_BACKEND=grouped_mm`.

## Verifying the stacks inside the container

```bash
docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined unsloth-amd \
  python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"

docker run --rm --device=/dev/dri unsloth-amd vulkaninfo --summary

docker run --rm --device=/dev/kfd --device=/dev/dri --security-opt seccomp=unconfined unsloth-amd amd-smi list
```

## Notes

- On first start Studio creates an admin account (user `unsloth`). Pass `-e UNSLOTH_STUDIO_PASSWORD=...` to set its password non-interactively; otherwise a bootstrap password is printed in the container logs and saved to `/opt/studio-data/auth/.bootstrap_password` (log in and change it within `UNSLOTH_STUDIO_BOOTSTRAP_TIMEOUT`, default 1 h, or Studio shuts itself down). Studio applies the variable to the *initial* password only and refuses to start if it is set once a password exists — the image's entrypoint drops it in that case, so it is safe to keep in a compose file.
- Downloaded models/datasets persist in the `./huggingface` folder (`HF_HOME=/workspace/.cache/huggingface`); your own files live in the `./work` bind mount. Studio's own state — accounts, chat history, download cache, datasets, outputs, runs, downloaded llama.cpp builds — is kept in `/opt/studio-data` (symlinked from `UNSLOTH_STUDIO_HOME=/opt/studio` by the entrypoint): mount a volume there (`-v unsloth-studio:/opt/studio-data`, as in `compose.yml`). Do **not** mount `/opt/studio` itself — it also holds the Python venv, which would then be frozen across image upgrades.
- Behind a reverse proxy that sets `X-Forwarded-For`, add `-e UNSLOTH_STUDIO_TRUST_FORWARDED=1` so login rate limiting keys on the real client.
- The container runs as root (standard for ROCm images — device access is simplest that way). Files created in `/workspace` will be root-owned on the host; `chown` them or run with `--user` plus matching `render`/`video` group IDs if that matters to you.
- `HF_HOME` points into `/workspace`, so downloaded models/datasets persist in the mounted volume.
- Pass `-e HF_TOKEN=...` for gated models.
- Expect a large image (~25 GB+): the `-complete` ROCm base plus PyTorch ROCm wheels are heavy.

## Sources

- [Unsloth AMD install guide](https://unsloth.ai/docs/get-started/install/amd)
- [Train & run models on AMD GPUs with Unsloth](https://unsloth.ai/docs/basics/amd)
- [Studio Vulkan llama.cpp support (unsloth#5819)](https://github.com/unslothai/unsloth/pull/5819)
- [ROCm-compatible bitsandbytes releases](https://github.com/bitsandbytes-foundation/bitsandbytes/releases/tag/continuous-release_main)
