# Running on a GPU server

The service uses the GPU automatically when PaddlePaddle is the CUDA build.
No code change, no config change — `OCR_DEVICE` defaults to `auto` and detects
it. Switching means installing one different package.

Expect roughly **5× faster**: median 40 s per invoice on CPU, target 10–15 s on
GPU.

---

## What you need

- An NVIDIA GPU with **4 GB VRAM or more**
- The NVIDIA **driver** (not the CUDA Toolkit — the wheel ships its own runtime)
- Python 3.11

---

## Step by step on your server

### 1. Check the GPU is visible

```bash
nvidia-smi
```

You should see the card and, top right, a CUDA version. It must read **12.6 or
higher** — that is the maximum the driver supports, not what is installed.

- `command not found` → the driver is missing, go to step 2.
- CUDA lower than 12.6 → update the driver, go to step 2.
- Card listed, CUDA 12.6+ → skip to step 3.

### 2. Install the driver (only if step 1 failed)

Ubuntu / Debian:

```bash
sudo apt update
sudo ubuntu-drivers install          # or: sudo apt install -y nvidia-driver-550
sudo reboot
```

RHEL / Rocky / Alma:

```bash
sudo dnf install -y epel-release
sudo dnf install -y nvidia-driver
sudo reboot
```

After the reboot, run `nvidia-smi` again. It must work before continuing.

### 3. Install the GPU build

Stop the service first, then, in the project's virtual environment:

```bash
pip install "paddlepaddle-gpu==3.3.1" \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Install it **over** the existing `paddlepaddle` — do not uninstall first.
About 1.5 GB downloaded, 3–4 GB on disk.

### 4. Confirm the switch

```bash
python -c "import paddle; print(paddle.device.is_compiled_with_cuda(), paddle.device.cuda.device_count())"
```

`True 1` means ready. `False 0` means the CPU build is still installed.

### 5. Start the service and check it is really using the GPU

```bash
pharmacy-ocr serve
```

Two confirmations, both worth doing:

- The startup log contains **`auto-selected GPU for OCR`**.
- While an invoice is processing, `nvidia-smi` in another terminal lists a
  `python` process under *Processes* and shows GPU memory in use.

Nothing else changes. Same commands, same API, same output format — only faster.

---

## Docker

The shipped `Dockerfile` builds a **CPU** image. For GPU, add one line after the
requirements install:

```dockerfile
RUN pip install "paddlepaddle-gpu==3.3.1" \
      -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

The base image does not need to be an NVIDIA CUDA image. The **host** needs the
driver plus the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/),
and the container needs GPU access:

```bash
docker run --gpus all -v ocr-data:/data -p 8000:8000 pharmacy-ocr
```

On Kubernetes, request `nvidia.com/gpu: 1`.

Mount the `/data` volume — the OCR models (~200 MB) live there, and without it
every restart re-downloads them.

---

## Rollback to CPU

Stop the service, then:

```bash
pip uninstall -y paddlepaddle-gpu
pip install --force-reinstall "paddlepaddle==3.3.1"
pip install -r requirements.txt
```

The third line is required — `--force-reinstall` upgrades dependencies and
breaks `paddlex`'s numpy pin. Check with `pip check` (expect no broken
requirements).

The OCR cache is keyed per device, so CPU entries survive a GPU run and are
reused on the way back.

---

## Notes

**Don't uninstall while the service is running.** It holds `libpaddle.pyd` open;
on Windows the uninstall aborts partway and deletes files permanently. If you
see `ImportError: cannot import name ... from partially initialized module
'paddle.base'`, that happened — fix with the rollback commands above.

**Out of memory?** Lower `MAX_LONG_SIDE` (default 3300) and set
`MAX_CONCURRENT_EXTRACTIONS=1` before concluding the card is too small.

**Choosing a server?** Any NVIDIA card with ≥4 GB VRAM works; a T4 is the
standard choice. Avoid burstable cloud instances (AWS `t3`, Azure `B`-series) —
sustained OCR drains their CPU credits and throughput collapses.

**Compare without reinstalling:** `OCR_DEVICE=cpu` forces CPU for one run.
