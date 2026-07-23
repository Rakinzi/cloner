# Local Training Runbook — WSL2 (Kali) + RTX 4050

End-to-end steps to segment the Shona dataset and fine-tune XTTS v2 on the
laptop GPU. Total GPU time: ~30–60 min segmentation + roughly 20–40 h training
(2–4 overnight runs with resume).

## 0. One-time Windows prep

- Install/update the normal **NVIDIA Windows driver** (Game Ready or Studio).
  Do **not** install any Linux NVIDIA driver inside WSL — WSL uses the Windows one.
- Power settings: plugged in, **sleep off** ("Never" when plugged in). Training
  dies if the laptop sleeps.
- If WSL/Kali is not installed yet, in an **admin** PowerShell:

```powershell
wsl --install -d kali-linux
wsl --update
```

## 1. Verify the GPU is visible inside Kali

```bash
nvidia-smi
```

You should see the RTX 4050 with ~6 GB memory. If this fails, update the
Windows driver and run `wsl --update`, then `wsl --shutdown` and reopen.

## 2. System packages + uv

```bash
sudo apt update
sudo apt install -y git unzip curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
```

## 3. Get the repo and dataset (into the Linux filesystem!)

Keep everything under `~` — NOT under `/mnt/c/...`. Cross-filesystem I/O in
WSL is slow and the training dataloader will crawl.

```bash
cd ~
git clone https://github.com/Rakinzi/cloner
cd cloner
uv sync
```

Plug the LaCie drive into the laptop. Windows gives it a letter (say `D:`),
which appears in WSL as `/mnt/d`:

```bash
mkdir -p ~/data
cp /mnt/d/WaxalNLP/sna_xtts_ft_v2.zip ~/data/
cd ~/data && unzip -q sna_xtts_ft_v2.zip && cd ~/cloner
```

Sanity check — expect `3250`:

```bash
ls ~/data/sna_xtts_ft_v2/wavs | wc -l
```

## 4. Verify CUDA works in the project env

```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Must print `True NVIDIA GeForce RTX 4050 ...`. If `False`, stop and fix this
first — training on CPU takes days.

## 5. Segment the dataset (once, ~30–60 min on GPU)

```bash
uv run python scripts/segment_dataset.py \
  --dataset ~/data/sna_xtts_ft_v2 \
  --output  ~/data/sna_xtts_seg10s
```

First run downloads the ~1.2 GB alignment model. At the end it prints a JSON
report — check that `output_clips` is in the thousands and `align_failed` is
small. Back the result up to LaCie:

```bash
cp -r ~/data/sna_xtts_seg10s /mnt/d/WaxalNLP/
```

## 6. Train

```bash
MPLBACKEND=Agg uv run python scripts/finetune_xtts.py \
  --dataset ~/data/sna_xtts_seg10s \
  --output  ~/checkpoints/shona_xtts \
  --xtts-language en \
  --batch-size 1 \
  --grad-accum 84 \
  --epochs 10 \
  --save-step 1000 \
  --workers 2
```

- Watch the first `print_step` logs (every 50 steps) for the real speed, then
  extrapolate: total iterations ≈ clips × epochs.
- **Close browsers/games first** — 6 GB VRAM is exactly the budget; anything
  else on the GPU risks an out-of-memory crash.
- If you get NaN losses: add `--no-half`.
- If the dataloader hangs: try `--workers 0`.

## 7. Resume after stopping (Ctrl-C, reboot, etc.)

Find the newest checkpoint and pass it back:

```bash
ls -t ~/checkpoints/shona_xtts/run-*/checkpoint_*.pth | head -1
```

```bash
MPLBACKEND=Agg uv run python scripts/finetune_xtts.py \
  --dataset ~/data/sna_xtts_seg10s \
  --output  ~/checkpoints/shona_xtts \
  --xtts-language en \
  --resume <path-from-above> \
  --batch-size 1 \
  --grad-accum 84 \
  --epochs 10 \
  --save-step 1000 \
  --workers 2
```

Checkpoints are ~5 GB each (2 kept + best model) — keep ~15 GB free under `~`.
Check WSL disk from Windows if unsure: WSL's disk grows inside
`%LOCALAPPDATA%\Packages\...\ext4.vhdx` on `C:`.

## 8. When training finishes

The best model is copied to `~/checkpoints/shona_xtts/shona_xtts_model.pth`.
Copy it to LaCie immediately:

```bash
cp ~/checkpoints/shona_xtts/shona_xtts_model.pth /mnt/d/WaxalNLP/
```

## If Kali fights you

CUDA-in-WSL is distro-agnostic, but the ML stack is best tested on Ubuntu.
If `uv sync` or system packages misbehave on Kali, don't debug for hours:

```powershell
wsl --install -d Ubuntu
```

and repeat from step 2 — everything else is identical.
