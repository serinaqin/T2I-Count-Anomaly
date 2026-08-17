"""Generate notebooks/05d_gray_amp_sweep.ipynb (Phase 5d).

Phase 5c: gaussian_gray gave coherent, untinted cats and nailed some counts
(asked 6 -> 6 clean white cats) but under-seeded on average (weaker than the
tinted bump). Here we sweep the gray amplitude to find the sweet spot that
forces the count reliably WHILE staying coherent. Thin Colab driver.

Run: python notebooks/_build_05d.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 5d - gaussian_gray amplitude sweep\n"
    "gaussian_gray is coherent + untinted but a weaker count-forcer than the "
    "tinted bump (the color DC was doing some object-seeding). Amplitude is a "
    "free knob: sweep `gray_amp` to find where the untinted bump matches plain "
    "gaussian's count control while keeping images clean.\n\n**Runtime:** GPU (~15 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml, torch\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import build_prompt\n"
    "from src.pipeline import load_sdxl\n"
    "from src.noise_layout import count_aware_latent\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase5d.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase5d.yaml'))\n"
    "amps = raw['gray_amps']; obj = cfg.objects[0]\n"
    "pk = dict(omega=raw['omega'], alpha=raw['alpha'], fill=raw['box_fill'])\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "SS = pipe.unet.config.sample_size\n"
    "def cnt(img):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "def base_latent(seed):\n"
    "    g = torch.Generator(device='cpu').manual_seed(seed)\n"
    "    z = torch.randn((1, pipe.unet.config.in_channels, SS, SS), generator=g)\n"
    "    return z.to(pipe.device, pipe.dtype)\n"
    "def gen_latent(prompt, lat):\n"
    "    return pipe(prompt, latents=lat, num_inference_steps=cfg.num_inference_steps).images[0]"))

c.append(nbf.v4.new_code_cell(
    "# Neutral gray latent direction (amp=1), scaled per-amp in the loop.\n"
    "@torch.no_grad()\n"
    "def gray_direction(probe=2.0):\n"
    "    vae = pipe.vae; orig = next(vae.parameters()).dtype\n"
    "    vae.to(torch.float32); sf = vae.config.scaling_factor\n"
    "    try:\n"
    "        z0 = torch.zeros(1, 4, 16, 16, device=pipe.device, dtype=torch.float32)\n"
    "        rgb0 = vae.decode(z0 / sf).sample.mean(dim=(0, 2, 3))\n"
    "        cols = []\n"
    "        for ch in range(4):\n"
    "            z = z0.clone(); z[:, ch] += probe\n"
    "            cols.append((vae.decode(z / sf).sample.mean(dim=(0, 2, 3)) - rgb0) / probe)\n"
    "        M = torch.stack(cols).cpu()\n"
    "    finally:\n"
    "        vae.to(orig)\n"
    "    if not torch.isfinite(M).all():\n"
    "        return torch.ones(4)\n"
    "    a = torch.linalg.pinv(M.t()) @ torch.ones(3)\n"
    "    return a / a.norm() * (4 ** 0.5)\n"
    "gray0 = gray_direction()\n"
    "print('gray direction (amp=1):', [round(x, 3) for x in gray0.tolist()])"))

c.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for N in cfg.counts:\n"
    "    prompt = build_prompt(N, obj)\n"
    "    for seed in cfg.seeds:\n"
    "        base = base_latent(seed)\n"
    "        rows.append({'N': N, 'seed': seed, 'amp': 0.0,\n"
    "                     'rendered': cnt(gen_latent(prompt, base.clone()))})\n"
    "        for amp in amps:\n"
    "            lat = count_aware_latent(base, N, scheme='gaussian_gray',\n"
    "                                     channel_weights=gray0 * amp, **pk)\n"
    "            rows.append({'N': N, 'seed': seed, 'amp': amp,\n"
    "                         'rendered': cnt(gen_latent(prompt, lat))})\n"
    "    print(f'N={N} done')\n"
    "df = pd.DataFrame(rows)\n"
    "df['correct'] = df['rendered'] == df['N']\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase5d_counts.csv', index=False)\n"
    "df.assign(e=(df['N'] - df['rendered']).abs()).groupby('amp').agg(\n"
    "    exact_acc=('correct', 'mean'), mae=('e', 'mean'))"))

c.append(nbf.v4.new_code_cell(
    "# Accuracy & MAE vs gray amplitude (amp=0 is baseline).\n"
    "g = df.assign(e=(df['N'] - df['rendered']).abs()).groupby('amp')\n"
    "acc, mae = g['correct'].mean(), g['e'].mean()\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
    "axes[0].plot(acc.index, acc.values, 'o-'); axes[0].set_xlabel('gray_amp (0=baseline)')\n"
    "axes[0].set_ylabel('exact-count accuracy'); axes[0].set_ylim(0, 1)\n"
    "axes[0].set_title('Accuracy vs amplitude'); axes[0].axhline(0.44, color='r', ls=':', label='plain gaussian 0.44')\n"
    "axes[0].legend(fontsize=8)\n"
    "axes[1].plot(mae.index, mae.values, 'o-'); axes[1].set_xlabel('gray_amp (0=baseline)')\n"
    "axes[1].set_ylabel('MAE'); axes[1].set_title('MAE vs amplitude'); axes[1].axhline(1.68, color='r', ls=':', label='plain gaussian 1.68')\n"
    "axes[1].legend(fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5d_sweep.png', dpi=100, bbox_inches='tight'); plt.show()\n"
    "print('acc:\\n', acc, '\\nMAE:\\n', mae)"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline vs gray at each amplitude (asked N0). Coherent + count?\n"
    "N0, seed0 = 6 if 6 in cfg.counts else cfg.counts[0], cfg.seeds[0]\n"
    "prompt0 = build_prompt(N0, obj); base = base_latent(seed0)\n"
    "cells = [('baseline', base.clone())] + \\\n"
    "        [(f'amp={a}', count_aware_latent(base, N0, scheme='gaussian_gray',\n"
    "                                         channel_weights=gray0 * a, **pk)) for a in amps]\n"
    "fig, axes = plt.subplots(1, len(cells), figsize=(3.0 * len(cells), 3.3))\n"
    "for ax, (name, lat) in zip(axes, cells):\n"
    "    im = gen_latent(prompt0, lat)\n"
    "    ax.imshow(im); ax.axis('off'); ax.set_title(f'{name} | asked {N0}->{cnt(im)}', fontsize=8)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5d_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Accuracy rises with gray_amp toward (or past) plain gaussian's 0.44, "
    "and the eyeball stays coherent/untinted** = the clean capstone: a "
    "training-free, quality-preserving, high-count mitigation. Pick the amp at "
    "the accuracy peak that still looks natural.\n"
    "- **Accuracy plateaus below gaussian, or the eyeball degrades at high amp** "
    "= there's a real coherence/accuracy tradeoff; report the best gray_amp as "
    "the quality-preserving option and plain gaussian as the max-accuracy option.\n"
    "- Watch for **too-high amp** re-introducing artifacts (over-strong injection)."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "05d_gray_amp_sweep.ipynb")
nbf.write(nb, out)
print("wrote", out)
