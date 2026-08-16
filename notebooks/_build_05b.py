"""Generate notebooks/05b_noise_injection_hi.ipynb (Phase 5b).

Phase 5's additive Gaussian bump imposed the count but tinted color / wrecked
quality, netting no accuracy gain at easy counts (baseline ceiling). Here:
(1) artifact-free schemes (noise_boost, gaussian_noise) that add layout WITHOUT
a color DC bias, and (2) the HIGH-count regime (4-8) where baseline collapses.
Thin Colab driver.

Run: python notebooks/_build_05b.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 5b - Artifact-free injection, high-count regime\n"
    "Phase 5: the additive Gaussian bump forced the count but tinted the image "
    "(color DC bias) and lost on the easy 1-5 set (baseline ceiling). Here we "
    "test **artifact-free** schemes - `noise_boost` (amplify in-box variance) and "
    "`gaussian_noise` (add Gaussian-enveloped fresh noise, zero-mean) - in the "
    "**high-count regime (4-8)** where baseline collapses.\n\n**Runtime:** GPU."))

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
    "cfg = load_config('configs/phase5b.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase5b.yaml'))\n"
    "schemes = raw['schemes']; obj = cfg.objects[0]\n"
    "pk = dict(gamma=raw['gamma'], omega=raw['omega'], alpha=raw['alpha'],\n"
    "          beta=raw['beta'], fill=raw['box_fill'])\n"
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
    "rows = []\n"
    "for N in cfg.counts:\n"
    "    prompt = build_prompt(N, obj)\n"
    "    for seed in cfg.seeds:\n"
    "        base = base_latent(seed)\n"
    "        rows.append({'N': N, 'seed': seed, 'scheme': 'baseline',\n"
    "                     'rendered': cnt(gen_latent(prompt, base.clone()))})\n"
    "        for sch in schemes:\n"
    "            lat = count_aware_latent(base, N, scheme=sch, noise_seed=seed, **pk)\n"
    "            rows.append({'N': N, 'seed': seed, 'scheme': sch,\n"
    "                         'rendered': cnt(gen_latent(prompt, lat))})\n"
    "    print(f'N={N} done')\n"
    "df = pd.DataFrame(rows)\n"
    "df['correct'] = df['rendered'] == df['N']\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase5b_counts.csv', index=False)\n"
    "df.groupby('scheme').agg(exact_acc=('correct', 'mean'),\n"
    "                         mae=('rendered', lambda s: (df.loc[s.index, 'N'] - s).abs().mean()))"))

c.append(nbf.v4.new_code_cell(
    "order = ['baseline'] + schemes\n"
    "acc = df.groupby('scheme')['correct'].mean().reindex(order)\n"
    "mae = df.assign(err=(df['N'] - df['rendered']).abs()).groupby('scheme')['err'].mean().reindex(order)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
    "axes[0].bar(acc.index, acc.values, color=['gray', 'C0', 'C1', 'C2'])\n"
    "axes[0].set_ylabel('exact-count accuracy'); axes[0].set_ylim(0, 1)\n"
    "axes[0].set_title('Exact accuracy (high-count regime)')\n"
    "axes[1].bar(mae.index, mae.values, color=['gray', 'C0', 'C1', 'C2'])\n"
    "axes[1].set_ylabel('MAE of count'); axes[1].set_title('Count MAE (lower better)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5b_accuracy.png', dpi=100, bbox_inches='tight'); plt.show()\n"
    "print('exact acc:\\n', acc, '\\nMAE:\\n', mae)"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline vs each artifact-free scheme at a high count.\n"
    "N0, seed0 = 6 if 6 in cfg.counts else cfg.counts[-1], cfg.seeds[0]\n"
    "prompt0 = build_prompt(N0, obj); base = base_latent(seed0)\n"
    "cells = [('baseline', base.clone())] + \\\n"
    "        [(s, count_aware_latent(base, N0, scheme=s, noise_seed=seed0, **pk)) for s in schemes]\n"
    "fig, axes = plt.subplots(1, len(cells), figsize=(3.2 * len(cells), 3.4))\n"
    "for ax, (name, lat) in zip(axes, cells):\n"
    "    im = gen_latent(prompt0, lat)\n"
    "    ax.imshow(im); ax.axis('off')\n"
    "    ax.set_title(f'{name} | asked {N0} -> {cnt(im)}', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5b_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **An artifact-free scheme (noise_boost / gaussian_noise) beats baseline on "
    "exact accuracy or MAE at high counts, in COHERENT images** = a working "
    "training-free mitigation where it matters -> the arc closes with a real fix.\n"
    "- **Eyeball is decisive:** do noise_boost / gaussian_noise images show the "
    "requested number of *photorealistic* cats (no green icons / box artifacts "
    "like plain gaussian)? Coherent + correct count = success.\n"
    "- **Still no gain / still degraded** = training-free noise injection is "
    "insufficient on SDXL even at high counts; a clean fix needs the fine-tuning "
    "the paper pairs with it. Honest negative; the causal finding stands. Then "
    "consolidate for the writeup."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "05b_noise_injection_hi.ipynb")
nbf.write(nb, out)
print("wrote", out)
