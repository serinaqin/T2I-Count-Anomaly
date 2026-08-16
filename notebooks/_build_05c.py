"""Generate notebooks/05c_noise_injection_gray.ipynb (Phase 5c).

Phase 5b: gaussian injection roughly doubled high-count accuracy but tinted
images green (additive DC across all 4 latent channels). Here `gaussian_gray`
projects the bump onto the VAE's NEUTRAL (gray) latent direction - seeding
objects without the color cast. Thin Colab driver.

Run: python notebooks/_build_05c.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 5c - Neutral-gray noise injection\n"
    "gaussian worked on the count (0.24->0.44, MAE 6.88->1.68 at counts 4-8) but "
    "tinted images green - the additive bump shifts all 4 latent channels, which "
    "decodes to a color. `gaussian_gray` shapes the SAME bump along the VAE's "
    "**neutral (gray) latent direction** (measured from the VAE here), so it "
    "seeds objects without the tint. Goal: coherent objects AND correct count.\n\n"
    "**Runtime:** GPU (~15 min)."))

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
    "cfg = load_config('configs/phase5c.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase5c.yaml'))\n"
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
    "# Measure the VAE's per-channel latent->RGB response, solve for the latent\n"
    "# direction that decodes to neutral gray (equal R=G=B).\n"
    "@torch.no_grad()\n"
    "def gray_direction(probe=2.0):\n"
    "    vae = pipe.vae; sf = vae.config.scaling_factor\n"
    "    z0 = torch.zeros(1, 4, 16, 16, device=pipe.device, dtype=pipe.dtype)\n"
    "    rgb0 = vae.decode(z0 / sf).sample.float().mean(dim=(0, 2, 3))\n"
    "    cols = []\n"
    "    for ch in range(4):\n"
    "        z = z0.clone(); z[:, ch] += probe\n"
    "        rgb = vae.decode(z / sf).sample.float().mean(dim=(0, 2, 3))\n"
    "        cols.append((rgb - rgb0) / probe)\n"
    "    M = torch.stack(cols).cpu()                       # (4 channels, 3 RGB)\n"
    "    a = torch.linalg.pinv(M.t()) @ torch.ones(3)      # min-norm a @ M = gray\n"
    "    a = a / a.norm() * (4 ** 0.5)                     # match L2 of ones(4)\n"
    "    return a * raw['gray_amp']\n"
    "gray = gray_direction()\n"
    "print('gray latent direction (per channel):', [round(x, 3) for x in gray.tolist()])"))

c.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for N in cfg.counts:\n"
    "    prompt = build_prompt(N, obj)\n"
    "    for seed in cfg.seeds:\n"
    "        base = base_latent(seed)\n"
    "        rows.append({'N': N, 'seed': seed, 'scheme': 'baseline',\n"
    "                     'rendered': cnt(gen_latent(prompt, base.clone()))})\n"
    "        for sch in schemes:\n"
    "            kw = dict(pk)\n"
    "            if sch == 'gaussian_gray':\n"
    "                kw['channel_weights'] = gray\n"
    "            lat = count_aware_latent(base, N, scheme=sch, noise_seed=seed, **kw)\n"
    "            rows.append({'N': N, 'seed': seed, 'scheme': sch,\n"
    "                         'rendered': cnt(gen_latent(prompt, lat))})\n"
    "    print(f'N={N} done')\n"
    "df = pd.DataFrame(rows)\n"
    "df['correct'] = df['rendered'] == df['N']\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase5c_counts.csv', index=False)\n"
    "order = ['baseline'] + schemes\n"
    "df.assign(err=(df['N'] - df['rendered']).abs()).groupby('scheme').agg(\n"
    "    exact_acc=('correct', 'mean'), mae=('err', 'mean')).reindex(order)"))

c.append(nbf.v4.new_code_cell(
    "order = ['baseline'] + schemes\n"
    "acc = df.groupby('scheme')['correct'].mean().reindex(order)\n"
    "mae = df.assign(e=(df['N'] - df['rendered']).abs()).groupby('scheme')['e'].mean().reindex(order)\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
    "axes[0].bar(acc.index, acc.values, color=['gray', 'C1', 'C2'])\n"
    "axes[0].set_ylim(0, 1); axes[0].set_ylabel('exact-count accuracy'); axes[0].set_title('Exact accuracy (4-8)')\n"
    "axes[1].bar(mae.index, mae.values, color=['gray', 'C1', 'C2'])\n"
    "axes[1].set_ylabel('MAE'); axes[1].set_title('Count MAE (lower better)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5c_accuracy.png', dpi=100, bbox_inches='tight'); plt.show()\n"
    "print('exact acc:\\n', acc, '\\nMAE:\\n', mae)"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: baseline | gaussian | gaussian_gray at a high count (is gray coherent?).\n"
    "N0, seed0 = 6 if 6 in cfg.counts else cfg.counts[-1], cfg.seeds[0]\n"
    "prompt0 = build_prompt(N0, obj); base = base_latent(seed0)\n"
    "cells = [('baseline', base.clone()),\n"
    "         ('gaussian', count_aware_latent(base, N0, scheme='gaussian', noise_seed=seed0, **pk)),\n"
    "         ('gaussian_gray', count_aware_latent(base, N0, scheme='gaussian_gray',\n"
    "                                              noise_seed=seed0, channel_weights=gray, **pk))]\n"
    "fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))\n"
    "for ax, (name, lat) in zip(axes, cells):\n"
    "    im = gen_latent(prompt0, lat)\n"
    "    ax.imshow(im); ax.axis('off'); ax.set_title(f'{name} | asked {N0} -> {cnt(im)}', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase5c_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **gaussian_gray matches gaussian's count accuracy/MAE but in COHERENT, "
    "natural (untinted) images** = the clean capstone mitigation: training-free, "
    "quality-preserving, high-count. The eyeball is the judge - gray should show "
    "~N real-ish cats, not green icons.\n"
    "- **gaussian_gray loses the count control** = the color DC was doing the "
    "object-seeding, not just tinting -> the two are inseparable in this simple "
    "scheme; report gaussian as the mitigation with an honest quality caveat.\n"
    "- **Detector note:** if gray images are more natural, the detector should "
    "also count them more reliably than the green icons - so gray may score "
    "*higher* than gaussian even at equal true count."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "05c_noise_injection_gray.ipynb")
nbf.write(nb, out)
print("wrote", out)
