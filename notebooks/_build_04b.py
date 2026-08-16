"""Generate notebooks/04b_template_transfer.ipynb (Phase 4b).

Phase 4 showed the count is spatial (linear steering failed) but donor patching
works. Here we test whether a SINGLE reference donor per count (a spatial
template) transfers ACROSS seeds: inject template[N] into different recipient
seeds and see if the output count follows N. If yes -> a donor-free per-count
template bank = a real spatial mitigation. Thin Colab driver.

Run: python notebooks/_build_04b.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Phase 4b - Cross-seed template transfer\n"
    "Donor patching (full spatial activation) moves the count; linear steering "
    "does not (count is spatial). Here: build ONE reference donor per count N "
    "(early up0 attn1 activations that render N), then inject it into "
    "**different recipient seeds**. If the output count follows N regardless of "
    "recipient -> a per-count **template bank** is a donor-free spatial "
    "mitigation.\n\n**Runtime:** GPU (~15 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import build_prompt\n"
    "from src.pipeline import (load_sdxl, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, generate_with_patch)\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/phase4b.yaml')\n"
    "raw = yaml.safe_load(open('configs/phase4b.yaml'))\n"
    "psteps = raw['patch_steps']; block, attn = raw['patch_block'], raw['patch_attn']\n"
    "obj = cfg.objects[0]\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def cnt(img):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "print('sites:', sites)"))

c.append(nbf.v4.new_code_cell(
    "# Build a template bank: one reference donor per count N (search seeds).\n"
    "templates, tinfo, timg = {}, {}, {}\n"
    "for N in cfg.counts:\n"
    "    prompt = build_prompt(N, obj)\n"
    "    best = None\n"
    "    for seed in raw['template_search_seeds']:\n"
    "        img, snaps = generate_and_capture(pipe, prompt, seed, sites, psteps,\n"
    "                                          cfg.num_inference_steps, reducer=raw_reducer)\n"
    "        rc = cnt(img)\n"
    "        if best is None or abs(rc - N) < abs(best[1] - N):\n"
    "            best = (seed, rc, snaps, img)\n"
    "        if rc == N:\n"
    "            break\n"
    "    templates[N] = best[2]; tinfo[N] = (best[0], best[1]); timg[N] = best[3]\n"
    "    print(f'template N={N}: donor seed {best[0]} rendered {best[1]}')"))

c.append(nbf.v4.new_code_cell(
    "# Transfer test: fixed recipient prompt, held-out seeds; inject each template.\n"
    "rprompt = build_prompt(raw['recipient_prompt_count'], obj)\n"
    "rows = []\n"
    "for rseed in raw['recipient_seeds']:\n"
    "    base = cnt(generate(pipe, rprompt, rseed, cfg.num_inference_steps))\n"
    "    for N in cfg.counts:\n"
    "        out = cnt(generate_with_patch(pipe, rprompt, rseed, templates[N],\n"
    "                                      cfg.num_inference_steps))\n"
    "        rows.append({'rseed': rseed, 'template_N': N, 'baseline': base, 'out': out})\n"
    "    print('recipient seed', rseed, 'done')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/phase4b_transfer.csv', index=False)\n"
    "df.groupby('template_N')[['out', 'baseline']].mean()"))

c.append(nbf.v4.new_code_cell(
    "# Plot: output count vs injected template N (diagonal = template controls it).\n"
    "fig, ax = plt.subplots(figsize=(6, 5))\n"
    "for rseed, g in df.groupby('rseed'):\n"
    "    ax.plot(g['template_N'], g['out'], marker='o', alpha=0.4, label=f's{rseed}')\n"
    "m = df.groupby('template_N')['out'].mean()\n"
    "ax.plot(m.index, m.values, 'k-o', lw=2.5, label='mean')\n"
    "ax.plot(cfg.counts, cfg.counts, 'k--', label='y=x (perfect transfer)')\n"
    "ax.axhline(df['baseline'].mean(), color='r', ls=':', label='baseline (no patch)')\n"
    "ax.set_xlabel('injected template count N'); ax.set_ylabel('recipient output count')\n"
    "ax.set_title(f\"Template transfer into '{rprompt}' (held-out seeds)\")\n"
    "ax.legend(fontsize=7); plt.tight_layout()\n"
    "plt.savefig('results/phase4b_transfer.png', dpi=100, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_code_cell(
    "# Eyeball: one recipient seed, inject templates N=1,3,5 (do we see 1,3,5 cats?).\n"
    "rseed0 = raw['recipient_seeds'][0]\n"
    "showN = [n for n in [1, 3, 5] if n in cfg.counts]\n"
    "fig, axes = plt.subplots(1, len(showN) + 1, figsize=(3.2 * (len(showN) + 1), 3.4))\n"
    "b = generate(pipe, rprompt, rseed0, cfg.num_inference_steps)\n"
    "axes[0].imshow(b); axes[0].axis('off'); axes[0].set_title(f'baseline={cnt(b)}', fontsize=9)\n"
    "for j, N in enumerate(showN):\n"
    "    im = generate_with_patch(pipe, rprompt, rseed0, templates[N], cfg.num_inference_steps)\n"
    "    axes[j + 1].imshow(im); axes[j + 1].axis('off')\n"
    "    axes[j + 1].set_title(f'template N={N} -> {cnt(im)}', fontsize=9)\n"
    "plt.tight_layout()\n"
    "plt.savefig('results/phase4b_eyeball.png', dpi=90, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Output count rises with injected template N (mean tracks the diagonal), "
    "above the baseline line** = the template **transfers across seeds** -> a "
    "small per-count template bank is a **donor-free spatial mitigation**. The "
    "eyeball should show recipient images with ~1, 3, 5 cats as the template "
    "changes, despite the prompt saying 'two'.\n"
    "- **Output stays near baseline regardless of N (flat)** = templates are "
    "**seed-specific**; the count is tied to each recipient's own initial noise, "
    "not transferable -> the spatial layout is noise-bound, and the mitigation "
    "route is a **noise-level** intervention (count-aware spatial noise "
    "injection), now causally motivated by the early-window localization.\n"
    "- **Partial (rises but shallow, or only some N)** = the template carries "
    "*some* transferable count structure but competes with the recipient noise -> "
    "combine template + noise control."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "04b_template_transfer.ipynb")
nbf.write(nb, out)
print("wrote", out)
