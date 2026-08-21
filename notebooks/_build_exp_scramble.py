"""Generate notebooks/exp_scramble.ipynb (spatial-scramble control).

Tests whether the causal patch effect is SPATIAL: inject the donor's early up0
self-attention intact vs with its tokens spatially permuted (same channel
content, destroyed layout). If intact moves the count but scrambled does not,
the effect depends on spatial arrangement, not channel statistics. Thin Colab
driver reusing the patch machinery.

Run: python notebooks/_build_exp_scramble.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Spatial-scramble control\n"
    "Is the causal patch effect **spatial**? For each seed we inject the donor's "
    "early up0 self-attention two ways: **intact**, and **spatially scrambled** "
    "(tokens randomly permuted -> same per-token channel content, destroyed "
    "layout). If intact shifts the count but scrambled does not, the effect "
    "depends on the **spatial arrangement**, not channel statistics.\n\n"
    "**Runtime:** GPU (~15 min)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "else:\n"
    "    !git pull\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml\n"
    "import matplotlib.pyplot as plt\n"
    "from src.prompts import build_prompt\n"
    "from src.pipeline import (load_sdxl, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, generate_with_patch, spatial_scramble)\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.analysis import bootstrap_ci, sign_flip_pvalue\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/exp_scramble.yaml')\n"
    "raw = yaml.safe_load(open('configs/exp_scramble.yaml'))\n"
    "psteps = raw['patch_steps']; pairs = raw['pairs']\n"
    "block, attn = raw['patch_block'], raw['patch_attn']\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def cnt(img, obj):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "def scramble_map(snaps, seed):\n"
    "    return {st: {s: spatial_scramble(v, seed) for s, v in d.items()} for st, d in snaps.items()}"))

c.append(nbf.v4.new_code_cell(
    "rows = []\n"
    "for obj in cfg.objects:\n"
    "    for src, dnr in pairs:\n"
    "        direction = 'up' if src < dnr else 'down'\n"
    "        sp, dp = build_prompt(src, obj), build_prompt(dnr, obj)\n"
    "        for seed in cfg.seeds:\n"
    "            _, snaps = generate_and_capture(pipe, dp, seed, sites, psteps,\n"
    "                                            cfg.num_inference_steps, reducer=raw_reducer)\n"
    "            cb = cnt(generate(pipe, sp, seed, cfg.num_inference_steps), obj)\n"
    "            cp_intact = cnt(generate_with_patch(pipe, sp, seed, snaps, cfg.num_inference_steps), obj)\n"
    "            cp_scram = cnt(generate_with_patch(pipe, sp, seed, scramble_map(snaps, seed),\n"
    "                                               cfg.num_inference_steps), obj)\n"
    "            f = 1.0 if direction == 'up' else -1.0\n"
    "            rows.append({'obj': obj, 'direction': direction, 'seed': seed,\n"
    "                         'intact': f * (cp_intact - cb), 'scrambled': f * (cp_scram - cb)})\n"
    "        print(obj, direction, 'done')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/exp_scramble.csv', index=False)\n"
    "df.head()"))

c.append(nbf.v4.new_code_cell(
    "# Donor-directed shift: intact vs spatially-scrambled donor.\n"
    "for cond in ['intact', 'scrambled']:\n"
    "    x = df[cond].to_numpy(float); lo, hi = bootstrap_ci(x)\n"
    "    print(f'{cond:>10}: mean {x.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  p={sign_flip_pvalue(x):.3f}')\n"
    "fig, ax = plt.subplots(figsize=(5.5, 4))\n"
    "for i, cond in enumerate(['intact', 'scrambled']):\n"
    "    x = df[cond].to_numpy(float); lo, hi = bootstrap_ci(x)\n"
    "    ax.errorbar(x.mean(), i, xerr=[[x.mean() - lo], [hi - x.mean()]], fmt='o', capsize=4,\n"
    "                color='C0' if cond == 'intact' else 'C3')\n"
    "ax.axvline(0, color='r', ls=':'); ax.set_yticks([0, 1]); ax.set_yticklabels(['intact', 'scrambled'])\n"
    "ax.set_xlabel('donor-directed count shift'); ax.set_title('Does the patch need spatial arrangement?')\n"
    "plt.tight_layout(); plt.savefig('results/exp_scramble.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **intact shift clearly > 0, scrambled shift ~0** = the effect requires the "
    "donor's **spatial arrangement** -> it is a spatial layout intervention, not a "
    "channel-statistics / activation-magnitude one. Strong support for the "
    "spatial object-layout account (and consistent with why uniform steering "
    "failed).\n"
    "- **scrambled shift ~ intact** = the effect is carried by channel/statistical "
    "content, not layout -> temper the 'spatial' claim.\n"
    "- **both ~0** = weak effect at this n; lean on the scaled replication."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_scramble.ipynb")
nbf.write(nb, out)
print("wrote", out)
