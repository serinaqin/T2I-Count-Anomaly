"""Generate notebooks/sd15_causal.ipynb (second-model generalization).

Does the early up-block self-attention causal locus replicate in SD 1.5? SD1.5
has attention in different up-blocks than SDXL, so we SWEEP each up-block's
self-attention (early steps) and measure the donor-directed count shift with
bootstrap CIs to re-localize and confirm. Thin Colab driver; reuses the same
patch machinery on a different U-Net.

Run: python notebooks/_build_sd15_causal.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Generalization to a second U-Net: SD 1.5\n"
    "Our SDXL result: the object count is causally set in **early up-block "
    "self-attention**. Does it replicate in **SD 1.5** (same U-Net family, "
    "different scale/blocks)? SD1.5's attention lives in different up-blocks, so "
    "we **sweep each up-block's self-attention** (early steps 0-5) with the same "
    "donor-patching test and re-localize. Effect with bootstrap CIs.\n\n"
    "**Runtime:** GPU (SD1.5 is light; ~20 min)."))

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
    "from src.pipeline import (load_sd15, generate, catalog_attention_sites,\n"
    "                          select_probe_sites, generate_and_capture,\n"
    "                          raw_reducer, generate_with_patch)\n"
    "from src.detector import Detector\n"
    "from src.scoring import count_from_detections\n"
    "from src.analysis import bootstrap_ci, sign_flip_pvalue\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/sd15_causal.yaml')\n"
    "raw = yaml.safe_load(open('configs/sd15_causal.yaml'))\n"
    "psteps = raw['patch_steps']; pairs = raw['pairs']; attn = raw['patch_attn']\n"
    "pipe = load_sd15(model_id=raw['model_id']); det = Detector()\n"
    "def cnt(img, obj):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "# up-block self-attention sites present in SD1.5, grouped by up-block\n"
    "up_sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "            if 'up_blocks' in s and s.endswith(attn)]\n"
    "blocks = sorted(set(s.split('.transformer_blocks')[0].rsplit('.attentions', 1)[0]\n"
    "                    for s in up_sites))\n"
    "block_sites = {b: [s for s in up_sites if s.startswith(b)] for b in blocks}\n"
    "print('SD1.5 up-block self-attn groups:')\n"
    "for b in blocks: print(' ', b, '->', len(block_sites[b]), 'sites')"))

c.append(nbf.v4.new_code_cell(
    "# Sweep: patch each up-block's self-attn (early steps) with a donor, both directions.\n"
    "rows = []\n"
    "for bi, b in enumerate(blocks):\n"
    "    sites = block_sites[b]\n"
    "    for obj in cfg.objects:\n"
    "        for src, dnr in pairs:\n"
    "            direction = 'up' if src < dnr else 'down'\n"
    "            sp, dp = build_prompt(src, obj), build_prompt(dnr, obj)\n"
    "            for seed in cfg.seeds:\n"
    "                _, snaps = generate_and_capture(pipe, dp, seed, sites, psteps,\n"
    "                                                cfg.num_inference_steps, reducer=raw_reducer)\n"
    "                cb = cnt(generate(pipe, sp, seed, cfg.num_inference_steps), obj)\n"
    "                cp = cnt(generate_with_patch(pipe, sp, seed, snaps, cfg.num_inference_steps), obj)\n"
    "                sd = (cp - cb) if direction == 'up' else (cb - cp)\n"
    "                rows.append({'block': b, 'obj': obj, 'direction': direction,\n"
    "                             'seed': seed, 'donor_directed': sd})\n"
    "    print(f'block {b} done ({bi+1}/{len(blocks)})')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/sd15_causal.csv', index=False)\n"
    "df.head()"))

c.append(nbf.v4.new_code_cell(
    "# Per-up-block donor-directed shift with bootstrap CI + permutation p.\n"
    "def summ(x):\n"
    "    x = np.asarray(x, float); lo, hi = bootstrap_ci(x)\n"
    "    return pd.Series({'n': len(x), 'mean': x.mean(), 'ci_lo': lo, 'ci_hi': hi,\n"
    "                      'p_perm': sign_flip_pvalue(x)})\n"
    "per_block = df.groupby('block')['donor_directed'].apply(summ).unstack().round(3)\n"
    "print(per_block)\n"
    "per_block.to_csv('results/sd15_causal_summary.csv')\n"
    "best = per_block['mean'].idxmax()\n"
    "print('\\nstrongest causal up-block in SD1.5:', best)"))

c.append(nbf.v4.new_code_cell(
    "# Forest plot across SD1.5 up-blocks.\n"
    "labels = list(per_block.index)\n"
    "means = per_block['mean'].values; los = per_block['ci_lo'].values; his = per_block['ci_hi'].values\n"
    "fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 1))\n"
    "for yi, (m, lo, hi) in enumerate(zip(means, los, his)):\n"
    "    ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='o', capsize=4, color='C0')\n"
    "ax.axvline(0, color='r', ls=':')\n"
    "ax.set_yticks(range(len(labels)))\n"
    "ax.set_yticklabels([l.replace('up_blocks', 'up') for l in labels])\n"
    "ax.set_xlabel('donor-directed count shift (toward donor)')\n"
    "ax.set_title('SD1.5: causal effect of patching each up-block self-attn (95% CI)')\n"
    "plt.tight_layout(); plt.savefig('results/sd15_causal_forest.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **At least one up-block has a clearly positive donor-directed shift (CI "
    "excludes 0, small p)** = the early up-block self-attention causal locus "
    "**replicates in SD 1.5** -> the mechanism generalizes beyond SDXL (a second "
    "U-Net), the key result for external validity. Note *which* up-block: SD1.5's "
    "attention blocks differ from SDXL, so a homologous (coarse-resolution) block "
    "is expected rather than literally 'up_blocks.0'.\n"
    "- **No block moves the count** = the SDXL locus does NOT transfer to SD1.5 -> "
    "report honestly as model-specific; still a finding, but tempers the "
    "generality claim.\n"
    "- Exclude degenerate collage generations if any block shows huge variance "
    "(the detector caveat from the validation experiment)."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "sd15_causal.ipynb")
nbf.write(nb, out)
print("wrote", out)
