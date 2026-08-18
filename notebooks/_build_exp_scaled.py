"""Generate notebooks/exp_scaled_causal.ipynb (scaled causal replication + stats).

Same-category donor patching across multiple objects, both directions, many
seeds, with bootstrap CIs and a permutation (sign-flip) test on the
donor-directed count shift. Fixes the pilot-scale / no-statistics critique.
Thin Colab driver.

Run: python notebooks/_build_exp_scaled.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Scaled causal replication + statistics\n"
    "Same-category donor patching (up0 early self-attn) across several objects, "
    "both directions (2->inject5 and 5->inject2), many seeds. We report the "
    "**donor-directed count shift** with **bootstrap 95% CIs** and a "
    "**permutation (sign-flip) p-value** per object and pooled. This puts error "
    "bars and significance on the core causal claim.\n\n"
    "**Runtime:** GPU (~45-60 min at default settings; reduce seeds/objects to pilot)."))

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
    "from src.analysis import bootstrap_ci, sign_flip_pvalue\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/exp_scaled.yaml')\n"
    "raw = yaml.safe_load(open('configs/exp_scaled.yaml'))\n"
    "psteps = raw['patch_steps']; pairs = raw['pairs']\n"
    "block, attn = raw['patch_block'], raw['patch_attn']\n"
    "pipe = load_sdxl(); det = Detector()\n"
    "sites = [s for s in select_probe_sites(catalog_attention_sites(pipe.unet))\n"
    "         if block in s and s.endswith(attn)]\n"
    "def cnt(img, obj):\n"
    "    return count_from_detections(det.detect(img, [obj]), obj, cfg.score_threshold)\n"
    "print(len(cfg.objects), 'objects x', len(pairs), 'directions x', len(cfg.seeds),\n"
    "      'seeds =', len(cfg.objects) * len(pairs) * len(cfg.seeds), 'patched runs')"))

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
    "            cp = cnt(generate_with_patch(pipe, sp, seed, snaps, cfg.num_inference_steps), obj)\n"
    "            sd = (cp - cb) if direction == 'up' else (cb - cp)   # donor-directed\n"
    "            rows.append({'obj': obj, 'direction': direction, 'seed': seed,\n"
    "                         'c_base': cb, 'c_patch': cp, 'donor_directed': sd})\n"
    "        print(obj, direction, 'done')\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/exp_scaled.csv', index=False)\n"
    "df.head()"))

c.append(nbf.v4.new_code_cell(
    "# Per-object and pooled donor-directed shift with bootstrap CI + permutation p.\n"
    "def summarize(x):\n"
    "    x = np.asarray(x, float)\n"
    "    lo, hi = bootstrap_ci(x)\n"
    "    return pd.Series({'n': len(x), 'mean': x.mean(), 'ci_lo': lo, 'ci_hi': hi,\n"
    "                      'p_perm': sign_flip_pvalue(x), 'frac_pos': (x > 0).mean()})\n"
    "per_obj = df.groupby('obj')['donor_directed'].apply(summarize).unstack().round(3)\n"
    "pooled = summarize(df['donor_directed']).round(3)\n"
    "print('Per-object donor-directed count shift:\\n', per_obj)\n"
    "print('\\nPOOLED:\\n', pooled)\n"
    "per_obj.to_csv('results/exp_scaled_summary.csv')"))

c.append(nbf.v4.new_code_cell(
    "# Forest plot: per-object mean donor-directed shift with 95% CI, + pooled.\n"
    "labels = list(per_obj.index) + ['POOLED']\n"
    "means = list(per_obj['mean']) + [pooled['mean']]\n"
    "los = list(per_obj['ci_lo']) + [pooled['ci_lo']]\n"
    "his = list(per_obj['ci_hi']) + [pooled['ci_hi']]\n"
    "fig, ax = plt.subplots(figsize=(7, 0.6 * len(labels) + 1))\n"
    "for yi, (m, lo, hi, lab) in enumerate(zip(means, los, his, labels)):\n"
    "    ax.errorbar(m, yi, xerr=[[m - lo], [hi - m]], fmt='o', capsize=4,\n"
    "                color='k' if lab == 'POOLED' else 'C0')\n"
    "ax.axvline(0, color='r', ls=':')\n"
    "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
    "ax.set_xlabel('donor-directed count shift (patched - baseline, toward donor)')\n"
    "ax.set_title('Causal effect of early up0 self-attn patch (95% bootstrap CI)')\n"
    "plt.tight_layout(); plt.savefig('results/exp_scaled_forest.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Pooled mean donor-directed shift > 0 with a CI excluding 0 and small "
    "permutation p** = the early up0 self-attention patch causally moves the "
    "count toward the donor, robustly across objects. This is the statistically "
    "grounded version of the core claim (replaces the single un-repped number).\n"
    "- **Per-object CIs** show whether the effect is general or driven by one or "
    "two categories (a reviewer will ask). Objects whose CI crosses 0 are "
    "individually inconclusive at this n.\n"
    "- Report mean [CI], p, and n; frame as *causal contribution*, not full "
    "control (the shift is partial). Combined with the cross-category result "
    "(category does not transfer), the representation is a **seed- and "
    "category-conditioned object-layout code**, not an abstract count."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_scaled_causal.ipynb")
nbf.write(nb, out)
print("wrote", out)
