"""Generate notebooks/00_setup_smoke.ipynb (Phase 0 smoke test).

Thin Colab driver: clone repo, install deps, run tests, then a tiny SDXL
pilot (generate -> detector-score -> capture activations). Includes an
OPTIONAL cell to re-score previously generated images from Google Drive
with the detector oracle (behavioral reuse; no regeneration needed).

Run: python notebooks/_build_00.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Phase 0 — Setup & Smoke Test\n"
    "Thin driver: clones the repo, installs deps, runs the unit tests, then a "
    "tiny SDXL pilot (generate -> detector-score -> capture activations). "
    "All logic lives in `src/`; this notebook only drives it.\n\n"
    "**Runtime:** GPU (T4 is enough)."))

cells.append(nbf.v4.new_code_cell(
    "# 1. Clone repo + install deps (skip clone if already inside the repo)\n"
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest groundingdino-py"))

cells.append(nbf.v4.new_code_cell(
    "# 2. Run the off-GPU unit tests (should all pass)\n"
    "!pytest -q"))

cells.append(nbf.v4.new_code_cell(
    "# 3. Imports\n"
    "import sys; sys.path.insert(0, '.')\n"
    "from src.prompts import generate_grid\n"
    "from src.pipeline import (load_sdxl, generate,\n"
    "                          catalog_attention_sites, ActivationCapture)\n"
    "from src.detector import Detector, count_objects\n"
    "from src.config import load_config\n"
    "import pandas as pd, os"))

cells.append(nbf.v4.new_code_cell(
    "# 4. Build the tiny pilot grid from the default config\n"
    "cfg = load_config('configs/default.yaml')\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\n"
    "print(len(grid), 'prompts in pilot')\n"
    "grid[:3]"))

cells.append(nbf.v4.new_code_cell(
    "# 5. Load SDXL and list the attention sites (the candidate probe catalog)\n"
    "pipe = load_sdxl()\n"
    "sites = catalog_attention_sites(pipe.unet)\n"
    "print(len(sites), 'attention sites')\n"
    "print(sites[:10])\n"
    "# ---> paste this full list into docs/ARCHITECTURE.md (site catalog)\n"
    "with open('results_sites.txt', 'w') as f:\n"
    "    f.write('\\n'.join(sites))"))

cells.append(nbf.v4.new_code_cell(
    "# 6. Generate the pilot, score each image with the detector oracle\n"
    "det = Detector()\n"
    "rows = []\n"
    "for p in grid:\n"
    "    img = generate(pipe, p.text, p.seed, cfg.num_inference_steps)\n"
    "    n = count_objects(det, img, p.obj, cfg.score_threshold)\n"
    "    rows.append({'text': p.text, 'obj': p.obj, 'count': p.count,\n"
    "                 'seed': p.seed, 'realized_count': n})\n"
    "df = pd.DataFrame(rows)\n"
    "os.makedirs('results', exist_ok=True)\n"
    "df.to_csv('results/smoke_summary.csv', index=False)\n"
    "df"))

cells.append(nbf.v4.new_code_cell(
    "# 7. Confirm activation hooks fire during generation (capture 3 sites, 2 steps)\n"
    "with ActivationCapture(pipe.unet, sites[:3]) as cap:\n"
    "    _ = generate(pipe, grid[0].text, grid[0].seed, num_inference_steps=2)\n"
    "print({k: tuple(v.shape) for k, v in cap.acts.items()})"))

cells.append(nbf.v4.new_markdown_cell(
    "## (Optional) Re-score previously generated images from Google Drive\n"
    "Re-scores your prior single-object images with the **detector oracle** "
    "(vs the old VLM scorer) — no regeneration needed. Set `USE_DRIVE = True` "
    "to run. Expects images at "
    "`MyDrive/ColabNotebooks/T2I-Count-Anomaly/generated-images-single-count/` "
    "and a manifest CSV (with an object/label column) in the same base folder."))

cells.append(nbf.v4.new_code_cell(
    "USE_DRIVE = False  # flip to True to re-score prior Drive images\n"
    "\n"
    "if USE_DRIVE:\n"
    "    from google.colab import drive\n"
    "    drive.mount('/content/drive')\n"
    "    from pathlib import Path\n"
    "    from PIL import Image\n"
    "    import glob\n"
    "    base = Path('/content/drive/MyDrive/ColabNotebooks/T2I-Count-Anomaly')\n"
    "    img_dir = base / 'generated-images-single-count'\n"
    "    # find a manifest CSV that maps image index -> object label\n"
    "    manifest = None\n"
    "    for cand in list(base.glob('*.csv')) + list(img_dir.glob('*.csv')):\n"
    "        try:\n"
    "            m = pd.read_csv(cand)\n"
    "        except Exception:\n"
    "            continue\n"
    "        if any(c in m.columns for c in ['obj', 'object', 'prompt']):\n"
    "            manifest = m; print('using manifest', cand.name); break\n"
    "    if manifest is None:\n"
    "        print('No manifest with an obj/object/prompt column found under', base)\n"
    "        print('Available CSVs:', [p.name for p in base.glob(\"*.csv\")])\n"
    "    else:\n"
    "        det = det if 'det' in dir() else Detector()\n"
    "        def label_for(row):\n"
    "            for c in ['obj', 'object']:\n"
    "                if c in row and isinstance(row[c], str):\n"
    "                    return row[c]\n"
    "            # fall back: last word of the prompt, singularized crudely\n"
    "            w = str(row.get('prompt', '')).split()[-1]\n"
    "            return w[:-1] if w.endswith('s') else w\n"
    "        out = []\n"
    "        for i, row in manifest.iterrows():\n"
    "            p = img_dir / f'{i:05d}.png'\n"
    "            if not p.exists():\n"
    "                continue\n"
    "            obj = label_for(row)\n"
    "            n = count_objects(det, Image.open(p).convert('RGB'), obj,\n"
    "                              cfg.score_threshold)\n"
    "            rec = {'idx': i, 'obj': obj, 'detector_count': n}\n"
    "            for c in ['count', 'prompt', 'seed']:\n"
    "                if c in row: rec[c] = row[c]\n"
    "            out.append(rec)\n"
    "        rescored = pd.DataFrame(out)\n"
    "        rescored.to_csv('results/drive_rescored.csv', index=False)\n"
    "        print('re-scored', len(rescored), 'prior images with the detector')\n"
    "        rescored.head()"))

nb["cells"] = cells
nb["metadata"] = {"accelerator": "GPU",
                  "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "00_setup_smoke.ipynb")
nbf.write(nb, out)
print("wrote", out)
