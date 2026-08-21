"""Generate notebooks/exp_seminvariance.ipynb (semantic-invariance of numerals).

Text-encoder only (no image generation -> fast). Tests whether the requested
count is represented in a FORM-INVARIANT way: train a probe on word-form
prompts ('three cats') and test on digit-form ('3 cats'), and vice versa. High
cross-form accuracy => an abstract-ish numeral representation; low => merely
lexical (the reviewer's caution that 'decodable numeral' != 'comprehended
number'). Thin Colab driver.

Run: python notebooks/_build_exp_seminvariance.py
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []

c.append(nbf.v4.new_markdown_cell(
    "# Is the numeral representation form-invariant? (text encoder only)\n"
    "The requested count is trivially decodable from the text embedding - but is "
    "that *numerical comprehension* or just *lexical numeral identity*? We probe "
    "the count from the fused text embedding and test **cross-form transfer**: "
    "train on **word** form ('three cats'), test on **digit** form ('3 cats'), "
    "and vice versa. High cross-form accuracy = a form-invariant (more abstract) "
    "numeral representation; low = lexical only.\n\n"
    "**Runtime:** GPU but fast (no image generation)."))

c.append(nbf.v4.new_code_cell(
    "import os\n"
    "if not os.path.exists('src'):\n"
    "    !git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\n"
    "    %cd T2I-Count-Anomaly\n"
    "else:\n"
    "    !git pull\n"
    "!pip install -q -r requirements.txt\n"
    "!pip install -q pytest"))

c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\n"
    "import numpy as np, pandas as pd, os, yaml, torch\n"
    "import matplotlib.pyplot as plt\n"
    "from sklearn.linear_model import LogisticRegression\n"
    "from sklearn.preprocessing import StandardScaler\n"
    "from sklearn.pipeline import make_pipeline\n"
    "from sklearn.metrics import balanced_accuracy_score\n"
    "from src.prompts import NUMBER_WORDS, pluralize\n"
    "from src.pipeline import load_sdxl\n"
    "from src.config import load_config"))

c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/exp_seminvariance.yaml')\n"
    "pipe = load_sdxl()\n"
    "@torch.no_grad()\n"
    "def text_embed(prompt):\n"
    "    # SDXL fused text embedding (mean-pooled over tokens)\n"
    "    out = pipe.encode_prompt(prompt=prompt, prompt_2=None, device=pipe.device,\n"
    "                             num_images_per_prompt=1, do_classifier_free_guidance=False)\n"
    "    return out[0][0].mean(0).detach().cpu().float().numpy()\n"
    "def word_prompt(n, obj):  return f'{NUMBER_WORDS[n]} {obj if n==1 else pluralize(obj)}'\n"
    "def digit_prompt(n, obj): return f'{n} {obj if n==1 else pluralize(obj)}'\n"
    "print('example forms:', word_prompt(3, 'cat'), '|', digit_prompt(3, 'cat'))"))

c.append(nbf.v4.new_code_cell(
    "# Build word-form and digit-form embeddings for every (count, object).\n"
    "Xw, Xd, y = [], [], []\n"
    "for n in cfg.counts:\n"
    "    for obj in cfg.objects:\n"
    "        Xw.append(text_embed(word_prompt(n, obj)))\n"
    "        Xd.append(text_embed(digit_prompt(n, obj)))\n"
    "        y.append(n)\n"
    "Xw, Xd, y = np.array(Xw), np.array(Xd), np.array(y)\n"
    "print('embeddings:', Xw.shape, '| counts', sorted(set(y)))"))

c.append(nbf.v4.new_code_cell(
    "# Within-form (train/test same form, held-out objects) vs cross-form transfer.\n"
    "def clf():\n"
    "    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))\n"
    "def bal(model, X, ytrue):\n"
    "    return balanced_accuracy_score(ytrue, model.predict(X))\n"
    "# fit on ALL word, test on ALL digit (and vice versa) = pure form transfer\n"
    "mw = clf().fit(Xw, y); md = clf().fit(Xd, y)\n"
    "res = {\n"
    "  'word->word (train-set fit)': bal(mw, Xw, y),\n"
    "  'digit->digit (train-set fit)': bal(md, Xd, y),\n"
    "  'word->DIGIT (cross-form)': bal(mw, Xd, y),\n"
    "  'digit->WORD (cross-form)': bal(md, Xw, y),\n"
    "}\n"
    "for k, v in res.items(): print(f'{k:>28}: balanced acc {v:.3f}')\n"
    "pd.Series(res).to_csv('results/exp_seminvariance.csv')"))

c.append(nbf.v4.new_code_cell(
    "chance = 1.0 / len(set(y))\n"
    "fig, ax = plt.subplots(figsize=(7, 4))\n"
    "ax.bar(range(len(res)), list(res.values()),\n"
    "       color=['gray', 'gray', 'C2', 'C2'])\n"
    "ax.axhline(chance, color='r', ls=':', label=f'chance={chance:.2f}')\n"
    "ax.set_xticks(range(len(res))); ax.set_xticklabels(list(res.keys()), rotation=20, ha='right')\n"
    "ax.set_ylabel('balanced accuracy'); ax.set_ylim(0, 1)\n"
    "ax.set_title('Numeral representation: within-form vs cross-form (word<->digit)')\n"
    "ax.legend(); plt.tight_layout()\n"
    "plt.savefig('results/exp_seminvariance.png', dpi=110, bbox_inches='tight'); plt.show()"))

c.append(nbf.v4.new_markdown_cell(
    "## How to read this\n"
    "- **Cross-form (word->digit, digit->word) accuracy well above chance and "
    "close to within-form** = the numeral is represented **form-invariantly** in "
    "the text embedding -> more than lexical; supports (some) numerical "
    "comprehension. We can then keep a measured 'the requested count is "
    "faithfully & form-invariantly represented' claim.\n"
    "- **Cross-form near chance while within-form is high** = the representation "
    "is **lexical** (word-token identity), not abstract number -> we downgrade "
    "the wording to 'the numeral token is faithfully represented' and drop any "
    "'comprehension' claim (the reviewer's exact point).\n"
    "- Note: this is the *text* side; it is orthogonal to the causal image-side "
    "finding and simply calibrates how strong the comprehension wording may be."))

nb["cells"] = c
nb["metadata"] = {"accelerator": "GPU", "colab": {"provenance": []},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}}

import os
out = os.path.join(os.path.dirname(__file__), "exp_seminvariance.ipynb")
nbf.write(nb, out)
print("wrote", out)
