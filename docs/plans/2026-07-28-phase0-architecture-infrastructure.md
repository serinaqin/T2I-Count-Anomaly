# Phase 0: Architecture & Infrastructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map SDXL's architecture (candidate count-signal sites) and build the shared, unit-tested `src/` infrastructure — prompt grid, detector scorer, probes, variance analysis, activation capture — culminating in a Colab smoke notebook that generates, scores, and captures on a tiny pilot.

**Architecture:** All logic lives in `src/` as small focused modules, unit-tested off-GPU with pytest on synthetic data. GPU-only code (SDXL generation, real detector inference) has its *logic* tested with fakes/synthetic inputs; its real execution is validated by the smoke notebook on Colab. Notebooks are thin drivers that add the repo to `sys.path` and import `src`.

**Tech Stack:** Python 3.10+, PyTorch, diffusers (SDXL base 1.0), transformers, scikit-learn, pandas, numpy, matplotlib, pytest. Detector: GroundingDINO (P0 default, pip-installable) with CountGD noted as the P1 upgrade.

## Global Constraints

- Counts supported: **1–7** only (`ValueError` outside this range).
- Single object category per prompt; template `"{number-word} {object}"`.
- **No VLM-only scoring** — counts come from a detector oracle.
- Notebooks are **thin**: clone repo → `pip install -r requirements.txt` → `sys.path` → `from src import ...`. No experiment logic inline.
- All SDXL/detector execution runs on **Colab GPU**; pytest never loads SDXL or a real detector.
- Every module gets a matching `tests/test_<module>.py`; pure-logic tests must pass locally (`pytest -q`).
- Package import root is `src` (add repo root to `sys.path`; no packaging step required).

---

### Task 1: Test scaffolding + prompt grid (`src/prompts.py`)

**Files:**
- Create: `src/__init__.py` (empty), `src/prompts.py`
- Create: `tests/__init__.py` (empty), `tests/conftest.py`, `tests/test_prompts.py`
- Create: `pytest.ini`

**Interfaces:**
- Produces:
  - `NUMBER_WORDS: dict[int,str]` (1→"one" … 7→"seven")
  - `DEFAULT_OBJECTS: list[str]`
  - `pluralize(noun: str) -> str`
  - `build_prompt(count: int, obj: str) -> str`
  - `@dataclass(frozen=True) PromptSpec(count:int, obj:str, seed:int, text:str)`
  - `generate_grid(counts: list[int], objects: list[str], seeds: list[int]) -> list[PromptSpec]`

- [ ] **Step 1: Create pytest config**

Create `pytest.ini`:
```ini
[pytest]
testpaths = tests
addopts = -q
```
Create empty `src/__init__.py`, `tests/__init__.py`, and `tests/conftest.py`:
```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_prompts.py`:
```python
import pytest
from src.prompts import (NUMBER_WORDS, DEFAULT_OBJECTS, pluralize,
                         build_prompt, PromptSpec, generate_grid)

def test_singular_no_pluralize():
    assert build_prompt(1, "apple") == "one apple"

def test_plural_regular():
    assert build_prompt(3, "cat") == "three cats"

def test_pluralize_edge_cases():
    assert pluralize("bus") == "buses"
    assert pluralize("berry") == "berries"
    assert pluralize("car") == "cars"

def test_count_out_of_range_raises():
    with pytest.raises(ValueError):
        build_prompt(8, "cat")
    with pytest.raises(ValueError):
        build_prompt(0, "cat")

def test_generate_grid_shape_and_content():
    counts, objects, seeds = [1, 2], ["cat", "bus"], [0, 1, 2]
    grid = generate_grid(counts, objects, seeds)
    assert len(grid) == 2 * 2 * 3
    assert all(isinstance(p, PromptSpec) for p in grid)
    one_cat = [p for p in grid if p.count == 1 and p.obj == "cat"]
    assert len(one_cat) == 3
    assert one_cat[0].text == "one cat"
    assert {p.seed for p in one_cat} == {0, 1, 2}

def test_default_objects_are_seven_plus_and_countable():
    assert len(DEFAULT_OBJECTS) >= 7
    assert len(NUMBER_WORDS) == 7
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.prompts'`.

- [ ] **Step 4: Implement `src/prompts.py`**

```python
from dataclasses import dataclass

NUMBER_WORDS = {1: "one", 2: "two", 3: "three", 4: "four",
                5: "five", 6: "six", 7: "seven"}

DEFAULT_OBJECTS = ["apple", "cat", "car", "bird", "bottle",
                   "chair", "cup", "dog", "banana", "clock"]


def pluralize(noun: str) -> str:
    if noun.endswith(("s", "x", "z", "ch", "sh")):
        return noun + "es"
    if noun.endswith("y") and noun[-2] not in "aeiou":
        return noun[:-1] + "ies"
    return noun + "s"


def build_prompt(count: int, obj: str) -> str:
    if count not in NUMBER_WORDS:
        raise ValueError(f"count {count} out of supported range 1-7")
    word = NUMBER_WORDS[count]
    noun = obj if count == 1 else pluralize(obj)
    return f"{word} {noun}"


@dataclass(frozen=True)
class PromptSpec:
    count: int
    obj: str
    seed: int
    text: str


def generate_grid(counts, objects, seeds):
    grid = []
    for obj in objects:
        for count in counts:
            text = build_prompt(count, obj)
            for seed in seeds:
                grid.append(PromptSpec(count=count, obj=obj, seed=seed, text=text))
    return grid
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_prompts.py -v`
Expected: PASS (6 tests).

- [ ] **Step 6: Commit**

```bash
git add src/__init__.py src/prompts.py tests/ pytest.ini
git commit -m "feat: prompt grid + test scaffolding (Phase 0 Task 1)"
```

---

### Task 2: Scoring metrics + detection→count (`src/scoring.py`)

**Files:**
- Create: `src/scoring.py`, `tests/test_scoring.py`

**Interfaces:**
- Consumes: nothing from prior tasks.
- Produces:
  - `exact_accuracy(pred, target) -> float`
  - `mae(pred, target) -> float`
  - `tolerance_accuracy(pred, target, tol: int = 1) -> float`
  - `count_from_detections(detections: list[dict], target_label: str, score_threshold: float = 0.3) -> int` where each detection is `{"label": str, "score": float, "box": [x0,y0,x1,y1]}`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_scoring.py`:
```python
from src.scoring import exact_accuracy, mae, tolerance_accuracy, count_from_detections

def test_exact_accuracy():
    assert exact_accuracy([1, 2, 3], [1, 2, 3]) == 1.0
    assert exact_accuracy([1, 2, 3], [1, 2, 4]) == 2 / 3

def test_mae():
    assert mae([1, 2, 3], [1, 2, 3]) == 0.0
    assert mae([1, 2], [3, 2]) == 1.0

def test_tolerance_accuracy():
    assert tolerance_accuracy([1, 5], [2, 3], tol=1) == 0.5
    assert tolerance_accuracy([1, 5], [2, 3], tol=2) == 1.0

def test_count_from_detections_filters_label_and_threshold():
    dets = [
        {"label": "cat", "score": 0.9, "box": [0, 0, 1, 1]},
        {"label": "cat", "score": 0.2, "box": [1, 1, 2, 2]},
        {"label": "dog", "score": 0.9, "box": [2, 2, 3, 3]},
    ]
    assert count_from_detections(dets, "cat", score_threshold=0.3) == 1
    assert count_from_detections(dets, "cat", score_threshold=0.1) == 2
    assert count_from_detections(dets, "bird") == 0
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_scoring.py -v`
Expected: FAIL (`No module named 'src.scoring'`).

- [ ] **Step 3: Implement `src/scoring.py`**

```python
import numpy as np


def exact_accuracy(pred, target) -> float:
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.mean(pred == target))


def mae(pred, target) -> float:
    pred, target = np.asarray(pred, float), np.asarray(target, float)
    return float(np.mean(np.abs(pred - target)))


def tolerance_accuracy(pred, target, tol: int = 1) -> float:
    pred, target = np.asarray(pred), np.asarray(target)
    return float(np.mean(np.abs(pred - target) <= tol))


def count_from_detections(detections, target_label, score_threshold=0.3) -> int:
    return sum(
        1 for d in detections
        if d["label"] == target_label and d["score"] >= score_threshold
    )
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_scoring.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add src/scoring.py tests/test_scoring.py
git commit -m "feat: scoring metrics + detection-to-count (Phase 0 Task 2)"
```

---

### Task 3: Detector oracle wrapper (`src/detector.py`)

**Files:**
- Create: `src/detector.py`, `tests/test_detector.py`

**Interfaces:**
- Consumes: `count_from_detections` from `src/scoring.py`.
- Produces:
  - `class Detector` with `detect(image, labels: list[str]) -> list[dict]` (GPU; loads GroundingDINO lazily).
  - `count_objects(detector, image, object_label: str, score_threshold: float = 0.3) -> int` — pure glue over `detect` + `count_from_detections`, testable with a fake detector.

**Note:** Real GroundingDINO inference is GPU-only and validated in the smoke notebook. Tests use a `FakeDetector` to verify the glue logic and label handling.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_detector.py`:
```python
from src.detector import count_objects

class FakeDetector:
    def __init__(self, dets):
        self._dets = dets
    def detect(self, image, labels):
        return [d for d in self._dets if d["label"] in labels]

def test_count_objects_uses_detect_and_threshold():
    dets = [
        {"label": "cat", "score": 0.8, "box": [0, 0, 1, 1]},
        {"label": "cat", "score": 0.25, "box": [1, 1, 2, 2]},
    ]
    det = FakeDetector(dets)
    assert count_objects(det, image=None, object_label="cat", score_threshold=0.3) == 1
    assert count_objects(det, image=None, object_label="cat", score_threshold=0.2) == 2
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_detector.py -v`
Expected: FAIL (`No module named 'src.detector'`).

- [ ] **Step 3: Implement `src/detector.py`**

```python
from src.scoring import count_from_detections


def count_objects(detector, image, object_label, score_threshold=0.3) -> int:
    dets = detector.detect(image, labels=[object_label])
    return count_from_detections(dets, object_label, score_threshold)


class Detector:
    """GroundingDINO wrapper. Loads lazily; GPU only (used in Colab)."""

    def __init__(self, device="cuda", box_threshold=0.3, text_threshold=0.25):
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is None:
            from transformers import (AutoProcessor,
                                      AutoModelForZeroShotObjectDetection)
            model_id = "IDEA-Research/grounding-dino-tiny"
            self._processor = AutoProcessor.from_pretrained(model_id)
            self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
                model_id).to(self.device)

    def detect(self, image, labels):
        self._ensure_loaded()
        import torch
        text = ". ".join(labels) + "."
        inputs = self._processor(images=image, text=text,
                                 return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(**inputs)
        results = self._processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[image.size[::-1]],
        )[0]
        dets = []
        for score, label, box in zip(results["scores"],
                                     results["labels"], results["boxes"]):
            dets.append({"label": label, "score": float(score),
                         "box": [float(x) for x in box]})
        return dets
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_detector.py -v`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add src/detector.py tests/test_detector.py
git commit -m "feat: detector oracle wrapper with fake-tested glue (Phase 0 Task 3)"
```

---

### Task 4: Variance decomposition (`src/analysis.py`)

**Files:**
- Create: `src/analysis.py`, `tests/test_analysis.py`

**Interfaces:**
- Produces:
  - `variance_explained(df, value: str, factor: str) -> float` — between-group SS / total SS (η²).
  - `noise_vs_text_decomposition(df, value="realized_count", noise="seed", text="prompt_id") -> dict` → `{"noise_var_explained": float, "text_var_explained": float}`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis.py`:
```python
import pandas as pd
from src.analysis import variance_explained, noise_vs_text_decomposition

def test_variance_explained_all_from_factor():
    # realized_count depends ONLY on seed
    rows = []
    for seed, val in {0: 2, 1: 5, 2: 7}.items():
        for prompt_id in range(4):
            rows.append({"realized_count": val, "seed": seed, "prompt_id": prompt_id})
    df = pd.DataFrame(rows)
    assert variance_explained(df, "realized_count", "seed") > 0.99
    assert variance_explained(df, "realized_count", "prompt_id") < 0.01

def test_decomposition_returns_both():
    df = pd.DataFrame({
        "realized_count": [2, 2, 5, 5],
        "seed": [0, 0, 1, 1],
        "prompt_id": [0, 1, 0, 1],
    })
    out = noise_vs_text_decomposition(df)
    assert out["noise_var_explained"] > 0.99
    assert out["text_var_explained"] < 0.01
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_analysis.py -v`
Expected: FAIL (`No module named 'src.analysis'`).

- [ ] **Step 3: Implement `src/analysis.py`**

```python
def variance_explained(df, value: str, factor: str) -> float:
    grand = df[value].mean()
    ss_total = ((df[value] - grand) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = 0.0
    for _, group in df.groupby(factor):
        ss_between += len(group) * (group[value].mean() - grand) ** 2
    return float(ss_between / ss_total)


def noise_vs_text_decomposition(df, value="realized_count",
                                noise="seed", text="prompt_id") -> dict:
    return {
        "noise_var_explained": variance_explained(df, value, noise),
        "text_var_explained": variance_explained(df, value, text),
    }
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_analysis.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/analysis.py tests/test_analysis.py
git commit -m "feat: variance decomposition for noise-vs-text (Phase 0 Task 4)"
```

---

### Task 5: Probes (`src/probes.py`)

**Files:**
- Create: `src/probes.py`, `tests/test_probes.py`

**Interfaces:**
- Produces:
  - `fit_eval_classifier(X, y, seed: int = 0) -> dict` → `{"bal_acc": float}` (logistic, stratified 70/30 split).
  - `fit_eval_magnitude(X, y, seed: int = 0) -> dict` → `{"r2": float}` (linear regression; tests whether count is a continuous magnitude direction).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_probes.py`:
```python
import numpy as np
from src.probes import fit_eval_classifier, fit_eval_magnitude

def test_classifier_separable():
    rng = np.random.default_rng(0)
    X0 = rng.normal(0, 0.1, (50, 5))
    X1 = rng.normal(3, 0.1, (50, 5))
    X = np.vstack([X0, X1]); y = np.array([0] * 50 + [1] * 50)
    assert fit_eval_classifier(X, y)["bal_acc"] > 0.95

def test_magnitude_linear_signal():
    rng = np.random.default_rng(0)
    counts = rng.integers(1, 8, 200)
    direction = np.array([1.0, 0, 0, 0, 0])
    X = counts[:, None] * direction + rng.normal(0, 0.05, (200, 5))
    assert fit_eval_magnitude(X, counts)["r2"] > 0.95
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_probes.py -v`
Expected: FAIL (`No module named 'src.probes'`).

- [ ] **Step 3: Implement `src/probes.py`**

```python
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.model_selection import train_test_split


def fit_eval_classifier(X, y, seed: int = 0) -> dict:
    X, y = np.asarray(X), np.asarray(y)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return {"bal_acc": float(balanced_accuracy_score(yte, clf.predict(Xte)))}


def fit_eval_magnitude(X, y, seed: int = 0) -> dict:
    X, y = np.asarray(X), np.asarray(y, float)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, random_state=seed)
    reg = LinearRegression().fit(Xtr, ytr)
    return {"r2": float(r2_score(yte, reg.predict(Xte)))}
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_probes.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/probes.py tests/test_probes.py
git commit -m "feat: linear/magnitude probes (Phase 0 Task 5)"
```

---

### Task 6: Activation capture + site catalog (`src/pipeline.py`)

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `catalog_attention_sites(unet) -> list[str]` — names of modules ending in `attn1`/`attn2`.
  - `class ActivationCapture(model, sites)` context manager; after a forward pass, `.acts: dict[str, Tensor]` holds each site's output; handles removed on exit.
  - `load_sdxl(device="cuda", dtype=torch.float16)` — returns a diffusers SDXL pipeline (GPU; smoke-tested only).
  - `generate(pipe, prompt: str, seed: int, num_inference_steps: int = 30) -> PIL.Image` (GPU; smoke-tested only).

**Note:** `load_sdxl`/`generate` run only on Colab. Tests cover `catalog_attention_sites` and `ActivationCapture` using a tiny fake `nn.Module` — no SDXL, no CUDA.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_pipeline.py`:
```python
import torch
import torch.nn as nn
from src.pipeline import catalog_attention_sites, ActivationCapture

class Attn(nn.Module):
    def __init__(self): super().__init__(); self.lin = nn.Linear(4, 4)
    def forward(self, x): return self.lin(x)

class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn1 = Attn()
        self.attn2 = Attn()
    def forward(self, x): return self.attn2(self.attn1(x))

def test_catalog_finds_attn_sites():
    m = Block()
    sites = catalog_attention_sites(m)
    assert set(sites) == {"attn1", "attn2"}

def test_activation_capture_records_and_cleans_up():
    m = Block()
    sites = catalog_attention_sites(m)
    with ActivationCapture(m, sites) as cap:
        m(torch.zeros(1, 4))
        assert set(cap.acts.keys()) == {"attn1", "attn2"}
        assert cap.acts["attn1"].shape == (1, 4)
    # handles removed: a second forward must not add new captures
    cap.acts.clear()
    m(torch.zeros(1, 4))
    assert cap.acts == {}
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_pipeline.py -v`
Expected: FAIL (`No module named 'src.pipeline'`).

- [ ] **Step 3: Implement `src/pipeline.py`**

```python
import torch


def catalog_attention_sites(unet):
    sites = []
    for name, _ in unet.named_modules():
        if name.endswith("attn1") or name.endswith("attn2"):
            sites.append(name)
    return sites


class ActivationCapture:
    def __init__(self, model, sites):
        self.model = model
        self.sites = sites
        self.acts = {}
        self._handles = []

    def _make_hook(self, name):
        def hook(_module, _inp, out):
            self.acts[name] = out.detach() if hasattr(out, "detach") else out
        return hook

    def __enter__(self):
        modmap = dict(self.model.named_modules())
        for s in self.sites:
            self._handles.append(modmap[s].register_forward_hook(self._make_hook(s)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def load_sdxl(device="cuda", dtype=torch.float16):
    from diffusers import StableDiffusionXLPipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=dtype)
    return pipe.to(device)


def generate(pipe, prompt, seed, num_inference_steps=30):
    g = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(prompt, generator=g,
                num_inference_steps=num_inference_steps).images[0]
```

- [ ] **Step 4: Run to verify pass**

Run: `pytest tests/test_pipeline.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: activation capture + site catalog + SDXL loaders (Phase 0 Task 6)"
```

---

### Task 7: Config + full local test run

**Files:**
- Create: `configs/default.yaml`, `src/config.py`, `tests/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass ExperimentConfig(counts, objects, seeds, num_inference_steps, score_threshold)`
  - `load_config(path: str) -> ExperimentConfig`

- [ ] **Step 1: Write the failing test**

Create `tests/test_config.py`:
```python
from src.config import load_config, ExperimentConfig

def test_load_default_config(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "counts: [1, 2, 3]\n"
        "objects: [cat, dog]\n"
        "seeds: [0, 1]\n"
        "num_inference_steps: 30\n"
        "score_threshold: 0.3\n"
    )
    cfg = load_config(str(p))
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.counts == [1, 2, 3]
    assert cfg.objects == ["cat", "dog"]
    assert cfg.num_inference_steps == 30
```

- [ ] **Step 2: Run to verify fail**

Run: `pytest tests/test_config.py -v`
Expected: FAIL (`No module named 'src.config'`).

- [ ] **Step 3: Implement config**

Create `configs/default.yaml` (small pilot defaults):
```yaml
# Phase 0/1 small pilot
counts: [1, 2, 3]
objects: [cat, dog]
seeds: [0, 1, 2]
num_inference_steps: 30
score_threshold: 0.3
```
Create `src/config.py`:
```python
from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    counts: list
    objects: list
    seeds: list
    num_inference_steps: int
    score_threshold: float


def load_config(path: str) -> ExperimentConfig:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f)
    return ExperimentConfig(**d)
```
Add `pyyaml` to `requirements.txt` (append a line `pyyaml`).

- [ ] **Step 4: Run the FULL suite to verify pass**

Run: `pytest -q`
Expected: PASS (all tests from Tasks 1–7).

- [ ] **Step 5: Commit**

```bash
git add configs/default.yaml src/config.py tests/test_config.py requirements.txt
git commit -m "feat: experiment config + full local test suite green (Phase 0 Task 7)"
```

---

### Task 8: Smoke notebook + architecture reference (Colab)

**Files:**
- Create: `notebooks/00_setup_smoke.ipynb` (via the generator script below)
- Create: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: everything in `src/`.
- Produces: a runnable Colab notebook that (a) installs deps, (b) runs the local test suite, (c) loads SDXL, (d) generates a tiny pilot grid, (e) scores counts with the detector, (f) captures activations for one generation and prints the site catalog, (g) writes `results/smoke_summary.csv`. And an architecture doc whose site catalog is filled from step (f).

- [ ] **Step 1: Generate the notebook**

Create and run `notebooks/_build_00.py` (one-off generator; committed for reproducibility):
```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
c = []
c.append(nbf.v4.new_markdown_cell(
    "# Phase 0 — Setup & Smoke Test\\n"
    "Thin driver: clones repo, installs deps, runs tests, then a tiny SDXL pilot."))
c.append(nbf.v4.new_code_cell(
    "!git clone https://github.com/serinaqin/T2I-Count-Anomaly.git\\n"
    "%cd T2I-Count-Anomaly\\n"
    "!pip install -q -r requirements.txt\\n"
    "!pip install -q pytest"))
c.append(nbf.v4.new_code_cell("!pytest -q"))
c.append(nbf.v4.new_code_cell(
    "import sys; sys.path.insert(0, '.')\\n"
    "from src.prompts import generate_grid\\n"
    "from src.pipeline import load_sdxl, generate, catalog_attention_sites, ActivationCapture\\n"
    "from src.detector import Detector, count_objects\\n"
    "from src.config import load_config\\n"
    "import pandas as pd, os"))
c.append(nbf.v4.new_code_cell(
    "cfg = load_config('configs/default.yaml')\\n"
    "grid = generate_grid(cfg.counts, cfg.objects, cfg.seeds)\\n"
    "print(len(grid), 'prompts in pilot'); grid[:3]"))
c.append(nbf.v4.new_code_cell(
    "pipe = load_sdxl()\\n"
    "sites = catalog_attention_sites(pipe.unet)\\n"
    "print(len(sites), 'attention sites'); print(sites[:10])"))
c.append(nbf.v4.new_code_cell(
    "det = Detector()\\n"
    "rows = []\\n"
    "for p in grid:\\n"
    "    img = generate(pipe, p.text, p.seed, cfg.num_inference_steps)\\n"
    "    n = count_objects(det, img, p.obj, cfg.score_threshold)\\n"
    "    rows.append({'text': p.text, 'obj': p.obj, 'count': p.count,\\n"
    "                 'seed': p.seed, 'realized_count': n})\\n"
    "df = pd.DataFrame(rows)\\n"
    "os.makedirs('results', exist_ok=True)\\n"
    "df.to_csv('results/smoke_summary.csv', index=False)\\n"
    "df"))
c.append(nbf.v4.new_code_cell(
    "# capture activations for ONE generation to confirm hooks fire\\n"
    "with ActivationCapture(pipe.unet, sites[:3]) as cap:\\n"
    "    _ = generate(pipe, grid[0].text, grid[0].seed, num_inference_steps=2)\\n"
    "print({k: tuple(v.shape) for k, v in cap.acts.items()})"))
nb["cells"] = c
nbf.write(nb, "notebooks/00_setup_smoke.ipynb")
print("wrote notebooks/00_setup_smoke.ipynb")
```
Run: `pip install nbformat && python notebooks/_build_00.py`
Expected: `wrote notebooks/00_setup_smoke.ipynb`.

- [ ] **Step 2: Write the architecture reference skeleton**

Create `docs/ARCHITECTURE.md`:
```markdown
# SDXL Architecture Reference (count-signal candidate sites)

## Text pathway
- Two text encoders: CLIP ViT-L/14 (`text_encoder`) + OpenCLIP ViT-bigG/14 (`text_encoder_2`).
- Their penultimate hidden states are concatenated → the conditioning fed to U-Net cross-attention. `text_encoder_2` also provides a pooled embedding.

## U-Net
- Blocks: `down_blocks[0..2]`, `mid_block`, `up_blocks[0..2]`.
- Each transformer block has `attn1` (self-attention, image↔image — instance identity per CountGen) and `attn2` (cross-attention, text→image — the matching junction).
- Prior Track A localization: `up_blocks[0].attentions[1]` cross-attn (1280-dim, 32×32). CountGen anchor: an up-block self-attn at t≈500.

## Timesteps
- Layout/count form early/high-noise (eDiff-I); capture emphasis on the first ~20% of steps.

## VAE
- Latents 128×128×4 → 1024×1024 RGB. Deprioritized as a cause.

## Candidate site catalog (filled from smoke run)
_Paste the printed `catalog_attention_sites(pipe.unet)` output here after running `notebooks/00_setup_smoke.ipynb` step (f)._
```

- [ ] **Step 3: Run the smoke notebook on Colab (GPU)**

Open `notebooks/00_setup_smoke.ipynb` in Colab (T4 is fine). Run all.
Expected: tests pass; site catalog prints (~140 attention modules); a small `results/smoke_summary.csv` with a `realized_count` column; activation-capture cell prints non-empty shapes. Paste the site catalog into `docs/ARCHITECTURE.md`.

- [ ] **Step 4: Commit**

```bash
git add notebooks/_build_00.py notebooks/00_setup_smoke.ipynb docs/ARCHITECTURE.md
git commit -m "feat: smoke notebook + architecture reference (Phase 0 Task 8)"
```

---

## Self-Review

**Spec coverage (§6 Phase 0 in the design):**
- "Map SDXL end-to-end + candidate probe-site catalog" → Task 6 (`catalog_attention_sites`) + Task 8 (`docs/ARCHITECTURE.md`). ✓
- "Build infrastructure: prompt grid, generation + capture, detector scorer, probe utilities" → Tasks 1, 6, 2/3, 5. ✓
- "Unit-tested off-GPU" → every task has pytest; GPU code fake/synthetic-tested. ✓
- "Validated with a tiny Colab smoke run" → Task 8. ✓
- Variance decomposition (needed by P1, built here as shared infra) → Task 4. ✓
- Detector oracle, no VLM (Global Constraint) → Tasks 2/3. ✓

**Placeholder scan:** No TBD/TODO. The only intentional fill-in is the site-catalog paste in `docs/ARCHITECTURE.md`, which is gated on running the notebook (Task 8 Step 3) — documented, not a code placeholder.

**Type consistency:** `count_from_detections` signature identical in Tasks 2 and 3; `catalog_attention_sites`/`ActivationCapture` names identical in Tasks 6 and 8; `ExperimentConfig` fields match `configs/default.yaml`; detection dict shape `{"label","score","box"}` consistent across scoring/detector tests.
