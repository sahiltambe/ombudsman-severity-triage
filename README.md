# Severity classification for automatic triage

## What this solves (assessment brief)

The brief asks for a **classification model that predicts severity score**, used for **automatic triage**: whether a case is **closed** or **investigated**.

| Severity | Triage action |
|---|---|
| 1, 2, 3 | Close (not progressed) |
| 4, 5, 6 | Investigate (progressed) |

Missing a severe case is treated as worse than reviewing a mild one. The model therefore predicts severity **1–6**, and a cost-aware rule on `P(severity >= 4)` sets the close / investigate decision.

**Shipped answer**

- Model: proportional-odds (ordered) logistic regression  
- Decision: FN:FP cost ratio **5:1**, threshold **0.19** on `P_Severe`  
- Holdback file: `data/predictions/holdback_predictions.csv`  
  - `CaseReference` + `PredictedSeverityScore` (required)  
  - also `ProgressDecision`, `P_Severe`, class probabilities  

How to read the prediction file without extra columns: **investigate if `PredictedSeverityScore` >= 4**, else close.

---

## Results (internal test, 400 cases held out)

| | Most likely class | Cost-aware triage (5:1) |
|---|---|---|
| Quadratic weighted kappa | 0.957 | 0.952 |
| Severe-case recall (severity >= 4) | 0.950 | 0.992 |
| Severe cases missed (of 120) | 6 | 1 |

Holdback (500 cases): severity counts 1→6 = 131, 118, 79, 100, 43, 29 (**34.4%** progressed).

---

## Deliverables

| Brief item | Location |
|---|---|
| Source code | `src/severity_triage/` |
| Notebooks | `notebooks/01_*.ipynb`, `02_*.ipynb`, `03_*.ipynb` |
| Holdback predictions | `data/predictions/holdback_predictions.csv` |
| Approach + assumptions | this README |
| Dependencies | `requirements.txt` |

---

## How to run (step by step)

Needs **Python 3.11+** (3.12 recommended). Open a terminal in the repo root.

### Step 1 — Create environment and install

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

**macOS / Linux**

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e . --no-deps
```

`pyproject.toml` is required for `pip install -e .` (registers the `severity-triage` command). It is not optional fluff.

### Step 2 — Quick check that install worked

```powershell
severity-triage --version
severity-triage --help
```

### Step 3 — Reproduce the holdback predictions (fast)

Uses the committed trained model under `models/`:

```powershell
severity-triage predict
```

You should see **500** rows written to `data/predictions/holdback_predictions.csv`.

### Step 4 — Full rebuild (optional, slower)

Runs cleaning → training → evaluation → explainability → predictions:

```powershell
severity-triage all
```

Faster smoke path (fewer CV repeats; numbers may differ slightly):

```powershell
severity-triage all --fast
```

### Step 5 — Run stages one by one (optional)

| Order | Command | What you get |
|---|---|---|
| 1 | `severity-triage validate` | Clean data + `reports/data_quality/` |
| 2 | `severity-triage train` | Model comparison tables in `reports/` |
| 3 | `severity-triage evaluate` | Holdout metrics + cost threshold |
| 4 | `severity-triage explain` | SHAP / importance artefacts |
| 5 | `severity-triage predict` | Holdback CSV + `models/run_manifest.json` |

### Step 6 — Tests

```powershell
pytest -m "not slow"
```

### Step 7 — Notebooks

After install (and ideally after `severity-triage all` or with committed `reports/` present):

```powershell
jupyter lab notebooks/
```

- **01** — data quality / leakage checks  
- **02** — models + triage evaluation  
- **03** — explainability + business risk  

---

## Approach (short)

1. Clean data; drop `OmbudsmanInvestigationRequired` (label leak; empty in holdback).  
2. Engineer features; compare models with repeated stratified CV.  
3. Select ordered logit (simplest among models with similar QWK).  
4. Tune triage threshold for FN:FP = 5:1.  
5. Score holdback and write the prediction file.

## Assumptions

1. A missed severe case costs five unnecessary reviews (brief states direction, not the exact ratio).  
2. Internal scores (`EstimatedImpactScore`, `EstimatedRiskScore`, `PredictedRemedyBand`) are available at triage time.  
3. Undocumented Yes/No columns are kept as features.  
4. Out-of-range values are treated as errors (set to missing).  
5. Train and holdback are from the same population.  
6. Data are synthetic; real-world performance will need live validation.

## Repository layout

```
configs/            settings (seed, cost ratio, paths)
data/raw/           assessment workbook + checksum
data/predictions/   holdback_predictions.csv
models/             trained pipeline + run_manifest.json
notebooks/          analysis notebooks
reports/            metric tables and figures
src/severity_triage/  Python package + CLI
tests/              automated checks
pyproject.toml      package install / CLI entrypoint
requirements.txt    pinned dependencies
```

## Licence

MIT. Synthetic assessment data; no personal data.
