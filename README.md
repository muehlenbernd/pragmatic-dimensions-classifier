# pragmatic-dimensions-classifier

**Status:** 🚧 Exploration phase — data familiarization and preliminary baselines. Final task design (classification vs. regression, binarization strategy, cross-corpus architecture) is not locked in yet. See `TASK.md` for the current working brief.

## What this is

A portfolio project investigating sentence-level **formality**, and how it relates to **informativeness** and **implicature** — three pragmatic dimensions central to my research at ZAS (Leibniz-Zentrum Allgemeine Sprachwissenschaft). The eventual goal is a classifier (or set of classifiers) trained on one formality corpus and evaluated for cross-corpus generalization on another, with a linguistically grounded error analysis.

This is a continuation of an existing research program — LLM calibration on social and pragmatic judgments (CMCL 2026) — rather than a standalone exercise, and connects to ongoing work on imprecision-in-context and computational pragmatics more broadly.

## Datasets

| Dataset | Dimensions | Size | License | Source |
|---|---|---|---|---|
| Pavlick & Tetreault (2016) | Formality (−3 to 3) | 11,274 sentences, 4 genres | CC-BY-3.0 | [HF: osyvokon/pavlick-formality-scores](https://huggingface.co/datasets/osyvokon/pavlick-formality-scores) |
| SQUINKY! (Lahiri 2015) | Formality, Informativeness, Implicature (1–7) | 7,032 sentences | Code: MIT (`meyersbs/squinky`); data license unconfirmed — see `data/README.md` | [arXiv:1506.02306](https://arxiv.org/abs/1506.02306), [github.com/meyersbs/squinky](https://github.com/meyersbs/squinky) |

Raw data files are **not redistributed in this repo** — see `data/README.md`. The notebook downloads them directly from the original sources.

## Current contents

- `notebooks/01_data_exploration_and_baselines.ipynb` — loads both datasets, inspects structure and distributions, and runs preliminary classifier tests (TF-IDF + logistic regression, in-domain and cross-corpus) to surface signal before finalizing the project design.
- `TASK.md` — working task description for this phase. Written to also serve as a standalone brief for an AI coding agent picking up the project without prior context.
- `data/README.md` — dataset provenance, citation requirements, licensing notes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook notebooks/01_data_exploration_and_baselines.ipynb
```

## Citation

If referencing the underlying data, cite the original papers — see `data/README.md` for exact BibTeX entries. Not this repository.

## License

Code in this repository is MIT licensed (see `LICENSE`). This does not extend to the datasets it loads.
