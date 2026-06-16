# Task: Data Familiarization & Preliminary Baselines

## Context (read this first)

This repo is the early-exploration phase of a portfolio project for Roland Mühlenbernd, a computational linguist transitioning from academia (ZAS, Berlin) to industry NLP roles. The project investigates sentence-level **formality**, and its relationship to **informativeness** and **implicature**, using two openly available datasets. It is the main remaining technical piece of "Level 1" of a structured career-transition plan, and will eventually become a public GitHub repo demonstrating both ML engineering skill and linguistic domain expertise.

**This is a familiarization phase, not the final project design.** The goal of this notebook is to get hands-on with both datasets, surface practical problems, and generate concrete signal (e.g. how hard is cross-corpus generalization, really?) to inform a separate design discussion later. Do not silently lock in modeling decisions beyond what's needed to run a preliminary test — flag open questions instead of resolving them.

## Datasets

### 1. Pavlick & Tetreault (2016) — Online Formality Corpus

- Load via HuggingFace: `load_dataset("osyvokon/pavlick-formality-scores")`
- Columns: `domain` (news / blog / email / answers), `avg_score` (float, −3 to 3, lower = less formal), `sentence`
- Splits: `train` (9,274 rows), `test` (2,000 rows)
- License: CC-BY-3.0. Cite both Pavlick & Tetreault (2016) and Lahiri (2015) — see `data/README.md`.

### 2. SQUINKY! (Lahiri 2015)

- Raw file: `http://people.rc.rit.edu/~bsm9339/corpora/squinky_corpus/mturk_merged.csv` (linked from the official `meyersbs/squinky` repo's `download.py`)
- **Note:** this is a personal academic homepage, not a stable archive. Verify the URL resolves before relying on it; if down, check `meyersbs/squinky` GitHub issues for a mirror, or contact the maintainer.
- **Column structure is not fully documented.** The reference implementation (`meyersbs/squinky/squinky/classifier.py`) accesses the CSV by position: column 0 = id, column 1 = formality (1–7), column 2 = informativeness (1–7), column 3 = implicature (1–7), last column = sentence text. This implies there may be additional undocumented columns (e.g. genre/source) in between. **Print and inspect `df.columns` after loading — do not assume the schema beyond this.**
- Reference binarization precedent (from the same repo): score > 4 → positive class (formal / informative / implicative), else negative. One possible convention, not necessarily the one we'll use for the final project.
- License: the wrapper code is MIT; the underlying corpus data has no explicit license stated in the repo or paper. Treat with the same caution as an unredistributable source until confirmed — same posture as GYAFC.

## Tasks for this notebook

1. **Environment & imports.** Standard stack: pandas, numpy, scikit-learn, matplotlib/seaborn, `datasets`, requests.

2. **Load both datasets.** Pavlick-Tetreault via `datasets`; SQUINKY! via direct download into `data/raw/` (gitignored, cached locally so re-runs don't re-download).

3. **Inspect both.**
   - Shape, columns, dtypes, missing values
   - Summary statistics per dimension (formality for both; informativeness/implicature for SQUINKY!)
   - Distribution plots per genre/domain
   - Note differences in scale (−3..3 vs. 1..7), genre composition, sentence length

4. **Preliminary classifier tests (2–3 models).** Goal is signal, not a finished pipeline:
   - **Model A:** TF-IDF + Logistic Regression, in-domain on Pavlick-Tetreault (binarize formality — flag the threshold choice explicitly in a markdown cell, including alternatives considered).
   - **Model B:** TF-IDF + Logistic Regression, in-domain on SQUINKY! formality.
   - **Model C (the interesting one):** cross-corpus test — train on one, evaluate on the other (both directions if time allows). This previews the cross-corpus generalization question that's central to the eventual project design.
   - Report accuracy, F1, confusion matrix for each. Don't over-engineer preprocessing at this stage.

5. **Document findings inline.** After each major step, add a markdown cell with observations — what's straightforward, what looks like a problem (class imbalance? genre mismatch? noisy implicature labels?), and ideas for the final design.

6. **Close with an "Open Questions for Design Discussion" cell** — concrete things to decide before building the final classifier (classification vs. regression framing, binarization rationale, how to handle implicature's known unreliability, whether genre should be a stratification variable or a feature).

## Out of scope for this phase

- Final model architecture / fine-tuning a transformer
- Final binarization or regression decision
- Cross-corpus generalization *architecture* (only a quick diagnostic test here, not the final method)
- Packaging, README polish, dataset card, or anything publication-facing

## Style notes

- Comment code for a reader who knows ML but isn't a specialist in computational pragmatics — explain *why* a step matters linguistically where relevant.
- Keep the notebook honest: if something doesn't work as expected (SQUINKY URL down, columns don't match expectations), note it in a markdown cell and propose a fallback rather than silently working around it.
- This notebook is meant to be published in the public repo eventually — write at that quality bar, but don't let that block fast iteration now.
