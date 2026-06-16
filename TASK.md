# Task: Data Familiarization & Preliminary Baselines

**Phase status: ✅ COMPLETE** (June 16, 2026). See `REPORT1.md` for full findings. Next step: design discussion to lock in binarization strategy, cross-corpus architecture, and transformer baseline plan before Phase 2.

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
- **Download note (confirmed June 2026):** the server redirects HTTP → HTTPS but has a broken certificate chain. Use `requests.get(..., verify=False)` with `urllib3.disable_warnings()`. Documented in the notebook.
- **Column structure (confirmed June 2026):** the CSV has a proper header row. Schema is exactly: `id` (int), `formality` (float), `informativeness` (float), `implicature` (float), `sentence` (str). No additional columns. No renaming needed.
- **Data quality note:** sentence values carry a leading integer prefix (e.g. `"10In High Bay 4…"`) — original sentence position markers from source documents concatenated without a space. Strip with `str.replace(r"^\d+", "")` before modeling.
- **Genre information:** SQUINKY! was compiled from blog (2,110), news (3,009), and forum (2,569) sentences, but **the CSV contains no genre column** and genre cannot be recovered by ID-range inference (row ordering does not match compilation order). Genre is unavailable for this corpus.
- **Reference classifier:** `meyersbs/squinky` uses POS + lemma + stem + chunk features (Vincze 2015) with `DictVectorizer + LogisticRegression`, 25% test split, no fixed random seed — not TF-IDF. Reported F1 (0.82/0.84/0.60) is not directly comparable to a TF-IDF baseline.
- Reference binarization (confirmed): score > 4 → positive class (formal / informative / implicative), else negative.
- License: the wrapper code is MIT; the underlying corpus data has no explicit license stated in the repo or paper. Treat as research-use only, do not redistribute.

## Tasks for this notebook

1. ✅ **Environment & imports.** Standard stack: pandas, numpy, scikit-learn, matplotlib/seaborn, `datasets`, requests.

2. ✅ **Load both datasets.** Pavlick-Tetreault via `datasets`; SQUINKY! via direct download into `data/raw/` (gitignored, cached locally so re-runs don't re-download).

3. ✅ **Inspect both.**
   - Shape, columns, dtypes, missing values
   - Summary statistics per dimension (formality for both; informativeness/implicature for SQUINKY!)
   - Distribution plots per genre/domain
   - Note differences in scale (−3..3 vs. 1..7), genre composition, sentence length

4. ✅ **Preliminary classifier tests (2–3 models).** Goal is signal, not a finished pipeline:
   - **Model A:** TF-IDF + Logistic Regression, in-domain on Pavlick-Tetreault — macro F1: **0.76**
   - **Model B:** TF-IDF + Logistic Regression, in-domain on SQUINKY! formality — macro F1: **0.81**
   - **Model C (the interesting one):** cross-corpus test, both directions:
     - Train PT → Test SQUINKY!: macro F1 **0.79**
     - Train SQUINKY! → Test PT: macro F1 **0.75**
   - Key finding: cross-corpus transfer is strong — essentially no degradation from in-domain.

5. ✅ **Document findings inline.** Observation cells completed throughout the notebook.

6. ✅ **Close with an "Open Questions for Design Discussion" cell** — 7 concrete open questions logged. See also `REPORT1.md` for the full writeup.

## Out of scope for this phase

- Final model architecture / fine-tuning a transformer
- Final binarization or regression decision
- Cross-corpus generalization *architecture* (only a quick diagnostic test here, not the final method)
- Packaging, README polish, dataset card, or anything publication-facing

## Style notes

- Comment code for a reader who knows ML but isn't a specialist in computational pragmatics — explain *why* a step matters linguistically where relevant.
- Keep the notebook honest: if something doesn't work as expected (SQUINKY URL down, columns don't match expectations), note it in a markdown cell and propose a fallback rather than silently working around it.
- This notebook is meant to be published in the public repo eventually — write at that quality bar, but don't let that block fast iteration now.
