# Phase 1 Report: Data Familiarization & Preliminary Baselines

**Project:** pragmatic-dimensions-classifier  
**Author:** Roland Mühlenbernd  
**Date:** June 16, 2026  
**Status:** Exploration phase complete — ready for design discussion.

This report summarizes findings from the first notebook (`notebooks/01_data_exploration_and_baselines.ipynb`), covering dataset inspection, reproducibility checks, and preliminary classifier results. It is intended as a handover document for the design discussion that follows this phase.

---

## 1. Datasets

### 1.1 Pavlick & Tetreault (2016) — Online Formality Corpus

- **Source:** HuggingFace (`osyvokon/pavlick-formality-scores`)
- **Size:** 11,274 sentences (9,274 train + 2,000 test)
- **Schema:** `domain` (news / blog / email / answers), `avg_score` (float, −3 to 3), `sentence`
- **License:** CC-BY-3.0

**Distribution findings:**

| Domain  | N      | Mean  | Std  |
|---------|--------|-------|------|
| answers | 4,977  | −0.73 | 1.30 |
| blog    | 1,821  | +0.21 | 1.06 |
| email   | 1,701  | +0.46 | 1.41 |
| news    | 2,775  | +0.72 | 0.86 |

- Clear genre hierarchy: news > email > blog > answers. The ranking is intuitive.
- `answers` (Yahoo Answers-style) dominates the corpus (~44%) and is the only genre with a negative mean — solidly informal.
- `news` has the tightest distribution (std 0.86), reflecting stylistic uniformity of newswire prose.
- `email` and `answers` show the most spread (std ~1.3–1.4), reflecting wider register variation within informal genres.
- **Scale note:** PT uses a continuous −3 to 3 scale (crowd-averaged). Any cross-corpus work must account for this vs. SQUINKY!'s 1–7 scale.

---

### 1.2 SQUINKY! (Lahiri 2015)

- **Source:** `http://people.rc.rit.edu/~bsm9339/corpora/squinky_corpus/mturk_merged.csv`
- **Size:** 7,032 sentences
- **Schema:** `id` (int), `formality` (float, 1–7), `informativeness` (float, 1–7), `implicature` (float, 1–7), `sentence` (str)
- **License:** Wrapper code MIT; underlying data license unconfirmed — treat as research-use only, do not redistribute.

**Data quality note:** Sentence values in the CSV carry a leading integer prefix (e.g. `"10In High Bay 4…"`), which are original sentence position markers from source documents concatenated without a space during corpus assembly. These are stripped in preprocessing (`str.replace(r"^\d+", "")`) before any modeling.

**Download note:** The server at `people.rc.rit.edu` has a broken SSL certificate chain. The download requires `verify=False` in `requests.get()` — this is a server-side misconfiguration, not a security issue with the data itself. Documented in the notebook.

**Distribution findings:**

| Dimension       | Mean | Std  | Min | Max |
|-----------------|------|------|-----|-----|
| Formality       | 3.91 | 1.12 | 1.0 | 6.6 |
| Informativeness | 4.48 | 1.05 | 1.2 | 6.7 |
| Implicature     | 4.00 | 0.67 | 1.6 | 6.2 |

- **Formality and informativeness** are strongly correlated (Pearson r = 0.77), consistent with Lahiri (2015) (Spearman ρ = 0.73) and theoretically grounded: formal registers tend to pack more propositional content. A formality classifier will inevitably learn partial informativeness signal.
- **Implicature** is uncorrelated with both formality (r = 0.01) and informativeness (r = −0.11 in our data; ρ = 0.05 in the paper). This is theoretically expected — implicature is orthogonal to register.
- **Implicature distribution is the tightest** (std 0.67 vs. 1.12/1.05 for the others). This is not annotation noise in the usual sense — it reflects *central tendency bias*: annotators defaulted to middling values for an inherently subjective concept. The paper's Table 1 confirms low inter-experiment reliability for implicature (Spearman ρ = 0.14 between MTurk rounds, vs. 0.68/0.64 for formality/informativeness). The practical consequence is a compressed score range with little discriminative signal near the binarization threshold.

**Genre information:** SQUINKY! was compiled from three genres — blog (2,110 sentences), news (3,009), and forum/Ubuntu Forums + TripAdvisor (2,569). However, **the publicly released `mturk_merged.csv` contains no genre column**. An attempt to recover genre by ID range (assuming blog → news → forum ordering) failed: the resulting formality ranking was inverted (forum > blog > news instead of expected news > blog > forum), confirming the rows are not stored in compilation order. Genre information is definitively unavailable from this source.

**Reference implementation note:** The `meyersbs/squinky` classifier uses **POS tags, lemmas, stems, and chunk tags** (Vincze 2015 feature set) with a `DictVectorizer + LogisticRegression` pipeline — not TF-IDF. The reported F1 scores (0.82 / 0.84 / 0.60) were achieved with a 25% test split and no fixed random seed. Our TF-IDF replication reaches 0.81 / 0.75 / 0.61 — close for formality and implicature, but the informativeness gap (0.75 vs. 0.84) reflects that informativeness is more syntactically grounded and less lexically predictable than the other two dimensions.

---

### 1.3 Structural Comparison

| Property            | Pavlick-Tetreault (2016) | SQUINKY! (Lahiri 2015) |
|---------------------|--------------------------|------------------------|
| N sentences         | 11,274                   | 7,032                  |
| Dimensions          | Formality only           | Formality, Informativeness, Implicature |
| Scale               | −3 to 3 (continuous)     | 1–7 (continuous)       |
| Genre labels        | ✓ (4 domains)            | ✗ (not in CSV release) |
| License             | CC-BY-3.0                | Data license unconfirmed |
| Source stability    | HuggingFace (stable)     | Personal academic server (fragile) |

---

## 2. Preliminary Classifier Results

All models: TF-IDF (max 10,000 features, unigrams + bigrams) + Logistic Regression (max_iter=1000). 80/20 train/test split, stratified, random_state=42.

**Binarization:**
- PT: median split (threshold = 0.0), yielding near-balanced classes (5,657 / 5,617).
- SQUINKY!: score > 4 → positive (formal/informative/implicative), following the `meyersbs/squinky` reference convention.

### Model A — Pavlick-Tetreault, in-domain

| Metric      | Class 0 (informal) | Class 1 (formal) | Macro avg |
|-------------|-------------------|-----------------|-----------|
| Precision   | 0.77              | 0.75            | 0.76      |
| Recall      | 0.75              | 0.77            | 0.76      |
| F1          | 0.76              | 0.76            | **0.76**  |

Confusion matrix: 844 TN / 865 TP / 288 FP / 258 FN. Symmetric — no class bias. The 0.76 ceiling reflects genuine difficulty: four genres with very different vocabularies, and median-split sentences near the decision boundary where surface form is uninformative.

### Model B — SQUINKY!, in-domain (formality)

| Metric      | Class 0 (informal) | Class 1 (formal) | Macro avg |
|-------------|-------------------|-----------------|-----------|
| Precision   | 0.82              | 0.79            | 0.81      |
| Recall      | 0.80              | 0.81            | 0.81      |
| F1          | 0.81              | 0.80            | **0.81**  |

Confusion matrix: 578 TN / 555 TP / 148 FP / 126 FN. Better than PT in-domain, likely because the >4 binarization threshold leaves more margin between classes than the median split, and SQUINKY!'s genre mix (news/blog/forum) provides stronger register contrast than PT's four-genre blend.

### Model C — Cross-corpus generalization

#### C1: Train on PT, test on SQUINKY!

| Metric      | Class 0 (informal) | Class 1 (formal) | Macro avg |
|-------------|-------------------|-----------------|-----------|
| Precision   | 0.90              | 0.73            | 0.82      |
| Recall      | 0.68              | 0.92            | 0.80      |
| F1          | 0.77              | 0.81            | **0.79**  |

#### C2: Train on SQUINKY!, test on PT

| Metric      | Class 0 (informal) | Class 1 (formal) | Macro avg |
|-------------|-------------------|-----------------|-----------|
| Precision   | 0.72              | 0.80            | 0.76      |
| Recall      | 0.84              | 0.67            | 0.75      |
| F1          | 0.77              | 0.73            | **0.75**  |

### Interpretation

**Cross-corpus transfer is surprisingly strong.** The drop from in-domain to cross-corpus is essentially zero in both directions:

- C1 macro F1 (0.79) *exceeds* Model A in-domain (0.76)
- C2 macro F1 (0.75) matches Model A in-domain (0.76)

This is the headline result: a TF-IDF surface-form classifier transfers across corpora with no meaningful degradation.

**Recall asymmetry is consistent across both directions.** In C1, informal recall is high (0.92) but formal recall is lower (0.68). In C2, the pattern partially inverts but informal recall remains stronger (0.84 vs. 0.67 for formal). This suggests informal vocabulary (forum/answers text: "lol", "thanks", question particles, contractions) is more distinctively lexical and therefore more transferable than formal register markers, which are subtler and more corpus-specific.

**Caution:** the strong generalization may be partly driven by genre-extreme sentences (newswire vs. forum/answers) that are easily separable in both corpora. A genre-stratified breakdown on PT — evaluating C2 predictions separately on news, blog, email, answers — would test whether generalization is uniform or masks failures in mid-register subsets.

---

## 3. Open Questions for Design Discussion

1. **What's driving the cross-corpus transfer — register extremes or genuine generalization?**  
   Both corpora contain strongly contrasting genres (formal newswire vs. informal forum/answers). The aggregate F1 may be flattering if the model succeeds on extremes and fails on mid-register sentences, which just don't move the aggregate number much. A genre-stratified evaluation on PT (which has domain labels) would test this. Decision: is the cross-corpus framing already validated by these numbers, or does this check need to happen first?

2. **Binarization strategy: median split vs. fixed threshold vs. continuous framing.**  
   Model A uses a median split (threshold = 0.0 for PT); Model B uses >4 (reference convention for SQUINKY!). These are not equivalent choices — the median split guarantees class balance regardless of score distribution, while >4 makes a substantive claim about what "formal" means. For the final design: should both corpora use a consistent, theoretically motivated threshold? Or should the project move to a regression framing throughout, avoiding the binarization decision entirely?

3. **Informativeness: co-target, control variable, or confound to report?**  
   Formality and informativeness correlate at r = 0.77. A formality classifier will partially learn informativeness signal and vice versa. Decision: should informativeness be included as a joint classification target, used as a feature, controlled for in the error analysis, or simply reported as a known confound with appropriate caveats?

4. **Implicature: in or out of the main pipeline?**  
   Low variance (std = 0.67), confirmed central tendency bias in annotations, near-zero correlation with the other two dimensions, and a reference classifier F1 of only 0.60 all suggest implicature is not a viable primary classification target. Decision: exclude from the main classifier pipeline and treat as a qualitative bonus analysis, or frame it explicitly as an upper-bound question ("what can surface form detect about implicature at all?")?

5. **Genre information for SQUINKY!: worth pursuing before the final design?**  
   Genre labels are not available in `mturk_merged.csv` and cannot be recovered by ID-range inference. The paper references a Google Drive link that may contain a richer version of the data. Decision: is it worth investigating this source before committing to a final cross-corpus architecture, or is operating without genre stratification on SQUINKY! acceptable given that PT provides the genre-labeled half of the cross-corpus pair?

6. **Transformer baseline: how much headroom is realistically available?**  
   TF-IDF + LogReg reaches 0.76–0.81 in-domain. The reference implementation's linguistic feature pipeline (POS + lemmas + chunks) reaches 0.82–0.84 on SQUINKY!. Before committing to transformer fine-tuning, it is worth asking whether the remaining error is primarily lexical (suggesting embeddings would help significantly) or structural/pragmatic (suggesting a harder ceiling regardless of model). This affects how to frame the transformer stage and what improvement to claim as a contribution.

7. **Cross-corpus architecture: harmonization vs. domain adaptation framing.**  
   The strong baseline transfer is encouraging, but the scale mismatch (−3..3 vs. 1..7) and binarization asymmetry mean the current setup is not a clean test. For the final design: should cross-corpus generalization be tested on raw binarized labels (current approach), on z-score normalized continuous scores, or via a domain adaptation method? The answer shapes the project's methodological contribution.

---

## 4. Data & Reproducibility Notes

- Raw data is gitignored (`data/raw/`). The notebook downloads both datasets at runtime.
- SQUINKY! download requires `verify=False` due to a server-side SSL certificate issue at `people.rc.rit.edu`. This is documented inline.
- The `meyersbs/squinky` reference classifier uses NLP features (POS, lemmas, stems, chunks), not TF-IDF. The reported F1 scores (0.82/0.84/0.60) are not directly comparable to our TF-IDF baseline — they use a 25% test split with no fixed random seed.
- `mturk_merged.csv` contains ratings merged across both MTurk annotation rounds, while the paper's statistics are computed on the second round only. This explains small discrepancies between our computed Spearman correlations and those in Lahiri (2015) Table 3.
