# Data Provenance & Licensing

Raw data files are downloaded into this folder (gitignored) and are **not redistributed in this repository**. This file documents where they come from and what's required to use them.

## Pavlick & Tetreault (2016) — Online Formality Corpus

- **Source:** https://huggingface.co/datasets/osyvokon/pavlick-formality-scores
- **License:** CC-BY-3.0
- **Citation (cite both):**

  ```bibtex
  @article{PavlickAndTetreault-2016:TACL,
    author    = {Ellie Pavlick and Joel Tetreault},
    title     = {An Empirical Analysis of Formality in Online Communication},
    journal   = {Transactions of the Association for Computational Linguistics},
    year      = {2016},
    publisher = {Association for Computational Linguistics}
  }

  @article{Lahiri-2015:arXiv,
    title   = {{SQUINKY! A} Corpus of Sentence-level Formality, Informativeness, and Implicature},
    author  = {Lahiri, Shibamouli},
    journal = {arXiv preprint arXiv:1506.02306},
    year    = {2015}
  }
  ```

- News and blog portions were originally collected by Lahiri; email and answers portions were collected by Pavlick & Tetreault. The HuggingFace version was detokenized by Oleksiy Syvokon (the `answers` and `email` subsets were originally tokenized).

## SQUINKY! (Lahiri 2015)

- **Paper:** https://arxiv.org/abs/1506.02306
- **Reference implementation / wrapper code:** https://github.com/meyersbs/squinky — MIT licensed (code only)
- **Raw data file:** `http://people.rc.rit.edu/~bsm9339/corpora/squinky_corpus/mturk_merged.csv` (per the wrapper repo's `download.py`)
- **Citation:** see the Lahiri (2015) entry above.
- **Licensing note:** the *code* wrapping the pre-trained classifiers is MIT licensed; the underlying annotated corpus carries no explicit license in the repo or paper. Treat it as available for research use (consistent with how it's been used in published work) but **do not assume redistribution rights** until confirmed — the same caution applied to GYAFC.
- **URL fragility:** this is hosted on a personal academic homepage, not a stable archive. If it 404s, check the `meyersbs/squinky` GitHub issues for a mirror, or contact the maintainer before assuming the data is gone.
