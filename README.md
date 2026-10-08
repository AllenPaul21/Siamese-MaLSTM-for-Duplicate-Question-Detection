# Duplicate Question Detection with Siamese MaLSTM

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20-orange.svg)](https://www.tensorflow.org/)
[![NLP](https://img.shields.io/badge/machine--learning-NLP-red.svg)](https://scikit-learn.org/)

## Overview

This project detects whether two questions have the same meaning. For example,
“How can I learn Python?” and “What is the best way to learn Python?” may be
different sentences but ask the same thing.

It implements a **Siamese MaLSTM** model. Two questions pass through the same
LSTM network, their meaning vectors are compared, and the model returns a
duplicate probability from 0 to 1. Three pretrained word embeddings are tested
and their predictions are also combined in an ensemble.

## Research paper

The reference paper is [`ATML_RP.pdf`](../ATML_RP.pdf). It uses a Siamese
Manhattan LSTM for duplicate-question classification and reports 91.14%
ensemble accuracy. The paper uses different experimental conditions, so that
number is a reference benchmark, not our reproduced test result.

## Dataset

The project uses [`Dataset_train.csv`](../Dataset_train.csv), containing:

| Column | Meaning |
|---|---|
| `question1` | First question |
| `question2` | Second question |
| `is_duplicate` | `1` for duplicate, `0` for different |

The split is 75% training, 10% of the training portion for validation, and 25%
final test. It is stratified with seed `42`. The saved original experiment has
101,073 test pairs.

## Word embeddings

Three separate models use 300-dimensional pretrained vectors:

1. Google News Word2Vec
2. FastText Crawl
3. FastText Subword

The large embedding files are in [`../embeddings/`](../embeddings/).

## Model pipeline

```text
Question pairs → cleaning → tokenization/padding to 20 words
→ 300d embeddings → shared Siamese LSTM
→ Manhattan similarity → duplicate probability
```

The similarity is calculated as:

```text
exp(-sum(abs(question1_vector - question2_vector)))
```

The original decision rule is: score `> 0.5` means duplicate.

### Main settings

| Setting | Value |
|---|---:|
| Maximum length | 20 tokens |
| Embedding size | 300 |
| LSTM units | 50 |
| Optimizer | Adam, learning rate 0.001 |
| Loss | Binary cross-entropy |
| Batch size | 64 |
| Epochs | 30 |
| Seed | 42 |

Text is lowercased, alphabetic words are kept, stopwords are removed, Porter
stemming is applied, and sequences are right-padded. The tokenizer is fitted
only on training data.

## Ensemble methods

The original ensemble is:

```text
33% Google + 33% FastText + 34% FastText Subword
```

The project also saves threshold tuning, weight tuning, calibrated blending
(sigmoid and isotonic), and logistic-regression stacking. The validation-
selected method is isotonic blending with weights `0.30 / 0.45 / 0.25` and
threshold `0.48`.

## Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| Google News | 81.11% | 77.12% | 69.45% | 73.08% |
| FastText | 81.59% | 77.08% | 71.35% | 74.11% |
| FastText Subword | 80.31% | 76.72% | 67.01% | 71.54% |
| Original ensemble | 81.81% | 78.97% | 69.16% | 73.74% |
| Selected isotonic blend | 82.35% | 75.88% | 76.53% | 76.20% |

Stacking reached 82.36% accuracy in the saved comparison, but isotonic
blending was selected using validation data.

## Plots

The repository-level `output` folder contains the generated comparisons:

![Paper vs our results](../output/paper_vs_our_results.png)

![Confusion matrices](../output/confusion_matrices.png)

![Calibration curves](../output/calibration_curves.png)

![Ensemble improvement](../output/ensemble_improvement_comparison.png)

## Repository structure

```text
prject_final/
├── ATML_RP.pdf                         # Reference paper
├── Dataset_train.csv                   # Dataset
├── embeddings/                         # Pretrained vectors
├── output/                             # Generated plots
└── DuplicateQuestion_MaLSTM/
    ├── README.md
    ├── main.py                          # Command-line commands
    ├── model.py                         # Cleaning, model, training, evaluation
    ├── ensemble.py                      # Blending and tuning
    ├── requirements.txt
    ├── data/quora_question_pairs.csv    # Link to dataset
    ├── embeddings/                      # Link to vectors
    └── results/
        ├── summary.csv
        ├── original/                     # Original models and predictions
        ├── tuned/                        # Tuned ensemble outputs
        └── preprocessing/               # Preserved-text validation run
```

`details.zip` files contain extra reports, plots, histories, and arrays.

## Installation

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r DuplicateQuestion_MaLSTM/requirements.txt
python -m nltk.downloader stopwords
```

## View and verify saved results

```bash
.venv/bin/python DuplicateQuestion_MaLSTM/main.py
.venv/bin/python DuplicateQuestion_MaLSTM/main.py --help
.venv/bin/python DuplicateQuestion_MaLSTM/main.py verify
```

The first command prints the saved result table. `verify` checks saved model
outputs and ensemble predictions without retraining or changing anything.

## Train a new experiment

Use a new directory so existing results are preserved:

```bash
.venv/bin/python DuplicateQuestion_MaLSTM/main.py prepare \
  --run-dir DuplicateQuestion_MaLSTM/results/new_run
.venv/bin/python DuplicateQuestion_MaLSTM/main.py train --model all \
  --run-dir DuplicateQuestion_MaLSTM/results/new_run
.venv/bin/python DuplicateQuestion_MaLSTM/main.py blend \
  --run-dir DuplicateQuestion_MaLSTM/results/new_run
```

For ensemble tuning:

```bash
.venv/bin/python DuplicateQuestion_MaLSTM/main.py tune \
  --output-dir DuplicateQuestion_MaLSTM/results/new_tuning
```

## Important notes

- Existing saved models and results were preserved; they were not retrained.
- The paper's 91.14% is a reference value, not our reproduced test accuracy.
- Tuned ensemble methods are exploratory validation-based extensions.
- Large embeddings are needed for new training, but not for viewing saved
  tables and plots.

## License

No license file is currently included. Add one before formally publishing or
redistributing the project.
