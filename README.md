# Real-Time Toxic Comment Filter with Explainability

A real-time toxicity detection system that classifies comments as **Toxic**, **Non-toxic**, or **Needs human review**, while highlighting the words that contribute to the model's decision.

Built for the **ValuePitch AI Hackathon** using the Jigsaw Unintended Bias in Toxicity Classification dataset.

## Features

* Real-time toxic comment classification
* Word-level explanations for model predictions
* Detection of obfuscated and misspelled abusive words
* Word and character TF-IDF features
* Bias-aware training to reduce false positives on identity-related comments
* Human-review band for uncertain predictions
* Low-latency predictions suitable for real-time moderation
* Streamlit interface for interactive demonstration

## Approach

### 1. Preprocessing

The text preprocessing pipeline performs:

* Lowercasing
* URL normalization
* Leetspeak normalization inside words
* Repeated-character reduction
* Removal of unnecessary special characters

Examples:

```text
id1ot     → idiot
stooopid  → stoopid
```

This helps the model handle common forms of obfuscated abusive language.

### 2. Feature Extraction

The model combines two TF-IDF representations:

* **Word n-grams (1–2):** capture individual toxic words and short toxic phrases.
* **Character n-grams (2–5):** capture misspellings, obfuscation, and variations such as `stooopid` and `id1ot`.

### 3. Classification

A **Logistic Regression** classifier is trained on the combined TF-IDF features.

The model uses:

* Balanced class weights
* Tuned decision threshold
* Sample weighting for bias mitigation

Logistic Regression was selected because it provides a strong balance between **accuracy, speed, and interpretability**.

### 4. Explainability

The application uses **word occlusion** to estimate which words drive the prediction.

Each word is removed from the comment and the model is evaluated again. The change in the model's decision score is used as the word's impact.

Words with stronger positive impact are highlighted more prominently in the interface.

### 5. Human Review Band

Instead of forcing every prediction into Toxic or Non-toxic, comments close to the classification threshold are labelled:

```text
Needs human review
```

This reduces overconfident decisions on ambiguous comments.

### 6. Bias Mitigation

Non-toxic comments that mention protected identity groups are given higher training weight.

The final model uses a **3x weight** for non-toxic identity-mention comments.

The goal is to reduce false positives caused by identity-related words while maintaining overall toxicity detection performance.

## Results

### Baseline — W=1.0

Training configuration:

* 120,000 comments
* 10% held out for testing
* Word + character TF-IDF
* Logistic Regression
* Balanced class weights

| Metric                               |      Result |
| ------------------------------------ | ----------: |
| Test F1                              |   **0.584** |
| Validation F1                        |   **0.597** |
| Overall AUC                          |    **0.91** |
| Tuned threshold                      |    **0.60** |
| Median prediction latency            | **0.98 ms** |
| Identity-mention non-toxic FPR       |   **10.5%** |
| Other non-toxic FPR                  |    **3.1%** |
| Automatically decided                |   **96.1%** |
| F1 on automatically decided comments |   **0.618** |

### Final bias-aware model

The final model is trained with increased weight on non-toxic identity-mention comments.

Results are recorded in:

```text
metrics_w3.0.json
```

The final model is selected based on the trade-off between overall F1, identity-related false positives, and real-time latency.

## Edge Cases

The application is tested against several challenging cases:

### Obfuscated abuse

```text
what a stooopid take, you f*cking id1ot
```

The character n-gram features help detect obfuscated toxic language.

### Misspelled but non-toxic text

```text
acident happenad
```

Misspelling alone should **not** make a comment toxic.

### Identity-related non-toxic text

```text
I am a proud Muslim woman.
```

The bias-aware training approach is designed to reduce false positives on comments like these.

### Sarcasm

```text
Oh great, another genius idea from the expert.
```

Sarcasm remains challenging because the model primarily relies on lexical and character-level features rather than deep contextual understanding.

See `edge_cases_results.md` for additional test cases.

## Limitations

* The model is trained on a subset of the full dataset because of local memory constraints.
* Sarcasm and context-dependent language remain challenging.
* Bias mitigation reduces false positives but does not eliminate bias.
* Toxicity probabilities are model scores and should not be treated as absolute truth.
* Character and word TF-IDF models have limited understanding of long-range context.

## Future Improvements

* Fine-tune a lightweight transformer such as DistilBERT
* Calibrate predicted probabilities
* Add richer contextual features
* Improve sarcasm detection
* Evaluate performance across additional demographic subgroups
* Add production monitoring and model drift detection

## Project Structure

```text
Real-Time-Toxic-Comment-Filter-with-Explainability/
│
├── data/
│   └── train.csv
│
├── src/
│   ├── __init__.py
│   ├── train.py
│   ├── text.py
│   └── app.py
│
├── predict.py
├── edge_cases.py
├── edge_cases_results.md
├── model.joblib
├── metrics_w1.0.json
├── metrics_w3.0.json
├── requirements.txt
├── README.md
└── .gitignore
```

## Run Locally

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add the Jigsaw dataset

Place the dataset at:

```text
data/train.csv
```

The training file should contain the `comment_text`, `target`, and identity-related columns used by the training pipeline.

### 3. Train the baseline

```bash
python -m src.train 1.0
```

### 4. Train the final bias-aware model

```bash
python -m src.train 3.0
```

This generates:

```text
model.joblib
```

### 5. Run the Streamlit application

From the project root:

```bash
streamlit run src/app.py
```

If required on Windows:

```powershell
$env:PYTHONPATH = (Get-Location).Path
streamlit run src/app.py
```

## Score a CSV

To generate predictions for a CSV file:

```bash
python predict.py test.csv comment_text predictions.csv
```

The output contains the toxicity probability and classification.

## Technologies

* Python
* scikit-learn
* TF-IDF
* Logistic Regression
* Pandas
* NumPy
* Streamlit
* Joblib

## Hackathon Focus

This project focuses on three practical requirements of real-time content moderation:

1. **Fast prediction** — suitable for interactive moderation.
2. **Explainability** — shows which words influenced the decision.
3. **Bias awareness** — specifically evaluates and reduces false positives on identity-related non-toxic comments.
