# Adaptive Query Reformation

This project builds an end-to-end NLP pipeline that takes a user query, checks if it is **strong** or **weak**, and then decides how much help the query needs.

- **Weak query** → fetch related terms from Google Trends and rewrite the query with a T5 model.  
- **Strong query** → show that it is strong and keep it as is (no extra rewriting for now).

Everything runs in a **Streamlit** web app.

```
streamlit run app.py
```
or 
```
python3 -m streamlit run app.py
```

---

## 1. Overall flow

1. The user types a query in the Streamlit app (`app.py`).
2. A **DistilRoBERTa classifier** (`infer.py`) labels the query as `WEAK` or `STRONG`.
3. If the query is **WEAK**:
   - `get_context_from_api.py` / `Trends.py` call **SerpApi Google Trends**.
   - The “top related” queries become context keywords.
   - These keywords are passed to a **T5 query rewriter**, which generates clearer, more specific versions of the query.
   - The UI shows a red “Weak query” card and the rewritten questions.
4. If the query is **STRONG**:
   - The UI shows a green “Strong query” card.
   - The query is treated as already well-formed (no extra rewriting in the current version).

---

## 2. Project structure

Typical layout of the repo:

```text
adaptive-query-reformation/
│
├── app.py                      # Streamlit app (main entry point)
├── get_context_from_api.py     # Google Trends context using SerpApi
├── Trends.py                   # Core trends helper functions + test harness
│
├── qw_strong_weak_classifier.py # Training script for DistilRoBERTa classifier
├── infer.py                     # Inference for classifier (CLI + used in app)
│
├── query-wellformedness-master/ # QWF dataset (NOT tracked in git)
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   └── README.md
│
├── t5-query-rewriter-final/    # T5 query rewriter model folder
│   └── ...                     # tokenizer + model weights
│
└── output/                     # Saved classifier model (NOT tracked in git)
    └── distilroberta/
        ├── config.json
        ├── tokenizer.json
        ├── model.safetensors
        └── routing_threshold.json
