# English → Amharic Neural Machine Translation

Two LSTM encoder–decoder models trained from scratch on the OPUS/MT560 English–Amharic corpus:

1. **Basic Seq2Seq + LSTM** (`models/baseline_lstm.pt`)
2. **Attention-based Seq2Seq + LSTM** (`models/attention_lstm.pt`, Bahdanau attention, bi-directional encoder)

This folder is a self-contained deployment: model weights, SentencePiece tokenizers, inference code, a Streamlit
web app and a FastAPI service.

## Installation

```bash
python -m venv venv && source venv/bin/activate      # optional
pip install -r requirements.txt
```

For correct display of Amharic letters in the attention plot on Linux, install an Ethiopic font, e.g.
`sudo apt-get install fonts-noto-core` (the translation itself works without it).

## Run the web app (Streamlit)

```bash
streamlit run app.py
```
Open http://localhost:8501, type an English sentence and press **Translate**.

## Run the REST API (FastAPI)

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/translate \
     -H "Content-Type: application/json" \
     -d '{"text": "I am going to the university."}'
# -> {"translation": "..."}
```
Interactive docs: http://localhost:8000/docs · Health check: `GET /health`

## Inference pipeline

`raw English text → normalisation (NFKC, whitespace, quotes) → SentencePiece tokenisation (<s> … </s>) →
encoder → decoder (greedy search, with attention for the attention model) → SentencePiece detokenisation → Amharic text`

## Files

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface |
| `api.py` | FastAPI service (`POST /translate`) |
| `nmt_models.py` | model classes, checkpoint loading, `translate()` |
| `nmt_text.py` | text normalisation and token helpers |
| `models/` | trained checkpoints (weights + architecture config) |
| `tokenizers/` | SentencePiece models for English and Amharic |

## Limitations

The models are small LSTM systems trained on ~100k noisy sentence pairs; translations are often imperfect,
especially for long sentences, rare words and names. Greedy decoding is used.
