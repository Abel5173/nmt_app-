import os
import sys

import sentencepiece as spm
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nmt_models import load_checkpoint, translate  # noqa: E402

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sp_en = spm.SentencePieceProcessor(model_file=os.path.join(BASE, "tokenizers", "tokenizer_en.model"))
sp_am = spm.SentencePieceProcessor(model_file=os.path.join(BASE, "tokenizers", "tokenizer_am.model"))
MODEL = load_checkpoint(os.path.join(BASE, "models", "attention_lstm.pt"), device)

app = FastAPI(title="English-Amharic NMT API", version="1.0")


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(req: TranslateRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' must not be empty")
    if len(text.split()) > 100:
        raise HTTPException(status_code=400, detail="'text' must be at most 100 words")
    result = translate(MODEL, sp_en, sp_am, text, device)
    return {"translation": result["translation"]}
