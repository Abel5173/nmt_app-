import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sentencepiece as spm
import streamlit as st
import torch

BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
from nmt_models import load_checkpoint, plot_attention, setup_amharic_font, translate  # noqa: E402

st.set_page_config(page_title="English to Amharic NMT", page_icon="🌍", layout="wide")

MODEL_FILES = {
    "Attention Seq2Seq + LSTM": "models/attention_lstm.pt",
    "Basic Seq2Seq + LSTM": "models/baseline_lstm.pt",
}
EXAMPLES = [
    "I am going to the university.",
    "Where is the hospital?",
    "Thank you very much for your help.",
    "The children are playing in the garden.",
    "What time does the meeting start tomorrow?",
]


@st.cache_resource(show_spinner="Loading models and tokenizers ...")
def load_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sp_en = spm.SentencePieceProcessor(model_file=os.path.join(BASE, "tokenizers", "tokenizer_en.model"))
    sp_am = spm.SentencePieceProcessor(model_file=os.path.join(BASE, "tokenizers", "tokenizer_am.model"))
    models = {name: load_checkpoint(os.path.join(BASE, path), device) for name, path in MODEL_FILES.items()}
    setup_amharic_font()
    return device, sp_en, sp_am, models


st.title("English → Amharic Neural Machine Translation")
st.caption("LSTM encoder-decoder models trained from scratch: Basic Seq2Seq vs. Attention-based Seq2Seq.")

device, sp_en, sp_am, models = load_system()

with st.sidebar:
    st.header("Settings")
    model_name = st.selectbox("Model", list(MODEL_FILES))
    example = st.selectbox("Try an example", ["(type your own)"] + EXAMPLES)
    show_att = st.checkbox("Show attention heat-map", value=True)
    st.markdown(f"**Device:** `{device}`")

default_text = EXAMPLES[0] if example == "(type your own)" else example
text = st.text_area("English sentence", value=default_text, height=110, max_chars=500)

if st.button("Translate", type="primary"):
    if not text.strip():
        st.warning("Please enter a sentence.")
    elif len(text.split()) > 60:
        st.warning("Please use at most 60 words (the models were trained on sentences of up to 40 words).")
    else:
        t0 = time.time()
        res = translate(models[model_name], sp_en, sp_am, text, device)
        elapsed = (time.time() - t0) * 1000

        st.subheader("Amharic translation")
        st.markdown(f"### {res['translation'] or '(empty output)'}")
        st.caption(f"Model: {model_name}  |  {elapsed:.0f} ms  |  preprocessed input: “{res['normalized_input']}”")

        with st.expander("Tokens (SentencePiece)"):
            st.write("**Source:**", " ".join(res["src_pieces"]))
            st.write("**Output:**", " ".join(res["tgt_pieces"]))

        if show_att:
            if res["attention"] is not None:
                st.subheader("Attention visualization")
                st.write("Each row is a generated Amharic token; darker cells show the English tokens it focused on.")
                fig = plot_attention(res["attention"], res["src_pieces"], res["tgt_pieces"])
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("The basic Seq2Seq model has no attention mechanism. Select the attention model to see the heat-map.")
