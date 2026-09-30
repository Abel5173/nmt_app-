# nmt_models.py - Basic Seq2Seq+LSTM and Attention Seq2Seq+LSTM (+ inference helpers)
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nmt_text import PAD_ID, BOS_ID, EOS_ID, normalize_en, encode_ids


def _lstm_dropout(n_layers, dropout):
    return dropout if n_layers > 1 else 0.0


# ----------------------------------------------------------------------------
# 1) BASIC SEQ2SEQ + LSTM  (no attention: the whole source is squeezed into (h, c))
# ----------------------------------------------------------------------------
class BaselineEncoder(nn.Module):
    def __init__(self, vocab, emb, hid, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True,
                           dropout=_lstm_dropout(n_layers, dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, c) = self.rnn(packed)
        return h, c


class BaselineDecoder(nn.Module):
    def __init__(self, vocab, emb, hid, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True,
                           dropout=_lstm_dropout(n_layers, dropout))
        self.fc_out = nn.Linear(hid, vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tok, h, c):
        x = self.dropout(self.embedding(tok)).unsqueeze(1)
        out, (h, c) = self.rnn(x, (h, c))
        return self.fc_out(out.squeeze(1)), h, c


class BaselineSeq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tgt_vocab = cfg["tgt_vocab"]
        self.encoder = BaselineEncoder(cfg["src_vocab"], cfg["emb_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"])
        self.decoder = BaselineDecoder(cfg["tgt_vocab"], cfg["emb_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"])

    def forward(self, src, src_len, tgt, teacher_forcing=1.0):
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.tgt_vocab, device=src.device)
        h, c = self.encoder(src, src_len)
        tok = tgt[:, 0]
        for t in range(1, T):
            logits, h, c = self.decoder(tok, h, c)
            outputs[:, t] = logits
            tok = tgt[:, t] if random.random() < teacher_forcing else logits.argmax(1)
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=80):
        self.eval()
        B = src.size(0)
        h, c = self.encoder(src, src_len)
        tok = torch.full((B,), BOS_ID, dtype=torch.long, device=src.device)
        out = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logits, h, c = self.decoder(tok, h, c)
            logits[:, PAD_ID] = -1e9
            logits[:, BOS_ID] = -1e9
            tok = logits.argmax(1).masked_fill(finished, PAD_ID)
            out[:, t] = tok
            finished |= tok == EOS_ID
            if finished.all():
                break
        return out, None


# ----------------------------------------------------------------------------
# 2) ATTENTION SEQ2SEQ + LSTM  (BiLSTM encoder + Bahdanau attention + LSTM decoder)
# ----------------------------------------------------------------------------
class AttnEncoder(nn.Module):
    def __init__(self, vocab, emb, hid, dec_hid, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.rnn = nn.LSTM(emb, hid, n_layers, batch_first=True, bidirectional=True,
                           dropout=_lstm_dropout(n_layers, dropout))
        self.fc_h = nn.Linear(hid * 2, dec_hid)
        self.fc_c = nn.Linear(hid * 2, dec_hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        x = self.dropout(self.embedding(src))
        packed = pack_padded_sequence(x, src_len.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, (h, c) = self.rnn(packed)
        outputs, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=src.size(1))
        h0 = torch.tanh(self.fc_h(torch.cat((h[-2], h[-1]), dim=1)))
        c0 = torch.tanh(self.fc_c(torch.cat((c[-2], c[-1]), dim=1)))
        return outputs, h0, c0


class BahdanauAttention(nn.Module):
    def __init__(self, enc_dim, dec_dim):
        super().__init__()
        self.attn = nn.Linear(enc_dim + dec_dim, dec_dim)
        self.v = nn.Linear(dec_dim, 1, bias=False)

    def forward(self, dec_h, enc_out, mask):
        S = enc_out.size(1)
        dec_h = dec_h.unsqueeze(1).expand(-1, S, -1)
        energy = torch.tanh(self.attn(torch.cat((dec_h, enc_out), dim=2)))
        scores = self.v(energy).squeeze(2).masked_fill(~mask, -1e9)
        return F.softmax(scores, dim=1)


class AttnDecoder(nn.Module):
    def __init__(self, vocab, emb, enc_hid, dec_hid, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=PAD_ID)
        self.attention = BahdanauAttention(enc_hid * 2, dec_hid)
        self.rnn = nn.LSTM(enc_hid * 2 + emb, dec_hid, n_layers, batch_first=True,
                           dropout=_lstm_dropout(n_layers, dropout))
        self.fc_out = nn.Linear(dec_hid + enc_hid * 2 + emb, vocab)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tok, h, c, enc_out, mask):
        x = self.dropout(self.embedding(tok)).unsqueeze(1)
        a = self.attention(h[-1], enc_out, mask)
        ctx = torch.bmm(a.unsqueeze(1), enc_out)
        out, (h, c) = self.rnn(torch.cat((x, ctx), dim=2), (h, c))
        logits = self.fc_out(torch.cat((out.squeeze(1), ctx.squeeze(1), x.squeeze(1)), dim=1))
        return logits, h, c, a


class AttentionSeq2Seq(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tgt_vocab = cfg["tgt_vocab"]
        self.n_layers = cfg["n_layers"]
        self.encoder = AttnEncoder(cfg["src_vocab"], cfg["emb_dim"], cfg["hid_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"])
        self.decoder = AttnDecoder(cfg["tgt_vocab"], cfg["emb_dim"], cfg["hid_dim"], cfg["hid_dim"], cfg["n_layers"], cfg["dropout"])

    def _init_state(self, h0, c0):
        return (h0.unsqueeze(0).repeat(self.n_layers, 1, 1).contiguous(),
                c0.unsqueeze(0).repeat(self.n_layers, 1, 1).contiguous())

    def forward(self, src, src_len, tgt, teacher_forcing=1.0):
        B, T = tgt.shape
        outputs = torch.zeros(B, T, self.tgt_vocab, device=src.device)
        enc_out, h0, c0 = self.encoder(src, src_len)
        h, c = self._init_state(h0, c0)
        mask = src != PAD_ID
        tok = tgt[:, 0]
        for t in range(1, T):
            logits, h, c, _ = self.decoder(tok, h, c, enc_out, mask)
            outputs[:, t] = logits
            tok = tgt[:, t] if random.random() < teacher_forcing else logits.argmax(1)
        return outputs

    @torch.no_grad()
    def greedy_decode(self, src, src_len, max_len=80):
        self.eval()
        B, S = src.shape
        enc_out, h0, c0 = self.encoder(src, src_len)
        h, c = self._init_state(h0, c0)
        mask = src != PAD_ID
        tok = torch.full((B,), BOS_ID, dtype=torch.long, device=src.device)
        out = torch.full((B, max_len), PAD_ID, dtype=torch.long, device=src.device)
        attn = torch.zeros(B, max_len, S, device=src.device)
        finished = torch.zeros(B, dtype=torch.bool, device=src.device)
        for t in range(max_len):
            logits, h, c, a = self.decoder(tok, h, c, enc_out, mask)
            attn[:, t] = a
            logits[:, PAD_ID] = -1e9
            logits[:, BOS_ID] = -1e9
            tok = logits.argmax(1).masked_fill(finished, PAD_ID)
            out[:, t] = tok
            finished |= tok == EOS_ID
            if finished.all():
                break
        return out, attn


# ----------------------------------------------------------------------------
# 3) Build / save / load
# ----------------------------------------------------------------------------
MODEL_TYPES = {"baseline": BaselineSeq2Seq, "attention": AttentionSeq2Seq}


def build_model(model_type, cfg):
    return MODEL_TYPES[model_type](cfg)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, path, model_type, extra=None):
    torch.save({"model_type": model_type, "cfg": model.cfg,
                "state_dict": model.state_dict(), "extra": extra or {}}, path)


def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["model_type"], ckpt["cfg"]).to(device)
    model.load_state_dict(ckpt["state_dict"])   # strict: fails loudly if architecture differs
    model.eval()
    return model


# ----------------------------------------------------------------------------
# 4) Inference pipeline: raw English text -> Amharic text (+ attention)
# ----------------------------------------------------------------------------
@torch.no_grad()
def translate(model, sp_src, sp_tgt, text, device, max_len=80, src_max_len=64):
    text = normalize_en(text)                                  # 1. preprocess
    ids = encode_ids(sp_src, text, src_max_len)                # 2. tokenise
    src = torch.tensor([ids], dtype=torch.long, device=device)
    src_len = torch.tensor([len(ids)])
    tokens, attn = model.greedy_decode(src, src_len, max_len)  # 3. decode
    gen, ended = [], False
    for i in tokens[0].tolist():
        if i == EOS_ID:
            ended = True
            break
        if i == PAD_ID:
            break
        gen.append(i)
    n_steps = len(gen) + (1 if ended else 0)
    tgt_pieces = [sp_tgt.id_to_piece(i) for i in gen] + (["</s>"] if ended else [])
    src_pieces = ["<s>"] + sp_src.encode(text, out_type=str)[: src_max_len - 2] + ["</s>"]
    attention = attn[0, :n_steps, :len(ids)].cpu().numpy() if attn is not None else None
    return {"normalized_input": text, "translation": sp_tgt.decode(gen),
            "src_pieces": src_pieces, "tgt_pieces": tgt_pieces, "attention": attention}


# ----------------------------------------------------------------------------
# 5) Plot helpers (Ethiopic font + attention heat-map)
# ----------------------------------------------------------------------------
def setup_amharic_font():
    """Find an installed Ethiopic font (Noto / Abyssinica) and make matplotlib use it."""
    import glob
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for pat in ["NotoSansEthiopic*", "NotoSerifEthiopic*", "AbyssinicaSIL*", "*Ethiopic*", "*Abyssinica*"]:
        files = glob.glob("/usr/share/fonts/**/" + pat + ".ttf", recursive=True)
        if files:
            fm.fontManager.addfont(files[0])
            name = fm.FontProperties(fname=files[0]).get_name()
            plt.rcParams["font.family"] = [name, "DejaVu Sans"]
            return name
    return None


def plot_attention(attention, src_pieces, tgt_pieces, title=None, save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(max(6, 0.55 * len(src_pieces)), max(4, 0.45 * len(tgt_pieces))))
    im = ax.imshow(attention, cmap="Blues", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(src_pieces)))
    ax.set_xticklabels(src_pieces, rotation=90)
    ax.set_yticks(range(len(tgt_pieces)))
    ax.set_yticklabels(tgt_pieces)
    ax.set_xlabel("Source (English) tokens")
    ax.set_ylabel("Generated (Amharic) tokens")
    if title:
        ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
