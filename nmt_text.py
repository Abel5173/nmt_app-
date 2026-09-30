# nmt_text.py - text normalisation + token helpers shared by training AND deployment
import re
import unicodedata

PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

_QUOTES = {"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"}
_INVISIBLE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")


def _base_clean(text):
    text = unicodedata.normalize("NFKC", str(text))
    text = _INVISIBLE.sub("", text)
    for k, v in _QUOTES.items():
        text = text.replace(k, v)
    return text


def normalize_en(text):
    """Unicode NFKC, remove invisible chars, straighten quotes, collapse whitespace."""
    return re.sub(r"\s+", " ", _base_clean(text)).strip()


def normalize_am(text):
    """Same as English + Ethiopic word-space (U+1361) -> space, double word-space -> full stop."""
    text = _base_clean(text)
    text = text.replace("\u1361\u1361", "\u1362")
    text = text.replace("\u1361", " ")
    return re.sub(r"\s+", " ", text).strip()


def ethiopic_ratio(text):
    """Share of alphabetic characters that are in the Ethiopic Unicode block."""
    letters = [ch for ch in str(text) if ch.isalpha()]
    if not letters:
        return 0.0
    return sum("\u1200" <= ch <= "\u137f" for ch in letters) / len(letters)


def encode_ids(sp, text, max_len):
    """[BOS] + sentencepiece ids (truncated) + [EOS]"""
    ids = sp.encode(text)[: max_len - 2]
    return [BOS_ID] + ids + [EOS_ID]


def ids_to_text(sp, ids):
    """Cut at EOS, drop PAD/BOS, detokenise with sentencepiece."""
    keep = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i in (PAD_ID, BOS_ID):
            continue
        keep.append(i)
    return sp.decode(keep)
