import re
import numpy as np
import pandas as pd

_QTY = re.compile(r"(\d+(?:\.\d+)?)\s*(ml|l|g|kg|mg|cm|mm|m)\b", re.I)
_PACK = re.compile(r"(\d+)\s*([x\*]|pack|packs|pcs|pieces|count)\b", re.I)
_IPQ  = re.compile(r"\b(?:IPQ|Item\s*Pack\s*Quantity)[:\s-]*(\d+(?:\.\d+)?)", re.I)
_SCALE = {"ml":1.0,"l":1000.0,"mg":0.001,"g":1.0,"kg":1000.0,"cm":1.0,"mm":0.1,"m":100.0}
_STOP = set(["pack","of","and","for","with","the","a","an"])

def clean(text: str) -> str:
    if not isinstance(text, str): return ""
    s = re.sub(r"<[^>]+>", " ", text)
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_brand(text: str) -> str:
    toks = re.findall(r"[A-Za-z][A-Za-z0-9\-\']+", text.lower())
    for t in toks[:5]:
        if t not in _STOP:
            return t
    return "unknown"

def extract_feats(text: str):
    s = (text or "").lower()
    qty = 0.0
    for m in _QTY.finditer(s):
        try: qty = max(qty, float(m.group(1)) * _SCALE.get(m.group(2).lower(), 1.0))
        except: pass
    pack = 1.0
    m = _PACK.search(s)
    if m:
        try: pack = float(m.group(1))
        except: pack = 1.0
    ipq = 0.0
    m = _IPQ.search(s)
    if m:
        try: ipq = float(m.group(1))
        except: ipq = 0.0
    length = float(len(s))
    digits = float(sum(ch.isdigit() for ch in s))
    return np.array([pack, qty, ipq, length, digits], dtype=np.float32)

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["text"] = df["catalog_content"].astype(str).map(clean)
    feats = np.vstack(df["text"].map(extract_feats).values)
    df[["pack","qty","ipq","len","digits"]] = feats
    df["brand"] = df["text"].map(extract_brand)
    df["qty_bucket"] = np.round(np.log1p(df["qty"]+1e-8), 1)
    return df