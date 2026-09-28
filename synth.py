"""Two synthetic-row generators (SDV stats model, local Ollama LLM) and a distribution comparison."""
import glob, hashlib, json, math, os, time, tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

import numpy as np
import pandas as pd


def split_columns(df):
    """-> (id_col or None, text_cols, cat_cols, num_cols). Text = free-text strings; ID = unique numeric per row."""
    id_col = next((c for c in df if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique() == len(df)), None)
    rest = [c for c in df if c != id_col]
    text = [c for c in rest if df[c].dtype == object and df[c].nunique() > 30]
    cat = [c for c in rest if c not in text and (df[c].dtype == object or df[c].nunique() <= 15)]
    num = [c for c in rest if c not in text and c not in cat]
    return id_col, text, cat, num


def _finish(gen, df, id_col):
    """Match original column order/dtypes; append fresh sequential IDs."""
    gen = gen.reset_index(drop=True)
    if id_col:
        gen[id_col] = df[id_col].max() + 1 + np.arange(len(gen))
    for c in df:
        gen[c] = gen[c].fillna("").astype(str) if df[c].dtype == object else pd.to_numeric(gen[c], errors="coerce")
    return gen[list(df.columns)]


# ---------- stats method ----------
MODEL_DIR = os.environ.get("MODEL_DIR", "/models")


def model_path(name, df, kind):
    """Saved-model file keyed on upload file name + data hash (same name, changed data -> refit)."""
    stem = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in os.path.splitext(name)[0])[:80]
    h = hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:8]
    return os.path.join(MODEL_DIR, f"{stem}__{kind}__{h}.pkl")


def gen_sdv(df, n, kind="GaussianCopula", seed=0, name="data.parquet"):
    """Returns (df, {'reused': bool, 'model': path}). Fitted model is saved/loaded via model_path()."""
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer

    cls = CTGANSynthesizer if kind == "CTGAN" else GaussianCopulaSynthesizer
    id_col, text, cat, num = split_columns(df)
    path = model_path(name, df, kind)
    reused = os.path.exists(path)
    if reused:
        synth = cls.load(path)
    else:
        train = df.drop(columns=[id_col] if id_col else [])
        md = SingleTableMetadata()
        md.detect_from_dataframe(train)
        for c in text + cat:
            md.update_column(c, sdtype="categorical")
        for c in num:
            md.update_column(c, sdtype="numerical")
        # ponytail: CTGAN runs on CPU with 100 epochs; raise epochs / use a CUDA torch image if quality matters
        synth = CTGANSynthesizer(md, epochs=100) if kind == "CTGAN" else GaussianCopulaSynthesizer(md)
        np.random.seed(seed)
        synth.fit(train)
        os.makedirs(MODEL_DIR, exist_ok=True)
        synth.save(path)
    np.random.seed(seed)
    return _finish(synth.sample(num_rows=n), df, id_col), {"reused": reused, "model": path}


# ---------- LLM method ----------
CLAUDE_MODEL = "claude-haiku-4-5"  # cheapest current Claude model; exact id, no date suffix
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.toml")


def _claude_enabled():
    """config.toml [claude] enabled — not a UI setting, deliberately not surfaced in app.py."""
    try:
        with open(_CONFIG_PATH, "rb") as f:
            return bool(tomllib.load(f).get("claude", {}).get("enabled"))
    except FileNotFoundError:
        return False


def _claude_rows(client, prompt, schema):
    """output_config.format guarantees the first text block is JSON matching schema. ANTHROPIC_API_KEY from env."""
    resp = client.messages.create(
        model=CLAUDE_MODEL, max_tokens=8192,
        output_config={"format": {"type": "json_schema", "schema": schema}},
        messages=[{"role": "user", "content": prompt}],
    )
    text = next((b.text for b in resp.content if b.type == "text"), None)
    if text is None:
        raise ValueError("Claude returned no text block")
    return json.loads(text)["rows"]


def _codebook_prompt(cols, cb):
    lines = [f"- {c}: {cb.get(c, {}).get('label', c)[:90]}" for c in cols]
    groups = defaultdict(list)
    for c in cols:
        if cb.get(c, {}).get("values"):
            groups[json.dumps(cb[c]["values"])].append(c)
    for vals, members in groups.items():
        v = json.loads(vals)
        lines.append(f"Allowed codes for {', '.join(members)}: " + "; ".join(f"{k}={s[:40]}" for k, s in v.items()))
    return "\n".join(lines)


def gen_llm(df, cb, n, model="ornith-1.5:9b", batch=15, seed=0, progress=None, free_text=False):
    """Batched few-shot generation. Returns (df, {'rejected': int, 'batches': int})."""
    use_claude = _claude_enabled()
    model = CLAUDE_MODEL if use_claude else model
    if use_claude:
        import anthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("config.toml [claude] enabled = true but ANTHROPIC_API_KEY is not set")
        client = anthropic.Anthropic()
    else:
        import ollama
        client = ollama.Client(host=os.environ.get("OLLAMA_HOST"))
    id_col, text, cat, num = split_columns(df)
    if not free_text:  # free-text answers dominate output tokens; leave them blank unless asked
        cols_text, text = text, []
    else:
        cols_text = []
    cols = [c for c in df if c != id_col and c not in cols_text]
    nullable = {c for c in cols if df[c].isna().any()}

    def codes(c):
        vals = {float(k) for k in cb.get(c, {}).get("values", {})} | set(df[c].dropna().unique())
        return sorted(int(v) if float(v).is_integer() else v for v in vals) + ([None] if c in nullable else [])

    allowed = {c: set(codes(c)) for c in cat}  # 6 == 6.0 in Python, so int/float codes both match
    props = {c: {"type": "string"} if c in text else {"enum": codes(c)} if c in cat else {"type": ["number", "null"]}
             for c in cols}
    schema = {"type": "object", "required": ["rows"], "additionalProperties": False, "properties": {"rows": {
        "type": "array", "items": {"type": "object", "required": cols, "additionalProperties": False, "properties": props}}}}
    rng = np.random.default_rng(seed)
    codebook = _codebook_prompt(cols, cb)
    good, rejected, batches = [], 0, 0

    def example(rec):
        return json.dumps({k: (None if isinstance(v, float) and math.isnan(v) else v[:100] if isinstance(v, str) else v)
                           for k, v in rec.items()}, ensure_ascii=False)

    def build_prompt(k):
        ex = "\n".join(example(r) for r in df[cols].iloc[rng.choice(len(df), 8, replace=False)].to_dict("records"))
        return (
            f"Generate {k} NEW synthetic survey respondents as JSON objects keyed by column name.\n{codebook}\n\n"
            f"Free-text columns {text} are strings ('' if no answer). All other columns are numeric answer codes.\n"
            f"Real respondents, for style and realistic answer patterns only (answers within a respondent are correlated):\n{ex}\n\n"
            f'Return {{"rows": [...]}} with exactly {k} rows. Do not copy the examples; vary answers like a real panel would.'
        )

    def keep(rows):
        """Append valid rows to `good`; return how many in this batch were rejected."""
        bad = 0
        for rec in rows:
            if isinstance(rec, dict) and all(c in rec for c in cols) and all(rec[c] in allowed[c] for c in cat):
                good.append(rec)
            else:
                bad += 1
        return bad

    max_batches = 3 * math.ceil(n / batch) + 3

    if use_claude:
        # Remote API, not GPU-bound like Ollama -> fire several batch calls at once for real wall-clock speedup.
        MAX_WORKERS = 5  # ponytail: fixed pool size; back off / lower this if Anthropic starts returning 429s
        while len(good) < n and batches < max_batches:
            round_n = min(math.ceil((n - len(good)) / batch), max_batches - batches, MAX_WORKERS)
            with ThreadPoolExecutor(max_workers=round_n) as pool:
                futures = [pool.submit(_claude_rows, client, build_prompt(batch), schema) for _ in range(round_n)]
                for f in as_completed(futures):
                    batches += 1
                    try:
                        rejected += keep(f.result())
                    except (ValueError, KeyError, TypeError):
                        rejected += batch
            good = good[:n]
            if progress:
                progress(len(good) / n)
    else:
        while len(good) < n and batches < max_batches:
            k = min(batch, n - len(good))
            prompt = build_prompt(k)
            batches += 1
            try:
                kw = dict(model=model, messages=[{"role": "user", "content": prompt}], format=schema,
                          options={"temperature": 0.9, "num_ctx": 16384, "seed": int(rng.integers(1 << 30))})
                try:
                    r = client.chat(think=False, **kw)
                except ollama.ResponseError:  # model has no thinking switch
                    r = client.chat(**kw)
                body = r["message"]["content"]  # model occasionally wraps JSON in a code fence despite the schema
                rows = json.loads(body[body.index("{"): body.rindex("}") + 1])["rows"]
            except (ValueError, KeyError, TypeError):
                rejected += k
                continue
            rejected += keep(rows)
            good = good[:n]
            if progress:
                progress(len(good) / n)
    if not good:
        raise RuntimeError("LLM produced no valid rows; check model / Ollama connection")
    out = pd.DataFrame(good).assign(**{c: "" for c in cols_text})
    return _finish(out, df, id_col), {"rejected": rejected, "batches": batches, "model": model}


# ---------- comparison ----------
def compare(orig, synth):
    """Per-column distance (0 = identical): total-variation for categoricals, KS for numerics. Free text skipped."""
    from scipy.stats import ks_2samp

    id_col, text, cat, num = split_columns(orig)
    rows = []
    for c in cat:
        p = orig[c].value_counts(normalize=True, dropna=False)
        q = synth[c].value_counts(normalize=True, dropna=False)
        rows.append((c, "categorical", 0.5 * p.sub(q, fill_value=0).abs().sum()))
    for c in num:
        rows.append((c, "numeric", ks_2samp(orig[c].dropna(), synth[c].dropna()).statistic))
    out = pd.DataFrame(rows, columns=["column", "type", "distance"])
    out["score"] = 1 - out["distance"]
    return out


if __name__ == "__main__":
    from convert import read_with_codebook

    df, cb = read_with_codebook(glob.glob("/parquet/*.parquet")[0])
    t = time.time()
    s, info = gen_sdv(df, 50, name="selftest.parquet")
    assert list(s.columns) == list(df.columns) and len(s) == 50 and s.iloc[:, 0].is_unique
    print(f"sdv ok {time.time() - t:.1f}s, {info}, score {compare(df, s).score.mean():.3f}")
    _, info2 = gen_sdv(df, 5, name="selftest.parquet")
    assert info2["reused"], "second call with same file should load the saved model"
    os.remove(info["model"])
    t = time.time()
    l, info = gen_llm(df, cb, 10, batch=10)
    assert list(l.columns) == list(df.columns) and len(l) == 10
    print(f"llm ok {time.time() - t:.1f}s, {info}, score {compare(df, l).score.mean():.3f}")
