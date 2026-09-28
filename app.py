import glob, io, os, time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from convert import read_with_codebook, write_with_codebook
from synth import compare, gen_llm, gen_sdv, split_columns

st.set_page_config(page_title="Synthetic data boost", layout="wide")
ss = st.session_state

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Data")
    up = st.file_uploader("Upload parquet", type="parquet")
    saved = sorted(glob.glob("/parquet/*.parquet"))
    pick = st.selectbox("...or pick from /parquet", [""] + saved, format_func=lambda p: os.path.basename(p) or "-")
    src, name = (io.BytesIO(up.getvalue()), up.name) if up else ((pick, os.path.basename(pick)) if pick else (None, None))

    st.header("Boost")
    if src:
        df, cb = read_with_codebook(src)
        pct = st.slider("Boost by %", 5, 300, 50, 5)
        n = st.number_input("New rows", 1, 20000, max(1, round(len(df) * pct / 100)))
    st.header("Methods")
    model = st.selectbox("Ollama model", ["ornith-1.5:9b", "gemma4:12b", "granite4.2:8b", "qwen3.5:9b"], accept_new_options=True,
                         help="qwen3.5 ignores the JSON schema, so most of its rows get rejected")
    free_text = st.checkbox("LLM writes free-text answers too", False, help="Much slower; otherwise left blank")
    batch = st.number_input("LLM rows per call", 5, 50, 15)
    kind = st.selectbox("Stats synthesizer", ["GaussianCopula", "CTGAN"])
    seed = st.number_input("Seed", 0, 10**6, 0)
    go_btn = st.button("Generate", type="primary", disabled=not src)

if not src:
    st.info("Upload a parquet file (or convert the .sav first: `docker compose run --rm convert`).")
    st.stop()

# ---------------- generate ----------------
if go_btn:
    ss.pop("res", None)
    res = {"name": name, "n": int(n)}
    with st.status("Generating...", expanded=True) as status:
        t = time.time()
        st.write(f"Stats method ({kind})...")
        res["sdv"], res["sdv_info"] = gen_sdv(df, int(n), kind, int(seed), name)
        res["sdv_info"]["secs"] = time.time() - t
        st.write("saved model reused" if res["sdv_info"]["reused"] else "model fitted and saved")
        t = time.time()
        bar = st.progress(0.0, "LLM method...")
        try:
            res["llm"], res["llm_info"] = gen_llm(df, cb, int(n), model, int(batch), int(seed),
                                                  lambda p: bar.progress(min(p, 1.0), "LLM method..."), free_text)
            res["llm_info"]["secs"] = time.time() - t
        except Exception as e:  # keep the SDV result usable if Ollama is down
            res["llm"], res["llm_info"] = None, {"error": str(e)}
        status.update(label="Done", state="complete")
    ss.res = res

res = ss.get("res")
if res and res["name"] != name:
    res = None  # results belong to a different upload

labels = {c: v["label"] for c, v in cb.items()}


def lab(col, x):
    """Code -> value label (6, 6.0 and '6' all resolve the same)."""
    try:
        f = float(x)
        x = str(int(f)) if f.is_integer() else str(f)
    except (TypeError, ValueError):
        return x
    return cb.get(col, {}).get("values", {}).get(x, x)


def show(d):
    """Optionally swap codes for value labels."""
    if not st.session_state.get("as_labels"):
        return d
    return d.apply(lambda s: s.map(lambda x: lab(s.name, x)) if cb.get(s.name, {}).get("values") else s)


def downloads(d, key):
    c1, c2, c3 = st.columns(3)
    buf = io.BytesIO()
    write_with_codebook(d, cb, buf)
    c1.download_button("Synthetic rows (parquet)", buf.getvalue(), f"{key}_synthetic.parquet", key=f"{key}p")
    c2.download_button("Synthetic rows (csv)", d.to_csv(index=False), f"{key}_synthetic.csv", key=f"{key}c")
    both = pd.concat([df.assign(is_synthetic=False), d.assign(is_synthetic=True)], ignore_index=True)
    buf2 = io.BytesIO()
    write_with_codebook(both, cb, buf2)
    c3.download_button("Original + synthetic (parquet)", buf2.getvalue(), f"{key}_combined.parquet", key=f"{key}b")


t1, t2, t3 = st.tabs(["1 · Uploaded data", "2 · Generated (LLM vs Stats)", "3 · Comparison"])

with t1:
    a, b, c = st.columns(3)
    a.metric("Rows", len(df))
    b.metric("Columns", df.shape[1])
    c.metric("Rows to add", int(n))
    st.toggle("Show value labels instead of codes", key="as_labels")
    st.dataframe(show(df), use_container_width=True, height=400)
    with st.expander("Codebook (embedded in parquet metadata)"):
        st.dataframe(pd.DataFrame([{"variable": k, "label": v["label"], "measure": v["measure"],
                                    "values": "; ".join(f"{a}={b}" for a, b in v["values"].items())}
                                   for k, v in cb.items()]), use_container_width=True)

with t2:
    if not res:
        st.write("Press **Generate** in the sidebar.")
    else:
        l, r = st.columns(2)
        with l:
            i = res["llm_info"]  # 'model' reflects what actually ran (config.toml can silently swap in Claude)
            st.subheader(f"LLM · {i.get('model', model)}")
            if res["llm"] is None:
                st.error(i["error"])
            else:
                st.caption(f"{len(res['llm'])} rows · {i['secs']:.0f}s · {i['rejected']} invalid rows rejected · {i['batches']} calls")
                st.dataframe(show(res["llm"]), use_container_width=True, height=350)
                downloads(res["llm"], "llm")
        with r:
            i = res["sdv_info"]
            st.subheader(f"Stats · {kind}")
            st.caption(f"{len(res['sdv'])} rows · {i['secs']:.1f}s · " +
                       ("saved model reused" if i["reused"] else "model fitted & saved") + f" (`{os.path.basename(i['model'])}`)")
            st.dataframe(show(res["sdv"]), use_container_width=True, height=350)
            downloads(res["sdv"], "sdv")

with t3:
    if not res:
        st.write("Generate first.")
    else:
        methods = {"Original": df, "Stats": res["sdv"], **({"LLM": res["llm"]} if res["llm"] is not None else {})}
        scores = {m: compare(df, d).set_index("column")["score"] for m, d in methods.items() if m != "Original"}
        tab = pd.DataFrame(scores)
        tab.insert(0, "label", [labels.get(c, c)[:70] for c in tab.index])
        cols = st.columns(len(scores))
        for col, m in zip(cols, scores):
            col.metric(f"{m} overall similarity", f"{scores[m].mean():.3f}", help="1 - avg distance to original (TV distance / KS statistic)")
        _, text, cat, num = split_columns(df)
        col = st.selectbox("Column", tab.index.tolist(), format_func=lambda c: f"{c} — {labels.get(c, c)[:70]}")
        fig = go.Figure()
        for m, d in methods.items():
            if col in cat:
                vc = d[col].value_counts(normalize=True).sort_index()
                fig.add_bar(x=[str(lab(col, k)) for k in vc.index], y=vc.values, name=m)
            else:
                fig.add_histogram(x=d[col], name=m, histnorm="probability", opacity=0.6)
        fig.update_layout(barmode="group" if col in cat else "overlay", yaxis_title="share", height=420)
        st.plotly_chart(fig, use_container_width=True)
        st.subheader("Per-column similarity (1 = identical)")
        st.dataframe(tab.sort_values(tab.columns[1]), use_container_width=True)
