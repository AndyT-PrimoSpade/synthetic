# synthetic-v2

Converts an SPSS `.sav` survey file to Parquet (with the SPSS codebook embedded
as metadata), then a Streamlit app lets you boost the dataset with synthetic
rows two ways — a local LLM (Ollama) and a statistical model (SDV) — and
compares both against the original distribution.

## Prerequisites

- **Docker Desktop**, with the WSL2 backend on Windows.
- **NVIDIA GPU** for a usable Ollama speed. CPU works but a 9B model is slow —
  if you don't have a GPU, delete the `deploy:` block under `ollama` in
  `docker-compose.yml`.
- **~15 GB free disk** for the Ollama models pulled on first run.

## Getting started

1. **Add your data.** Put the `.sav` file in `data/`.

2. **Build the image:**
   ```bash
   docker compose build
   ```

3. **Convert SPSS → Parquet** (writes to `parquet/`, embeds the codebook):
   ```bash
   docker compose run --rm convert
   ```

4. **Start the app** (also brings up Ollama and pulls the default model,
   `ornith-1.5:9b`, into a Docker volume — first run takes a few minutes):
   ```bash
   docker compose up -d app
   ```

5. Open **http://localhost:8501**.

To stop everything: `docker compose stop`. Your converted Parquet file, saved
models, and pulled Ollama models persist (nothing is deleted) — `docker
compose up -d app` starts it all again without re-downloading anything.

## Using the app

- **Tab 1 — Uploaded data**: the dataset, row/column counts, and the codebook
  (variable + value labels) read back out of the Parquet file's metadata.
- **Tab 2 — Generated**: pick a boost size in the sidebar and click
  **Generate**. Runs both methods and shows their output side by side, with
  download buttons (synthetic rows alone, or combined with the original).
- **Tab 3 — Comparison**: per-column and overall similarity scores (1 =
  identical distribution) between the original data and each method's output.

The sidebar's Ollama model dropdown lets you try other pulled models
(`ornith-1.5:9b` is the fastest/default; `gemma4:12b` scored highest in
informal testing but is slower). The SDV model for a given upload is fit once
and cached in `models/` (keyed by filename + a hash of the data), so
re-generating against the same file is near-instant on the stats side.

## Optional: Claude API backend

`synth.py`'s LLM method can run against the Claude API (`claude-haiku-4-5`)
instead of local Ollama — much faster (parallel batch calls), at real API
cost. This is a deliberately hidden, developer-only toggle — not exposed in
the UI. To use it:

1. Create a `.env` file next to `docker-compose.yml`:
   ```
   ANTHROPIC_API_KEY=sk-ant-...
   ```
2. Edit `config.toml`: set `[claude] enabled = true`.
3. `docker compose up -d app` (recreates the container so it picks up the
   key; `config.toml` itself is bind-mounted, so flipping it back doesn't
   need a rebuild or recreate).

While enabled, every LLM-method run uses Claude instead of Ollama; the only
visible sign is the model name shown in Tab 2 (`LLM · claude-haiku-4-5`).

## Project structure

```
Dockerfile            image for convert.py / app.py
docker-compose.yml    services: ollama, ollama-pull, convert, app
requirements.txt
convert.py            .sav -> parquet, embeds codebook in schema metadata
synth.py              SDV + Ollama/Claude generators, distribution comparison
app.py                Streamlit UI
config.toml           hidden Claude on/off switch (see above)
data/                 put your .sav here (gitignored)
parquet/              conversion output (gitignored, regenerate anytime)
models/               saved SDV models (gitignored - fit on the real data, regenerate anytime)
```

## Troubleshooting

- **GPU not detected / Ollama container won't start**: delete the `deploy:`
  block under the `ollama` service in `docker-compose.yml` to run on CPU.
- **Port 11434 already in use**: this project publishes its own Ollama on
  `11435` for that reason; if you also run Ollama on the host, both can
  coexist.
- **LLM rows keep getting rejected**: a small/older Ollama model may not
  respect the JSON schema. Try `ornith-1.5:9b`, `gemma4:12b`, or
  `granite4.2:8b` — pull with `docker exec <ollama container> ollama pull <model>`.
