"""SPSS .sav -> parquet, with the SPSS codebook embedded in the parquet schema metadata."""
import glob, json, os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyreadstat

CODEBOOK_KEY = b"codebook"


def _key(v):
    return str(int(v)) if isinstance(v, float) and v.is_integer() else str(v)


def build_codebook(meta):
    labels = meta.column_names_to_labels
    return {
        c: {
            "label": labels.get(c) or c,
            "measure": meta.variable_measure.get(c, "unknown"),
            "type": meta.original_variable_types.get(c),
            "values": {_key(k): v for k, v in meta.variable_value_labels.get(c, {}).items()},
        }
        for c in meta.column_names
    }


def write_with_codebook(df, codebook, path):
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), CODEBOOK_KEY: json.dumps(codebook)})
    pq.write_table(table, path)


def read_with_codebook(source):
    table = pq.read_table(source)
    raw = (table.schema.metadata or {}).get(CODEBOOK_KEY)
    return table.to_pandas(), json.loads(raw) if raw else {}


if __name__ == "__main__":
    for sav in glob.glob("/data/*.sav"):
        df, meta = pyreadstat.read_sav(sav)
        out = os.path.join("/parquet", os.path.splitext(os.path.basename(sav))[0] + ".parquet")
        write_with_codebook(df, build_codebook(meta), out)
        back, cb = read_with_codebook(out)
        assert back.shape == df.shape and set(cb) == set(df.columns)
        print(f"{out}: {df.shape[0]} rows x {df.shape[1]} cols, {sum(bool(v['values']) for v in cb.values())} vars with value labels")
