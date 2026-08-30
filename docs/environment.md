# Environment notes

## Python 3.14

The local interpreter is **Python 3.14.7** on Windows. That is newer than much
of the data-science stack targets, so a few things were resolved deliberately
rather than by accident.

Verified working on 3.14 as of 2026-08-30:

| Package | Version | Note |
|---|---|---|
| `nflreadpy` | 0.1.5 | nflverse data access |
| `pandas` | 3.0.5 | **major version 3** — see below |
| `polars` | 1.44.1 | `nflreadpy` returns Polars frames |
| `numpy` | 2.5.2 | |
| `xgboost` | 3.4.1 | wheels available, no source build needed |
| `lightgbm` | 4.7.0 | |
| `scikit-learn` | 1.9.0 | |
| `pyarrow` | 25.0.1 | parquet |

### `nfl_data_py` is intentionally not used

The original plan named `nfl_data_py`. It has no Python 3.14 wheels and is the
older of the two nflverse Python paths. **`nflreadpy` is the nflverse-maintained
port and is what this repo uses.**

The practical difference: `nflreadpy` returns **Polars** DataFrames, not pandas.
Call `.to_pandas()` at the ingest boundary. Everything downstream of
`src/ingest/` is pandas, so the conversion happens once, in one place.

```python
import nflreadpy as nfl
pbp = nfl.load_pbp([2024]).to_pandas()
```

### pandas 3.x

pandas 3 is a major release with real behavior changes from 2.x — copy-on-write
is the default, and some silent-upcast behavior is gone. Code written against
pandas 2 tutorials may need adjusting. The modules here are written against 3.x
and the test suite passes on it.

## Setup

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m pytest tests/ -q
```

On macOS/Linux the interpreter path is `.venv/bin/python` instead.

## Rate limits worth remembering

- **Sleeper** — stay under 1000 req/min. The client in `src/ingest/sleeper_api.py`
  self-throttles well below that. The live draft monitor polls every 5s (12/min).
- **Pro-Football-Reference** — 20 req/min, and exceeding it gets your session
  blocked for up to a day. Prefer nflverse, which already ingests most
  PFR-derived data.
- **The Odds API** — free tier is credit-metered; historical endpoints cost
  10 credits per region per market. Budget before backfilling.
