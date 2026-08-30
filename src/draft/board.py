"""Draft board construction and export.

The board is the Thursday deliverable. It answers one question on every pick:
"who is the best available player, adjusted for what my roster still needs?"

Output goes to three places, deliberately:
  - parquet, for the backtester and any downstream analysis
  - CSV, as the fallback you can open in a spreadsheet if Python breaks
  - a single self-contained HTML file, because you will be drafting from your
    phone and a terminal table is useless there

The HTML export has no external dependencies and works offline.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BOARD_COLUMNS = [
    "rank", "player_id", "player_name", "position", "nfl_team",
    "projection", "projection_calibrated", "vorp", "replacement_points",
    "adp_rank", "adp_delta", "injury_status", "available",
    "bye_week", "blend_components_used",
]


def finalize_board(df: pd.DataFrame, *, drop_unavailable: bool = True
                   ) -> pd.DataFrame:
    """Sort by VORP, rank, and compute value-vs-ADP.

    `adp_delta` is positive when the board likes a player more than the market
    does — those are the picks where you gain ground. Negative means the market
    is higher on him than you are; taking him is paying retail.
    """
    out = df.copy()

    if "available" not in out.columns:
        out["available"] = True
    if drop_unavailable:
        out = out[out["available"]]

    out = out.sort_values("vorp", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)

    if "adp_rank" in out.columns:
        out["adp_delta"] = out["adp_rank"] - out["rank"]

    for col in BOARD_COLUMNS:
        if col not in out.columns:
            out[col] = None

    return out[BOARD_COLUMNS]


def best_available(board: pd.DataFrame, drafted_ids: set, *,
                   position: str | None = None, n: int = 10) -> pd.DataFrame:
    df = board[~board["player_id"].isin(drafted_ids)]
    if position:
        df = df[df["position"] == position]
    return df.head(n)


def positional_run(recent_picks: list[dict], window: int = 6,
                   threshold: float = 0.5) -> dict[str, float]:
    """Detect a positional run in the last `window` picks.

    Returns positions whose share of recent picks exceeds `threshold`. A run is
    a reason to move a position up, not a reason to panic — the useful signal
    is "the tier I want will be gone before my next pick", which the caller
    checks against picks-until-next-turn.
    """
    if not recent_picks:
        return {}
    recent = recent_picks[-window:]
    counts: dict[str, int] = {}
    for p in recent:
        pos = p.get("position")
        if pos:
            counts[pos] = counts.get(pos, 0) + 1
    return {
        pos: c / len(recent)
        for pos, c in counts.items()
        if c / len(recent) >= threshold
    }


def export(board: pd.DataFrame, out_dir: Path | str = "outputs/projections",
           *, stem: str = "draft_board", meta: dict | None = None) -> dict[str, Path]:
    """Write parquet + CSV + HTML. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "parquet": out / f"{stem}.parquet",
        "csv": out / f"{stem}.csv",
        "html": out / f"{stem}.html",
    }
    board.to_parquet(paths["parquet"], index=False)
    board.to_csv(paths["csv"], index=False)
    paths["html"].write_text(_render_html(board, meta or {}), encoding="utf-8")
    return paths


def _render_html(board: pd.DataFrame, meta: dict) -> str:
    """Self-contained, phone-readable board with position filtering and search."""
    records = board.fillna("").to_dict(orient="records")
    payload = json.dumps(records)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = " · ".join(
        f"{k}: {v}" for k, v in meta.items()
    ) or "value-based draft board"

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Draft Board</title>
<style>
  :root {{
    --bg:#fbfbfa; --fg:#1a1a19; --muted:#6b6b68; --line:#e4e4e1;
    --card:#fff; --pos-qb:#b45309; --pos-rb:#15803d; --pos-wr:#1d4ed8;
    --pos-te:#7c3aed; --pos-k:#6b6b68; --pos-def:#0f766e; --good:#15803d; --bad:#b91c1c;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg:#191918; --fg:#f2f2ef; --muted:#a3a3a0; --line:#333331;
      --card:#222221; --pos-qb:#f0b429; --pos-rb:#4ade80; --pos-wr:#60a5fa;
      --pos-te:#c084fc; --pos-k:#a3a3a0; --pos-def:#2dd4bf; --good:#4ade80; --bad:#f87171;
    }}
  }}
  *{{box-sizing:border-box}}
  body{{margin:0;background:var(--bg);color:var(--fg);
    font:15px/1.45 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}}
  header{{padding:16px;border-bottom:1px solid var(--line);
    position:sticky;top:0;background:var(--bg);z-index:10}}
  h1{{margin:0 0 2px;font-size:17px;letter-spacing:-.01em}}
  .sub{{color:var(--muted);font-size:12px}}
  .controls{{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}}
  button{{font:inherit;font-size:13px;padding:6px 11px;border:1px solid var(--line);
    background:var(--card);color:var(--fg);border-radius:999px;cursor:pointer}}
  button.on{{background:var(--fg);color:var(--bg);border-color:var(--fg)}}
  input{{font:inherit;font-size:14px;padding:7px 11px;border:1px solid var(--line);
    background:var(--card);color:var(--fg);border-radius:8px;flex:1;min-width:130px}}
  .wrap{{overflow-x:auto}}
  table{{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}}
  th,td{{padding:9px 10px;text-align:right;border-bottom:1px solid var(--line);
    white-space:nowrap}}
  th{{font-size:11px;text-transform:uppercase;letter-spacing:.05em;
    color:var(--muted);font-weight:600}}
  td.l,th.l{{text-align:left}}
  tr.taken{{opacity:.32}}
  tr.taken .nm{{text-decoration:line-through}}
  .pos{{font-weight:700;font-size:11px;padding:2px 6px;border-radius:4px;
    border:1px solid currentColor}}
  .QB{{color:var(--pos-qb)}} .RB{{color:var(--pos-rb)}} .WR{{color:var(--pos-wr)}}
  .TE{{color:var(--pos-te)}} .K{{color:var(--pos-k)}} .DEF{{color:var(--pos-def)}}
  .up{{color:var(--good)}} .dn{{color:var(--bad)}}
  .nm{{font-weight:600;cursor:pointer}}
  .hint{{padding:10px 16px;color:var(--muted);font-size:12px}}
</style></head><body>
<header>
  <h1>Draft Board</h1>
  <div class="sub">{subtitle} · generated {generated}</div>
  <div class="controls">
    <input id="q" placeholder="Search player…" autocomplete="off">
    <button data-p="ALL" class="on">All</button>
    <button data-p="QB">QB</button><button data-p="RB">RB</button>
    <button data-p="WR">WR</button><button data-p="TE">TE</button>
    <button data-p="K">K</button><button data-p="DEF">DEF</button>
  </div>
</header>
<div class="hint">Tap a player to mark them drafted. Marks persist on this device.</div>
<div class="wrap"><table>
<thead><tr>
  <th class="l">#</th><th class="l">Player</th><th class="l">Pos</th>
  <th class="l">Tm</th><th>VORP</th><th>Proj</th><th>ADP</th><th>Δ</th><th class="l">Note</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<script>
const DATA = {payload};
const KEY = 'drafted_v1';
let taken = new Set();
try {{ taken = new Set(JSON.parse(localStorage.getItem(KEY) || '[]')); }} catch (e) {{}}
let pos = 'ALL', q = '';

function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify([...taken])); }} catch (e) {{}}
}}
function num(v, d) {{
  return (v === '' || v === null || v === undefined || isNaN(v))
    ? '—' : Number(v).toFixed(d);
}}
function render() {{
  const tb = document.getElementById('tb');
  const rows = DATA.filter(r =>
    (pos === 'ALL' || r.position === pos) &&
    (!q || String(r.player_name).toLowerCase().includes(q))
  );
  tb.innerHTML = rows.map(r => {{
    const id = String(r.player_id);
    const d = Number(r.adp_delta);
    const dc = !isFinite(d) || r.adp_delta === '' ? '' : (d > 0 ? 'up' : d < 0 ? 'dn' : '');
    const ds = !isFinite(d) || r.adp_delta === '' ? '—' : (d > 0 ? '+' + d : d);
    const note = r.injury_status ? String(r.injury_status) : '';
    return `<tr class="${{taken.has(id) ? 'taken' : ''}}" data-id="${{id}}">
      <td class="l">${{r.rank}}</td>
      <td class="l nm">${{r.player_name}}</td>
      <td class="l"><span class="pos ${{r.position}}">${{r.position}}</span></td>
      <td class="l">${{r.nfl_team || '—'}}</td>
      <td><b>${{num(r.vorp, 1)}}</b></td>
      <td>${{num(r.projection_calibrated || r.projection, 1)}}</td>
      <td>${{num(r.adp_rank, 0)}}</td>
      <td class="${{dc}}">${{ds}}</td>
      <td class="l">${{note}}</td>
    </tr>`;
  }}).join('');
}}
document.getElementById('tb').addEventListener('click', e => {{
  const tr = e.target.closest('tr'); if (!tr) return;
  const id = tr.dataset.id;
  taken.has(id) ? taken.delete(id) : taken.add(id);
  save(); render();
}});
document.querySelectorAll('button[data-p]').forEach(b => {{
  b.onclick = () => {{
    document.querySelectorAll('button[data-p]').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); pos = b.dataset.p; render();
  }};
}});
document.getElementById('q').addEventListener('input', e => {{
  q = e.target.value.toLowerCase().trim(); render();
}});
render();
</script></body></html>"""
