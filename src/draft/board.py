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

# The run detector lives with the pick policy now (`ValueDrafter.survivors`
# consumes it, and the simulator has to compute the same number the monitor
# shows); re-exported here because this is where callers found it.
from ..backtest.draft_sim import positional_run  # noqa: F401

# `player_id` is the *Sleeper* id, because that is what the live draft feed
# returns and what the monitor matches on. `player_key` is the canonical
# gsis-anchored id (src/ingest/player_ids.py) that everything else joins on.
# They are both here on purpose; neither one can do the other's job.
BOARD_COLUMNS = [
    "rank", "player_id", "player_key", "player_name", "position", "nfl_team",
    "projection", "projection_calibrated", "vorp", "replacement_points",
    "adp_rank", "adp_delta", "injury_status", "available",
    "bye_week", "blend_components_used",
    # What pure VORP thought before the market got a vote, and how hard the
    # market was allowed to vote. K and DEF are anchored; everyone else is 0.
    "vorp_unanchored", "market_anchor",
    # How much the experts disagree. A tight best/worst is a consensus player;
    # a wide one is contested, which is where both the value and the busts are.
    "ecr_sd", "ecr_best", "ecr_worst",
    # Share of the season this player has been available for, recent history,
    # shrunk toward his position. `durability_flag` fires only past ECR ~120,
    # where the consensus stops pricing it — see features/availability.py.
    "durability", "durability_seasons", "durability_flag",
    # Best-case value if the most bullish expert is right, on the VORP
    # scale. What `upside_tiebreak` ranks bench picks on.
    "upside_vorp",
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
                   position: str | None = None, n: int = 10,
                   roster_counts: dict[str, int] | None = None,
                   caps: dict[str, int] | None = None) -> pd.DataFrame:
    """Best players still on the board.

    `roster_counts` + `caps` drop positions you can no longer use. Without them
    this is a raw VORP list, which is the thing `ValueDrafter` exists to stop
    you doing: in the back half of a draft the top of an uncapped VORP list is
    a backup quarterback you can never start. The live monitor always passes
    them; the argument stays optional so a bare board dump still works.
    """
    df = board[~board["player_id"].isin(drafted_ids)]
    if caps:
        counts = roster_counts or {}
        full = {p for p, cap in caps.items() if counts.get(p, 0) >= cap}
        if full:
            trimmed = df[~df["position"].isin(full)]
            # Never hand back nothing: if the caps exclude everyone left, the
            # caps are wrong for this board, not the board wrong for the caps.
            if not trimmed.empty:
                df = trimmed
    if position:
        df = df[df["position"] == position]
    return df.head(n)




def export(board: pd.DataFrame, out_dir: Path | str = "outputs/projections",
           *, stem: str = "draft_board", meta: dict | None = None,
           drafted: set | None = None) -> dict[str, Path]:
    """Write parquet + CSV + HTML + a meta sidecar. Returns the paths written.

    The sidecar (`<stem>.meta.json`) exists because parquet and CSV carry only
    the table — nothing about *how* the board was built. The live monitor needs
    to know the scoring format the board was priced under (full PPR, in this
    league) so it can compare that against the draft room it is pointed at and
    warn if they disagree; without a durable record of it, that comparison is
    impossible once the process that built the board has exited.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "parquet": out / f"{stem}.parquet",
        "csv": out / f"{stem}.csv",
        "html": out / f"{stem}.html",
        "meta": out / f"{stem}.meta.json",
    }
    board.to_parquet(paths["parquet"], index=False)
    board.to_csv(paths["csv"], index=False)
    paths["html"].write_text(
        _render_html(board, meta or {}, drafted=drafted), encoding="utf-8")
    paths["meta"].write_text(json.dumps(meta or {}, indent=2, default=str),
                             encoding="utf-8")
    return paths


def refresh_html(board: pd.DataFrame, path: Path | str, drafted: set,
                 *, meta: dict | None = None, candidates=None,
                 preview_pick: int | None = None) -> Path:
    """Rewrite the phone board with the live draft's picks struck through.

    The HTML shipped as a tap-to-strike sheet backed by localStorage, which is
    fine for the picks *you* make and useless for the other nine teams: between
    two of your turns the room takes nine players and you would have to find and
    tap all nine on a phone while your clock runs. Nobody does that, so by round
    three the board on the phone is a list of players who are mostly gone.

    So the monitor writes the drafted set into the page itself. Local taps stay
    — they are how you mark a player you have decided against — but they are now
    a *union* with the feed, not the only source. A player the feed says is gone
    cannot be un-struck by tapping, because he is gone.

    `candidates` is `ValueDrafter.rank()`'s output — the pick plus a couple of
    runners-up, each carrying its own `.reason` — the same thing the terminal
    prints as `>> TAKE:` / `also considered:`. Without it the phone board was
    just a sortable table with no opinion on it — on the one device you are
    actually drafting from, the thing that tells you what to do was
    terminal-only.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_render_html(board, meta or {}, drafted=drafted,
                                 candidates=candidates,
                                 preview_pick=preview_pick),
                    encoding="utf-8")
    return path


def _render_html(board: pd.DataFrame, meta: dict,
                 *, drafted: set | None = None, candidates=None,
                 preview_pick: int | None = None) -> str:
    """Self-contained, phone-readable board with position filtering and search."""
    records = board.fillna("").to_dict(orient="records")
    payload = json.dumps(records)
    drafted_payload = json.dumps(sorted(str(d) for d in (drafted or set())))
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    subtitle = " · ".join(
        f"{k}: {v}" for k, v in meta.items()
    ) or "value-based draft board"

    # `drafted is not None` is exactly "this call came from the live monitor,
    # not the Thursday build" — `export()` never passes `drafted` at all, and
    # `refresh_html()` always passes a set (even an empty one, on the first
    # poll). The file on disk was already being rewritten every pick; nothing
    # in the page told an already-open browser tab to go look. A meta refresh
    # is the whole fix — no server, no JS polling, works for a local file.
    # Position filter and search reset on each reload, which is an acceptable
    # cost against actually seeing who is still on the board.
    refresh_tag = ('<meta http-equiv="refresh" content="6">'
                   if drafted is not None else "")

    rec_html = ""
    cands = list(candidates or [])
    if cands:
        top, rest = cands[0], cands[1:]
        top_reason = f'<div class="rec-why">{top.reason}</div>' if top.reason else ""
        alt_html = ""
        # A forced pick (scarcity / end-of-draft) is always alone in the list
        # — there is no real alternative to show, only a reason there wasn't
        # one, and `top_reason` already says so.
        if rest:
            rows = "".join(
                f"""<div class="alt">
      <span class="pos {a.position}">{a.position}</span> {a.player_name}
      <span class="alt-vorp">VORP {a.raw_value:.1f}</span>
      <div class="alt-why">{a.reason}</div>
    </div>"""
                for a in rest
            )
            alt_html = f'<div class="rec-alts">{rows}</div>'
        label = (f"Likely pick at your turn (#{preview_pick})"
                 if preview_pick else "Suggested pick")
        rec_html = f"""
  <div class="rec">
    <div class="rec-label">{label}</div>
    <div class="rec-player"><span class="pos {top.position}">{top.position}</span> {top.player_name}</div>
    <div class="rec-stats">VORP {top.raw_value:.1f}</div>
    {top_reason}
    {alt_html}
  </div>"""

    return f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
{refresh_tag}
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
  .meta{{color:var(--muted);font-size:11px;letter-spacing:.02em}}
  .inj{{color:var(--bad);font-weight:600}}
  /* Everyone still on the board is at worst a game-time decision — the
     genuinely unavailable were filtered out upstream. A late-August
     "Questionable" is paperwork, so it stays grey and does not shout. */
  .inj.soft{{color:var(--muted);font-weight:400}}
  .hint{{padding:10px 16px;color:var(--muted);font-size:12px}}
  /* The one thing that was missing from the phone board entirely: what to
     actually do. The terminal always had this; the device you draft from
     did not. */
  .rec{{margin-top:10px;padding:10px 12px;border:1px solid var(--fg);
    border-radius:10px;background:var(--card)}}
  .rec-label{{font-size:10px;text-transform:uppercase;letter-spacing:.08em;
    color:var(--muted);font-weight:700}}
  .rec-player{{font-size:17px;font-weight:700;margin:2px 0}}
  .rec-stats{{font-size:12px;color:var(--muted)}}
  .rec-why{{font-size:12.5px;color:var(--fg);margin-top:4px;line-height:1.4}}
  .rec-alts{{margin-top:8px;padding-top:8px;border-top:1px solid var(--line)}}
  .alt{{font-size:13px;margin-top:6px}}
  .alt:first-child{{margin-top:0}}
  .alt-vorp{{color:var(--muted);font-size:11.5px;margin-left:4px}}
  .alt-why{{font-size:11.5px;color:var(--muted);line-height:1.3;margin-top:1px}}
  /* On a phone the projection column is the first thing to go: VORP already
     carries it, and ADP and the delta are what you actually compare. */
  @media (max-width:430px) {{
    th,td{{padding:8px 6px}}
    th:nth-child(7),td:nth-child(7){{display:none}}
  }}
</style></head><body>
<header>
  <h1>Draft Board</h1>
  <div class="sub">{subtitle} · generated {generated}</div>{rec_html}
  <div class="controls">
    <input id="q" placeholder="Search player…" autocomplete="off">
    <button data-p="ALL" class="on">All</button>
    <button data-p="QB">QB</button><button data-p="RB">RB</button>
    <button data-p="WR">WR</button><button data-p="TE">TE</button>
    <button data-p="K">K</button><button data-p="DEF">DEF</button>
    <button id="hide">Hide taken</button>
  </div>
  <div class="sub" id="cnt"></div>
</header>
<div class="hint">Drafted players strike out automatically while the monitor is
running. Tap anyone else to mark them yourself; taps persist on this device.</div>
<div class="wrap"><table>
<thead><tr>
  <th class="l">#</th><th class="l">Player</th><th class="l">Pos</th>
  <th>VORP</th><th>ADP</th><th>Δ</th><th>Proj</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<script>
const DATA = {payload};
// Written by src/draft/monitor.py from the live Sleeper feed: every player the
// room has taken, not just the ones you tapped. Empty on a pre-draft board.
const DRAFTED = new Set({drafted_payload});
const KEY = 'drafted_v1';
let taps = new Set();
try {{ taps = new Set(JSON.parse(localStorage.getItem(KEY) || '[]')); }} catch (e) {{}}

// Position filter, search text and the hide-taken toggle used to live only in
// memory, so the page's own auto-refresh (every 6s, so struck-out picks stay
// current) silently reset all three on every reload — "Hide taken" undid
// itself every six seconds. Persisted the same way the taps already were.
const UI_KEY = 'ui_state_v1';
let pos = 'ALL', q = '', hide = false;
try {{
  const ui = JSON.parse(localStorage.getItem(UI_KEY) || '{{}}');
  if (typeof ui.pos === 'string') pos = ui.pos;
  if (typeof ui.q === 'string') q = ui.q;
  if (typeof ui.hide === 'boolean') hide = ui.hide;
}} catch (e) {{}}

const isTaken = id => DRAFTED.has(id) || taps.has(id);
function save() {{
  try {{ localStorage.setItem(KEY, JSON.stringify([...taps])); }} catch (e) {{}}
}}
function saveUI() {{
  try {{ localStorage.setItem(UI_KEY, JSON.stringify({{pos, q, hide}})); }} catch (e) {{}}
}}
function num(v, d) {{
  return (v === '' || v === null || v === undefined || isNaN(v))
    ? '—' : Number(v).toFixed(d);
}}
function render() {{
  const tb = document.getElementById('tb');
  const rows = DATA.filter(r =>
    (pos === 'ALL' || r.position === pos) &&
    (!q || String(r.player_name).toLowerCase().includes(q)) &&
    (!hide || !isTaken(String(r.player_id)))
  );
  const gone = DATA.filter(r => isTaken(String(r.player_id))).length;
  document.getElementById('cnt').textContent =
    gone ? gone + ' off the board · ' + (DATA.length - gone) + ' left' : '';
  tb.innerHTML = rows.map(r => {{
    const id = String(r.player_id);
    const d = Number(r.adp_delta);
    const dc = !isFinite(d) || r.adp_delta === '' ? '' : (d > 0 ? 'up' : d < 0 ? 'dn' : '');
    const ds = !isFinite(d) || r.adp_delta === '' ? '—' : (d > 0 ? '+' + d : d);
    // Team, bye and injury ride under the name instead of taking three more
    // columns. On a 375px phone the columns that decide a pick — VORP, ADP and
    // the gap between them — have to be visible without scrolling sideways.
    const bits = [r.nfl_team || '—'];
    if (num(r.bye_week, 0) !== '—') bits.push('bye ' + num(r.bye_week, 0));
    // Expert range. Wide means contested, which is a different bet from a
    // consensus pick at the same rank — worth seeing before you spend a pick.
    if (num(r.ecr_best, 0) !== '—' && num(r.ecr_worst, 0) !== '—')
      bits.push(num(r.ecr_best, 0) + '–' + num(r.ecr_worst, 0));
    // Only shown where it carries information the consensus has not priced.
    if (r.durability_flag === true || r.durability_flag === 'True')
      bits.push('<span class="inj">' + Math.round(r.durability * 100) + '% avail</span>');
    const sub = bits.join(' · ');
    const soft = String(r.injury_status) === 'Questionable' ? ' soft' : '';
    const inj = r.injury_status
      ? ` <span class="inj${{soft}}">${{r.injury_status}}</span>` : '';
    return `<tr class="${{isTaken(id) ? 'taken' : ''}}" data-id="${{id}}">
      <td class="l">${{r.rank}}</td>
      <td class="l">
        <div class="nm">${{r.player_name}}</div>
        <div class="meta">${{sub}}${{inj}}</div>
      </td>
      <td class="l"><span class="pos ${{r.position}}">${{r.position}}</span></td>
      <td><b>${{num(r.vorp, 1)}}</b></td>
      <td>${{num(r.adp_rank, 0)}}</td>
      <td class="${{dc}}">${{ds}}</td>
      <td>${{num(r.projection_calibrated || r.projection, 1)}}</td>
    </tr>`;
  }}).join('');
}}
document.getElementById('tb').addEventListener('click', e => {{
  const tr = e.target.closest('tr'); if (!tr) return;
  const id = tr.dataset.id;
  // A player the live feed says is drafted stays struck: tapping him is a
  // misclick, not a correction, and un-striking him would put a player who is
  // genuinely gone back at the top of your board.
  if (DRAFTED.has(id)) return;
  taps.has(id) ? taps.delete(id) : taps.add(id);
  save(); render();
}});
document.getElementById('hide').onclick = e => {{
  hide = !hide; e.target.classList.toggle('on', hide); saveUI(); render();
}};
document.querySelectorAll('button[data-p]').forEach(b => {{
  b.onclick = () => {{
    document.querySelectorAll('button[data-p]').forEach(x => x.classList.remove('on'));
    b.classList.add('on'); pos = b.dataset.p; saveUI(); render();
  }};
}});
document.getElementById('q').addEventListener('input', e => {{
  q = e.target.value.toLowerCase().trim(); saveUI(); render();
}});

// Reflect the restored filter/search/hide state into the controls themselves
// before the first render — otherwise the *filtering* would persist correctly
// but the buttons and search box would silently disagree with what the page
// is actually showing.
document.getElementById('q').value = q;
document.getElementById('hide').classList.toggle('on', hide);
document.querySelectorAll('button[data-p]').forEach(b => {{
  b.classList.toggle('on', b.dataset.p === pos);
}});

render();
</script></body></html>"""
