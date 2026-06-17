"""Generate an interactive feature UMAP explorer HTML.

Loads all non-edge cells, runs UMAP over z-scored features, embeds cell
thumbnails from zarr, and writes a standalone HTML with:
  - Feature dropdown  → recolors points by z-scored feature value (RdBu)
  - Hover             → shows condition label
  - Click             → right panel: thumbnail + all feature values
  - Lasso/box select  → right panel: grid of thumbnails

Usage
-----
    python "Joaquin'scripts/feature_umap_explorer.py" \\
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table outputs/cell_table.csv \\
        --out   outputs/feature_umap
"""

import argparse, io, json, base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
import umap
from PIL import Image
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# HTML template  (single-quoted JS/HTML to minimise JSON escaping)
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'/>
  <title>Feature UMAP Explorer</title>
  <script src='https://cdn.plot.ly/plotly-2.30.0.min.js'></script>
  <style>
    *, *::before, *::after { box-sizing: border-box }
    body  { font-family: Arial, sans-serif; margin: 0; padding: 12px 16px;
            background: #f0f0f0; color: #222 }
    h2    { margin: 0 0 2px }
    .sub  { color: #999; font-size: 13px; margin: 0 0 10px }
    .ctrl { display: flex; align-items: center; gap: 12px; margin-bottom: 10px;
            background: #fff; border-radius: 8px; padding: 8px 14px;
            box-shadow: 0 1px 4px #ccc }
    .ctrl label { font-size: 13px; font-weight: bold; white-space: nowrap }
    .ctrl select { font-size: 13px; padding: 4px 8px; border: 1px solid #ddd;
                   border-radius: 4px; min-width: 260px; cursor: pointer }
    .ctrl .hint  { font-size: 11px; color: #aaa }
    #wrap      { display: flex; gap: 14px; align-items: flex-start }
    #plot-col  { flex: 3 }
    #panel-col { flex: 2; min-width: 340px; max-height: 760px; overflow-y: auto;
                 background: #fff; border-radius: 8px; padding: 14px;
                 box-shadow: 0 2px 8px #bbb }
    #panel-col h4 { margin: 0 0 10px; font-size: 15px }
    .hint-p { color: #ccc; font-style: italic }
    .ibox b   { font-size: 14px }
    .ibox .meta { font-size: 11px; color: #999; word-break: break-all; margin: 2px 0 8px }
    .img-bg { background: #0a0a0a; padding: 4px; border-radius: 4px;
              width: 100%; margin-top: 6px }
    .img-bg img { width: 100%; display: block; image-rendering: pixelated }
    .ch-labels { display: flex; width: 100%; margin-top: 2px; margin-bottom: 8px }
    .ch-labels span { flex: 1; text-align: center; font-size: 10px; color: #999 }
    table  { border-collapse: collapse; width: 100%; font-size: 12px }
    th     { text-align: left; color: #888; font-weight: normal;
             border-bottom: 1px solid #eee; padding: 3px 6px }
    td     { padding: 3px 6px; border-bottom: 1px solid #f4f4f4 }
    td.hi  { font-weight: bold; color: #c0392b }
    td:last-child { text-align: right; font-family: monospace }
    .ghdr  { font-size: 13px; font-weight: bold; margin-bottom: 8px }
    .grid  { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px }
    .tile  { background: #111; border-radius: 3px; padding: 3px 3px 0 3px }
    .tile img  { width: 100%; display: block; image-rendering: pixelated }
    .tile .tlbl { font-size: 9px; color: #bbb; text-align: center; padding: 2px 0 3px }
  </style>
</head>
<body>
<h2>Feature UMAP Explorer</h2>
<p class='sub'>
  __N_CELLS__ cells &nbsp;|&nbsp; __N_FEATS__ features (geometry + texture)
  &nbsp;|&nbsp; color = z-scored feature value
</p>
<div class='ctrl'>
  <label for='feat-sel'>Color by feature:</label>
  <select id='feat-sel'></select>
  <span class='hint'>blue = low &nbsp; &middot; &nbsp; red = high &nbsp; (z-score, &plusmn;3)</span>
</div>
<div id='wrap'>
  <div id='plot-col'><div id='umap-plot'></div></div>
  <div id='panel-col'>
    <h4>Cell Inspector</h4>
    <div id='panel-content'>
      <p class='hint-p'>&#x1F446; Click a point to inspect &nbsp; &middot; &nbsp; lasso/box to see a grid.</p>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;

// ── populate dropdown ──────────────────────────────────────────────────────
function humanName(n) {
  return n.replace('gFeat_cell_', 'cell · ')
          .replace('gFeat_nuc_',  'nuc · ')
          .replace('tFeat_mem_',  'mem · ')
          .replace('tFeat_nuc_',  'nuc · ');
}
const sel = document.getElementById('feat-sel');
DATA.feat_names.forEach(function(name) {
  const opt = document.createElement('option');
  opt.value = name;
  opt.textContent = humanName(name);
  sel.appendChild(opt);
});

// ── build single trace ────────────────────────────────────────────────────
const firstFeat = DATA.feat_names[0];
const trace = {
  x: DATA.x, y: DATA.y,
  mode: 'markers', type: 'scatter',
  text: DATA.condition,
  customdata: DATA.x.map(function(_, i) { return i; }),
  hovertemplate: '<b>%{text}</b><extra></extra>',
  marker: {
    size: 4, opacity: 0.75,
    color: DATA.zscores[firstFeat],
    colorscale: 'RdBu',
    reversescale: true,
    showscale: true,
    cmin: -3, cmax: 3,
    colorbar: {
      title: { text: humanName(firstFeat), side: 'right' },
      thickness: 14, len: 0.6, x: 1.01
    }
  }
};

Plotly.newPlot('umap-plot', [trace], {
  height: 700,
  margin: { l: 50, r: 80, t: 20, b: 50 },
  clickmode: 'event+select',
  dragmode: 'pan',
  plot_bgcolor: '#fff', paper_bgcolor: '#fff',
  xaxis: { title: 'UMAP 1', gridcolor: '#eee', zeroline: false },
  yaxis: { title: 'UMAP 2', gridcolor: '#eee', zeroline: false },
}, { scrollZoom: true, responsive: true });

// ── dropdown → recolor ────────────────────────────────────────────────────
sel.addEventListener('change', function() {
  const name = sel.value;
  Plotly.restyle('umap-plot', {
    'marker.color':              [DATA.zscores[name]],
    'marker.colorbar.title.text': humanName(name)
  });
});

// ── panel helpers ─────────────────────────────────────────────────────────
const panel = document.getElementById('panel-content');

function thumbHtml(ci, cond) {
  const src = DATA.thumbnails[String(ci)];
  if (!src) return '';
  return '<div class=\'img-bg\'><img src=\'' + src + '\'></div>' +
         '<div class=\'ch-labels\'><span>membrane</span><span>nuclei</span></div>';
}

function singleView(idx) {
  const ci   = DATA.cell_idx[idx];
  const cond = DATA.condition[idx];
  const feat = sel.value;

  const rows = DATA.feat_names.map(function(f) {
    const val = DATA.feat_raw[f][idx];
    const cls = (f === feat) ? ' class=\'hi\'' : '';
    return '<tr><td' + cls + '>' + humanName(f) + '</td><td>' + val + '</td></tr>';
  }).join('');

  panel.innerHTML =
    '<div class=\'ibox\'><b>' + cond + '</b></div>' +
    thumbHtml(ci, cond) +
    '<table><tr><th>feature</th><th>value</th></tr>' + rows + '</table>';
}

function gridView(pts) {
  const MAX = 16, n = pts.length;
  let h = '<div class=\'ghdr\'>' + n + ' cell' + (n !== 1 ? 's' : '') + ' selected' +
          (n > MAX ? ' — first ' + MAX + ' shown' : '') + '</div><div class=\'grid\'>';
  for (let i = 0; i < Math.min(n, MAX); i++) {
    const idx  = pts[i].customdata;
    const ci   = DATA.cell_idx[idx];
    const cond = DATA.condition[idx];
    const src  = DATA.thumbnails[String(ci)];
    h += '<div class=\'tile\'>' +
         (src ? '<img src=\'' + src + '\'>' : '<div style=\'height:48px;background:#333\'></div>') +
         '<div class=\'tlbl\'>' + cond + '</div></div>';
  }
  panel.innerHTML = h + '</div>';
}

const el = document.getElementById('umap-plot');
el.on('plotly_click',    function(ev) { singleView(ev.points[0].customdata); });
el.on('plotly_selected', function(ev) { if (ev && ev.points.length) gridView(ev.points); });
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def _normalize(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    denom = float(hi) - float(lo)
    if denom <= 0:
        return np.zeros_like(arr, dtype=np.float32)
    return np.clip((arr.astype(np.float32) - lo) / denom, 0.0, 1.0)


def _thumb_b64(mem: np.ndarray, nuc: np.ndarray, size: int) -> str:
    panels = []
    for arr in (mem, nuc):
        pil = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
        pil = pil.resize((size, size), Image.LANCZOS)
        panels.append(np.array(pil))
    composite = np.concatenate(panels, axis=1)
    buf = io.BytesIO()
    Image.fromarray(composite, mode="L").save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _process_group(args):
    zarr_path, rep, cond, img, rows, thumb_px = args
    root = zarr.open_group(zarr_path, mode="r")
    pg = root[f"patches/{rep}/{cond}/{img}"]
    cnp = pg["cnPatches"]   # (N, H, W, 2) float32
    results = []
    for _, row in rows.iterrows():
        loc = int(row["local_cell_index"])
        ci  = int(row["cell_idx"])
        mem = _normalize(cnp[loc, :, :, 0], row["norm_mem_lo"], row["norm_mem_hi"])
        nuc = _normalize(cnp[loc, :, :, 1], row["norm_nuc_lo"], row["norm_nuc_hi"])
        results.append((ci, _thumb_b64(mem, nuc, thumb_px)))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",     required=True)
    p.add_argument("--table",    required=True)
    p.add_argument("--out",      default="outputs/feature_umap")
    p.add_argument("--thumb-px", type=int, default=48,
                   help="Thumbnail side length in pixels (default 48; larger → bigger file)")
    p.add_argument("--workers",  type=int, default=8)
    p.add_argument("--edge-threshold", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── load + filter ──────────────────────────────────────────────────────
    df = pd.read_csv(args.table)
    print(f"  loaded       {len(df)} cells")

    df = df[df["edge_run_px"] < args.edge_threshold].reset_index(drop=True)
    print(f"  edge filter  {len(df)} cells remaining")

    feat_cols = [c for c in df.columns
                 if c.startswith(("gFeat_", "tFeat_")) and "orientation" not in c]
    if not feat_cols:
        raise RuntimeError("No gFeat_/tFeat_ columns found — run ComputeFeatures.py first")
    print(f"  features     {len(feat_cols)}")

    n_before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    print(f"  NaN drop     {len(df)} cells ({n_before - len(df)} removed)")

    # ── scale + UMAP ───────────────────────────────────────────────────────
    X_raw = df[feat_cols].values.astype(np.float32)
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    print("  running UMAP …")
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1,
        n_components=2, random_state=42, verbose=True,
    )
    xy = reducer.fit_transform(X)
    print(f"  UMAP done    {xy.shape}")

    # ── thumbnails ─────────────────────────────────────────────────────────
    norm_cols = ["norm_mem_lo", "norm_mem_hi", "norm_nuc_lo", "norm_nuc_hi"]
    groups: dict = {}
    for _, row in df.iterrows():
        key = (str(row["replicate"]), str(row["condition"]), str(row["image_name"]))
        groups.setdefault(key, []).append(row)

    work = [
        (args.zarr, rep, cond, img, pd.DataFrame(rows), args.thumb_px)
        for (rep, cond, img), rows in groups.items()
    ]
    print(f"  thumbnails   {len(work)} groups × {args.workers} workers …")

    thumbnails: dict = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_process_group, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            for ci, b64 in fut.result():
                thumbnails[str(ci)] = b64
            done += 1
            if done % 50 == 0 or done == len(work):
                print(f"    {done}/{len(work)} groups", end="\r")
    print(f"\n  generated    {len(thumbnails)} thumbnails")

    # ── build data dict ────────────────────────────────────────────────────
    cell_idxs = df["cell_idx"].astype(int).tolist()

    data_dict = {
        "x":          xy[:, 0].tolist(),
        "y":          xy[:, 1].tolist(),
        "condition":  df["condition"].tolist(),
        "cell_idx":   cell_idxs,
        "feat_names": feat_cols,
        "zscores":    {col: X[:, i].tolist() for i, col in enumerate(feat_cols)},
        "feat_raw":   {col: [round(float(v), 4) for v in df[col]] for col in feat_cols},
        "thumbnails": {str(ci): thumbnails.get(str(ci), "") for ci in cell_idxs},
    }

    # ── write HTML ─────────────────────────────────────────────────────────
    print("  serialising data …")
    data_json = json.dumps(data_dict, separators=(",", ":"))
    html = (HTML_TEMPLATE
            .replace("__DATA_JSON__", data_json)
            .replace("__N_CELLS__",   str(len(df)))
            .replace("__N_FEATS__",   str(len(feat_cols))))

    out_path = out / "feature_umap_explorer.html"
    out_path.write_text(html, encoding="utf-8")
    mb = out_path.stat().st_size / 1e6
    print(f"  saved        {mb:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
