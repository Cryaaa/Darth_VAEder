"""Generate an interactive feature UMAP explorer HTML.

Loads all non-edge cells, applies QuantileTransformer normalisation (robust to
skewed size features), runs UMAP, embeds zarr thumbnails, and writes a
standalone HTML with:
  - Feature dropdown  → recolors points by normalised feature value (RdBu)
  - Condition legend  → click to toggle / double-click to isolate
  - Hover             → condition label
  - Click             → right panel: thumbnail + all feature values
  - Lasso/box select  → right panel: grid of thumbnails

Usage
-----
    python "Joaquin'scripts/feature_umap_explorer.py" \\
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table outputs/cell_table.csv
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
import umap

sys.path.insert(0, str(Path(__file__).parent))
from _feature_explorer_common import (
    load_and_filter, normalize_features, build_classes,
    generate_thumbnails, save_normalized_npz,
)

_SCRIPT_DIR  = Path(__file__).parent
_DEFAULT_OUT = str(_SCRIPT_DIR / "outputs" / "feature_umap")

# ---------------------------------------------------------------------------
# HTML template
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
            box-shadow: 0 1px 4px #ccc; flex-wrap: wrap }
    .ctrl label  { font-size: 13px; font-weight: bold; white-space: nowrap }
    .ctrl select { font-size: 13px; padding: 4px 8px; border: 1px solid #ddd;
                   border-radius: 4px; min-width: 240px; cursor: pointer }
    .ctrl .hint  { font-size: 11px; color: #aaa }
    #wrap      { display: flex; gap: 14px; align-items: flex-start }
    #plot-col  { flex: 3 }
    #panel-col { flex: 2; min-width: 340px; max-height: 760px; overflow-y: auto;
                 background: #fff; border-radius: 8px; padding: 14px;
                 box-shadow: 0 2px 8px #bbb }
    #panel-col h4 { margin: 0 0 10px; font-size: 15px }
    .hint-p { color: #ccc; font-style: italic }
    .ibox b   { font-size: 14px }
    .img-bg   { background: #0a0a0a; padding: 4px; border-radius: 4px;
                width: 100%; margin-top: 6px }
    .img-bg img { width: 100%; display: block; image-rendering: pixelated }
    .ch-labels  { display: flex; width: 100%; margin-top: 2px; margin-bottom: 8px }
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
    .tile img   { width: 100%; display: block; image-rendering: pixelated }
    .tile .tlbl { font-size: 9px; color: #bbb; text-align: center; padding: 2px 0 3px }
  </style>
</head>
<body>
<h2>Feature UMAP Explorer</h2>
<p class='sub'>
  __N_CELLS__ cells &nbsp;|&nbsp; __N_FEATS__ features
  &nbsp;|&nbsp; normalisation: QuantileTransformer &rarr; Gaussian
  &nbsp;|&nbsp; color = normalised value (RdBu &plusmn;3)
</p>
<div class='ctrl'>
  <label for='feat-sel'>Color by feature:</label>
  <select id='feat-sel'></select>
  <span class='hint'>blue = low &nbsp;&middot;&nbsp; red = high &nbsp;
    (click legend to toggle conditions)</span>
</div>
<div id='wrap'>
  <div id='plot-col'><div id='umap-plot'></div></div>
  <div id='panel-col'>
    <h4>Cell Inspector</h4>
    <div id='panel-content'>
      <p class='hint-p'>&#x1F446; Click a point to inspect
        &nbsp;&middot;&nbsp; lasso/box for a grid.</p>
    </div>
  </div>
</div>
<script>
const DATA = __DATA_JSON__;

function humanName(n) {
  return n.replace('gFeat_cell_', 'cell · ')
          .replace('gFeat_nuc_',  'nuc · ')
          .replace('tFeat_mem_',  'mem · ')
          .replace('tFeat_nuc_',  'nuc · ');
}

// ── feature dropdown ───────────────────────────────────────────────────────
const sel = document.getElementById('feat-sel');
DATA.feat_names.forEach(function(name) {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = humanName(name);
  sel.appendChild(opt);
});

// ── one trace per condition ────────────────────────────────────────────────
const firstFeat = DATA.feat_names[0];
const traces = DATA.classes.map(function(cls, i) {
  const cidx = cls.indices;
  return {
    x: cidx.map(function(j) { return DATA.x[j]; }),
    y: cidx.map(function(j) { return DATA.y[j]; }),
    mode: 'markers', type: 'scatter',
    name: cls.name,
    text: cidx.map(function() { return cls.name; }),
    customdata: cidx,
    hovertemplate: '<b>%{text}</b><extra></extra>',
    marker: {
      size: 4, opacity: 0.75,
      color: cidx.map(function(j) { return DATA.qnorm[firstFeat][j]; }),
      colorscale: 'RdBu', reversescale: true,
      showscale: i === 0, cmin: -3, cmax: 3,
      colorbar: i === 0 ? {
        title: { text: humanName(firstFeat), side: 'right' },
        thickness: 14, len: 0.6, x: 1.01
      } : {}
    }
  };
});

Plotly.newPlot('umap-plot', traces, {
  height: 700,
  margin: { l: 50, r: 80, t: 20, b: 50 },
  clickmode: 'event+select', dragmode: 'pan',
  plot_bgcolor: '#fff', paper_bgcolor: '#fff',
  xaxis: { title: 'UMAP 1', gridcolor: '#eee', zeroline: false },
  yaxis: { title: 'UMAP 2', gridcolor: '#eee', zeroline: false },
  legend: { bgcolor: '#fff', bordercolor: '#ddd', borderwidth: 1 },
}, { scrollZoom: true, responsive: true });

// ── dropdown → recolor all traces ─────────────────────────────────────────
sel.addEventListener('change', function() {
  const name = sel.value;
  const colors = DATA.classes.map(function(cls) {
    return cls.indices.map(function(j) { return DATA.qnorm[name][j]; });
  });
  Plotly.restyle('umap-plot', { 'marker.color': colors });
  Plotly.restyle('umap-plot', { 'marker.colorbar.title.text': humanName(name) }, [0]);
});

// ── panel ──────────────────────────────────────────────────────────────────
const panel = document.getElementById('panel-content');

function singleView(idx) {
  const ci = DATA.cell_idx[idx], cond = DATA.condition[idx], feat = sel.value;
  const src = DATA.thumbnails[String(ci)] || '';
  const rows = DATA.feat_names.map(function(f) {
    const cls = f === feat ? ' class=\'hi\'' : '';
    return '<tr><td' + cls + '>' + humanName(f) + '</td><td>' +
           DATA.feat_raw[f][idx] + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class=\'ibox\'><b>' + cond + '</b></div>' +
    (src ? '<div class=\'img-bg\'><img src=\'' + src + '\'></div>' +
           '<div class=\'ch-labels\'><span>membrane</span><span>nuclei</span></div>' : '') +
    '<table><tr><th>feature</th><th>raw value</th></tr>' + rows + '</table>';
}

function gridView(pts) {
  const MAX = 16, n = pts.length;
  let h = '<div class=\'ghdr\'>' + n + ' cell' + (n !== 1 ? 's' : '') + ' selected' +
          (n > MAX ? ' — first ' + MAX + ' shown' : '') + '</div><div class=\'grid\'>';
  for (let i = 0; i < Math.min(n, MAX); i++) {
    const idx = pts[i].customdata;
    const src = DATA.thumbnails[String(DATA.cell_idx[idx])] || '';
    h += '<div class=\'tile\'>' +
         (src ? '<img src=\'' + src + '\'>' : '<div style=\'height:48px;background:#333\'></div>') +
         '<div class=\'tlbl\'>' + DATA.condition[idx] + '</div></div>';
  }
  panel.innerHTML = h + '</div>';
}

const el = document.getElementById('umap-plot');
el.on('plotly_click',    function(ev) { singleView(ev.points[0].customdata); });
el.on('plotly_selected', function(ev) { if (ev && ev.points && ev.points.length) gridView(ev.points); });
</script>
</body>
</html>'''


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zarr",           required=True)
    p.add_argument("--table",          required=True)
    p.add_argument("--out",            default=_DEFAULT_OUT)
    p.add_argument("--thumb-px",       type=int, default=48)
    p.add_argument("--workers",        type=int, default=8)
    p.add_argument("--edge-threshold", type=int, default=5)
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df, feat_cols = load_and_filter(args.table, args.edge_threshold)

    X_raw  = df[feat_cols].values.astype("float32")
    X_norm = normalize_features(X_raw)
    print(f"  normalised   QuantileTransformer → Gaussian")

    print("  running UMAP …")
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1,
                   n_components=2, random_state=42, verbose=True).fit_transform(X_norm)
    print(f"  UMAP done    {xy.shape}")

    thumbnails = generate_thumbnails(df, args.zarr, args.thumb_px, args.workers)
    save_normalized_npz(df, X_norm, feat_cols, out)

    classes   = build_classes(df)
    cell_idxs = df["cell_idx"].astype(int).tolist()

    data_dict = {
        "x":         xy[:, 0].tolist(),
        "y":         xy[:, 1].tolist(),
        "condition": df["condition"].tolist(),
        "cell_idx":  cell_idxs,
        "classes":   classes,
        "feat_names": feat_cols,
        "qnorm":     {col: X_norm[:, i].tolist() for i, col in enumerate(feat_cols)},
        "feat_raw":  {col: [round(float(v), 4) for v in df[col]] for col in feat_cols},
        "thumbnails": {str(ci): thumbnails.get(str(ci), "") for ci in cell_idxs},
    }

    print("  serialising …")
    data_json = json.dumps(data_dict, separators=(",", ":"))
    html = (HTML_TEMPLATE
            .replace("__DATA_JSON__", data_json)
            .replace("__N_CELLS__",   str(len(df)))
            .replace("__N_FEATS__",   str(len(feat_cols))))

    out_path = out / "feature_umap_explorer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved        {out_path.stat().st_size / 1e6:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
