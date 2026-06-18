"""Generate an interactive feature PCA explorer HTML.

Same concept as feature_umap_explorer but using PCA instead of UMAP:
  - Feature dropdown  → recolors points by normalised feature value (RdBu)
  - Axis-pair selector→ switch between PC1-PC2, PC1-PC3, PC2-PC3, …
  - Condition legend  → click to toggle / double-click to isolate
  - Hover             → condition label
  - Default panel     → top loadings for the shown PC axes
  - Click             → thumbnail + feature table
  - Lasso/box select  → grid of thumbnails

Usage
-----
    python "Joaquin'scripts/feature_pca_explorer.py" \\
        --zarr  /mnt/efs/dl_jrc/student_data/S-JS/multinucleation.zarr \\
        --table outputs/cell_table.csv
"""

import argparse, json, sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).parent))
from _feature_explorer_common import (
    load_and_filter, normalize_features, build_classes,
    generate_thumbnails,
)

_SCRIPT_DIR  = Path(__file__).parent
_DEFAULT_OUT = str(_SCRIPT_DIR / "outputs" / "feature_pca")

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'/>
  <title>Feature PCA Explorer</title>
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
                   border-radius: 4px; cursor: pointer }
    .ctrl #feat-sel  { min-width: 220px }
    .ctrl #axis-sel  { min-width: 150px }
    .ctrl .hint  { font-size: 11px; color: #aaa }
    #wrap      { display: flex; gap: 14px; align-items: flex-start }
    #plot-col  { flex: 3 }
    #panel-col { flex: 2; min-width: 340px; max-height: 760px; overflow-y: auto;
                 background: #fff; border-radius: 8px; padding: 14px;
                 box-shadow: 0 2px 8px #bbb }
    #panel-col h4 { margin: 0 0 10px; font-size: 15px }
    .hint-p { color: #ccc; font-style: italic }
    .ibox b  { font-size: 14px }
    .img-bg  { background: #0a0a0a; padding: 4px; border-radius: 4px;
               width: 100%; margin-top: 6px }
    .img-bg img { width: 100%; display: block; image-rendering: pixelated }
    .ch-labels  { display: flex; width: 100%; margin-top: 2px; margin-bottom: 8px }
    .ch-labels span { flex: 1; text-align: center; font-size: 10px; color: #999 }
    table  { border-collapse: collapse; width: 100%; font-size: 12px }
    th     { text-align: left; color: #888; font-weight: normal;
             border-bottom: 1px solid #eee; padding: 3px 6px }
    td     { padding: 3px 6px; border-bottom: 1px solid #f4f4f4 }
    td.hi  { font-weight: bold; color: #c0392b }
    td.pos { color: #c0392b }
    td.neg { color: #2980b9 }
    td:last-child { text-align: right; font-family: monospace }
    .ghdr  { font-size: 13px; font-weight: bold; margin: 10px 0 6px }
    .ghdr:first-child { margin-top: 0 }
    .grid  { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px }
    .tile  { background: #111; border-radius: 3px; padding: 3px 3px 0 3px }
    .tile img   { width: 100%; display: block; image-rendering: pixelated }
    .tile .tlbl { font-size: 9px; color: #bbb; text-align: center; padding: 2px 0 3px }
    .ev-bar { display: inline-block; height: 6px; background: #4e9de0;
              border-radius: 3px; margin-left: 6px; vertical-align: middle }
  </style>
</head>
<body>
<h2>Feature PCA Explorer</h2>
<p class='sub'>
  __N_CELLS__ cells &nbsp;|&nbsp; __N_FEATS__ features
  &nbsp;|&nbsp; normalisation: QuantileTransformer &rarr; Gaussian
  &nbsp;|&nbsp; color = normalised value (RdBu &plusmn;3)
</p>
<div class='ctrl'>
  <label for='feat-sel'>Color by:</label>
  <select id='feat-sel'></select>
  <label for='axis-sel'>Axes:</label>
  <select id='axis-sel'></select>
  <span class='hint'>click legend to toggle conditions</span>
</div>
<div id='wrap'>
  <div id='plot-col'><div id='pca-plot'></div></div>
  <div id='panel-col'>
    <h4>PCA Inspector</h4>
    <div id='panel-content'></div>
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

function pcLabel(i) {
  return 'PC' + (i + 1) + ' (' + (DATA.ev[i] * 100).toFixed(1) + '%)';
}

// ── feature dropdown ───────────────────────────────────────────────────────
const featSel = document.getElementById('feat-sel');
DATA.feat_names.forEach(function(name) {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = humanName(name);
  featSel.appendChild(opt);
});

// ── axis-pair dropdown (populated from DATA.ev length) ────────────────────
const axisSel = document.getElementById('axis-sel');
for (let a = 0; a < DATA.ev.length - 1; a++) {
  for (let b = a + 1; b < DATA.ev.length; b++) {
    const opt = document.createElement('option');
    opt.value = a + ',' + b;
    opt.textContent = 'PC' + (a+1) + ' vs PC' + (b+1);
    axisSel.appendChild(opt);
  }
}

// ── build initial traces (PC1 vs PC2) ─────────────────────────────────────
const firstFeat = DATA.feat_names[0];
let curA = 0, curB = 1;

function makeTraces(pcA, pcB, featName) {
  return DATA.classes.map(function(cls, i) {
    const cidx = cls.indices;
    return {
      x: cidx.map(function(j) { return DATA.pca[j][pcA]; }),
      y: cidx.map(function(j) { return DATA.pca[j][pcB]; }),
      mode: 'markers', type: 'scatter',
      name: cls.name,
      text: cidx.map(function() { return cls.name; }),
      customdata: cidx,
      hovertemplate: '<b>%{text}</b><extra></extra>',
      marker: {
        size: 4, opacity: 0.75,
        color: cidx.map(function(j) { return DATA.qnorm[featName][j]; }),
        colorscale: 'RdBu', reversescale: true,
        showscale: i === 0, cmin: -3, cmax: 3,
        colorbar: i === 0 ? {
          title: { text: humanName(featName), side: 'right' },
          thickness: 14, len: 0.6, x: 1.01
        } : {}
      }
    };
  });
}

Plotly.newPlot('pca-plot', makeTraces(0, 1, firstFeat), {
  height: 700,
  margin: { l: 50, r: 80, t: 20, b: 50 },
  clickmode: 'event+select', dragmode: 'pan',
  plot_bgcolor: '#fff', paper_bgcolor: '#fff',
  xaxis: { title: pcLabel(0), gridcolor: '#eee', zeroline: false },
  yaxis: { title: pcLabel(1), gridcolor: '#eee', zeroline: false },
  legend: { bgcolor: '#fff', bordercolor: '#ddd', borderwidth: 1 },
}, { scrollZoom: true, responsive: true });

// ── feature dropdown → recolor ─────────────────────────────────────────────
featSel.addEventListener('change', function() {
  const name = featSel.value;
  const colors = DATA.classes.map(function(cls) {
    return cls.indices.map(function(j) { return DATA.qnorm[name][j]; });
  });
  Plotly.restyle('pca-plot', { 'marker.color': colors });
  Plotly.restyle('pca-plot', { 'marker.colorbar.title.text': humanName(name) }, [0]);
});

// ── axis selector → restyle x/y + relayout titles ─────────────────────────
axisSel.addEventListener('change', function() {
  const parts = axisSel.value.split(',');
  curA = Number(parts[0]); curB = Number(parts[1]);
  const xArr = DATA.classes.map(function(cls) {
    return cls.indices.map(function(j) { return DATA.pca[j][curA]; });
  });
  const yArr = DATA.classes.map(function(cls) {
    return cls.indices.map(function(j) { return DATA.pca[j][curB]; });
  });
  Plotly.restyle('pca-plot', { x: xArr, y: yArr });
  Plotly.relayout('pca-plot', {
    'xaxis.title': pcLabel(curA),
    'yaxis.title': pcLabel(curB),
  });
  showLoadings(curA, curB);
});

// ── panel: loadings view (default) ────────────────────────────────────────
const panel = document.getElementById('panel-content');

function topLoadings(pcIdx, n) {
  return DATA.feat_names
    .map(function(name, i) { return { name: name, val: DATA.loadings[pcIdx][i] }; })
    .sort(function(a, b) { return Math.abs(b.val) - Math.abs(a.val); })
    .slice(0, n);
}

function loadingRow(f) {
  const cls = f.val > 0 ? 'pos' : 'neg';
  const bar = (f.val > 0 ? '+' : '') + f.val.toFixed(3);
  return '<tr><td>' + humanName(f.name) + '</td><td class=\'' + cls + '\'>' + bar + '</td></tr>';
}

function evRow(i) {
  const pct = (DATA.ev[i] * 100).toFixed(1);
  const w   = Math.round(DATA.ev[i] / DATA.ev[0] * 80);
  return '<tr><td>PC' + (i+1) + '</td><td>' + pct + '%' +
         '<span class=\'ev-bar\' style=\'width:' + w + 'px\'></span></td></tr>';
}

function showLoadings(pcA, pcB) {
  const aTop = topLoadings(pcA, 6), bTop = topLoadings(pcB, 6);
  const evRows = DATA.ev.map(function(_, i) { return evRow(i); }).join('');
  panel.innerHTML =
    '<div class=\'ghdr\'>Explained variance</div>' +
    '<table><tr><th>PC</th><th>variance</th></tr>' + evRows + '</table>' +
    '<div class=\'ghdr\'>PC' + (pcA+1) + ' top loadings</div>' +
    '<table><tr><th>feature</th><th>loading</th></tr>' +
    aTop.map(loadingRow).join('') + '</table>' +
    '<div class=\'ghdr\'>PC' + (pcB+1) + ' top loadings</div>' +
    '<table><tr><th>feature</th><th>loading</th></tr>' +
    bTop.map(loadingRow).join('') + '</table>';
}
showLoadings(0, 1);

// ── click → inspect ────────────────────────────────────────────────────────
function singleView(idx) {
  const ci = DATA.cell_idx[idx], cond = DATA.condition[idx], feat = featSel.value;
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

// ── lasso/box → grid ───────────────────────────────────────────────────────
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

const el = document.getElementById('pca-plot');
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
    p.add_argument("--n-pcs",          type=int, default=10,
                   help="Number of PCs to compute and store (default 10)")
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

    n_pcs = min(args.n_pcs, len(feat_cols))
    pca   = PCA(n_components=n_pcs, random_state=42)
    scores = pca.fit_transform(X_norm)   # (N, n_pcs)
    ev     = pca.explained_variance_ratio_
    print(f"  PCA done     {n_pcs} PCs  |  cumulative variance: "
          f"{ev.sum()*100:.1f}%  (PC1={ev[0]*100:.1f}%, PC2={ev[1]*100:.1f}%)")

    thumbnails = generate_thumbnails(df, args.zarr, args.thumb_px, args.workers)

    classes   = build_classes(df)
    cell_idxs = df["cell_idx"].astype(int).tolist()

    data_dict = {
        "x":         scores[:, 0].tolist(),
        "y":         scores[:, 1].tolist(),
        "pca":       scores.tolist(),              # (N, n_pcs) — all axes
        "ev":        ev.tolist(),                  # (n_pcs,)
        "loadings":  pca.components_.tolist(),     # (n_pcs, n_features)
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

    out_path = out / "feature_pca_explorer.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved        {out_path.stat().st_size / 1e6:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
