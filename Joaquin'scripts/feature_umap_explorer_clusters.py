"""Interactive feature UMAP explorer with hand-drawn cluster analysis.

Same UMAP layout as feature_umap_explorer.html, plus:
  - Hand-draw clusters by lasso/box → "Assign to cluster" button
  - Per-cluster condition composition, normalised by condition population
  - Per-cluster feature enrichment (mean qnorm score, ranked bar chart)
  - All existing features preserved: feature dropout recolor, condition legend
    toggle, hover=condition, click → thumbnail + feature table, lasso → grid

Usage
-----
    python "Joaquin'scripts/feature_umap_explorer_clusters.py" \\
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

# ─────────────────────────────────────────────────────────────────────────────
HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang='en'>
<head>
  <meta charset='UTF-8'/>
  <title>Feature UMAP Explorer — Clusters</title>
  <script src='https://cdn.plot.ly/plotly-2.30.0.min.js'></script>
  <style>
    *, *::before, *::after { box-sizing: border-box }
    body  { font-family: Arial, sans-serif; margin: 0; padding: 10px 14px;
            background: #f0f0f0; color: #222; font-size: 13px }
    h2    { margin: 0 0 2px }
    .sub  { color: #999; font-size: 12px; margin: 0 0 8px }

    /* ── control bars ───────────────────────────────────────────────── */
    .ctrl { display: flex; align-items: center; gap: 10px; margin-bottom: 6px;
            background: #fff; border-radius: 8px; padding: 7px 12px;
            box-shadow: 0 1px 4px #ccc; flex-wrap: wrap }
    .ctrl label  { font-weight: bold; white-space: nowrap }
    .ctrl select { font-size: 13px; padding: 3px 7px; border: 1px solid #ddd;
                   border-radius: 4px; min-width: 220px; cursor: pointer }
    .ctrl .hint  { font-size: 11px; color: #aaa }

    /* cluster pills */
    .cpill { display: inline-flex; align-items: center; gap: 4px;
             padding: 3px 10px; border-radius: 12px; cursor: pointer;
             font-size: 12px; font-weight: bold; border: 2px solid transparent;
             transition: opacity .15s }
    .cpill.active  { opacity: 1 }
    .cpill.inactive{ opacity: .45 }
    .cbadge { background: rgba(0,0,0,.18); border-radius: 8px;
              padding: 0 5px; font-size: 10px }

    /* buttons */
    button { font-size: 12px; padding: 4px 10px; border-radius: 5px;
             cursor: pointer; border: 1px solid #ccc; background: #fff }
    button:hover { background: #f5f5f5 }
    button.primary { background: #2c7be5; color: #fff; border-color: #2c7be5 }
    button.primary:hover { background: #1a6dd6 }
    button.danger  { background: #e74c3c; color: #fff; border-color: #e74c3c }
    button.danger:hover  { background: #c0392b }
    #sel-badge { font-size: 11px; color: #888 }

    /* ── main layout ────────────────────────────────────────────────── */
    #wrap      { display: flex; gap: 12px; align-items: flex-start }
    #plot-col  { flex: 3 }
    #panel-col { flex: 2; min-width: 340px; max-height: 780px; overflow-y: auto;
                 background: #fff; border-radius: 8px;
                 box-shadow: 0 2px 8px #bbb }

    /* panel tabs */
    .tab-bar   { display: flex; border-bottom: 2px solid #eee }
    .tab-btn   { flex: 1; padding: 9px 0; text-align: center; cursor: pointer;
                 font-size: 13px; font-weight: bold; color: #888;
                 background: none; border: none; border-bottom: 3px solid transparent;
                 margin-bottom: -2px }
    .tab-btn.active { color: #2c7be5; border-bottom-color: #2c7be5 }
    .tab-content { padding: 12px }
    #tab-cell  {}
    #tab-stats {}

    /* cell inspector (tab-cell) */
    .hint-p { color: #ccc; font-style: italic }
    .ibox b  { font-size: 14px }
    .img-bg  { background: #0a0a0a; padding: 4px; border-radius: 4px;
               width: 100%; margin-top: 6px }
    .img-bg img { width: 100%; display: block; image-rendering: pixelated }
    .ch-labels  { display: flex; width: 100%; margin-top: 2px; margin-bottom: 8px }
    .ch-labels span { flex: 1; text-align: center; font-size: 10px; color: #999 }
    table  { border-collapse: collapse; width: 100%; font-size: 11px }
    th     { text-align: left; color: #888; font-weight: normal;
             border-bottom: 1px solid #eee; padding: 3px 5px }
    td     { padding: 3px 5px; border-bottom: 1px solid #f4f4f4 }
    td.hi  { font-weight: bold; color: #c0392b }
    td:last-child { text-align: right; font-family: monospace }
    .ghdr  { font-size: 13px; font-weight: bold; margin-bottom: 8px }
    .grid  { display: grid; grid-template-columns: repeat(2, 1fr); gap: 6px }
    .tile  { background: #111; border-radius: 3px; padding: 3px 3px 0 3px }
    .tile img   { width: 100%; display: block; image-rendering: pixelated }
    .tile .tlbl { font-size: 9px; color: #bbb; text-align: center; padding: 2px 0 3px }

    /* cluster stats (tab-stats) */
    .stats-section { margin-bottom: 16px }
    .stats-section h5 { margin: 0 0 6px; font-size: 13px; color: #555;
                        border-bottom: 1px solid #eee; padding-bottom: 4px }
    .comp-table { font-size: 11px }
    .comp-table td { text-align: center }
    .comp-table td.cname { text-align: left; font-weight: bold }
    .comp-table td.pct  { font-family: monospace }
    .no-data { color: #bbb; font-style: italic; font-size: 12px }
    #comp-chart   { width: 100%; min-height: 160px }
    #enrich-chart { width: 100%; min-height: 200px }
  </style>
</head>
<body>
<h2>Feature UMAP Explorer — Clusters</h2>
<p class='sub'>
  __N_CELLS__ cells &nbsp;|&nbsp; __N_FEATS__ features
  &nbsp;|&nbsp; normalisation: QuantileTransformer &rarr; Gaussian
</p>

<!-- feature dropdown -->
<div class='ctrl'>
  <label for='feat-sel'>Color by feature:</label>
  <select id='feat-sel'></select>
  <span class='hint'>blue = low &middot; red = high &nbsp;(click legend to toggle conditions)</span>
</div>

<!-- cluster controls -->
<div class='ctrl' id='cluster-bar'>
  <label>Cluster:</label>
  <div id='cpills'></div>
  <button id='btn-add' title='Add cluster (max 6)'>＋</button>
  <button id='btn-assign' class='primary'>Assign &nbsp;<span id='sel-badge'>0 pts</span></button>
  <button id='btn-clear'>Clear active</button>
  <button id='btn-reset' class='danger'>Reset all</button>
  <button id='btn-toggle-ov'>Hide overlays</button>
  <button id='btn-save-csv'>⬇ CSV</button>
</div>

<!-- main area -->
<div id='wrap'>
  <div id='plot-col'><div id='umap-plot'></div></div>
  <div id='panel-col'>
    <div class='tab-bar'>
      <button class='tab-btn active' id='tabBtn-cell'  onclick="showTab('cell')">🖼 Cell View</button>
      <button class='tab-btn'        id='tabBtn-stats' onclick="showTab('stats')">📊 Cluster Stats</button>
    </div>
    <div id='tab-cell'  class='tab-content'>
      <div id='panel-content'><p class='hint-p'>&#x1F446; Click a point to inspect &middot; lasso/box for a grid.</p></div>
    </div>
    <div id='tab-stats' class='tab-content' style='display:none'>
      <div class='stats-section'>
        <h5>Condition composition (% of each condition's population)</h5>
        <div id='comp-table-wrap'><p class='no-data'>No clusters assigned yet.</p></div>
        <div id='comp-chart'></div>
      </div>
      <div class='stats-section'>
        <h5>Feature enrichment — active cluster (mean qnorm)</h5>
        <div id='enrich-chart'></div>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = __DATA_JSON__;

// ── helpers ────────────────────────────────────────────────────────────────
function humanName(n) {
  return n.replace('gFeat_cell_','cell · ').replace('gFeat_nuc_','nuc · ')
          .replace('tFeat_mem_','mem · ').replace('tFeat_nuc_','nuc · ');
}

// ── cluster state ──────────────────────────────────────────────────────────
const MAX_CL = 6;
const CL_COLORS = ['#e6194b','#3cb44b','#4363d8','#f58231','#911eb4','#42d4f4'];
let nClusters      = 1;
let activeCI       = 0;
let clusterSets    = Array.from({length: MAX_CL}, () => new Set());
let selIndices     = [];   // current lasso/click selection
let overlaysOn     = true;

const condTotals = {};
DATA.classes.forEach(cls => { condTotals[cls.name] = cls.indices.length; });
const nCond = DATA.classes.length;

// ── feature dropdown ───────────────────────────────────────────────────────
const sel = document.getElementById('feat-sel');
DATA.feat_names.forEach(function(name) {
  const opt = document.createElement('option');
  opt.value = name; opt.textContent = humanName(name);
  sel.appendChild(opt);
});
const firstFeat = DATA.feat_names[0];

// ── base condition traces ─────────────────────────────────────────────────
const baseTraces = DATA.classes.map(function(cls, i) {
  const cidx = cls.indices;
  return {
    x: cidx.map(j => DATA.x[j]),
    y: cidx.map(j => DATA.y[j]),
    mode: 'markers', type: 'scatter',
    name: cls.name,
    text: cidx.map(() => cls.name),
    customdata: cidx,
    hovertemplate: '<b>%{text}</b><extra></extra>',
    marker: {
      size: 4, opacity: 0.75,
      color: cidx.map(j => DATA.qnorm[firstFeat][j]),
      colorscale: 'RdBu', reversescale: true,
      showscale: i === 0, cmin: -3, cmax: 3,
      colorbar: i === 0 ? {
        title: { text: humanName(firstFeat), side: 'right' },
        thickness: 14, len: 0.6, x: 1.01
      } : {}
    }
  };
});

// ── cluster overlay traces (pre-created, empty) ───────────────────────────
const overlayTraces = Array.from({length: MAX_CL}, (_, ci) => ({
  x: [], y: [],
  mode: 'markers', type: 'scatter',
  name: 'Cluster ' + (ci + 1),
  showlegend: false,
  hoverinfo: 'skip',
  marker: {
    symbol: 'circle-open', size: 11, opacity: 1,
    line: { width: 2.5, color: CL_COLORS[ci] },
    color: 'rgba(0,0,0,0)'
  }
}));

Plotly.newPlot('umap-plot', [...baseTraces, ...overlayTraces], {
  height: 700,
  margin: { l: 50, r: 80, t: 20, b: 50 },
  clickmode: 'event+select', dragmode: 'lasso',
  plot_bgcolor: '#fff', paper_bgcolor: '#fff',
  xaxis: { title: 'UMAP 1', gridcolor: '#eee', zeroline: false },
  yaxis: { title: 'UMAP 2', gridcolor: '#eee', zeroline: false },
  legend: { bgcolor: '#fff', bordercolor: '#ddd', borderwidth: 1 },
}, { scrollZoom: true, responsive: true });

// ── cluster pill UI ────────────────────────────────────────────────────────
function renderPills() {
  const wrap = document.getElementById('cpills');
  wrap.innerHTML = '';
  for (let ci = 0; ci < nClusters; ci++) {
    const n = clusterSets[ci].size;
    const pill = document.createElement('span');
    pill.className = 'cpill ' + (ci === activeCI ? 'active' : 'inactive');
    pill.style.background = CL_COLORS[ci];
    pill.style.color = '#fff';
    pill.innerHTML = 'Cluster ' + (ci + 1) +
      (n ? ' <span class="cbadge">' + n + '</span>' : '');
    const _ci = ci;
    pill.onclick = () => { activeCI = _ci; renderPills(); updateEnrichChart(); };
    wrap.appendChild(pill);
  }
  document.getElementById('btn-add').disabled = nClusters >= MAX_CL;
}
renderPills();

document.getElementById('btn-add').onclick = function() {
  if (nClusters >= MAX_CL) return;
  nClusters++;
  activeCI = nClusters - 1;
  renderPills();
};

document.getElementById('btn-assign').onclick = function() {
  if (!selIndices.length) return;
  // exclusive membership: remove from all other clusters
  for (let ci = 0; ci < MAX_CL; ci++) {
    if (ci !== activeCI) selIndices.forEach(i => clusterSets[ci].delete(i));
  }
  selIndices.forEach(i => clusterSets[activeCI].add(i));
  selIndices = [];
  document.getElementById('sel-badge').textContent = '0 pts';
  renderPills();
  updateOverlays();
  updateStatsPanel();
  showTab('stats');
};

document.getElementById('btn-clear').onclick = function() {
  clusterSets[activeCI] = new Set();
  renderPills();
  updateOverlays();
  updateStatsPanel();
};

document.getElementById('btn-reset').onclick = function() {
  clusterSets = Array.from({length: MAX_CL}, () => new Set());
  nClusters = 1; activeCI = 0;
  renderPills();
  updateOverlays();
  updateStatsPanel();
};

document.getElementById('btn-toggle-ov').onclick = function() {
  overlaysOn = !overlaysOn;
  this.textContent = overlaysOn ? 'Hide overlays' : 'Show overlays';
  const vis = overlaysOn ? true : 'legendonly';
  const traceIds = Array.from({length: MAX_CL}, (_, i) => nCond + i);
  Plotly.restyle('umap-plot', { visible: overlaysOn ? true : false }, traceIds);
};

document.getElementById('btn-save-csv').onclick = function() {
  const rows = ['cell_idx,condition,cluster'];
  for (let ci = 0; ci < nClusters; ci++) {
    clusterSets[ci].forEach(i => {
      rows.push(DATA.cell_idx[i] + ',' + DATA.condition[i] + ',' + (ci + 1));
    });
  }
  const blob = new Blob([rows.join('\n')], {type: 'text/csv'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'clusters.csv';
  a.click();
};

// ── feature dropdown recolor (only base traces) ───────────────────────────
sel.addEventListener('change', function() {
  const name = sel.value;
  const colors = DATA.classes.map(cls => cls.indices.map(j => DATA.qnorm[name][j]));
  const traceIds = DATA.classes.map((_, i) => i);
  Plotly.restyle('umap-plot', { 'marker.color': colors }, traceIds);
  Plotly.restyle('umap-plot', { 'marker.colorbar.title.text': humanName(name) }, [0]);
});

// ── cluster overlay update ────────────────────────────────────────────────
function updateOverlays() {
  if (!overlaysOn) return;
  for (let ci = 0; ci < MAX_CL; ci++) {
    const indices = [...clusterSets[ci]];
    Plotly.restyle('umap-plot', {
      x: [indices.map(i => DATA.x[i])],
      y: [indices.map(i => DATA.y[i])],
    }, [nCond + ci]);
  }
}

// ── tab switching ─────────────────────────────────────────────────────────
function showTab(tab) {
  document.getElementById('tab-cell').style.display  = tab === 'cell'  ? '' : 'none';
  document.getElementById('tab-stats').style.display = tab === 'stats' ? '' : 'none';
  document.getElementById('tabBtn-cell').classList.toggle('active',  tab === 'cell');
  document.getElementById('tabBtn-stats').classList.toggle('active', tab === 'stats');
  if (tab === 'stats') { updateStatsPanel(); }
}

// ── stats panel ───────────────────────────────────────────────────────────
function updateStatsPanel() {
  updateCompositionTable();
  updateCompChart();
  updateEnrichChart();
}

function activeClusters() {
  return Array.from({length: nClusters}, (_, i) => i).filter(i => clusterSets[i].size > 0);
}

function updateCompositionTable() {
  const wrap = document.getElementById('comp-table-wrap');
  const aci  = activeClusters();
  if (!aci.length) {
    wrap.innerHTML = '<p class="no-data">No cells assigned yet.</p>';
    return;
  }
  const conds = DATA.classes.map(c => c.name);
  let html = '<table class="comp-table"><thead><tr><th>Cluster</th>';
  conds.forEach(c => { html += '<th>' + c + '</th>'; });
  html += '</tr></thead><tbody>';
  aci.forEach(ci => {
    html += '<tr><td class="cname" style="color:' + CL_COLORS[ci] + '">' +
            '■ Cluster ' + (ci + 1) + '</td>';
    conds.forEach(cname => {
      const cls  = DATA.classes.find(c => c.name === cname);
      const n    = cls.indices.filter(i => clusterSets[ci].has(i)).length;
      const pct  = condTotals[cname] > 0 ? (n / condTotals[cname] * 100).toFixed(1) : '0.0';
      html += '<td class="pct" title="' + n + ' cells">' + pct + '%</td>';
    });
    html += '</tr>';
  });
  html += '</tbody></table>';
  wrap.innerHTML = html;
}

function updateCompChart() {
  const el  = document.getElementById('comp-chart');
  const aci = activeClusters();
  if (!aci.length) { Plotly.purge(el); return; }
  const traces = aci.map(ci => ({
    x: DATA.classes.map(c => c.name),
    y: DATA.classes.map(c => {
      const n = c.indices.filter(i => clusterSets[ci].has(i)).length;
      return condTotals[c.name] > 0 ? +(n / condTotals[c.name] * 100).toFixed(2) : 0;
    }),
    name: 'Cluster ' + (ci + 1),
    type: 'bar',
    marker: { color: CL_COLORS[ci] },
  }));
  Plotly.react(el, traces, {
    barmode: 'group',
    height: 180,
    margin: {l: 40, r: 10, t: 6, b: 50},
    yaxis: { title: '% of condition', rangemode: 'tozero' },
    plot_bgcolor: '#fff', paper_bgcolor: '#fff',
    legend: { orientation: 'h', y: -0.35, font: {size: 11} },
    font: { size: 11 },
  }, { responsive: true });
}

function updateEnrichChart() {
  const el  = document.getElementById('enrich-chart');
  const set = clusterSets[activeCI];
  if (!set.size) { Plotly.purge(el); return; }
  const indices = [...set];
  const scores  = DATA.feat_names.map(f => {
    const vals = indices.map(i => DATA.qnorm[f][i]);
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  });
  // sort ascending so highest appears at top of horizontal bar chart
  const order = scores.map((s, i) => [s, i]).sort((a, b) => a[0] - b[0]);
  const yNames = order.map(([, i]) => humanName(DATA.feat_names[i]));
  const xVals  = order.map(([s]) => +s.toFixed(3));
  const colors = xVals.map(s => s >= 0 ? '#c0392b' : '#2980b9');

  Plotly.react(el, [{
    x: xVals, y: yNames,
    type: 'bar', orientation: 'h',
    marker: { color: colors },
  }], {
    height: Math.max(220, yNames.length * 16 + 60),
    margin: {l: 160, r: 20, t: 6, b: 40},
    xaxis: { title: 'mean qnorm (effect size)', zeroline: true,
             zerolinecolor: '#999', zerolinewidth: 1.5 },
    plot_bgcolor: '#fff', paper_bgcolor: '#fff',
    font: { size: 10 },
  }, { responsive: true });
}

// ── point events ──────────────────────────────────────────────────────────
const panel = document.getElementById('panel-content');

function singleView(idx) {
  const ci = DATA.cell_idx[idx], cond = DATA.condition[idx], feat = sel.value;
  const src = DATA.thumbnails[String(ci)] || '';
  const rows = DATA.feat_names.map(f => {
    const cls = f === feat ? ' class="hi"' : '';
    return '<tr><td' + cls + '>' + humanName(f) + '</td><td>' +
           DATA.feat_raw[f][idx] + '</td></tr>';
  }).join('');
  panel.innerHTML =
    '<div class="ibox"><b>' + cond + '</b></div>' +
    (src ? '<div class="img-bg"><img src="' + src + '"></div>' +
           '<div class="ch-labels"><span>membrane</span><span>nuclei</span></div>' : '') +
    '<table><tr><th>feature</th><th>raw value</th></tr>' + rows + '</table>';
}

function gridView(pts) {
  const MAX = 16, n = pts.length;
  let h = '<div class="ghdr">' + n + ' cell' + (n !== 1 ? 's' : '') + ' selected' +
          (n > MAX ? ' — first ' + MAX + ' shown' : '') + '</div><div class="grid">';
  for (let i = 0; i < Math.min(n, MAX); i++) {
    const idx = pts[i].customdata;
    const src = DATA.thumbnails[String(DATA.cell_idx[idx])] || '';
    h += '<div class="tile">' +
         (src ? '<img src="' + src + '">' : '<div style="height:48px;background:#333"></div>') +
         '<div class="tlbl">' + DATA.condition[idx] + '</div></div>';
  }
  panel.innerHTML = h + '</div>';
}

const el = document.getElementById('umap-plot');

el.on('plotly_click', function(ev) {
  const pt = ev.points[0];
  // ignore clicks on overlay traces
  if (pt.fullData && pt.fullData.hoverinfo === 'skip') return;
  selIndices = [pt.customdata];
  document.getElementById('sel-badge').textContent = '1 pt';
  showTab('cell');
  singleView(pt.customdata);
});

el.on('plotly_selected', function(ev) {
  if (!ev || !ev.points || !ev.points.length) return;
  // filter out overlay trace points (hoverinfo='skip')
  const pts = ev.points.filter(p => !(p.fullData && p.fullData.hoverinfo === 'skip'));
  if (!pts.length) return;
  selIndices = pts.map(p => p.customdata);
  document.getElementById('sel-badge').textContent = selIndices.length + ' pts';
  showTab('cell');
  gridView(pts);
});
</script>
</body>
</html>'''


# ─────────────────────────────────────────────────────────────────────────────

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
    print("  normalised   QuantileTransformer → Gaussian")

    print("  running UMAP …")
    xy = umap.UMAP(n_neighbors=15, min_dist=0.1,
                   n_components=2, random_state=42, verbose=True).fit_transform(X_norm)
    print(f"  UMAP done    {xy.shape}")

    thumbnails = generate_thumbnails(df, args.zarr, args.thumb_px, args.workers)

    classes   = build_classes(df)
    cell_idxs = df["cell_idx"].astype(int).tolist()

    data_dict = {
        "x":          xy[:, 0].tolist(),
        "y":          xy[:, 1].tolist(),
        "condition":  df["condition"].tolist(),
        "cell_idx":   cell_idxs,
        "classes":    classes,
        "feat_names": feat_cols,
        "qnorm":      {col: X_norm[:, i].tolist() for i, col in enumerate(feat_cols)},
        "feat_raw":   {col: [round(float(v), 4) for v in df[col]] for col in feat_cols},
        "thumbnails": {str(ci): thumbnails.get(str(ci), "") for ci in cell_idxs},
    }

    print("  serialising …")
    data_json = json.dumps(data_dict, separators=(",", ":"))
    html = (HTML_TEMPLATE
            .replace("__DATA_JSON__", data_json)
            .replace("__N_CELLS__",   str(len(df)))
            .replace("__N_FEATS__",   str(len(feat_cols))))

    out_path = out / "feature_umap_explorer_clusters.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"  saved        {out_path.stat().st_size / 1e6:.1f} MB → {out_path}")


if __name__ == "__main__":
    main()
