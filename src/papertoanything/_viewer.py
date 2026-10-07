"""The built-in raw viewer, served at / when no Lab build is bundled.

Plain numbers from /spec, /health and /events, no diagnoses (those live in
the Lab's engine) and no external requests of any kind.
"""

VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="referrer" content="no-referrer">
<title>pta watch</title>
<style>
:root { color-scheme: dark; --bg:#101114; --fg:#e6e6e6; --dim:#8a8f98; --rule:#2a2d33; --bad:#ff6b5b; --warn:#f2b84b; --ok:#5fb3ff; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui, sans-serif; }
main { max-width:1200px; margin:0 auto; padding:16px; }
header { display:flex; flex-wrap:wrap; gap:16px; align-items:baseline; border-bottom:1px solid var(--rule); padding-bottom:8px; }
h1 { font-size:18px; margin:0; font-weight:600; }
.mono, td, .num { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-variant-numeric: tabular-nums; }
.dim { color:var(--dim); }
#status.live { color:var(--ok); } #status.down { color:var(--bad); }
.charts { display:grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap:16px; margin:16px 0; }
figure { margin:0; border-top:1px solid var(--rule); padding-top:6px; }
figcaption { color:var(--dim); font-size:12px; }
canvas { width:100%; height:160px; display:block; }
.wrap { overflow-x:auto; }
table { border-collapse:collapse; width:100%; font-size:12.5px; }
th, td { text-align:right; padding:3px 8px; border-bottom:1px solid var(--rule); white-space:nowrap; }
th:nth-child(-n+3), td:nth-child(-n+3) { text-align:left; }
th { color:var(--dim); font-weight:500; }
td.bad { color:var(--bad); font-weight:600; }
p.note { color:var(--dim); font-size:12px; }
</style>
</head>
<body>
<main>
<header>
  <h1 id="name">pta watch</h1>
  <span id="status" class="mono">connecting</span>
  <span class="mono">step <b id="step">-</b></span>
  <span class="mono">loss <b id="loss">-</b></span>
  <span class="mono dim" id="extra"></span>
</header>
<div class="charts">
  <figure><canvas id="c-loss"></canvas><figcaption>loss (white) and validation loss (blue), per emitted step</figcaption></figure>
  <figure><canvas id="c-grad"></canvas><figcaption>global gradient norm, log10</figcaption></figure>
</div>
<div class="wrap"><table>
  <thead><tr><th>block</th><th>kind</th><th>label</th><th>act mean</th><th>act std</th><th>act |max|</th><th>zero %</th><th>sat %</th><th>mlp hidden zero %</th><th>grad std</th><th>|W|</th><th>|dW/dL|</th><th>update ratio</th><th>head entropy (nats)</th></tr></thead>
  <tbody id="rows"></tbody>
</table></div>
<p class="note">Raw numbers from the latest frame, measured in your Python process. This is the package's built-in viewer; the Lab (when bundled) draws the model and derives diagnoses from the same frames. Everything here is served from 127.0.0.1 and nothing is sent anywhere else.</p>
<p class="note" id="trace"></p>
</main>
<script>
"use strict";
const token = new URLSearchParams(location.search).get("token") || "";
const q = (p) => p + (p.includes("?") ? "&" : "?") + "token=" + encodeURIComponent(token);
const $ = (id) => document.getElementById(id);
let spec = null, frames = [], seen = new Set();
const fmt = (v, d=3) => v === undefined || v === null ? "" : (Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : v.toFixed(d));
function setSpec(hello) {
  spec = hello.spec; $("name").textContent = spec.name + " · pta watch"; document.title = spec.name + " · pta";
}
function addFrame(f) { if (seen.has(f.step)) return; seen.add(f.step); frames.push(f); frames.sort((a,b)=>a.step-b.step); render(); }
function line(canvas, series, colors, log) {
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w*dpr; canvas.height = h*dpr; const g = canvas.getContext("2d"); g.scale(dpr, dpr); g.clearRect(0,0,w,h);
  const pts = series.map(s => s.filter(p => p[1] !== null && p[1] !== undefined && isFinite(p[1])).map(p => [p[0], log ? Math.log10(Math.max(p[1], 1e-12)) : p[1]]));
  const all = pts.flat(); if (!all.length) return;
  let x0 = Math.min(...all.map(p=>p[0])), x1 = Math.max(...all.map(p=>p[0])), y0 = Math.min(...all.map(p=>p[1])), y1 = Math.max(...all.map(p=>p[1]));
  if (x1 === x0) x1 = x0 + 1; if (y1 === y0) { y1 += 1; y0 -= 1; }
  const X = x => 40 + (x-x0)/(x1-x0)*(w-48), Y = y => 8 + (1-(y-y0)/(y1-y0))*(h-24);
  g.fillStyle = "#8a8f98"; g.font = "11px ui-monospace, monospace"; g.fillText(fmt(y1,2), 0, 12); g.fillText(fmt(y0,2), 0, h-14); g.fillText(String(x1), w-40, h-2);
  pts.forEach((s, i) => { g.strokeStyle = colors[i]; g.lineWidth = 1.5; g.beginPath(); s.forEach((p, j) => j ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1]))); g.stroke(); });
}
function render() {
  const f = frames[frames.length-1]; if (!f) return;
  $("step").textContent = f.step; $("loss").textContent = f.loss === null || f.loss === undefined ? "n/a" : fmt(f.loss, 4);
  const ex = []; if (f.lr !== undefined) ex.push("lr " + fmt(f.lr)); if (f.gradNorm !== undefined) ex.push("|g| " + fmt(f.gradNorm));
  if (f.timing) ex.push(Object.entries(f.timing).map(([k,v]) => k + " " + v.toFixed(1) + "ms").join(" "));
  if (f.memory) ex.push((f.memory/1048576).toFixed(0) + " MiB"); $("extra").textContent = ex.join("  ·  ");
  line($("c-loss"), [frames.map(x=>[x.step,x.loss]), frames.map(x=>[x.step,x.valLoss])], ["#e6e6e6", "#5fb3ff"], false);
  line($("c-grad"), [frames.map(x=>[x.step,x.gradNorm])], ["#f2b84b"], true);
  const byId = {}; (spec ? spec.blocks : []).forEach(b => byId[b.id] = b);
  const rows = f.blocks.map(b => {
    const blk = byId[b.id] || {}; const a = b.act || {}, gr = b.grad || {};
    const bad = (s) => s && (s.nanCount || s.infCount) ? " class=\"bad\"" : "";
    const cells = [b.id, blk.kind || "", blk.label || "", fmt(a.mean), fmt(a.std), fmt(a.absMax), a.zeroFrac === undefined ? "" : (a.zeroFrac*100).toFixed(1), a.satFrac === undefined ? "" : (a.satFrac*100).toFixed(1), b.hidden ? (b.hidden.zeroFrac*100).toFixed(1) : "", fmt(gr.std), fmt(b.weightNorm), fmt(b.weightGradNorm), fmt(b.updateRatio), (b.headEntropy || []).map(e => e.toFixed(2)).join(" ")];
    return "<tr>" + cells.map((c, i) => "<td" + (i >= 3 && i <= 5 ? bad(b.act) : i === 8 ? bad(b.hidden) : i === 9 ? bad(b.grad) : "") + ">" + String(c).replace(/[&<>]/g, s => ({"&":"&amp;","<":"&lt;",">":"&gt;"}[s])) + "</td>").join("") + "</tr>";
  });
  $("rows").innerHTML = rows.join("");
}
async function boot() {
  try { const r = await fetch(q("/spec")); if (r.status === 200) setSpec(await r.json()); } catch (e) {}
  try { const r = await fetch(q("/health?since=-1")); if (r.ok) (await r.json()).frames.forEach(addFrame); } catch (e) {}
  const es = new EventSource(q("/events"));
  es.onopen = () => { $("status").textContent = "live"; $("status").className = "mono live"; };
  es.onerror = () => { $("status").textContent = "disconnected"; $("status").className = "mono down"; };
  es.addEventListener("hello", e => setSpec(JSON.parse(e.data)));
  es.addEventListener("spec", e => { setSpec(JSON.parse(e.data)); render(); });
  es.addEventListener("health", e => addFrame(JSON.parse(e.data)));
  es.addEventListener("frame", e => { const t = JSON.parse(e.data); $("trace").textContent = "trace frame at step " + t.step + ": " + t.tensors.length + " tensors" + (t.tensors.some(x => x.clipped) ? " (some clipped to their leading corner)" : ""); });
  es.addEventListener("bye", () => { es.close(); $("status").textContent = "run ended"; $("status").className = "mono dim"; });
}
window.addEventListener("resize", render);
boot();
</script>
</body>
</html>
"""
