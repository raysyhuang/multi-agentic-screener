/* MAS dashboard — hand-rolled SVG charts, no dependencies.
   Chart rules follow the dataviz method: validated fixed categorical palette,
   one axis, thin marks, 2px gaps, legends for >=2 series, tooltips, tnum text. */

const STREAM_META = {
  "sniper|mas_official":          { label: "Sniper (official)",    color: "#533afd" },
  "mean_reversion|mas_official":  { label: "MR (official)",        color: "#2874ad" },
  "mean_reversion|mr_manual_sleeve": { label: "MR (manual sleeve)", color: "#ea2261" },
  "pead|mas_official":            { label: "PEAD (paper)",         color: "#0f8a6d" },
};
const EXIT_COLORS = { trail_stop: "#533afd", stop: "#ea2261", target: "#0f8a6d",
                      time_stop: "#9b6829", expiry: "#2874ad", other: "#d94fc6" };
const NAVY = "#061b31", BODY = "#64748d";

const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, text) => {
  const n = tag === "svg" || SVG_TAGS.has(tag)
    ? document.createElementNS("http://www.w3.org/2000/svg", tag)
    : document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
  if (text != null) n.textContent = text;
  return n;
};
const SVG_TAGS = new Set(["rect", "line", "path", "circle", "text", "g", "polyline"]);
const fmt = (x, d = 2) => (x == null ? "–" : Number(x).toFixed(d));
const pct = (x, d = 1) => (x == null ? "–" : `${(x * 100).toFixed(d)}%`);
const cls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");

const tooltip = {
  node: null,
  show(html, evt) {
    this.node.innerHTML = html; this.node.hidden = false;
    const pad = 12, w = this.node.offsetWidth;
    let x = evt.clientX + pad;
    if (x + w > window.innerWidth - 8) x = evt.clientX - w - pad;
    this.node.style.left = `${x}px`; this.node.style.top = `${evt.clientY + pad}px`;
  },
  hide() { this.node.hidden = true; },
};

function streamMeta(key) {
  return STREAM_META[key] || { label: key, color: "#d94fc6" };
}
function legend(container, keys) {
  const box = el("div"); box.className = "legend";
  for (const k of keys) {
    const m = streamMeta(k);
    const chip = el("span"); chip.className = "chip";
    const dot = el("span"); dot.className = "dot"; dot.style.background = m.color;
    chip.append(dot, document.createTextNode(m.label));
    box.append(chip);
  }
  container.append(box);
}

/* ---------- generic scales ---------- */
function linScale(d0, d1, r0, r1) {
  const d = d1 - d0 || 1;
  return (x) => r0 + ((x - d0) / d) * (r1 - r0);
}
function niceTicks(lo, hi, n = 4) {
  const span = hi - lo || 1, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n + 1) || mag * 10;
  const t = []; for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) t.push(+v.toFixed(10));
  return t;
}

/* ---------- line chart (multi-series, crosshair tooltip) ---------- */
function measureWidth(container, fallback = 640) {
  const w = container.getBoundingClientRect().width || container.clientWidth || 0;
  return Math.max(Math.round(w) || fallback, 240);
}
function lineChart(container, seriesList, { height = 260, band = null, yFmt = (v) => fmt(v, 1), unit = "", directLabels = true, rightPad = 110 } = {}) {
  const W = measureWidth(container), H = height;
  const M = { t: 14, r: rightPad, b: 24, l: 44 };
  const allPts = seriesList.flatMap((s) => s.points);
  if (!allPts.length) { container.append(el("p", {}, "No closed trades yet.")); return; }
  const xs = allPts.map((p) => p.x), ys = allPts.map((p) => p.y).concat(band ? [band.lo, band.hi] : []);
  const x = linScale(Math.min(...xs), Math.max(...xs), M.l, W - M.r);
  const yLo = Math.min(0, ...ys), yHi = Math.max(0, ...ys);
  const y = linScale(yLo, yHi, H - M.b, M.t);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img" });

  for (const t of niceTicks(yLo, yHi)) {
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: t === 0 ? "zero-line" : "grid-line" }));
    svg.append(el("text", { x: M.l - 6, y: y(t) + 3, "text-anchor": "end", class: "axis-label" }, yFmt(t)));
  }
  if (band) {
    svg.append(el("rect", { x: M.l, y: y(band.hi), width: W - M.r - M.l, height: Math.abs(y(band.lo) - y(band.hi)),
      fill: band.color, opacity: 0.1 }));
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: y(band.mid), y2: y(band.mid),
      stroke: band.color, "stroke-dasharray": "4 4", "stroke-width": 1.5, opacity: 0.7 }));
    if (rightPad > 40)
      svg.append(el("text", { x: W - M.r + 6, y: y(band.mid) + 3, class: "series-label", fill: band.color }, band.label));
    else  // narrow: annotate the band inside the plot, top-left
      svg.append(el("text", { x: M.l + 4, y: y(band.hi) - 3, class: "axis-label", fill: band.color }, `${band.label} ${yFmt(band.mid)}`));
  }
  const dLabels = [];
  for (const s of seriesList) {
    const pts = s.points.map((p) => `${x(p.x).toFixed(1)},${y(p.y).toFixed(1)}`).join(" ");
    svg.append(el("polyline", { points: pts, fill: "none", stroke: s.color, "stroke-width": 2,
      "stroke-linejoin": "round", "stroke-linecap": "round" }));
    const last = s.points[s.points.length - 1];
    // emphasized endpoint
    svg.append(el("circle", { cx: x(last.x), cy: y(last.y), r: 3, fill: s.color, stroke: "#fff", "stroke-width": 1.5 }));
    dLabels.push({ y: y(last.y), text: `${s.label} ${yFmt(last.y)}${unit}`, color: s.color });
  }
  if (directLabels) {
    dLabels.sort((a, b) => a.y - b.y);           // de-collide direct labels
    for (let i = 1; i < dLabels.length; i++) if (dLabels[i].y - dLabels[i - 1].y < 13) dLabels[i].y = dLabels[i - 1].y + 13;
    for (const L of dLabels) svg.append(el("text", { x: W - M.r + 6, y: L.y + 3, class: "series-label", fill: L.color }, L.text));
  }

  // crosshair + tooltip
  const cross = el("line", { y1: M.t, y2: H - M.b, stroke: "#cbd5e1", "stroke-width": 1, visibility: "hidden" });
  svg.append(cross);
  svg.addEventListener("mousemove", (evt) => {
    const r = svg.getBoundingClientRect(), mx = evt.clientX - r.left;
    cross.setAttribute("x1", mx); cross.setAttribute("x2", mx); cross.setAttribute("visibility", "visible");
    const rows = seriesList.map((s) => {
      let best = s.points[0];
      for (const p of s.points) if (Math.abs(x(p.x) - mx) < Math.abs(x(best.x) - mx)) best = p;
      return `<span style="color:${s.color}">●</span> ${s.label}: ${yFmt(best.y)}${unit} <span style="opacity:.7">${best.tip || ""}</span>`;
    });
    tooltip.show(rows.join("<br>"), evt);
  });
  svg.addEventListener("mouseleave", () => { cross.setAttribute("visibility", "hidden"); tooltip.hide(); });
  container.append(svg);
}

/* ---------- horizontal stacked bar (2px gaps, labels) ---------- */
function stackedBar(container, rows, order) {
  const W = Math.max(container.clientWidth || 640, 320), rowH = 30, gap = 14, labelW = 150;
  const H = rows.length * (rowH + gap) + 24;
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  rows.forEach((row, i) => {
    const total = order.reduce((a, k) => a + (row.counts[k] || 0), 0) || 1;
    const yTop = i * (rowH + gap) + 4;
    svg.append(el("text", { x: 0, y: yTop + rowH / 2 + 4, class: "axis-label", fill: NAVY,
      "font-size": 12 }, `${row.label} (${total})`));
    let xCur = labelW;
    for (const k of order) {
      const c = row.counts[k] || 0; if (!c) continue;
      const w = Math.max(((W - labelW) * c) / total - 2, 1);   // 2px surface gap
      const rect = el("rect", { x: xCur, y: yTop, width: w, height: rowH, rx: 3, fill: EXIT_COLORS[k] || EXIT_COLORS.other });
      rect.addEventListener("mousemove", (e) => tooltip.show(`${row.label}<br>${k}: <b>${c}</b> (${pct(c / total)})`, e));
      rect.addEventListener("mouseleave", () => tooltip.hide());
      svg.append(rect);
      if (w > 44) svg.append(el("text", { x: xCur + w / 2, y: yTop + rowH / 2 + 4, "text-anchor": "middle",
        fill: "#fff", "font-size": 11, "font-weight": 400 }, `${k} ${Math.round((c / total) * 100)}%`));
      xCur += w + 2;
    }
  });
  container.append(svg);
  const lg = el("div"); lg.className = "legend";
  for (const k of order) {
    const chip = el("span"); chip.className = "chip";
    const dot = el("span"); dot.className = "dot"; dot.style.background = EXIT_COLORS[k] || EXIT_COLORS.other;
    chip.append(dot, document.createTextNode(k)); lg.append(chip);
  }
  container.append(lg);
}

/* ---------- MFE/MAE scatter ---------- */
function scatter(container, streams) {
  const W = Math.max(container.clientWidth || 460, 300), H = 300, M = { t: 12, r: 14, b: 30, l: 44 };
  const pts = [];
  for (const [key, trades] of Object.entries(streams))
    for (const t of trades) if (t.mfe != null && t.mae != null)
      pts.push({ x: -t.mae, y: t.mfe, t, key });
  if (!pts.length) { container.append(el("p", {}, "No MFE/MAE data yet.")); return; }
  const hi = Math.max(...pts.map((p) => Math.max(p.x, p.y)), 2);
  const x = linScale(0, hi, M.l, W - M.r), y = linScale(0, hi, H - M.b, M.t);
  const svg = el("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}` });
  for (const t of niceTicks(0, hi)) {
    svg.append(el("line", { x1: M.l, x2: W - M.r, y1: y(t), y2: y(t), class: "grid-line" }));
    svg.append(el("text", { x: M.l - 6, y: y(t) + 3, "text-anchor": "end", class: "axis-label" }, fmt(t, 0)));
    svg.append(el("text", { x: x(t), y: H - M.b + 14, "text-anchor": "middle", class: "axis-label" }, fmt(t, 0)));
  }
  svg.append(el("line", { x1: x(0), y1: y(0), x2: x(hi), y2: y(hi), stroke: "#cbd5e1", "stroke-dasharray": "4 4" }));
  svg.append(el("text", { x: W / 2, y: H - 4, "text-anchor": "middle", class: "axis-label" }, "max adverse |MAE| %"));
  svg.append(el("text", { x: 12, y: H / 2, transform: `rotate(-90 12 ${H / 2})`, "text-anchor": "middle", class: "axis-label" }, "max favorable MFE %"));
  for (const p of pts) {
    const c = el("circle", { cx: x(p.x), cy: y(p.y), r: 4.5, fill: streamMeta(p.key).color,
      stroke: "#fff", "stroke-width": 2 });   // 2px surface ring
    c.addEventListener("mousemove", (e) => tooltip.show(
      `<b>${p.t.ticker}</b> ${streamMeta(p.key).label}<br>PnL ${fmt(p.t.pnl_pct)}% · ${p.t.exit_reason}<br>MFE +${fmt(p.t.mfe)}% · MAE ${fmt(p.t.mae)}%`, e));
    c.addEventListener("mouseleave", () => tooltip.hide());
    svg.append(c);
  }
  container.append(svg);
  legend(container, Object.keys(streams));
}

/* ---------- helpers over trades ---------- */
const cum = (trades) => { let a = 0; return trades.map((t) => ({ x: Date.parse(t.exit_date), y: (a += t.pnl_pct), tip: `${t.exit_date} ${t.ticker}` })); };
const rollWR = (trades, w = 15) => trades.map((t, i) => {
  const win = trades.slice(Math.max(0, i - w + 1), i + 1);
  return { x: i, y: win.filter((u) => u.pnl_pct > 0).length / win.length, tip: `${t.exit_date} (n=${win.length})` };
}).slice(Math.min(w, trades.length) - 1);

/* ---------- render ---------- */
async function main() {
  tooltip.node = $("tooltip");
  const data = window.__MAS_DATA__ || await (await fetch(`data.json?ts=${Date.now()}`)).json();
  const streams = data.trades || {};
  const keys = Object.keys(streams).filter((k) => streams[k].length);

  $("hero-sub").textContent =
    `Snapshot ${data.generated_at?.slice(0, 16).replace("T", " ")} UTC · latest run ${data.latest_run?.date} · regime ${data.latest_run?.regime?.toUpperCase()}`;
  $("footer-meta").textContent =
    `Window: last ${data.window_days} days · generated ${data.generated_at} · streams: ${keys.length}`;

  // tiles
  const tiles = [];
  tiles.push({ k: "Regime", v: (data.latest_run?.regime || "–").toUpperCase() });
  tiles.push({ k: "Universe", v: data.latest_run?.universe ?? "–" });
  tiles.push({ k: "Picks today", v: String(data.today_picks?.length ?? 0) });
  for (const k of keys) {
    const t = streams[k], wr = t.filter((u) => u.pnl_pct > 0).length / t.length;
    const avg = t.reduce((a, u) => a + u.pnl_pct, 0) / t.length;
    const b = data.baselines?.[k];
    tiles.push({ k: streamMeta(k).label, v: `${pct(wr, 0)} · ${avg >= 0 ? "+" : ""}${fmt(avg)}%`,
      s: b ? `expect ~${pct(b.wr, 0)} · ${b.avg >= 0 ? "+" : ""}${fmt(b.avg)}%` : "", color: streamMeta(k).color });
  }
  tiles.push({ k: "Open positions", v: String(data.open_positions?.length ?? 0) });
  $("today-tiles").append(...tiles.map((t) => {
    const d = el("div"); d.className = "tile";
    d.append(Object.assign(el("div"), { className: "k", textContent: t.k }));
    const v = Object.assign(el("div"), { className: "v", textContent: t.v });
    if (t.color) v.style.color = t.color;
    d.append(v);
    if (t.s) d.append(Object.assign(el("div"), { className: "s hint", textContent: t.s }));
    return d;
  }));

  // picks
  const picksBox = $("picks");
  if (!data.today_picks?.length) picksBox.append(el("p", {}, "No picks in the latest run."));
  for (const p of data.today_picks || []) {
    const key = `${p.model}|${p.source}`;
    const row = el("div"); row.className = "pick";
    const left = el("div");
    left.append(Object.assign(el("div"), { className: "tkr", textContent: p.ticker }));
    const chip = el("span"); chip.className = "chip";
    const dot = el("span"); dot.className = "dot"; dot.style.background = streamMeta(key).color;
    chip.append(dot, document.createTextNode(streamMeta(key).label));
    left.append(chip);
    left.append(Object.assign(el("div"), { className: "meta", textContent: `score ${fmt(p.raw_score ?? p.confidence, 1)} · ${p.hold_days}d hold` }));
    const comps = el("div");
    for (const [nm, vl] of Object.entries(p.components || {})) {
      const r = el("div"); r.className = "comp-row";
      r.append(Object.assign(el("span"), { className: "nm", textContent: nm }));
      const bg = el("div"); bg.className = "bar-bg";
      const fg = el("div"); fg.className = "bar-fg";
      fg.style.width = `${Math.min(Number(vl) || 0, 100)}%`; fg.style.background = streamMeta(key).color;
      bg.append(fg); r.append(bg);
      r.append(Object.assign(el("span"), { className: "vl", textContent: String(Math.round(vl)) }));
      comps.append(r);
    }
    const lv = el("div"); lv.className = "levels";
    const riskPct = p.entry ? ((p.entry - p.stop) / p.entry) * 100 : null;
    const rewPct = p.entry ? ((p.target1 - p.entry) / p.entry) * 100 : null;
    lv.innerHTML = `Entry <b style="color:${NAVY}">$${fmt(p.entry)}</b><br>` +
      `Target $${fmt(p.target1)} (+${fmt(rewPct, 1)}%)<br>Stop $${fmt(p.stop)} (−${fmt(riskPct, 1)}%)`;
    row.append(left, comps, lv);
    picksBox.append(row);
  }

  // Charts render on the NEXT frame — after the browser has done a full layout
  // pass — so every container (especially the auto-fit small-multiples grid)
  // reports its true width. Measuring mid-build in this same tick gives the
  // grid a stale (too-wide) column count.
  requestAnimationFrame(() => renderCharts(data, streams, keys));
}

function renderCharts(data, streams, keys) {
  // equity
  lineChart($("equity-chart"), keys.map((k) => ({
    label: streamMeta(k).label, color: streamMeta(k).color, points: cum(streams[k]),
  })), { unit: "%", yFmt: (v) => `${v >= 0 ? "+" : ""}${fmt(v, 0)}` });
  legend($("equity-chart"), keys);

  // rolling WR small multiples with expectation band (cards already in the DOM)
  for (const k of keys) {
    if (streams[k].length < 5) continue;
    const cardDiv = el("div");
    cardDiv.append(Object.assign(el("h3"), { textContent: streamMeta(k).label, style: "font-size:15px;margin-bottom:4px" }));
    $("wr-multiples").append(cardDiv);
    const b = data.baselines?.[k];
    lineChart(cardDiv, [{ label: "live", color: streamMeta(k).color, points: rollWR(streams[k]) }], {
      height: 180, unit: "", yFmt: (v) => pct(v, 0), directLabels: false, rightPad: 12,
      band: b ? { lo: b.wr - 0.05, hi: b.wr + 0.05, mid: b.wr, color: streamMeta(k).color, label: "expected" } : null,
    });
  }

  // exit mix
  const reasons = ["trail_stop", "stop", "target", "time_stop", "expiry"];
  stackedBar($("exit-mix"), keys.map((k) => ({
    label: streamMeta(k).label,
    counts: streams[k].reduce((a, t) => { const r = reasons.includes(t.exit_reason) ? t.exit_reason : "other"; a[r] = (a[r] || 0) + 1; return a; }, {}),
  })), reasons.concat(["other"]));

  // scatter + open positions
  scatter($("scatter"), Object.fromEntries(keys.map((k) => [k, streams[k]])));
  const op = $("open-positions");
  if (!data.open_positions?.length) op.append(el("p", {}, "None."));
  else {
    const tb = el("table");
    tb.innerHTML = "<tr><th>Ticker</th><th>Stream</th><th>Entry</th><th class=num>Days</th><th class=num>Unrealized</th></tr>";
    for (const o of data.open_positions) {
      const tr = el("tr");
      tr.innerHTML = `<td style="color:${NAVY}">${o.ticker}</td><td>${streamMeta(o.stream).label}</td>` +
        `<td>${o.entry_date}</td><td class=num>${o.days_held ?? "–"}</td>` +
        `<td class="num ${cls(o.unrealized_pnl_pct)}">${o.unrealized_pnl_pct == null ? "–" : fmt(o.unrealized_pnl_pct) + "%"}</td>`;
      tb.append(tr);
    }
    op.append(tb);
  }

  // trades table + one filter row
  let filter = "all";
  const renderTable = () => {
    const box = $("trades-table"); box.innerHTML = "";
    const rows = keys.flatMap((k) => (filter === "all" || filter === k) ? streams[k].map((t) => ({ ...t, key: k })) : [])
      .sort((a, b) => (b.exit_date || "").localeCompare(a.exit_date || "")).slice(0, 40);
    const tb = el("table");
    tb.innerHTML = "<tr><th>Exit</th><th>Ticker</th><th>Stream</th><th>Reason</th><th class=num>Hold</th><th class=num>MFE</th><th class=num>MAE</th><th class=num>PnL</th></tr>";
    for (const t of rows) {
      const tr = el("tr");
      tr.innerHTML = `<td>${t.exit_date}</td><td style="color:${NAVY}">${t.ticker}</td>` +
        `<td><span class="chip"><span class="dot" style="background:${streamMeta(t.key).color}"></span>${streamMeta(t.key).label}</span></td>` +
        `<td>${t.exit_reason}</td><td class=num>${t.hold_days ?? "–"}d</td>` +
        `<td class=num>+${fmt(t.mfe)}%</td><td class=num>${fmt(t.mae)}%</td>` +
        `<td class="num ${cls(t.pnl_pct)}">${t.pnl_pct > 0 ? "+" : ""}${fmt(t.pnl_pct)}%</td>`;
      tb.append(tr);
    }
    box.append(tb);
  };
  const fr = $("trade-filters");
  for (const [val, lbl] of [["all", "All streams"], ...keys.map((k) => [k, streamMeta(k).label])]) {
    const b = el("button"); b.className = `fbtn${val === "all" ? " active" : ""}`; b.textContent = lbl;
    b.onclick = () => { filter = val; fr.querySelectorAll(".fbtn").forEach((x) => x.classList.remove("active")); b.classList.add("active"); renderTable(); };
    fr.append(b);
  }
  renderTable();

  // run history
  const rh = $("run-history");
  const tb = el("table");
  tb.innerHTML = "<tr><th>Date</th><th>Regime</th><th class=num>Universe</th><th class=num>Candidates</th><th class=num>Duration</th><th>Health</th></tr>";
  for (const r of (data.run_history || []).slice(-14).reverse()) {
    const badge = r.health === "OK" ? `<span class="badge ok">✓ OK</span>`
      : r.health === "WARN" ? `<span class="badge warn">⚠ WARN</span>`
      : r.health ? `<span class="badge crit">✕ ${r.health}</span>` : `<span class="badge neutral">–</span>`;
    const warns = (r.warnings || []).length ? ` <span class="hint">${(r.warnings || []).length} warning(s)</span>` : "";
    const tr = el("tr");
    tr.innerHTML = `<td>${r.date}</td><td>${(r.regime || "–").toUpperCase()}</td>` +
      `<td class=num>${r.universe ?? "–"}</td><td class=num>${r.candidates ?? "–"}</td>` +
      `<td class=num>${r.duration_s ? fmt(r.duration_s, 0) + "s" : "–"}</td><td>${badge}${warns}</td>`;
    tb.append(tr);
  }
  rh.append(tb);
}

main().catch((e) => {
  document.querySelector("main").prepend(Object.assign(document.createElement("p"), {
    textContent: `Failed to load data.json — ${e}. The first pipeline run after deployment publishes it.`,
  }));
});
