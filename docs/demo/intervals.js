/* UnKGCP: intervals for the confidence of triples of uncertain knowledge graphs. Plain CP
   gives every triple the same half-width; UnKGCP scales it by the entropy of the
   prediction. Both bands are drawn over the predicted and true confidences. */
"use strict";

(async () => {
  const { $, $$, h, s, pretty, percent, number, integer } = UI;
  const NAMES = { absolute: "Plain CP", entropy: "UnKGCP" };
  const COLORS = { absolute: "var(--blue)", entropy: "var(--orange)" };
  const WASH = { absolute: "var(--wash)", entropy: "var(--wash-orange)" };
  const SHOWN = 8; // example triples
  const index = window.KGCP_DATA.index.intervals;
  const state = { dataset: "nl27k", model: "beurre", view: "entropy", alpha: 0.1, seed: 0, offset: 0 };
  let current = null;
  let blend = 1; // 0: the band of plain CP, 1: that of UnKGCP
  let run = { cancelled: false };

  // ---------------------------------------------------------------- controls
  const datasetSelect = $("#int-dataset");
  const modelSelect = $("#int-model");
  const datasets = [...new Set(index.map((entry) => entry.dataset))];
  datasetSelect.replaceChildren(...datasets.map((d) => h("option", { value: d, text: UI.label(d) })));
  datasetSelect.value = state.dataset;
  function listModels() {
    const models = index.filter((entry) => entry.dataset === state.dataset);
    modelSelect.replaceChildren(
      ...models.map((entry) => h("option", { value: entry.model, text: `${UI.label(entry.model)} · error ${number(entry.mae, 2)}` })),
    );
    modelSelect.value = state.model;
  }
  listModels();
  datasetSelect.addEventListener("change", () => {
    state.dataset = datasetSelect.value;
    state.offset = 0;
    listModels();
    update();
  });
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    update();
  });
  state.alpha = UI.level("int-level", (alpha) => {
    state.alpha = alpha;
    schedule();
  });
  for (const button of $$("#int-view button")) {
    button.addEventListener("click", () => {
      run.cancelled = true;
      show(button.dataset.view, true);
    });
  }
  $("#int-resample").addEventListener("click", () => {
    state.seed++;
    update();
  });
  $("#int-shuffle").addEventListener("click", () => {
    state.offset += SHOWN;
    examples.render();
  });
  let pending = 0;
  function schedule() {
    window.cancelAnimationFrame(pending);
    pending = window.requestAnimationFrame(update);
  }

  // ---------------------------------------------------------------- compute
  let ticket = 0;
  async function compute() {
    const mine = ++ticket;
    const data = await UI.model(`intervals-${state.dataset}-${state.model}`);
    if (mine !== ticket) return false;
    const split = KGCP.split(data.N, state.seed);
    const methods = {};
    for (const name of ["absolute", "entropy"]) {
      const S = KGCP.residuals(data, name);
      const model = KGCP.calibrateIntervals(S, split.cal, { method: "split", alpha: state.alpha });
      const summary = KGCP.summary(KGCP.evaluateIntervals(model, split.test), data.relation, state.alpha);
      methods[name] = { S, model, summary, threshold: model.thresholds.get(0) };
    }
    current = { data, split, methods, alpha: state.alpha };
    return true;
  }

  /* Half the width of the interval around a prediction M, between the two methods. */
  function halfWidth(M, t) {
    const { methods } = current;
    return UI.lerp(methods.absolute.threshold, methods.entropy.threshold * KGCP.entropy(M), t);
  }

  // ------------------------------------------------------------------ tiles
  function tiles() {
    const { methods, data, alpha } = current;
    const target = 1 - alpha;
    const a = methods.absolute.summary;
    const e = methods.entropy.summary;
    const change = e.size / a.size - 1;
    const tile = (label, value, ...sub) => h("div", { class: "tile" }, h("div", { class: "label" }, ...[label].flat()), h("div", { class: "value", text: value }), h("div", { class: "sub" }, ...sub));
    const key = (name) => h("span", { class: "key", style: `background:${COLORS[name]}` });
    $("#int-tiles").replaceChildren(
      tile([key("absolute"), "Plain CP: mean width"], number(a.size, 3), `coverage ${percent(a.coverage)} · target ${percent(target, 0)}`),
      tile([key("entropy"), "UnKGCP: mean width"], number(e.size, 3), `coverage ${percent(e.coverage)} · target ${percent(target, 0)}`),
      tile("UnKGCP against plain CP", `${change > 0 ? "+" : "−"}${percent(Math.abs(change), 0)}`, change > 0.005 ? "wider intervals on this graph" : change < -0.005 ? "narrower intervals, same guarantee" : "the same width"),
      tile("Intervals that are all of [0,\u00a01]", percent(e.full, 0), `UnKGCP; plain CP: ${percent(a.full, 0)}`),
      tile("The model's error", number(data.raw.mae, 3), `mean distance from the true confidence, ${integer(data.raw.testTriples)} test triples`),
    );
  }

  // --------------------------------------------------------------- the plot
  let band = null;
  let scale = null; // the plot's x and y scales
  let dots = [];
  const plot = UI.chart($("#int-plot"), (svg, width) => {
    dots = [];
    if (!current) return 320;
    const { data, split } = current;
    const left = 40;
    const right = 12;
    const top = 26;
    const size = Math.min(width - left - right, 470);
    const x = UI.linear(0, 1, left, left + size);
    const y = UI.linear(0, 1, top + size, top);
    const height = top + size + 40;
    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("line", { class: "grid", x1: x(tick), x2: x(tick), y1: y(0), y2: y(1) }));
      svg.append(s("line", { class: "grid", x1: x(0), x2: x(1), y1: y(tick), y2: y(tick) }));
      svg.append(s("text", { class: "data", x: x(tick), y: y(0) + 15, "text-anchor": "middle", text: tick.toFixed(2) }));
      svg.append(s("text", { class: "data", x: left - 6, y: y(tick) + 4, "text-anchor": "end", text: tick.toFixed(2) }));
    }
    svg.append(s("text", { x: x(0.5), y: y(0) + 32, "text-anchor": "middle", text: "confidence the model predicts, M" }));
    svg.append(s("text", { x: left, y: 12, text: "true confidence, c" }));
    band = s("path", { stroke: "none" });
    svg.append(band);
    svg.append(s("line", { class: "axis", x1: x(0), y1: y(0), x2: x(1), y2: y(1) }));
    const points = s("g", {});
    for (const i of split.test) {
      const dot = s("circle", { cx: x(data.predicted[i]).toFixed(1), cy: y(data.confidence[i]).toFixed(1), r: 1.7 });
      points.append(dot);
      dots.push({ i, dot });
    }
    svg.append(points);
    for (const name of ["absolute", "entropy"]) {
      for (const edge of edges(name === "entropy" ? 1 : 0, x, y)) {
        svg.append(s("path", { d: `M${edge.join("L")}`, fill: "none", stroke: COLORS[name], "stroke-width": 2, "stroke-linejoin": "round" }));
      }
    }
    // direct labels at the bands' upper edges, where the two differ most
    const at = (M, t, dy, text) => svg.append(s("text", { class: "strong halo", x: x(M), y: y(Math.min(1, M + halfWidth(M, t))) + dy, "text-anchor": "middle", text }));
    at(0.2, 0, -7, NAMES.absolute);
    at(0.62, 1, -7, NAMES.entropy);

    // the crosshair reads both intervals at a predicted confidence
    const hair = s("line", { stroke: "var(--ink)", "stroke-width": 1, y1: y(0), y2: y(1), opacity: 0 });
    const pane = s("rect", { x: x(0), y: y(1), width: size, height: size, fill: "transparent" });
    const read = (event) => {
      const box = svg.getBoundingClientRect();
      const M = Math.min(1, Math.max(0, x.invert(((event.clientX - box.left) / box.width) * width)));
      hair.setAttribute("x1", x(M));
      hair.setAttribute("x2", x(M));
      hair.setAttribute("opacity", 0.6);
      const range = (t) => `[${number(Math.max(0, M - halfWidth(M, t)), 2)}, ${number(Math.min(1, M + halfWidth(M, t)), 2)}]`;
      UI.showTip(event, [
        h("div", {}, "predicted ", h("b", { text: number(M, 2) })),
        UI.tipLine(range(0), NAMES.absolute),
        UI.tipLine(range(1), NAMES.entropy),
      ]);
    };
    pane.addEventListener("pointermove", read);
    pane.addEventListener("pointerleave", () => {
      hair.setAttribute("opacity", 0);
      UI.hideTip();
    });
    svg.append(hair, pane);
    scale = { x, y };
    paint();
    return height;
  }, "Predicted against true confidence, with the bands of plain CP and UnKGCP");

  /* The upper and lower edge of the band M ± w(M), within [0, 1], as points. */
  function edges(t, x, y) {
    const upper = [];
    const lower = [];
    for (let step = 0; step <= 100; step++) {
      const M = step / 100;
      const w = halfWidth(M, t);
      upper.push(`${x(M).toFixed(1)} ${y(Math.min(1, M + w)).toFixed(1)}`);
      lower.push(`${x(M).toFixed(1)} ${y(Math.max(0, M - w)).toFixed(1)}`);
    }
    return [upper, lower];
  }
  function outline(t, x, y) {
    const [upper, lower] = edges(t, x, y);
    return `M${upper.join("L")}L${lower.reverse().join("L")}Z`;
  }

  /* Fills the band at the current blend and marks the confidences outside it. */
  function paint() {
    if (!band || !current) return;
    const { data } = current;
    band.setAttribute("d", outline(blend, scale.x, scale.y));
    band.setAttribute("fill", blend > 0.5 ? WASH.entropy : WASH.absolute);
    for (const { i, dot } of dots) {
      const M = data.predicted[i];
      const outside = Math.abs(data.confidence[i] - M) > halfWidth(M, blend);
      dot.setAttribute("fill", outside ? "var(--bad)" : "var(--ink-3)");
      dot.setAttribute("fill-opacity", outside ? 0.95 : 0.3);
      dot.setAttribute("r", outside ? 2.3 : 1.7);
    }
  }

  function texts() {
    const { methods, split, alpha } = current;
    const active = methods[state.view];
    const q = methods.entropy.threshold;
    const missed = dots.filter(({ i }) => !KGCP.interval(active.model, i).covered).length;
    $("#int-plot-sub").textContent =
      `${integer(split.test.length)} test triples; ${integer(split.cal.length)} others calibrated the bands. Red: outside the ${NAMES[state.view]} interval (${integer(missed)}).`;
    $("#int-legend").replaceChildren(
      h("span", {}, h("span", { class: "key line", style: `background:${COLORS.absolute}` }), `${NAMES.absolute}: M ± ${number(methods.absolute.threshold, 2)}`),
      h("span", {}, h("span", { class: "key line", style: `background:${COLORS.entropy}` }), `${NAMES.entropy}: M ± ${number(q, 2)} × H(M)`),
      h("span", {}, h("span", { class: "key", style: "background:var(--bad)" }), "true confidence outside the interval"),
    );
    $("#int-caption").textContent =
      `At a ${percent(1 - alpha, 0)} target, plain CP adds the same ±${number(methods.absolute.threshold, 2)} to every prediction. ` +
      `UnKGCP scales its margin by the entropy H(M) of the prediction: ±${number(q * KGCP.entropy(0.5), 2)} at M = 0.5, ` +
      `±${number(q * KGCP.entropy(0.95), 2)} at M = 0.95. Move over the plot to read both intervals.`;
  }

  // ------------------------------------------------------- example triples
  const examples = UI.chart($("#int-list"), (svg, width) => {
    if (!current) return 200;
    const { data, split, methods } = current;
    const active = methods[state.view];
    const tested = new Set(split.test);
    const pool = data.raw.examples.filter(([i]) => tested.has(i));
    const shown = Array.from({ length: Math.min(SHOWN, pool.length) }, (_, k) => pool[(state.offset + k) % pool.length]);
    const row = 50;
    const left = 4;
    const right = 34;
    const x = UI.linear(0, 1, left, width - right);
    let covered = 0;
    shown.forEach(([i, head, tail], position) => {
      const top = position * row;
      const interval = KGCP.interval(active.model, i);
      covered += interval.covered;
      // PPI5k numbers its proteins and its seven kinds of interaction
      const numbered = data.raw.dataset === "ppi5k";
      const relation = numbered ? `interaction ${data.relation[i]}` : pretty(data.raw.relations[data.relation[i]]);
      const entity = (name) => (numbered ? `protein ${name}` : pretty(name));
      const label = s("text", { class: "data", x: left, y: top + 13, text: `(${entity(head)}, ${relation}, ${entity(tail)})` });
      svg.append(label);
      let text = label.textContent;
      while (text.length > 8 && label.getComputedTextLength() > width - right) {
        text = text.slice(0, -2);
        label.textContent = `${text}…`;
      }
      const cy = top + 30;
      svg.append(s("line", { class: "axis", x1: x(0), x2: x(1), y1: cy, y2: cy }));
      svg.append(s("line", { x1: x(interval.lower), x2: Math.max(x(interval.upper), x(interval.lower) + 2), y1: cy, y2: cy, stroke: COLORS[state.view], "stroke-width": 7, "stroke-linecap": "round", opacity: 0.85 }));
      svg.append(s("line", { x1: x(data.predicted[i]), x2: x(data.predicted[i]), y1: cy - 8, y2: cy + 8, stroke: "var(--ink)", "stroke-width": 2 }));
      svg.append(s("circle", { cx: x(data.confidence[i]), cy, r: 4.5, fill: "var(--surface)", stroke: "var(--ink)", "stroke-width": 2 }));
      const mark = UI.icon(interval.covered ? "check" : "cross", 15);
      mark.setAttribute("x", width - 16);
      mark.setAttribute("y", cy - 8);
      svg.append(mark);
      const hit = s("rect", { x: 0, y: top, width, height: row - 4, fill: "transparent" });
      UI.tip(hit, () => [
        UI.tipLine(`[${number(interval.lower, 2)}, ${number(interval.upper, 2)}]`, `${NAMES[state.view]} interval`),
        UI.tipLine(number(data.predicted[i], 2), "predicted confidence"),
        UI.tipLine(number(data.confidence[i], 2), interval.covered ? "true confidence: inside" : "true confidence: outside"),
      ], { focusable: true });
      svg.append(hit);
    });
    const base = shown.length * row;
    for (const tick of [0, 0.5, 1]) svg.append(s("text", { class: "data", x: x(tick), y: base + 8, "text-anchor": tick === 0 ? "start" : tick === 1 ? "end" : "middle", text: tick.toFixed(1) }));
    $("#int-list-sub").textContent =
      `Bar: the ${NAMES[state.view]} interval. Tick: the prediction. Ring: the true confidence. ${covered} of these ${shown.length} intervals contain it.`;
    return base + 14;
  }, "Intervals of some test triples");

  // ------------------------------------------------------------------ views
  async function show(view, animate) {
    state.view = view;
    for (const button of $$("#int-view button")) button.setAttribute("aria-pressed", String(button.dataset.view === view));
    const target = view === "entropy" ? 1 : 0;
    const from = blend;
    texts();
    examples.render();
    await UI.tween(animate ? 900 : 0, (t) => {
      blend = UI.lerp(from, target, t);
      paint();
    });
  }
  async function update() {
    if (!(await compute())) return;
    tiles();
    plot.render();
    texts();
    examples.render();
  }

  await update();
  UI.onFirstView($("#int-plot"), async () => {
    const mine = (run = { cancelled: false });
    await show("absolute", false);
    await UI.sleep(1100, mine);
    if (!mine.cancelled) show("entropy", true);
  });
})();
