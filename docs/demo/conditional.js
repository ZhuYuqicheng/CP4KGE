/* CondKGCP in four steps on a map of the relations: split CP covers them unevenly,
   calibrating each alone gives rare relations every entity, merging them into similar
   relations gives every group a threshold, and rank calibration shrinks the sets. */
"use strict";

(async () => {
  const { $, $$, h, s, pretty, percent, number, integer } = UI;
  const SPLITS = 30;
  const METHODS = ["split", "mondrian", "merged", "condkgcp"]; // what each step shows
  const NAMES = { split: "Split CP", condkgcp: "CondKGCP", mondrian: "Mondrian CP" };
  const COLORS = { split: "var(--blue)", condkgcp: "var(--orange)", mondrian: "var(--aqua)" };
  const index = window.KGCP_DATA.index;
  const state = { key: "sets-umls-distmult", alpha: 0.1, minSize: 20, gamma: 1, step: 4, seed: 0 };
  const scores = new Map();
  const pooledCache = new Map();
  let current = null;
  let nodes = [];
  let run = { cancelled: false };

  // ---------------------------------------------------------------- controls
  const select = $("#cond-model");
  select.replaceChildren(
    ...index.sets.map((entry) => h("option", { value: entry.key, text: `${UI.label(entry.dataset)} · ${UI.label(entry.model)}` })),
  );
  select.value = state.key;
  select.addEventListener("change", () => {
    state.key = select.value;
    update(true);
  });
  state.alpha = UI.level("cond-level", (alpha) => {
    state.alpha = alpha;
    schedule();
  });
  const slider = (id, key, format) => {
    const input = $(`#${id}`);
    input.addEventListener("input", () => {
      state[key] = Number(input.value);
      $(`#${id}-out`).textContent = format(state[key]);
      schedule();
    });
  };
  slider("cond-minsize", "minSize", integer);
  slider("cond-gamma", "gamma", (v) => number(v, 2));
  let pending = 0;
  function schedule() {
    window.cancelAnimationFrame(pending);
    pending = window.requestAnimationFrame(() => update(false));
  }

  // ---------------------------------------------------------------- compute
  function options(method) {
    const { alpha, minSize, gamma } = state;
    if (method === "mondrian") return { method: "mondrian", groups: "relation", alpha };
    if (method === "merged") return { method: "mondrian", groups: "merged", minSize, alpha };
    if (method === "condkgcp") return { method: "condkgcp", groups: "merged", minSize, gamma, alpha };
    return { method: "split", alpha };
  }

  let ticket = 0;
  async function compute() {
    const mine = ++ticket;
    const data = await UI.model(state.key);
    if (mine !== ticket) return false;
    if (!scores.has(state.key)) scores.set(state.key, KGCP.scores(data, "softmax"));
    const S = scores.get(state.key);
    const pooled = {};
    const models = {};
    const split = KGCP.split(data.N, state.seed);
    for (const method of METHODS) {
      const opts = options(method);
      const id = `${state.key}/${JSON.stringify(opts)}`; // the sliders revisit settings
      if (!pooledCache.has(id)) pooledCache.set(id, KGCP.repeat(S, opts, SPLITS, state.seed));
      pooled[method] = pooledCache.get(id);
      models[method] = KGCP.calibrate(S, split.cal, opts);
    }
    if (pooledCache.size > 400) pooledCache.clear();
    const held = new Int32Array(data.R); // held-out (calibration) queries per relation
    for (const i of split.cal) held[data.relation[i]]++;
    const total = new Int32Array(data.R);
    for (let i = 0; i < data.N; i++) total[data.relation[i]]++;
    current = { data, S, pooled, models, split, held, total, alpha: state.alpha, key: state.key };
    return true;
  }

  // ---------------------------------------------------------- the relation map
  const map = UI.chart($("#cond-map"), drawMap, "Map of the relations, colored by their coverage");

  function drawMap(svg, width) {
    nodes = [];
    if (!current) return 320;
    const { data, held } = current;
    const height = Math.round(Math.min(Math.max(width * 0.78, 300), 470));
    const pad = 30;
    svg.setAttribute("class", "map diverging");
    const largest = Math.max(...held, 1);
    const radius = (r) => 4 + 15 * Math.sqrt(held[r] / largest);
    // the layout keeps similar relations close; ranks spread its dense middle over the map
    const spread = (axis) => {
      const values = data.raw.layout.map((xy) => xy[axis]);
      const order = values.map((_, r) => r).sort((a, b) => values[a] - values[b]);
      const rank = new Float64Array(values.length);
      order.forEach((r, position) => (rank[r] = position / Math.max(values.length - 1, 1)));
      return values.map((value, r) => 0.3 * value + 0.7 * rank[r]);
    };
    const [xs, ys] = [spread(0), spread(1)];
    const points = data.raw.layout.map((_, r) => ({
      r, x: pad + xs[r] * (width - 2 * pad), y: pad + ys[r] * (height - 2 * pad), radius: radius(r),
    }));
    separate(points, width, height);
    const spokes = s("g", {});
    const circles = s("g", {});
    svg.append(spokes, circles);
    for (const point of points) {
      if (current.total[point.r] === 0) continue;
      const spoke = s("line", { class: "spoke", stroke: "var(--axis)", "stroke-width": 1, opacity: 0 });
      const node = s("g", { class: "node" });
      const all = s("circle", { class: "all", r: point.radius + 3.5, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, opacity: 0 });
      const circle = s("circle", { r: point.radius, stroke: "var(--surface)", "stroke-width": 1.5 });
      node.append(all, circle);
      UI.tip(node, () => describe(point.r), { focusable: true });
      spokes.append(spoke);
      circles.append(node);
      nodes.push({ ...point, node, circle, all, spoke });
    }
    applyMap();
    return height;
  }

  /* Pushes overlapping circles apart, so that every relation can be seen and pointed at. */
  function separate(points, width, height) {
    for (let round = 0; round < 80; round++) {
      for (let a = 0; a < points.length; a++) {
        for (let b = a + 1; b < points.length; b++) {
          const p = points[a];
          const q = points[b];
          let dx = q.x - p.x;
          let dy = q.y - p.y;
          let distance = Math.hypot(dx, dy);
          const wanted = p.radius + q.radius + 3;
          if (distance >= wanted) continue;
          if (distance < 1e-6) [dx, dy, distance] = [Math.cos(a + b), Math.sin(a + b), 1];
          const push = (wanted - distance) / 2;
          p.x -= (dx / distance) * push;
          p.y -= (dy / distance) * push;
          q.x += (dx / distance) * push;
          q.y += (dy / distance) * push;
        }
      }
      for (const p of points) {
        p.x = Math.min(Math.max(p.x, p.radius + 6), width - p.radius - 6);
        p.y = Math.min(Math.max(p.y, p.radius + 6), height - p.radius - 6);
      }
    }
  }

  const stepMethod = () => METHODS[state.step - 1];

  /* Sets every node to the current step: where it sits, its color, its marks. The CSS
     transitions animate from the previous state. */
  function applyMap() {
    if (!current || !nodes.length) return;
    const { pooled, models, alpha } = current;
    const method = stepMethod();
    const coverage = new Map(pooled[method].everyRelation.map((x) => [x.relation, x]));
    const model = models[method];
    const table = state.step >= 3 ? models.merged.merged.table : null;
    const byRelation = new Map(nodes.map((n) => [n.r, n]));
    const satellites = new Map(); // host -> the relations merged into it
    if (table) {
      for (const n of nodes) {
        if (table[n.r] === n.r || !byRelation.has(table[n.r])) continue;
        if (!satellites.has(table[n.r])) satellites.set(table[n.r], []);
        satellites.get(table[n.r]).push(n);
      }
    }
    const place = new Map(nodes.map((n) => [n.r, { x: n.x, y: n.y }]));
    for (const [hostRelation, members] of satellites) {
      const host = byRelation.get(hostRelation);
      members.sort((a, b) => Math.atan2(a.y - host.y, a.x - host.x) - Math.atan2(b.y - host.y, b.x - host.x));
      members.forEach((member, position) => {
        // rings around the host, 2 pi / count apart
        const ring = Math.floor(position / 10);
        const inRing = Math.min(10, members.length - ring * 10);
        const angle = ((position % 10) / inRing) * 2 * Math.PI - Math.PI / 2 + ring * 0.3;
        const distance = host.radius + member.radius + 3 + ring * (2 * member.radius + 3);
        place.set(member.r, { x: host.x + Math.cos(angle) * distance, y: host.y + Math.sin(angle) * distance });
      });
    }
    for (const n of nodes) {
      const at = place.get(n.r);
      n.node.style.transform = `translate(${at.x.toFixed(1)}px, ${at.y.toFixed(1)}px)`;
      const entry = coverage.get(n.r);
      n.circle.style.fill = entry ? `var(--d${band(entry.coverage - (1 - alpha))})` : "var(--surface)";
      n.circle.style.stroke = entry ? "var(--surface)" : "var(--axis)";
      // a relation (or its group) without a threshold at this target gets every entity
      const group = model.groupOf(nodeQuery(n.r));
      const capped = model.topK.has(group) && model.topK.get(group) < current.data.E;
      const unbounded = method !== "split" && !capped && !(model.thresholds.get(group) < Infinity);
      n.all.style.opacity = unbounded ? 1 : 0;
      const merged = table && table[n.r] !== n.r && byRelation.has(table[n.r]);
      const host = merged ? byRelation.get(table[n.r]) : n;
      n.spoke.setAttribute("x1", n.x.toFixed(1));
      n.spoke.setAttribute("y1", n.y.toFixed(1));
      n.spoke.setAttribute("x2", host.x.toFixed(1));
      n.spoke.setAttribute("y2", host.y.toFixed(1));
      n.spoke.style.opacity = merged ? 0.9 : 0;
    }
    legend();
    caption();
  }

  /* Any query of the relation, to ask a predictor for the relation's group. */
  const firstQuery = new Map();
  function nodeQuery(relation) {
    const id = `${current.key}/${relation}`;
    if (!firstQuery.has(id)) firstQuery.set(id, current.data.relation.indexOf(relation));
    return firstQuery.get(id);
  }

  const band = (gap) => (gap <= -0.1 ? -2 : gap <= -0.03 ? -1 : gap < 0.03 ? 0 : gap < 0.1 ? 1 : 2);

  function describe(relation) {
    const { data, pooled, models, held, alpha } = current;
    const method = stepMethod();
    const entry = pooled[method].everyRelation.find((x) => x.relation === relation);
    const lines = [h("div", {}, h("b", { text: pretty(data.raw.relations[relation]) }))];
    lines.push(UI.tipLine(integer(held[relation]), "held-out queries in this split"));
    if (entry) {
      lines.push(UI.tipLine(percent(entry.coverage), `coverage (target ${percent(1 - alpha, 0)})`));
      lines.push(UI.tipLine(number(entry.size, 1), `answers per set, of ${data.E}`));
      if (entry.queries < 10) lines.push(h("div", { class: "dim", text: "fewer than 10 test queries: not counted in the gap" }));
    }
    if (state.step >= 3) {
      const host = models.merged.merged.table[relation];
      if (host !== relation) lines.push(h("div", { class: "dim", text: `calibrated with ${pretty(data.raw.relations[host])}` }));
    }
    return lines;
  }

  function legend() {
    const { alpha } = current;
    const target = percent(1 - alpha, 0);
    const scale = h("span", { class: "scale diverging", "aria-hidden": "true" }, [-2, -1, 0, 1, 2].map((b) => h("i", { style: `background: var(--d${b})` })));
    $("#cond-map-legend").replaceChildren(
      h("span", {}, scale, `coverage: 10 points or more below ${target} … on target (±3) … 10 or more above`),
      h("span", {}, h("span", { class: "swatch ring" }), "no threshold: gets every entity"),
      h("span", {}, "size: held-out queries"),
    );
    $("#cond-map").classList.add("diverging");
  }

  function caption() {
    const { data, pooled, models, held, alpha } = current;
    const target = percent(1 - alpha, 0);
    const b = (text) => h("b", { text });
    const under = (method) => pooled[method].relations.filter((x) => 1 - alpha - x.coverage > 1e-12);
    const counted = pooled.split.relations.length;
    const present = Array.from(held.keys()).filter((r) => current.total[r] > 0);
    const noThreshold = present.filter((r) => !(models.mondrian.thresholds.get(r) < Infinity)).length;
    const groups = new Set(present.map((r) => models.merged.merged.table[r])).size;
    const worst = pooled.split.relations.slice().sort((x, y) => x.coverage - y.coverage)[0];
    const size = (method) => number(pooled[method].size, 1);
    const texts = {
      1: ["One threshold for all relations. Coverage is ", b(percent(pooled.split.coverage)), " overall, but ", b(`${under("split").length} of the ${counted} relations`), ` with enough test queries stay below ${target}`, worst ? [": ", b(pretty(data.raw.relations[worst.relation])), ` is covered ${percent(worst.coverage, 0)} of the time.`] : "."],
      2: ["A threshold per relation evens this out: the gap falls from ", b(number(pooled.split.covgap, 3)), " to ", b(number(pooled.mondrian.covgap, 3)), ". But ", b(`${noThreshold} of ${present.length} relations`), ` have too few held-out queries for a threshold at ${target} (ringed): their queries get all ${data.E} entities, and the mean set grows from ${size("split")} to `, b(size("mondrian")), " entities."],
      3: ["Relations with fewer than ", b(`${state.minSize} held-out queries`), " join the most similar relation that has enough, by the distance of their embeddings. ", b(`${groups} groups`), ` remain, each with a threshold: the mean set is `, b(size("merged")), " entities, at a gap of ", b(number(pooled.merged.covgap, 3)), "."],
      4: ["CondKGCP also caps each group's sets at its top k candidates, with k calibrated on the held-out queries, and relaxes the score threshold in return (γ). Sets: ", b(size("condkgcp")), " entities, gap ", b(number(pooled.condkgcp.covgap, 3)), ", coverage ", b(percent(pooled.condkgcp.coverage)), state.gamma < 1 ? ". Below γ = 1, coverage may fall under the target by design." : "."],
    };
    $("#cond-caption").replaceChildren(...texts[state.step].flat(Infinity));
    $("#cond-map-sub").textContent = `${present.length} relations of ${UI.label(data.raw.dataset)}; one calibration split for the map, ${SPLITS} for the numbers.`;
  }

  // --------------------------------------------------------------- the trade
  const last = new Map(); // method -> where its point was drawn
  const trade = UI.chart($("#cond-trade"), (svg, width) => {
    if (!current) return 200;
    const { pooled, data } = current;
    const methods = ["split", "condkgcp", "mondrian"];
    const left = 46;
    const right = 22;
    const top = 16;
    const base = 196;
    const highest = Math.max(0.05, ...methods.map((m) => pooled[m].covgap)) * 1.18;
    const x = UI.logarithmic(1, data.E, left, width - right);
    const y = UI.linear(0, highest, base, top);
    for (const tick of UI.ticks(0, highest, 4)) {
      svg.append(s("line", { class: tick === 0 ? "axis" : "grid", x1: left, x2: width - right, y1: y(tick), y2: y(tick) }));
      svg.append(s("text", { class: "data", x: left - 6, y: y(tick) + 4, "text-anchor": "end", text: String(+tick.toFixed(3)) }));
    }
    for (const tick of [1, 2, 5, 10, 20, 50, 100, 200, 500].filter((t) => t <= data.E)) {
      svg.append(s("line", { class: "axis", x1: x(tick), x2: x(tick), y1: base, y2: base + 4 }));
      svg.append(s("text", { class: "data", x: x(tick), y: base + 16, "text-anchor": "middle", text: String(tick) }));
    }
    svg.append(s("text", { x: left + (width - left - right) / 2, y: base + 33, "text-anchor": "middle", text: "mean set size per relation (log scale)" }));
    svg.append(s("text", { x: left, y: 10, text: "coverage gap across relations" }));
    for (const method of methods) {
      const px = x(pooled[method].condSize);
      const py = y(pooled[method].covgap);
      const from = last.get(method) || [px, py];
      const point = s("g", { class: "move", style: `transform: translate(${from[0].toFixed(1)}px, ${from[1].toFixed(1)}px)` });
      point.append(s("circle", { r: 7, fill: COLORS[method], stroke: "var(--surface)", "stroke-width": 2 }));
      const flip = px > width - right - 110;
      point.append(s("text", { class: "strong", x: flip ? -12 : 12, y: 4, "text-anchor": flip ? "end" : "start", text: NAMES[method] }));
      UI.tip(point, () => [
        h("div", {}, h("b", { text: NAMES[method] })),
        UI.tipLine(number(pooled[method].covgap, 3), "coverage gap across relations"),
        UI.tipLine(number(pooled[method].condSize, 1), "answers per set, mean over relations"),
        UI.tipLine(percent(pooled[method].coverage), "coverage of all queries"),
      ], { focusable: true });
      svg.append(point);
      if (from[0] !== px || from[1] !== py) {
        window.requestAnimationFrame(() => window.requestAnimationFrame(() => (point.style.transform = `translate(${px.toFixed(1)}px, ${py.toFixed(1)}px)`)));
      }
      last.set(method, [px, py]);
    }
    return base + 40;
  }, "Coverage gap against set size for split CP, CondKGCP and Mondrian CP");
  UI.table("cond-trade", () => ({
    columns: ["Method", "Coverage", "Answers per set", "Per relation", "Coverage gap", "Relations below target"],
    rows: ["split", "condkgcp", "mondrian"].map((m) => {
      const p = current.pooled[m];
      const under = p.relations.filter((x) => 1 - current.alpha - x.coverage > 1e-12).length;
      return [NAMES[m], p.coverage, p.size, p.condSize, p.covgap, `${under} of ${p.relations.length}`];
    }),
  }));

  // ------------------------------------------------------- the benchmarks
  const BENCH = { wn18: { name: "WN18", entities: 40943 }, fb15k: { name: "FB15k", entities: 14951 } };
  const benchRows = index.benchmark.conditional;
  UI.chart($("#cond-bench"), (svg, width) => {
    const methods = ["split", "condkgcp", "mondrian"];
    const stacked = width < 520;
    const panels = Object.keys(BENCH);
    const gap = 26;
    const panelWidth = stacked ? width : (width - gap) / 2;
    const row = 20;
    const head = 42;
    const panelHeight = head + 6 * row + 30;
    panels.forEach((dataset, position) => {
      const ox = stacked ? 0 : position * (panelWidth + gap);
      const oy = stacked ? position * (panelHeight + 8) : 0;
      const left = 66;
      const right = 10;
      const x = UI.logarithmic(1, BENCH[dataset].entities, ox + left, ox + panelWidth - right);
      svg.append(s("text", { class: "strong", x: ox, y: oy + 12, text: `${BENCH[dataset].name} · ${integer(BENCH[dataset].entities)} entities` }));
      const ticksAt = [1, 10, 100, 1000, 10000].filter((t) => t <= BENCH[dataset].entities);
      for (const tick of ticksAt) {
        svg.append(s("line", { class: "grid", x1: x(tick), x2: x(tick), y1: oy + head - 8, y2: oy + head + 6 * row }));
        svg.append(s("text", { class: "data", x: x(tick), y: oy + head + 6 * row + 14, "text-anchor": "middle", text: tick >= 1000 ? `${tick / 1000}K` : String(tick) }));
      }
      benchRows.filter((entry) => entry.dataset === dataset).forEach((entry, line) => {
        const cy = oy + head + line * row + row / 2;
        svg.append(s("text", { class: "data", x: ox + left - 8, y: cy + 4, "text-anchor": "end", text: UI.label(entry.model) }));
        const xs = methods.map((m) => x(entry[m].condSize));
        svg.append(s("line", { x1: Math.min(...xs), x2: Math.max(...xs), y1: cy, y2: cy, stroke: "var(--axis)", "stroke-width": 1.5 }));
        methods.forEach((method) => {
          const dot = s("circle", { cx: x(entry[method].condSize), cy, r: 5, fill: COLORS[method], stroke: "var(--surface)", "stroke-width": 2 });
          UI.tip(dot, () => [
            h("div", {}, h("b", { text: `${NAMES[method]} · ${UI.label(entry.model)} on ${BENCH[dataset].name}` })),
            UI.tipLine(number(entry[method].condSize, 1), "answers per set, mean over relations"),
            UI.tipLine(number(entry[method].covgap, 3), "coverage gap across relations"),
            UI.tipLine(percent(entry[method].coverage), "coverage of all queries"),
          ], { focusable: true });
          svg.append(dot);
        });
      });
      if (position === 0 || stacked) {
        let lx = ox;
        for (const method of methods) {
          svg.append(s("circle", { cx: lx + 5, cy: oy + 27, r: 4.5, fill: COLORS[method] }));
          const text = s("text", { x: lx + 14, y: oy + 31, text: NAMES[method] });
          svg.append(text);
          lx += 22 + text.getComputedTextLength();
        }
      }
    });
    return stacked ? panels.length * (panelHeight + 8) : panelHeight;
  }, "Mean set size per relation of split CP, CondKGCP and Mondrian CP on WN18 and FB15k");
  UI.table("cond-bench", () => ({
    columns: ["Graph", "Model", "Method", "Answers per set (per relation)", "Coverage gap", "Coverage"],
    rows: benchRows.flatMap((entry) =>
      ["split", "condkgcp", "mondrian"].map((m) => [BENCH[entry.dataset].name, UI.label(entry.model), NAMES[m], entry[m].condSize, entry[m].covgap, entry[m].coverage]),
    ),
  }));

  // ------------------------------------------------------------------ steps
  function mark() {
    for (const button of $$("#cond-steps button")) {
      if (Number(button.dataset.step) === state.step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    }
  }
  function go(step) {
    state.step = step;
    mark();
    applyMap();
  }
  async function update(rebuild) {
    if (!(await compute())) return;
    if (rebuild || !nodes.length) {
      last.clear();
      map.render();
    } else applyMap();
    trade.render();
    mark();
  }
  async function play() {
    run.cancelled = true;
    const mine = (run = { cancelled: false });
    $("#cond-play").textContent = "Playing…";
    for (const step of [1, 2, 3, 4]) {
      if (mine.cancelled) return;
      go(step);
      await UI.sleep(step === 4 ? 0 : 3200, mine);
    }
    if (!mine.cancelled) $("#cond-play").textContent = "Play again";
  }
  for (const button of $$("#cond-steps button")) {
    button.addEventListener("click", () => {
      run.cancelled = true;
      $("#cond-play").textContent = "Play the four steps";
      go(Number(button.dataset.step));
    });
  }
  $("#cond-play").addEventListener("click", play);

  await update(true);
  UI.onFirstView($("#cond-map"), play);
})();
