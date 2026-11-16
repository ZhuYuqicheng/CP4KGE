/* The playground: any exported model, score and predictor; metrics pooled over random
   calibration splits as in the library's experiments, and the answer sets of one split. */
"use strict";

(async () => {
  const { $, $$, h, s, pretty, percent, number, integer } = UI;
  const SPLITS = 100;
  const index = window.KGCP_DATA.index.sets;
  const state = {
    dataset: "umls", model: "distmult", score: "softmax", method: "split",
    alpha: 0.1, minSize: 20, gamma: 0.5, seed: 0, selected: null, missed: false, search: "",
  };
  const cache = new Map(); // "key/score" -> scores
  let current = null; // what the charts show

  // ---------------------------------------------------------------- controls
  const datasets = [...new Set(index.map((entry) => entry.dataset))];
  const datasetSelect = $("#try-dataset");
  const modelSelect = $("#try-model");
  datasetSelect.replaceChildren(...datasets.map((d) => h("option", { value: d, text: UI.label(d) })));
  datasetSelect.value = state.dataset;
  function listModels() {
    const models = index.filter((entry) => entry.dataset === state.dataset);
    modelSelect.replaceChildren(
      ...models.map((entry) => h("option", { value: entry.model, text: `${UI.label(entry.model)} · MRR ${number(entry.mrr, 2)}` })),
    );
    if (!models.some((entry) => entry.model === state.model)) state.model = models[0].model;
    modelSelect.value = state.model;
  }
  listModels();
  datasetSelect.addEventListener("change", () => {
    state.dataset = datasetSelect.value;
    state.selected = null;
    listModels();
    refresh();
  });
  modelSelect.addEventListener("change", () => {
    state.model = modelSelect.value;
    state.selected = null;
    refresh();
  });
  $("#try-score").addEventListener("change", (event) => {
    state.score = event.target.value;
    refresh();
  });
  for (const button of $$("#try-method button")) {
    button.addEventListener("click", () => {
      state.method = button.dataset.method;
      for (const other of $$("#try-method button")) other.setAttribute("aria-pressed", String(other === button));
      $("#try-minsize-field").hidden = $("#try-gamma-field").hidden = state.method !== "condkgcp";
      refresh();
    });
  }
  state.alpha = UI.level("try-level", (alpha) => {
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
  slider("try-minsize", "minSize", (v) => integer(v));
  slider("try-gamma", "gamma", (v) => number(v, 2));
  $("#try-resample").addEventListener("click", () => {
    state.seed += SPLITS;
    state.selected = null;
    refresh();
  });
  $("#try-missed").addEventListener("change", (event) => {
    state.missed = event.target.checked;
    list();
  });
  $("#try-search").addEventListener("input", (event) => {
    state.search = event.target.value.trim().toLowerCase();
    list();
  });

  let pending = 0;
  function schedule() {
    window.cancelAnimationFrame(pending);
    pending = window.requestAnimationFrame(refresh);
  }

  // ---------------------------------------------------------------- compute
  function options() {
    const { method, alpha, minSize, gamma } = state;
    if (method === "mondrian") return { method, alpha, groups: "relation" };
    if (method === "condkgcp") return { method, alpha, groups: "merged", minSize, gamma };
    return { method, alpha };
  }

  let ticket = 0;
  async function refresh() {
    const mine = ++ticket;
    const key = `sets-${state.dataset}-${state.model}`;
    const data = await UI.model(key);
    if (mine !== ticket) return;
    const id = `${key}/${state.score}`;
    if (!cache.has(id)) cache.set(id, KGCP.scores(data, state.score, { seed: 0 }));
    const S = cache.get(id);
    const pooled = KGCP.repeat(S, options(), SPLITS, state.seed);
    const split = KGCP.split(data.N, state.seed);
    const model = KGCP.calibrate(S, split.cal, options());
    const queries = Array.from(split.test, (i) => ({ i, ...KGCP.predict(model, i), text: UI.queryText(data, i) }));
    current = { data, S, pooled, split, model, queries, alpha: state.alpha };
    if (state.selected === null || !queries.some((q) => q.i === state.selected)) {
      const typical = queries.find((q) => q.covered && q.size >= 2 && q.size <= 6) || queries[0];
      state.selected = typical.i;
    }
    tiles();
    splitsChart.render();
    sizesChart.render();
    relationsChart.render();
    list();
    detail();
  }

  // ------------------------------------------------------------------ tiles
  function tiles() {
    const { pooled, data, alpha } = current;
    const target = 1 - alpha;
    const meets = pooled.coverage >= target - 0.003; // within the noise of 100 splits
    const under = pooled.relations.filter((r) => target - r.coverage > 1e-12).length;
    const tile = (label, value, ...sub) => h("div", { class: "tile" }, h("div", { class: "label", text: label }), h("div", { class: "value", text: value }), h("div", { class: "sub" }, ...sub));
    $("#try-tiles").replaceChildren(
      tile("Coverage", percent(pooled.coverage), UI.icon(meets ? "check" : "cross", 13), meets ? ` meets the ${percent(target, 0)} target` : ` below the ${percent(target, 0)} target`),
      tile("Mean set size", number(pooled.size, 2), `of ${data.E} entities`),
      tile("Sets with one answer", percent(pooled.singleton), `empty sets: ${percent(pooled.empty)}`),
      tile("Coverage gap across relations", number(pooled.covgap, 3), "mean distance from the target (CovGap)"),
      tile("Relations below the target", `${under} of ${pooled.relations.length}`, "relations with 10 or more test queries"),
    );
  }

  // ------------------------------------------------- coverage of each split
  const splitsChart = UI.chart($("#try-splits"), (svg, width) => {
    if (!current) return 120;
    const { pooled, alpha } = current;
    const target = 1 - alpha;
    const coverages = pooled.coverages;
    const step = 0.005;
    const low = Math.floor((Math.min(...coverages, target) - 0.012) / step) * step;
    const high = Math.min(1, Math.ceil((Math.max(...coverages, target) + 0.012) / step) * step);
    const left = 14;
    const right = 14;
    const x = UI.linear(low, high, left, width - right);
    const bins = new Map();
    coverages.forEach((coverage, split) => {
      const bin = Math.min(Math.floor((coverage - low) / step), Math.round((high - low) / step) - 1);
      if (!bins.has(bin)) bins.set(bin, []);
      bins.get(bin).push({ coverage, split });
    });
    const tallest = Math.max(...Array.from(bins.values(), (b) => b.length));
    const radius = Math.min(4, Math.max(2.5, (x(low + step) - x(low)) / 2 - 1));
    const gap = radius * 2 + 1.5;
    const top = 30;
    const base = top + Math.max(tallest, 6) * gap + 6;
    const height = base + 30;
    svg.append(s("line", { class: "axis", x1: left, x2: width - right, y1: base, y2: base }));
    for (const tick of UI.ticks(low, high, width < 420 ? 4 : 7)) {
      svg.append(s("line", { class: "axis", x1: x(tick), x2: x(tick), y1: base, y2: base + 4 }));
      svg.append(s("text", { class: "data", x: x(tick), y: base + 17, "text-anchor": "middle", text: percent(tick, 1) }));
    }
    for (const [bin, members] of bins) {
      members.forEach((member, level) => {
        const dot = s("circle", {
          cx: x(low + (bin + 0.5) * step), cy: base - radius - 2 - level * gap, r: radius,
          fill: "var(--blue)", stroke: "var(--surface)", "stroke-width": 1,
        });
        UI.tip(dot, () => UI.tipLine(percent(member.coverage), `of the sets of split ${member.split + 1} contain the answer`));
        svg.append(dot);
      });
    }
    // the target, and the mean over the splits
    svg.append(s("line", { x1: x(target), x2: x(target), y1: 16, y2: base, stroke: "var(--ink)", "stroke-width": 1.5 }));
    const flip = x(target) > width / 2;
    svg.append(s("text", { class: "strong", x: x(target) + (flip ? -6 : 6), y: 12, "text-anchor": flip ? "end" : "start", text: `target ${percent(target, 0)}` }));
    const mean = pooled.coverage;
    svg.append(s("path", { d: `M${x(mean)} ${base + 22}l-5 8h10z`, fill: "var(--ink-2)" }));
    svg.append(s("text", { x: x(mean) + (flip ? 9 : -9), y: base + 30, "text-anchor": flip ? "start" : "end", text: `mean ${percent(mean)}` }));
    $("#try-splits-sub").textContent = `${SPLITS} random halves of the test queries calibrate; each dot is the coverage on the other half.`;
    return height + 4;
  }, "Coverage of each of 100 calibration splits");
  UI.table("try-splits", () => ({
    columns: ["Split", "Coverage"],
    rows: current.pooled.coverages.map((coverage, split) => [String(split + 1), coverage]),
  }));

  // ------------------------------------------------------------- set sizes
  const LAST = 12; // sizes above are one column
  function sizeShares() {
    const { sizes } = current.pooled;
    const shares = sizes.slice(0, LAST + 1);
    shares.push(sizes.slice(LAST + 1).reduce((a, b) => a + b, 0));
    return shares;
  }
  const sizesChart = UI.chart($("#try-sizes"), (svg, width) => {
    if (!current) return 120;
    const shares = sizeShares();
    const left = 38;
    const right = 8;
    const top = 16;
    const base = 150;
    const peak = Math.max(...shares);
    const yTicks = UI.ticks(0, peak, 4);
    const y = UI.linear(0, Math.max(yTicks[yTicks.length - 1], peak), base, top);
    const band = (width - left - right) / shares.length;
    const bar = Math.min(24, band - 2);
    for (const tick of yTicks) {
      svg.append(s("line", { class: tick === 0 ? "axis" : "grid", x1: left, x2: width - right, y1: y(tick), y2: y(tick) }));
      svg.append(s("text", { class: "data", x: left - 6, y: y(tick) + 4, "text-anchor": "end", text: percent(tick, 0) }));
    }
    const tallest = shares.indexOf(peak);
    shares.forEach((share, size) => {
      const cx = left + (size + 0.5) * band;
      const name = size > LAST ? `${LAST + 1}+` : String(size);
      svg.append(s("text", { class: "data", x: cx, y: base + 15, "text-anchor": "middle", text: name }));
      if (share <= 0) return;
      const topY = Math.min(y(share), base - 1);
      const r = Math.min(4, (base - topY) / 2);
      const column = s("path", {
        d: `M${cx - bar / 2} ${base}V${topY + r}q0 ${-r} ${r} ${-r}H${cx + bar / 2 - r}q${r} 0 ${r} ${r}V${base}Z`,
        fill: "var(--blue)",
      });
      UI.tip(column, () => UI.tipLine(percent(share), `of the sets have ${size > LAST ? `${LAST + 1} or more answers` : size === 1 ? "one answer" : `${size} answers`}`), { focusable: true });
      svg.append(column);
      if (size === tallest || (size > LAST && share > 0.02)) {
        svg.append(s("text", { class: "strong data", x: cx, y: topY - 5, "text-anchor": "middle", text: percent(share, 0) }));
      }
    });
    svg.append(s("text", { x: left + (width - left - right) / 2, y: base + 31, "text-anchor": "middle", text: "answers in the set" }));
    $("#try-sizes-sub").textContent = `Share of the sets with each size, over the ${SPLITS} splits.`;
    return base + 38;
  }, "Distribution of the size of the answer sets");
  UI.table("try-sizes", () => ({
    columns: ["Answers in the set", "Share of the sets"],
    rows: sizeShares().map((share, size) => [size > LAST ? `${LAST + 1} or more` : String(size), share]),
  }));

  // --------------------------------------------------- coverage per relation
  const relationsChart = UI.chart($("#try-relations"), (svg, width) => {
    if (!current) return 120;
    const { pooled, data, alpha } = current;
    const target = 1 - alpha;
    const relations = pooled.relations.slice().sort((a, b) => a.coverage - b.coverage);
    const left = Math.min(170, Math.round(width * 0.36));
    const right = 44;
    const top = 26;
    const row = 17;
    const low = Math.min(0.6, Math.floor(Math.min(...relations.map((r) => r.coverage)) * 10) / 10);
    const x = UI.linear(low, 1, left, width - right);
    const height = top + relations.length * row + 26;
    for (const tick of UI.ticks(low, 1, width < 420 ? 3 : 5)) {
      svg.append(s("line", { class: "grid", x1: x(tick), x2: x(tick), y1: top - 4, y2: height - 22 }));
      svg.append(s("text", { class: "data", x: x(tick), y: height - 8, "text-anchor": "middle", text: percent(tick, 0) }));
    }
    svg.append(s("line", { x1: x(target), x2: x(target), y1: 14, y2: height - 22, stroke: "var(--ink)", "stroke-width": 1.5 }));
    svg.append(s("text", { class: "strong", x: x(target) - 6, y: 11, "text-anchor": "end", text: `target ${percent(target, 0)}` }));
    relations.forEach((relation, position) => {
      const cy = top + position * row + row / 2;
      const under = target - relation.coverage > 1e-12;
      const name = s("text", { class: "data", x: left - 10, y: cy + 4, "text-anchor": "end", text: pretty(data.raw.relations[relation.relation]) });
      svg.append(name);
      shorten(name, left - 14);
      svg.append(s("line", { x1: x(target), x2: x(relation.coverage), y1: cy, y2: cy, stroke: "var(--blue)", "stroke-width": 2 }));
      // below the target: a hollow dot, so the state is not carried by position alone
      const dot = s("circle", {
        cx: x(relation.coverage), cy, r: 4.5,
        fill: under ? "var(--surface)" : "var(--blue)", stroke: "var(--blue)", "stroke-width": 2,
      });
      svg.append(dot);
      svg.append(s("text", { class: "data", x: width - 2, y: cy + 4, "text-anchor": "end", text: integer(relation.queries) }));
      const hit = s("rect", { x: 0, y: cy - row / 2, width, height: row, fill: "transparent" });
      UI.tip(hit, () => [
        UI.tipLine(percent(relation.coverage), `coverage of ${pretty(data.raw.relations[relation.relation])}`),
        UI.tipLine(number(relation.size, 1), "answers per set"),
        UI.tipLine(integer(relation.queries), "test queries"),
      ], { focusable: true });
      svg.append(hit);
    });
    svg.append(s("text", { x: width - 2, y: 11, "text-anchor": "end", text: "queries" }));
    const below = relations.filter((r) => target - r.coverage > 1e-12).length;
    $("#try-relations-sub").textContent =
      `The ${relations.length} of ${data.R} relations with 10 or more test queries. Hollow: below the target (${below}).`;
    return height;
  }, "Coverage of each relation");
  UI.table("try-relations", () => ({
    columns: ["Relation", "Test queries", "Coverage", "Answers per set"],
    rows: current.pooled.relations
      .slice()
      .sort((a, b) => a.coverage - b.coverage)
      .map((r) => [pretty(current.data.raw.relations[r.relation]), String(r.queries), r.coverage, r.size]),
  }));

  function shorten(text, width) {
    const full = text.textContent;
    let length = full.length;
    while (length > 4 && text.getComputedTextLength() > width) {
      length--;
      text.textContent = `${full.slice(0, length).trimEnd()}…`;
    }
  }

  // ----------------------------------------------- the answer sets of a split
  function list() {
    const { queries } = current;
    const shown = queries.filter((q) => (!state.missed || !q.covered) && (!state.search || q.text.toLowerCase().includes(state.search)));
    const covered = queries.filter((q) => q.covered).length;
    $("#try-list-sub").textContent =
      `Split 1 of ${SPLITS}: ${integer(covered)} of ${integer(queries.length)} sets contain the answer (${percent(covered / queries.length)}). Showing ${integer(shown.length)}.`;
    $("#try-list").replaceChildren(
      ...shown.map((q) =>
        h(
          "button",
          {
            type: "button", role: "option", "aria-selected": String(q.i === state.selected),
            title: q.covered ? "contains the answer" : "misses the answer",
            onclick: () => {
              state.selected = q.i;
              for (const other of $$("#try-list button")) other.setAttribute("aria-selected", "false");
              $(`#try-list button[data-i="${q.i}"]`).setAttribute("aria-selected", "true");
              detail();
            },
            "data-i": q.i,
          },
          UI.icon(q.covered ? "check" : "cross", 14),
          h("span", { class: "text", text: q.text }),
          h("span", { class: "size", text: q.size === 1 ? "1 answer" : `${integer(q.size)} answers` }),
        ),
      ),
      ...(shown.length ? [] : [h("p", { class: "note", style: "padding: 14px", text: "No query matches the filter." })]),
    );
  }

  function detail() {
    const { data, S, model, queries } = current;
    const q = queries.find((entry) => entry.i === state.selected) || queries[0];
    const i = q.i;
    const members = new Set(KGCP.members(model, i));
    const ranked = Array.from({ length: data.E }, (_, e) => e)
      .filter((e) => S.values[i * data.E + e] !== Infinity)
      .sort((a, b) => S.values[i * data.E + a] - S.values[i * data.E + b] || a - b);
    const target = data.target[i];
    const names = data.raw.entities;
    const shown = ranked.slice(0, Math.max(10, Math.min(members.size + 3, 16)));
    if (!shown.includes(target)) shown.push(target);

    const group = model.groupOf(i);
    const calibrated = (model.members.get(group) || []).length;
    const facts = [];
    if (model.options.method === "split") facts.push(`One threshold for all queries, from ${integer(calibrated)} held-out queries.`);
    else if (model.groups === "relation") facts.push(`Threshold of this relation alone, from its ${integer(calibrated)} held-out queries.`);
    else {
      const host = pretty(data.raw.relations[group]);
      const merged = Array.from(model.merged.table).filter((g) => g === group).length;
      facts.push(`Threshold of the group of ${host} (${merged} relation${merged === 1 ? "" : "s"}, ${integer(calibrated)} held-out queries).`);
    }
    if (q.threshold === Infinity) facts.push("Too few held-out queries for a threshold at this target: every candidate is kept.");
    if (model.topK.has(group)) facts.push(`Rank calibration keeps at most the top ${model.topK.get(group)} candidates.`);

    const rows = [];
    let cutDrawn = false;
    shown.forEach((e) => {
      const rank = ranked.indexOf(e) + 1;
      if (!members.has(e) && !cutDrawn && members.size > 0 && rank > members.size) {
        rows.push(h("tr", {}, h("td", { colspan: 4, class: "note", style: "border-bottom: 2px solid var(--ink); padding: 0" })));
        cutDrawn = true;
      }
      rows.push(
        h(
          "tr",
          {},
          h("td", { class: "num", text: String(rank) }),
          h("td", { style: members.has(e) ? "font-weight: 650" : "color: var(--ink-3)" }, pretty(names[e]), e === target ? h("span", { class: "note", text: "  ← held-out answer" }) : null),
          h("td", { class: "num", text: number(S.values[i * data.E + e], 3) }),
          h("td", { text: members.has(e) ? "in the set" : "" }),
        ),
      );
    });

    $("#try-detail").replaceChildren(
      h("p", { class: "query" }, ...UI.query(data, i)),
      h("p", { class: "verdict" }, UI.icon(q.covered ? "check" : "cross"), h("span", {}, q.covered ? "The set contains the held-out answer, " : "The set misses the held-out answer, ", h("b", { text: pretty(names[target]) }), `. It has ${integer(q.size)} of ${S.finite[i]} candidates.`)),
      h("p", { class: "note", text: facts.join(" ") }),
      h(
        "table",
        { class: "plain" },
        h("thead", {}, h("tr", {}, h("th", { class: "num", text: "Rank" }), h("th", { text: "Candidate" }), h("th", { class: "num", text: "Score" }), h("th", { text: "" }))),
        h("tbody", {}, rows),
      ),
      h("p", { class: "note", text: `Lower scores are more plausible. The line marks the cut${q.threshold === Infinity ? "" : ` at ${number(q.threshold, 3)}`}; candidates above it form the set.` }),
    );
  }

  await refresh();
})();
