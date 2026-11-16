/* How split conformal prediction works, in four animated steps, on Nations: its 14
   countries are few enough to draw every candidate of a query. One score axis carries the
   calibration answers (top) and the query's candidates (bottom); the cut crosses both. */
"use strict";

(async () => {
  const { $, $$, h, s, pretty, percent, number, integer } = UI;
  const READABLE = ["embassy", "tourism", "treaties", "students", "officialvisits", "economicaid", "militaryalliance", "conferences", "booktranslations", "exportbooks"];
  const data = await UI.model("sets-nations-distmult");
  const S = KGCP.scores(data, "minmax"); // 0: the model's top candidate, 1: its last
  const names = data.raw.entities.map(pretty);
  const stage = $("#how-stage");

  let step = 4;
  let entering = 0; // the step whose entrance is being animated
  let seed = 1;
  let position = 0;
  let split;
  let order;
  let run = { cancelled: false };
  let lastCut = null; // where the cut was drawn last, in pixels
  let alpha = UI.level("how-level", (a) => {
    alpha = a;
    entering = 0;
    render();
  });

  function resplit() {
    split = KGCP.split(data.N, seed);
    const random = KGCP.rng(seed + 100);
    // first the queries that show the idea best: a relation with a plain name, and at a
    // 90% target a set of a few countries that contains the answer
    const typical = KGCP.calibrate(S, split.cal, { method: "split", alpha: 0.1 });
    const good = (i) => {
      const { covered, size } = KGCP.predict(typical, i);
      const readable = READABLE.includes(data.raw.relations[data.relation[i]]) && S.finite[i] >= 9;
      return readable + (readable && covered && size >= 3 && size <= 5);
    };
    order = Array.from(split.test).sort(() => random() - 0.5).sort((a, b) => good(b) - good(a));
    position = 0;
  }
  resplit();

  function state() {
    const i = order[position];
    const model = KGCP.calibrate(S, split.cal, { method: "split", alpha });
    const threshold = model.thresholds.get(0);
    const sorted = Float64Array.from(split.cal, (j) => S.truth[j]).sort();
    const n = sorted.length;
    const k = Math.ceil((n + 1) * (1 - alpha) - 1e-9);
    const candidates = Array.from({ length: data.E }, (_, e) => e)
      .filter((e) => S.values[i * data.E + e] !== Infinity)
      .sort((a, b) => S.values[i * data.E + a] - S.values[i * data.E + b] || a - b);
    const tested = Array.from(split.test, (j) => KGCP.predict(model, j).covered);
    return { i, model, threshold, sorted, n, k, candidates, tested, target: data.target[i] };
  }

  const view = UI.chart(stage, draw, "Split conformal prediction on one query");

  function draw(svg, width) {
    const st = state();
    const { i, threshold, sorted, n, k, candidates, tested, target } = st;
    const narrow = width < 560;
    const left = narrow ? 96 : 150;
    const right = 18;
    const x = UI.linear(0, 1, left, width - right);
    const top = 30; // the curve of the calibration answers
    const bottom = 132;
    const y = UI.linear(0, 1, bottom, top);
    const rug = bottom + 16;
    const axis = rug + 16;
    const rows = axis + 60; // below the tick labels, the axis' ends and the cut's label
    const row = 21;
    const rowsEnd = rows + candidates.length * row;
    const cell = narrow ? 7 : 8;
    const perLine = Math.floor((width - left - right) / cell);
    const waffle = rowsEnd + 46;
    const height = waffle + Math.ceil(tested.length / perLine) * cell + 10;
    const cutX = x(Math.min(threshold, 1));
    const finite = threshold !== Infinity;
    svg.setAttribute("class", `how${entering ? ` enter-${entering}` : ""}`);
    const hidden = (from) => (step < from ? "0" : "1");

    // ---- the score axis, shared by both halves
    svg.append(s("line", { class: "axis", x1: left, x2: width - right, y1: axis, y2: axis }));
    for (const tick of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      svg.append(s("line", { class: "axis", x1: x(tick), x2: x(tick), y1: axis, y2: axis + 5 }));
      svg.append(s("text", { class: "data", x: x(tick), y: axis + 17, "text-anchor": "middle", text: tick.toFixed(1) }));
    }
    svg.append(s("text", { x: left, y: axis + 31, text: "0: the model's top candidate" }));
    svg.append(s("text", { x: width - right, y: axis + 31, "text-anchor": "end", text: narrow ? "1: its last" : "nonconformity score · 1: its last candidate" }));

    // ---- step 2: where the answers of the held-out queries fall
    const calibration = s("g", { class: "fade", opacity: hidden(2) });
    for (const share of [0, 0.5, 1]) {
      calibration.append(s("line", { class: "grid", x1: left, x2: width - right, y1: y(share), y2: y(share) }));
      calibration.append(s("text", { class: "data", x: left - 10, y: y(share) + 4, "text-anchor": "end", text: percent(share, 0) }));
    }
    calibration.append(s("text", { x: left, y: 12, text: narrow ? `share of ${n} held-out answers up to a score` : `share of the ${n} held-out answers that score at most this` }));
    let path = `M${x(0)} ${y(0)}`;
    sorted.forEach((value, j) => (path += `H${x(value).toFixed(1)}V${y((j + 1) / n).toFixed(1)}`));
    path += `H${x(1)}`;
    const curve = s("path", { class: "curve", d: path, fill: "none", stroke: "var(--ink-2)", "stroke-width": 2, "stroke-linejoin": "round" });
    calibration.append(curve);
    const dots = s("g", { class: "rug" });
    const random = KGCP.rng(seed);
    sorted.forEach((value, j) => {
      const counted = j < k && finite;
      dots.append(
        s("circle", {
          cx: x(value).toFixed(1),
          cy: (rug - 6 + random() * 12).toFixed(1),
          r: 2.2,
          fill: step >= 3 && counted ? "var(--blue)" : "var(--ink-3)",
          "fill-opacity": 0.55,
          style: `animation-delay:${Math.round((j / n) * 900)}ms`,
        }),
      );
    });
    calibration.append(dots);
    svg.append(calibration);
    curve.style.setProperty("--length", curve.getTotalLength());
    curve.setAttribute("stroke-dasharray", curve.getTotalLength());

    // ---- step 4: the candidates left of the cut are the answer set
    const members = candidates.filter((e) => S.values[i * data.E + e] <= threshold);
    const result = s("g", { class: "fade", opacity: hidden(4) });
    if (members.length) {
      result.append(s("rect", { x: left, y: rows - 2, width: Math.max(cutX - left, 2), height: members.length * row, fill: "var(--wash)" }));
    }
    svg.append(result);

    // ---- step 1: the candidates of the query, by score
    const candidateRows = s("g", { class: "rows" });
    candidates.forEach((e, index) => {
      const cy = rows + index * row + row / 2 - 2;
      const value = S.values[i * data.E + e];
      const inSet = step >= 4 && value <= threshold;
      const isTarget = e === target;
      candidateRows.append(s("line", { class: "grid", x1: left, x2: width - right, y1: cy, y2: cy }));
      candidateRows.append(
        s("text", { class: `data${inSet || isTarget ? " strong" : ""}`, x: left - 12, y: cy + 4, "text-anchor": "end", text: names[e] }),
      );
      const dot = s("circle", {
        class: "dot",
        cx: x(value).toFixed(1),
        cy,
        r: 5,
        fill: inSet ? "var(--blue)" : "var(--mute)",
        stroke: "var(--surface)",
        "stroke-width": 2,
        style: `--dx:${(x(1) - x(value)).toFixed(0)}px; animation-delay:${index * 45}ms; transition-delay:${entering === 4 ? index * 60 : 0}ms`,
      });
      UI.tip(dot, () => [UI.tipLine(number(value, 3), `score of ${names[e]}`), isTarget ? "the answer that was held out" : ""], { focusable: true });
      candidateRows.append(dot);
      if (isTarget) {
        candidateRows.append(s("circle", { class: "dot", cx: x(value).toFixed(1), cy, r: 8.5, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5, style: `--dx:${(x(1) - x(value)).toFixed(0)}px; animation-delay:${index * 45}ms` }));
        const label = x(value) > width - right - 110;
        candidateRows.append(s("text", { class: "strong", x: x(value) + (label ? -14 : 14), y: cy + 4, "text-anchor": label ? "end" : "start", text: "held-out answer" }));
      }
    });
    svg.append(candidateRows);

    // ---- step 3: the cut at the k-th smallest calibration score
    const level = finite ? y(k / n) : y(1);
    const cut = s("g", { class: "fade", opacity: hidden(3) });
    cut.append(s("line", { class: "level", x1: left, x2: cutX, y1: level, y2: level, stroke: "var(--blue)", "stroke-width": 1.5 }));
    cut.append(s("circle", { cx: cutX, cy: level, r: 4, fill: "var(--blue)", stroke: "var(--surface)", "stroke-width": 2 }));
    // the cut slides from where it was when only the target changed
    const from = lastCut === null || entering ? cutX : lastCut;
    const blade = s("g", { class: "blade move", style: `transform: translateX(${from.toFixed(1)}px)` });
    if (from !== cutX) {
      window.requestAnimationFrame(() => window.requestAnimationFrame(() => (blade.style.transform = `translateX(${cutX.toFixed(1)}px)`)));
    }
    lastCut = cutX;
    blade.append(s("line", { x1: 0, x2: 0, y1: level, y2: rowsEnd + 4, stroke: "var(--ink)", "stroke-width": 1.5 }));
    blade.append(s("path", { d: `M-5 ${rowsEnd + 4}h10l-5 -7z`, fill: "var(--ink)" }));
    const flip = cutX > width - right - 150;
    blade.append(
      s("text", { class: "strong data halo", x: flip ? -8 : 8, y: axis + 47, "text-anchor": flip ? "end" : "start", text: finite ? `cut q̂ = ${number(threshold, 2)}` : "no cut: too few queries" }),
    );
    cut.append(blade);
    const levelText = finite ? `${integer(k)} of ${n} answers` : `all ${n} answers`;
    cut.append(s("text", { class: "strong", x: left + 6, y: level - 6, text: levelText }));
    svg.append(cut);

    // ---- step 4: the same cut on the test queries that did not calibrate
    const check = s("g", { class: "fade waffle", opacity: hidden(4) });
    const covered = tested.filter(Boolean).length;
    check.append(
      s("text", { x: left, y: waffle - 12, text: narrow ? `${integer(tested.length)} other test queries:` : `The same cut on the ${integer(tested.length)} test queries that did not calibrate:` }),
    );
    tested.forEach((hit, j) => {
      const cx = left + (j % perLine) * cell;
      const cy = waffle + Math.floor(j / perLine) * cell;
      check.append(
        s("rect", {
          x: cx + 0.75, y: cy + 0.75, width: cell - 2.5, height: cell - 2.5, rx: 1,
          fill: hit ? "var(--blue)" : "none",
          "fill-opacity": 0.7,
          stroke: hit ? "none" : "var(--bad)",
          "stroke-width": 1.5,
          style: `animation-delay:${Math.round((j / tested.length) * 700)}ms`,
        }),
      );
    });
    svg.append(check);

    caption(st, members, covered);
    return height;
  }

  function caption({ i, threshold, n, k, candidates, tested, target }, members, covered) {
    const node = $("#how-caption");
    const level = percent(1 - alpha, 0);
    const b = (text) => h("b", { text });
    const texts = {
      1: ["The model scores all ", b(`${candidates.length} candidate countries`), " for this query (answers it already knows from training are left out). The score is turned into a nonconformity: 0 for its top candidate, 1 for its last. A ranking alone gives no place to stop."],
      2: ["Now take ", b(`${n} held-out queries`), " whose answer is known. Each tick marks where the true answer of one of them scored; the curve counts them from the left. Most answers sit near 0, but some are far down the ranking."],
      3: threshold === Infinity
        ? [`A ${level} target needs more held-out queries than ${n}: the cut would lie beyond the last answer, so every candidate must be kept.`]
        : ["Walk along the curve to the ", b(`${UI.ordinal(k)} of the ${n} answers`), ` (⌈(n + 1) × ${level}⌉), and cut there: `, b(`q̂ = ${number(threshold, 2)}`), `. A new answer falls left of this cut with a probability of at least ${level}, whatever the model.`],
      4: ["Keep every candidate left of the cut: ", b(`{${members.map((e) => names[e]).join(", ")}}`), members.includes(target) ? ", which contains the held-out answer. " : ", which misses the held-out answer this time. ", "On the test queries that did not calibrate, ", b(`${integer(covered)} of ${integer(tested.length)} sets (${percent(covered / tested.length)})`), " contain theirs: the filled squares."],
    };
    node.replaceChildren(...texts[step]);
    $("#how-query").replaceChildren(...UI.query(data, i));
  }

  function render() {
    for (const button of $$("#how-steps button")) {
      if (Number(button.dataset.step) === step) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    }
    view.render();
  }

  /* Shows a step; with `animate`, plays its entrance. */
  function go(target, animate = false) {
    step = target;
    entering = animate && !UI.reduced() ? target : 0;
    lastCut = null;
    render();
    // a later redraw (a resize, the target) must not replay the entrance
    window.setTimeout(() => {
      if (entering === target) entering = 0;
    }, 1800);
  }

  const PAUSE = { 1: 2300, 2: 3000, 3: 3000, 4: 0 };
  async function play() {
    run.cancelled = true;
    const mine = (run = { cancelled: false });
    $("#how-play").textContent = "Playing…";
    for (const target of [1, 2, 3, 4]) {
      if (mine.cancelled) return;
      go(target, true);
      await UI.sleep(PAUSE[target], mine);
    }
    if (!mine.cancelled) $("#how-play").textContent = "Play again";
  }
  const stop = () => {
    run.cancelled = true;
    $("#how-play").textContent = "Play the four steps";
  };

  for (const button of $$("#how-steps button")) {
    button.addEventListener("click", () => {
      stop();
      go(Number(button.dataset.step), true);
    });
  }
  $("#how-play").addEventListener("click", play);
  $("#how-next").addEventListener("click", () => {
    stop();
    position = (position + 1) % order.length;
    go(step);
  });
  $("#how-split").addEventListener("click", () => {
    stop();
    seed++;
    resplit();
    go(Math.max(step, 2), true);
  });

  render();
  UI.onFirstView(stage, play);
})();
