/* The hero: one real query, its answer set, and the coverage target as a dial. */
"use strict";

(async () => {
  const { $, h, s, pretty, percent, number, integer } = UI;
  const ROWS = 8; // candidates shown
  const data = await UI.model("sets-umls-distmult");
  const S = KGCP.scores(data, "softmax");
  const names = data.raw.entities.map(pretty);

  // The query of the paper's example, in the half that is tested.
  const wanted = (i) =>
    data.direction[i] === 0 &&
    data.raw.entities[data.anchor[i]] === "pharmacologic_substance" &&
    data.raw.relations[data.relation[i]] === "treats" &&
    data.raw.entities[data.target[i]] === "neoplastic_process";
  let { cal, test } = KGCP.split(data.N, 0);
  if (Array.from(cal).some(wanted)) [cal, test] = [test, cal];
  const first = Array.from(test).find(wanted);
  const random = KGCP.rng(7);
  const order = Array.from(test).sort(() => random() - 0.5);
  if (first !== undefined) order.unshift(first);
  let position = 0;

  let cutX = null; // the cut's last position, to animate from
  let alpha = UI.level("hero-level", (a) => {
    alpha = a;
    render();
  });
  $("#hero-next").addEventListener("click", () => {
    position = (position + 1) % order.length;
    cutX = null;
    render();
  });
  const bars = UI.chart($("#hero-bars"), (svg, width) => draw(svg, width), "The model's probability of its top candidates");

  function state() {
    const i = order[position];
    const model = KGCP.calibrate(S, cal, { method: "split", alpha });
    const threshold = model.thresholds.get(0);
    const members = KGCP.members(model, i);
    const ranked = Array.from({ length: data.E }, (_, e) => e)
      .filter((e) => S.values[i * data.E + e] !== Infinity)
      .sort((a, b) => S.values[i * data.E + a] - S.values[i * data.E + b] || a - b);
    return { i, model, threshold, members, ranked, target: data.target[i] };
  }

  function draw(svg, width) {
    const { i, threshold, ranked, target } = state();
    const shown = ranked.slice(0, ROWS);
    const targetRank = ranked.indexOf(target) + 1;
    if (!shown.includes(target)) shown.push(target);
    const row = 27;
    const top = 22;
    const left = Math.min(Math.round(width * 0.44), 200);
    const right = 46;
    const x = UI.linear(0, 1, left, width - right);
    const height = top + shown.length * row + 8;
    const probability = (e) => 1 - S.values[i * data.E + e];
    const cut = Math.max(0, 1 - threshold); // candidates with a probability of at least this

    for (const tick of [0, 0.25, 0.5, 0.75, 1]) {
      svg.append(s("line", { class: "grid", x1: x(tick), x2: x(tick), y1: top - 4, y2: height - 6 }));
      svg.append(s("text", { x: x(tick), y: 10, "text-anchor": "middle", text: tick === 0 ? "0" : percent(tick, 0) }));
    }
    const line = s("g", {}, s("line", { x1: 0, x2: 0, y1: top - 6, y2: height - 4, stroke: "var(--ink)", "stroke-width": 1.5 }),
      s("path", { d: `M-5 ${top - 6}h10l-5 6z`, fill: "var(--ink)" }));
    svg.append(line);
    shown.forEach((e, index) => {
      const y = top + index * row;
      const inSet = S.values[i * data.E + e] <= threshold;
      const p = probability(e);
      const isTarget = e === target;
      const beyond = index >= ROWS; // the answer, ranked below the rows shown
      const name = (beyond ? `#${targetRank}  ` : "") + names[e];
      const text = s("text", { x: left - 10, y: y + 13, "text-anchor": "end", class: isTarget ? "strong data" : "data", text: name });
      svg.append(text);
      fit(text, left - 14);
      const end = Math.max(x(p), left + 2);
      // 4px rounded data end, square at the baseline
      svg.append(
        s("path", {
          d: `M${left} ${y + 4}H${end - 3}q3 0 3 3v6q0 3 -3 3H${left}Z`,
          fill: inSet ? "var(--blue)" : "var(--mute)",
          style: "transition: fill .25s",
        }),
      );
      svg.append(s("text", { x: end + 6, y: y + 13, class: "data halo", text: p >= 0.0995 ? percent(p, 0) : percent(p, 1) }));
      if (isTarget) {
        // a ring marks the answer: shape, not only color
        svg.append(s("circle", { cx: left - 5, cy: y + 10, r: 3, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5 }));
        text.setAttribute("x", left - 14);
        fit(text, left - 18);
      }
    });

    const target_ = x(Math.min(cut, 1));
    const from = cutX === null ? target_ : cutX;
    UI.tween(260, (t) => line.setAttribute("transform", `translate(${UI.lerp(from, target_, t)} 0)`), { easing: UI.easeOut });
    cutX = target_;
    return height;
  }

  /* Shortens the text of an SVG label until it is at most `width` wide. */
  function fit(text, width) {
    const full = text.textContent;
    let length = full.length;
    while (length > 4 && text.getComputedTextLength() > width) {
      length--;
      text.textContent = `${full.slice(0, length).trimEnd()}…`;
    }
  }

  function render() {
    const { i, model, threshold, members, target } = state();
    $("#hero-query").replaceChildren(...UI.query(data, i));
    bars.render();

    const set = $("#hero-set");
    const shown = members.slice(0, 9);
    const more = members.length - shown.length;
    set.replaceChildren(
      h("span", { class: "brace", "aria-hidden": "true", text: "{" }),
      ...shown.map((e) => h("span", { class: e === target ? "chip answer" : "chip", text: names[e] })),
      ...(more > 0 ? [h("span", { class: "chip more", text: `+ ${more} more` })] : []),
      ...(members.length === 0 ? [h("span", { class: "chip more", text: "empty: no candidate is plausible enough" })] : []),
      h("span", { class: "brace", "aria-hidden": "true", text: "}" }),
    );
    set.setAttribute("aria-label", `Answer set with ${members.length} of ${data.E} entities`);

    const covered = members.includes(target);
    const level = percent(1 - alpha, 0);
    $("#hero-verdict").replaceChildren(
      UI.icon(covered ? "check" : "cross"),
      h(
        "span",
        {},
        covered ? "Contains the answer that was held out from the model, " : "Misses the answer that was held out from the model, ",
        h("b", { text: names[target] }),
        covered ? "." : `. At a ${level} target, that is allowed for ${integer(Math.round(alpha * 100))} in 100 queries.`,
      ),
    );

    const sums = KGCP.summary(KGCP.evaluate(model, test), data.relation, alpha);
    const cut = threshold === Infinity ? "every candidate" : `candidates with a probability of at least ${percent(Math.max(0, 1 - threshold), 1)}`;
    $("#hero-caption").textContent =
      `Calibrated on ${integer(cal.length)} held-out queries, the cut keeps ${cut}. ` +
      `Over the ${integer(test.length)} other test queries, ${percent(sums.coverage)} of the sets contain the answer, ` +
      `with ${number(sums.size, 1)} of ${data.E} entities on average.`;
  }

  render();
})();
