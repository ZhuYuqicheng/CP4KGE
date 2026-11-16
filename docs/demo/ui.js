/* Shared pieces of the demo page: DOM and SVG builders, scales, animation, the tooltip,
   table views, loading of the model files. */
"use strict";

const UI = (() => {
  const SVG = "http://www.w3.org/2000/svg";
  const motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const reduced = () => motion.matches;

  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

  function build(node, attributes, children) {
    for (const [name, value] of Object.entries(attributes || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (name === "text") node.textContent = value;
      else if (name.startsWith("on")) node.addEventListener(name.slice(2), value);
      else node.setAttribute(name, value === true ? "" : value);
    }
    for (const child of children.flat(Infinity)) {
      if (child === null || child === undefined || child === false) continue;
      node.append(child.nodeType ? child : document.createTextNode(String(child)));
    }
    return node;
  }
  /* h("p", {class: "note"}, "text", child): names from the data go in as text, never HTML */
  const h = (tag, attributes, ...children) => build(document.createElement(tag), attributes, children);
  const s = (tag, attributes, ...children) => build(document.createElementNS(SVG, tag), attributes, children);

  /* An SVG as wide as its container, redrawn when the width changes. draw(svg, width)
     returns the height. */
  function chart(container, draw, label) {
    const svg = s("svg", { role: "img", "aria-label": label });
    container.replaceChildren(svg);
    let width = 0;
    const render = () => {
      width = Math.max(container.clientWidth, 260);
      svg.replaceChildren();
      const height = draw(svg, width);
      svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
      svg.setAttribute("width", width);
      svg.setAttribute("height", height);
    };
    render();
    new ResizeObserver(() => {
      if (Math.abs(container.clientWidth - width) > 1) window.requestAnimationFrame(render);
    }).observe(container);
    return { svg, render };
  }

  // ---------------------------------------------------------------- numbers

  const percent = (x, digits = 1) => (Number.isFinite(x) ? `${(100 * x).toFixed(digits)}%` : "–");
  const number = (x, digits = 2) => {
    if (!Number.isFinite(x)) return x === Infinity ? "∞" : "–";
    return x.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
  };
  const integer = (x) => number(x, 0);
  /* 1st, 2nd, 3rd, 4th, 11th, 182nd */
  const ordinal = (n) => {
    const tens = n % 100;
    const suffix = tens >= 11 && tens <= 13 ? "th" : ["th", "st", "nd", "rd"][n % 10] || "th";
    return `${integer(n)}${suffix}`;
  };

  function linear(d0, d1, r0, r1) {
    const scale = (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
    scale.invert = (y) => d0 + ((y - r0) / (r1 - r0)) * (d1 - d0);
    return scale;
  }
  function logarithmic(d0, d1, r0, r1) {
    const inner = linear(Math.log(d0), Math.log(d1), r0, r1);
    return (x) => inner(Math.log(Math.max(x, d0)));
  }
  /* About `count` round tick values in [d0, d1]. */
  function ticks(d0, d1, count = 5) {
    const raw = (d1 - d0) / count;
    const power = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((v) => v >= raw);
    const out = [];
    for (let v = Math.ceil(d0 / step - 1e-9) * step; v <= d1 + 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }

  // ----------------------------------------------------------------- names

  /* "concept:athlete:kobe_bryant" -> "kobe bryant"; "disease_or_syndrome" -> "disease or syndrome" */
  function pretty(name) {
    return String(name).replace(/^concept:/, "").replace(/^[a-z]+:/, "").replace(/_/g, " ");
  }
  /* "(netherlands, exports, ?)" as nodes, with the unknown marked. */
  function query(data, i) {
    const anchor = h("b", { text: pretty(data.raw.entities[data.anchor[i]]) });
    const relation = pretty(data.raw.relations[data.relation[i]]);
    const slot = h("span", { class: "slot", text: "?" });
    return data.direction[i] === 0 ? ["(", anchor, ", ", relation, ", ", slot, ")"] : ["(", slot, ", ", relation, ", ", anchor, ")"];
  }
  function queryText(data, i) {
    const anchor = pretty(data.raw.entities[data.anchor[i]]);
    const relation = pretty(data.raw.relations[data.relation[i]]);
    return data.direction[i] === 0 ? `(${anchor}, ${relation}, ?)` : `(?, ${relation}, ${anchor})`;
  }

  // ------------------------------------------------------------- animation

  const ease = (t) => (t < 0.5 ? 4 * t * t * t : 1 - (-2 * t + 2) ** 3 / 2);
  const easeOut = (t) => 1 - (1 - t) ** 3;
  const lerp = (a, b, t) => a + (b - a) * t;

  /* Calls update(t) with t from 0 to 1 over `duration` ms. With reduced motion, or when
     `run` (an object with `cancelled`) is cancelled, it jumps to the end. */
  function tween(duration, update, { delay = 0, easing = ease, run = null } = {}) {
    return new Promise((resolve) => {
      if (reduced() || duration <= 0) {
        update(1);
        resolve();
        return;
      }
      const start = performance.now() + delay;
      const frame = (now) => {
        if (run && run.cancelled) {
          resolve();
          return;
        }
        const t = Math.min(Math.max((now - start) / duration, 0), 1);
        update(easing(t));
        if (t < 1) window.requestAnimationFrame(frame);
        else resolve();
      };
      window.requestAnimationFrame(frame);
    });
  }
  const sleep = (ms, run = null) =>
    new Promise((resolve) => window.setTimeout(resolve, reduced() || (run && run.cancelled) ? 0 : ms));

  /* Runs `start` the first time the node is mostly in view (at once without motion). */
  function onFirstView(node, start) {
    if (reduced() || !("IntersectionObserver" in window)) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          observer.disconnect();
          start();
        }
      },
      { threshold: 0.55 },
    );
    observer.observe(node);
  }

  // --------------------------------------------------------------- tooltip

  const tooltip = $("#tooltip");
  function showTip(event, ...content) {
    tooltip.replaceChildren(...content.flat().map((c) => (c.nodeType ? c : document.createTextNode(String(c)))));
    const box = event.target.getBoundingClientRect ? event.target.getBoundingClientRect() : null;
    const x = event.clientX ?? (box ? box.left + box.width / 2 : 0);
    const y = event.clientY ?? (box ? box.top : 0);
    tooltip.classList.add("on");
    const width = tooltip.offsetWidth;
    const height = tooltip.offsetHeight;
    tooltip.style.left = `${Math.min(Math.max(x - width / 2, 8), window.innerWidth - width - 8)}px`;
    tooltip.style.top = `${y - height - 12 < 8 ? y + 16 : y - height - 12}px`;
  }
  const hideTip = () => tooltip.classList.remove("on");
  /* The same readout on hover and on keyboard focus. content() returns nodes or text. */
  function tip(node, content, { focusable = false } = {}) {
    const show = (event) => showTip(event, content());
    node.addEventListener("pointerenter", show);
    node.addEventListener("pointermove", show);
    node.addEventListener("pointerleave", hideTip);
    if (focusable) {
      node.setAttribute("tabindex", "0");
      node.addEventListener("focus", show);
      node.addEventListener("blur", hideTip);
    }
    return node;
  }
  const tipLine = (value, label) => h("div", {}, h("b", { text: value }), " ", h("span", { class: "dim", text: label }));

  // ------------------------------------------------------------------ icons

  /* Covered and missed are a state: an icon and words, not color alone. */
  function icon(kind, size = 16) {
    const color = kind === "check" ? "var(--good)" : "var(--bad)";
    const path = kind === "check" ? "M4.2 8.4l2.5 2.6 5.1-5.6" : "M5 5l6 6M11 5l-6 6";
    return s(
      "svg",
      { width: size, height: size, viewBox: "0 0 16 16", "aria-hidden": "true" },
      s("circle", { cx: 8, cy: 8, r: 7.25, fill: "none", stroke: color, "stroke-width": 1.5 }),
      s("path", { d: path, fill: "none", stroke: color, "stroke-width": 1.7, "stroke-linecap": "round", "stroke-linejoin": "round" }),
    );
  }

  // ------------------------------------------------------------ table views

  /* Every chart has a table twin: a button with data-table="<id>" swaps the chart with
     id <id> for the table that `rows` returns: {columns: [...], rows: [[...], ...]}. */
  const tables = new Map();
  function table(id, rows) {
    tables.set(id, rows);
    const view = $(`#${id}-table`);
    if (view && !view.hidden) fillTable(view, rows());
  }
  function fillTable(view, { columns, rows }) {
    const numeric = (value) => typeof value === "number";
    view.replaceChildren(
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col", text: c })))),
        h(
          "tbody",
          {},
          rows.map((row) =>
            h("tr", {}, row.map((cell) => h("td", { class: numeric(cell) ? "num" : null, text: numeric(cell) ? number(cell, 3) : cell }))),
          ),
        ),
      ),
    );
  }
  for (const button of $$("button[data-table]")) {
    const id = button.dataset.table;
    const chartNode = $(`#${id}`);
    const view = h("div", { class: "table-view", id: `${id}-table`, hidden: true });
    chartNode.after(view);
    button.setAttribute("aria-pressed", "false");
    button.addEventListener("click", () => {
      const show = view.hidden;
      if (show && tables.has(id)) fillTable(view, tables.get(id)());
      view.hidden = !show;
      chartNode.hidden = show;
      button.setAttribute("aria-pressed", String(show));
      button.textContent = show ? "Chart" : "Table";
    });
  }

  // ---------------------------------------------------------------- loading

  const loading = new Map();
  /* The model file `key` (data/<key>.js), loaded once. */
  function load(key) {
    const have = window.KGCP_DATA && window.KGCP_DATA[key];
    if (have) return Promise.resolve(have);
    if (!loading.has(key)) {
      loading.set(
        key,
        new Promise((resolve, reject) => {
          const script = h("script", { src: `data/${key}.js` });
          script.onload = () => resolve(window.KGCP_DATA[key]);
          script.onerror = () => reject(new Error(`could not load data/${key}.js`));
          document.head.append(script);
        }),
      );
    }
    return loading.get(key);
  }
  const parsed = new Map();
  /* The parsed model file: KGCP.loadSets or KGCP.loadIntervals of it. */
  async function model(key) {
    if (!parsed.has(key)) {
      const raw = await load(key);
      parsed.set(key, raw.kind === "sets" ? KGCP.loadSets(raw) : KGCP.loadIntervals(raw));
    }
    return parsed.get(key);
  }

  const LABELS = {
    nations: "Nations", umls: "UMLS", kinship: "Kinship", nl27k: "NL27k (NELL)", cn15k: "CN15k (ConceptNet)",
    ppi5k: "PPI5k (STRING)", distmult: "DistMult", complex: "ComplEx", rescal: "RESCAL",
    ukge_logi: "UKGE", passleaf: "PASSLEAF", beurre: "BEUrRE", conve: "ConvE", rotate: "RotatE", transe: "TransE",
  };
  const label = (name) => LABELS[name] || name;

  /* A range input for the coverage target in percent: calls change(alpha). */
  function level(id, change) {
    const input = $(`#${id}`);
    const output = $(`#${id}-out`);
    const read = () => {
      output.textContent = `${input.value}%`;
      return 1 - Number(input.value) / 100;
    };
    input.addEventListener("input", () => change(read()));
    return read();
  }

  return {
    $, $$, h, s, chart, percent, number, integer, ordinal, linear, logarithmic, ticks, pretty, query, queryText,
    ease, easeOut, lerp, tween, sleep, onFirstView, reduced, tip, tipLine, showTip, hideTip, icon, table,
    load, model, label, level,
  };
})();
