/* The conformal prediction of kgcp, in JavaScript, for the demo page.

   The page runs no model. It loads what the models predicted (data/*.js, written by
   scripts/export_demo.py) and repeats here what src/kgcp does with such predictions:
   scores.py (nonconformity scores, set sizes), conformal.py (split and Mondrian CP,
   CondKGCP), groups.py (MergedRelations), intervals.py and metrics.py. `verify` checks
   the result against metrics that kgcp itself computed from the same numbers. */
"use strict";

const KGCP = (() => {
  // ---------------------------------------------------------------- decoding

  function typed(base64, Type) {
    const text = atob(base64);
    const bytes = new Uint8Array(text.length);
    for (let i = 0; i < text.length; i++) bytes[i] = text.charCodeAt(i);
    return new Type(bytes.buffer, 0, bytes.length / Type.BYTES_PER_ELEMENT);
  }

  /* 16-bit floats, which not every browser reads natively */
  function halfFloats(base64) {
    const half = typed(base64, Uint16Array);
    const out = new Float32Array(half.length);
    for (let i = 0; i < half.length; i++) {
      const x = half[i];
      const exponent = (x >> 10) & 0x1f;
      const fraction = x & 0x3ff;
      let value;
      if (exponent === 0) value = fraction * 2 ** -24;
      else if (exponent === 31) value = fraction ? NaN : Infinity;
      else value = (1 + fraction / 1024) * 2 ** (exponent - 15);
      out[i] = x & 0x8000 ? -value : value;
    }
    return out;
  }

  /* A small seeded generator (mulberry32): the same splits on every visit. */
  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* Random split of n rows: `fraction` of them calibrate, the others are tested. */
  function split(n, seed, fraction = 0.5) {
    const random = rng(seed);
    const order = Int32Array.from({ length: n }, (_, i) => i);
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const cut = Math.round(fraction * n);
    return { cal: order.slice(0, cut).sort(), test: order.slice(cut).sort() };
  }

  /* The split of the reference results: even rows calibrate, odd rows are tested. */
  function evenOdd(n) {
    const cal = [];
    const test = [];
    for (let i = 0; i < n; i++) (i % 2 === 0 ? cal : test).push(i);
    return { cal: Int32Array.from(cal), test: Int32Array.from(test) };
  }

  // ------------------------------------------------- answer sets: the scores

  function loadSets(raw) {
    const anchor = typed(raw.anchor, Int16Array);
    return {
      raw,
      N: anchor.length, // queries: the tail queries of the test triples, then the head queries
      E: raw.entities.length,
      R: raw.relations.length,
      anchor,
      relation: typed(raw.relation, Int16Array),
      target: typed(raw.target, Int16Array),
      direction: typed(raw.direction, Uint8Array), // 0: (anchor, relation, ?), 1: (?, relation, anchor)
      logits: halfFloats(raw.logits), // [N, E]; -Infinity for filtered known answers
      distances: typed(raw.distances, Float32Array), // [R, R]
    };
  }

  /* Nonconformity of every candidate (scores.py): lower = more plausible, +Infinity for
     filtered candidates. `u` is APS's random number of a query (1 if not randomized). */
  const SCORE_FUNCTIONS = {
    neg(x, out) {
      for (let e = 0; e < x.length; e++) out[e] = -x[e];
    },
    minmax(x, out) {
      let low = Infinity;
      let high = -Infinity;
      for (const v of x) {
        if (v > high) high = v;
        if (v < low && v !== -Infinity) low = v;
      }
      const range = Math.max(high - low, 1e-12);
      for (let e = 0; e < x.length; e++) out[e] = (high - x[e]) / range;
    },
    softmax(x, out) {
      softmax(x, out);
      for (let e = 0; e < x.length; e++) out[e] = 1 - out[e];
    },
    aps(x, out, u) {
      adaptive(x, out, u, 0, 0);
    },
    raps(x, out, u) {
      adaptive(x, out, u, 0.01, 5);
    },
    rank(x, out) {
      const order = descending(x);
      for (let position = 0; position < order.length; position++) out[order[position]] = position + 1;
    },
  };

  function softmax(x, out) {
    let high = -Infinity;
    for (const v of x) if (v > high) high = v;
    let sum = 0;
    for (let e = 0; e < x.length; e++) sum += out[e] = Math.exp(x[e] - high);
    for (let e = 0; e < x.length; e++) out[e] /= sum;
  }

  /* Indices by decreasing value; equal values keep their order. */
  function descending(x) {
    return Array.from(x.keys()).sort((a, b) => x[b] - x[a] || a - b);
  }

  /* APS: the softmax mass of the candidates ranked above, plus u times the own mass;
     RAPS adds `lam` per rank beyond `kReg`. */
  function adaptive(x, out, u, lam, kReg) {
    const probs = new Float64Array(x.length);
    softmax(x, probs);
    const order = descending(probs);
    let cumulative = 0;
    for (let position = 0; position < order.length; position++) {
      const p = probs[order[position]];
      cumulative += p;
      out[order[position]] = cumulative - (1 - u) * p + lam * Math.max(position + 1 - kReg, 0);
    }
  }

  function scores(data, name, { randomized = true, seed = 0 } = {}) {
    const { N, E, logits, target } = data;
    const values = new Float32Array(N * E);
    const sorted = new Float32Array(N * E); // each query's scores in increasing order
    const truth = new Float32Array(N); // the score of each query's target
    const finite = new Int32Array(N); // candidates left after filtering
    const random = rng(seed);
    const row = new Float64Array(E);
    for (let i = 0; i < N; i++) {
      const x = logits.subarray(i * E, (i + 1) * E);
      const u = randomized && (name === "aps" || name === "raps") ? random() : 1;
      SCORE_FUNCTIONS[name](x, row, u);
      const out = values.subarray(i * E, (i + 1) * E);
      for (let e = 0; e < E; e++) {
        const filtered = x[e] === -Infinity;
        out[e] = filtered ? Infinity : row[e];
        if (!filtered) finite[i]++;
      }
      sorted.set(out, i * E);
      sorted.subarray(i * E, (i + 1) * E).sort();
      truth[i] = out[target[i]];
    }
    return { data, name, N, E, values, sorted, truth, finite };
  }

  /* Candidates of query i that score at most the threshold (or below it, if strict):
     a binary search in the sorted row, among the finite scores. */
  function count(S, i, threshold, strict = false) {
    const offset = i * S.E;
    let low = 0;
    let high = S.finite[i];
    while (low < high) {
      const middle = (low + high) >> 1;
      const value = S.sorted[offset + middle];
      if (strict ? value < threshold : value <= threshold) low = middle + 1;
      else high = middle;
    }
    return low;
  }

  // ------------------------------------------------------------ calibration

  /* conformal.py: the ceil((n + 1)(1 - alpha))-th smallest of n scores, +Infinity if
     that exceeds n. A new exchangeable score is at most it with probability >= 1 - alpha. */
  function conformalQuantile(values, alpha) {
    const n = values.length;
    const k = Math.ceil((n + 1) * (1 - alpha) - 1e-9);
    if (k > n) return Infinity;
    return Float64Array.from(values).sort()[k - 1];
  }

  /* groups.py, MergedRelations: a relation with fewer than minSize calibration queries
     joins the nearest relation (L1 distance of the embeddings) that has enough. */
  function mergedRelations(data, calibrationRelations, minSize) {
    const { R, distances } = data;
    const counts = new Int32Array(R);
    for (const r of calibrationRelations) counts[r]++;
    const large = [];
    for (let r = 0; r < R; r++) if (counts[r] >= minSize) large.push(r);
    const table = new Int32Array(R);
    if (large.length === 0) return { table: table.fill(counts.indexOf(Math.max(...counts))), counts };
    for (let r = 0; r < R; r++) {
      if (counts[r] >= minSize) {
        table[r] = r;
        continue;
      }
      let nearest = large[0];
      for (const other of large) {
        if (distances[r * R + other] < distances[r * R + nearest]) nearest = other;
      }
      table[r] = nearest;
    }
    return { table, counts };
  }

  /* The predictors of conformal.py. options:
       method   "split" (one threshold), "mondrian" (one per group), "condkgcp"
                (per group, with rank calibration)
       groups   "relation", "direction" or "merged" (MergedRelations; needs minSize)
       alpha, gamma (CondKGCP)                                                    */
  function calibrate(S, rows, options) {
    const { data } = S;
    const { method, alpha, gamma = 0.01 } = options;
    const groups = method === "split" ? "marginal" : options.groups;
    let merged = null;
    if (groups === "merged") {
      merged = mergedRelations(data, Array.from(rows, (i) => data.relation[i]), options.minSize);
    }
    const groupOf = (i) => {
      if (groups === "relation") return data.relation[i];
      if (groups === "direction") return data.direction[i];
      if (groups === "merged") return merged.table[data.relation[i]];
      return 0;
    };

    const members = new Map(); // group -> its calibration queries
    for (const i of rows) {
      const group = groupOf(i);
      if (!members.has(group)) members.set(group, []);
      members.get(group).push(i);
    }
    const thresholds = new Map();
    const topK = new Map();
    const miscoverage = new Map();
    for (const [group, queries] of members) {
      const truth = queries.map((i) => S.truth[i]);
      if (method !== "condkgcp") {
        thresholds.set(group, conformalQuantile(truth, alpha));
        continue;
      }
      // k(g): the smallest k such that fewer than a fraction alpha of the targets rank
      // beyond it; eps(g): that fraction. Ranks count ties against the target.
      const n = queries.length;
      const ranks = queries.map((i) => count(S, i, S.truth[i]));
      const position = Math.min(Math.floor(n * (1 - alpha) + 1e-9) + 1, n);
      const k = ranks.slice().sort((a, b) => a - b)[position - 1];
      const eps = ranks.filter((rank) => rank > k).length / n;
      topK.set(group, k);
      miscoverage.set(group, eps);
      thresholds.set(group, conformalQuantile(truth, alpha - gamma * eps));
    }
    return { S, options, groups, groupOf, merged, members, thresholds, topK, miscoverage };
  }

  /* The answer set of query i: its score threshold, the score that caps it at the top
     k(g) candidates (CondKGCP), its size, and whether it contains the target. */
  function predict(model, i) {
    const { S } = model;
    const group = model.groupOf(i);
    const threshold = model.thresholds.has(group) ? model.thresholds.get(group) : Infinity;
    let size = count(S, i, threshold);
    let cap = Infinity; // candidates must score below it
    if (model.topK.has(group) && model.topK.get(group) < S.E) {
      cap = S.sorted[i * S.E + model.topK.get(group)];
      size = Math.min(size, count(S, i, cap, true));
    }
    const truth = S.truth[i];
    return { group, threshold, cap, size, covered: truth <= threshold && truth < cap };
  }

  /* The entities of the answer set of query i, most plausible first. */
  function members(model, i) {
    const { S } = model;
    const { threshold, cap } = predict(model, i);
    const out = [];
    for (let e = 0; e < S.E; e++) {
      const value = S.values[i * S.E + e];
      if (value <= threshold && value < cap) out.push(e);
    }
    return out.sort((a, b) => S.values[i * S.E + a] - S.values[i * S.E + b] || a - b);
  }

  // ---------------------------------------------------------------- metrics

  /* metrics.py: sums over evaluated queries, which may be pooled over several splits. */
  function accumulator(units, R, largest = 0) {
    return {
      sizes: new Float64Array(largest + 1), // answer sets: how many have each size
      n: 0,
      covered: 0,
      size: 0,
      empty: 0,
      singleton: 0,
      full: 0,
      side: [0, 0, 0, 0, 0, 0], // per direction: queries, covered, size
      relation: { n: new Float64Array(R), covered: new Float64Array(R), size: new Float64Array(R) },
      seen: new Uint8Array(units),
    };
  }

  function add(sums, unit, relation, covered, size, direction = 0) {
    sums.n++;
    sums.covered += covered;
    sums.size += size;
    sums.empty += size === 0;
    sums.singleton += size === 1;
    sums.full += size >= 1; // intervals: the whole of [0, 1]
    if (size < sums.sizes.length && Number.isInteger(size)) sums.sizes[size]++;
    sums.side[3 * direction]++;
    sums.side[3 * direction + 1] += covered;
    sums.side[3 * direction + 2] += size;
    sums.relation.n[relation]++;
    sums.relation.covered[relation] += covered;
    sums.relation.size[relation] += size;
    sums.seen[unit] = 1;
  }

  /* Coverage per relation, over the relations with at least `minimum` distinct queries. */
  function perRelation(sums, relationOf, alpha, minimum = 10) {
    const R = sums.relation.n.length;
    const distinct = new Int32Array(R);
    sums.seen.forEach((seen, unit) => (distinct[relationOf[unit]] += seen));
    const relations = [];
    for (let r = 0; r < R; r++) {
      if (distinct[r] < minimum) continue;
      const coverage = sums.relation.covered[r] / sums.relation.n[r];
      relations.push({ relation: r, queries: distinct[r], coverage, size: sums.relation.size[r] / sums.relation.n[r] });
    }
    const gaps = relations.map((x) => 1 - alpha - x.coverage);
    const under = gaps.filter((gap) => gap > 1e-12);
    const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
    return {
      relations,
      covgap: mean(gaps.map(Math.abs)),
      ucg: under.length ? mean(under) : 0,
      ucr: under.length / gaps.length,
      condSize: mean(relations.map((x) => x.size)),
    };
  }

  function evaluate(model, rows, sums = null) {
    const { data } = model.S;
    sums = sums || accumulator(data.N, data.R, data.E);
    for (const i of rows) {
      const { covered, size } = predict(model, i);
      add(sums, i, data.relation[i], covered, size, data.direction[i]);
    }
    return sums;
  }

  function summary(sums, relationOf, alpha) {
    const side = sums.side;
    return {
      n: sums.n,
      coverage: sums.covered / sums.n,
      size: sums.size / sums.n,
      empty: sums.empty / sums.n,
      singleton: sums.singleton / sums.n,
      full: sums.full / sums.n,
      sizes: Array.from(sums.sizes, (count) => count / sums.n),
      coverageTail: side[1] / side[0],
      sizeTail: side[2] / side[0],
      coverageHead: side[4] / side[3],
      sizeHead: side[5] / side[3],
      ...perRelation(sums, relationOf, alpha),
      everyRelation: perRelation(sums, relationOf, alpha, 1).relations, // also the rare ones
    };
  }

  /* Calibrate on random halves and evaluate on the other halves, pooled as in kgcp's
     experiments. Returns the summary and the coverage of each split. */
  function repeat(S, options, splits, seed = 0) {
    const sums = accumulator(S.N, S.data.R, S.E);
    const coverages = [];
    for (let s = 0; s < splits; s++) {
      const { cal, test } = split(S.N, seed + s);
      const before = sums.covered;
      evaluate(calibrate(S, cal, options), test, sums);
      coverages.push((sums.covered - before) / test.length);
    }
    return { ...summary(sums, S.data.relation, options.alpha), coverages };
  }

  // -------------------------------------------------------------- intervals

  function loadIntervals(raw) {
    const predicted = typed(raw.predicted, Float32Array);
    return {
      raw,
      N: predicted.length,
      R: raw.relations.length,
      predicted, // M: the model's confidence of each triple
      confidence: typed(raw.confidence, Float32Array), // c: its true confidence
      relation: typed(raw.relation, Int16Array),
    };
  }

  /* intervals.py: the nonconformity |c - M| / s(M) of each triple's confidence, with
     s = 1 ("absolute": plain CP) or the binary entropy H(M) ("entropy": UnKGCP). */
  function residuals(data, name, eps = 1e-6) {
    const { N, predicted, confidence } = data;
    const scale = new Float64Array(N).fill(1);
    const truth = new Float64Array(N);
    for (let i = 0; i < N; i++) {
      if (name === "entropy") scale[i] = entropy(predicted[i], eps);
      truth[i] = Math.abs(confidence[i] - predicted[i]) / scale[i];
    }
    return { data, name, N, scale, truth };
  }

  function entropy(p, eps = 1e-6) {
    p = Math.min(Math.max(p, eps), 1 - eps);
    return -p * Math.log(p) - (1 - p) * Math.log(1 - p);
  }

  /* options: method "split" or "mondrian" (a threshold per relation), alpha */
  function calibrateIntervals(S, rows, options) {
    const groupOf = (i) => (options.method === "mondrian" ? S.data.relation[i] : 0);
    const groups = new Map();
    for (const i of rows) {
      const group = groupOf(i);
      if (!groups.has(group)) groups.set(group, []);
      groups.get(group).push(S.truth[i]);
    }
    const thresholds = new Map();
    for (const [group, truth] of groups) thresholds.set(group, conformalQuantile(truth, options.alpha));
    return { S, options, groupOf, thresholds };
  }

  /* The interval M ± t s(M) of triple i, within [0, 1]. */
  function interval(model, i) {
    const { S } = model;
    const group = model.groupOf(i);
    const threshold = model.thresholds.has(group) ? model.thresholds.get(group) : Infinity;
    const center = S.data.predicted[i];
    const lower = Math.min(Math.max(center - threshold * S.scale[i], 0), 1);
    const upper = Math.min(Math.max(center + threshold * S.scale[i], 0), 1);
    return { threshold, lower, upper, width: upper - lower, covered: S.truth[i] <= threshold };
  }

  function evaluateIntervals(model, rows, sums = null) {
    const { data } = model.S;
    sums = sums || accumulator(data.N, data.R);
    for (const i of rows) {
      const { covered, width } = interval(model, i);
      add(sums, i, data.relation[i], covered, width);
    }
    return sums;
  }

  function repeatIntervals(S, options, splits, seed = 0) {
    const sums = accumulator(S.N, S.data.R);
    for (let s = 0; s < splits; s++) {
      const { cal, test } = split(S.N, seed + s);
      evaluateIntervals(calibrateIntervals(S, cal, options), test, sums);
    }
    return summary(sums, S.data.relation, options.alpha);
  }

  // ----------------------------------------------------------- verification

  const REFERENCE_METHODS = {
    split: { method: "split" },
    "mondrian-relation": { method: "mondrian", groups: "relation" },
    "mondrian-direction": { method: "mondrian", groups: "direction" },
    "mondrian-merged": { method: "mondrian", groups: "merged" },
    "condkgcp-merged": { method: "condkgcp", groups: "merged" },
    "condkgcp-relation": { method: "condkgcp", groups: "relation", gamma: 1 },
  };
  // the names of kgcp's metrics, and ours
  const SET_METRICS = {
    coverage: "coverage", size: "size", empty: "empty", singleton: "singleton",
    coverage_tail: "coverageTail", size_tail: "sizeTail", coverage_head: "coverageHead",
    size_head: "sizeHead", covgap: "covgap", ucg: "ucg", ucr: "ucr", cond_size: "condSize",
  };
  const INTERVAL_METRICS = {
    coverage: "coverage", width: "size", full: "full", covgap: "covgap", ucg: "ucg",
    ucr: "ucr", cond_width: "condSize",
  };

  /* Recompute the reference results of a data file: the largest difference from kgcp's
     value of each metric, relative to the value's size. */
  function verify(raw) {
    const cases = [];
    const sets = raw.kind === "sets";
    const data = sets ? loadSets(raw) : loadIntervals(raw);
    const { cal, test } = evenOdd(data.N);
    const cache = new Map();
    for (const expected of raw.reference) {
      if (!cache.has(expected.score)) {
        cache.set(expected.score, sets ? scores(data, expected.score, { randomized: false }) : residuals(data, expected.score));
      }
      const S = cache.get(expected.score);
      let ours;
      if (sets) {
        const options = {
          gamma: raw.referenceSettings.gamma,
          minSize: raw.referenceSettings.minSize,
          ...REFERENCE_METHODS[expected.method],
          alpha: expected.alpha,
        };
        ours = summary(evaluate(calibrate(S, cal, options), test), data.relation, expected.alpha);
      } else {
        const options = { method: expected.method === "split" ? "split" : "mondrian", alpha: expected.alpha };
        ours = summary(evaluateIntervals(calibrateIntervals(S, cal, options), test), data.relation, expected.alpha);
      }
      let worst = 0;
      let metric = "";
      for (const [theirs, name] of Object.entries(sets ? SET_METRICS : INTERVAL_METRICS)) {
        const [a, b] = [ours[name], expected[theirs]];
        const same = Number.isNaN(a) && Number.isNaN(b); // no relation with enough queries
        const difference = same ? 0 : Math.abs(a - b) / Math.max(1, Math.abs(b));
        if (!(difference <= worst)) [worst, metric] = [difference, theirs];
      }
      const relations = ours.relations.length === expected.relations;
      cases.push({ score: expected.score, method: expected.method, alpha: expected.alpha, worst, metric, relations });
    }
    return cases;
  }

  return {
    rng, split, evenOdd, conformalQuantile, entropy,
    loadSets, scores, count, calibrate, predict, members, evaluate, summary, repeat, mergedRelations,
    loadIntervals, residuals, calibrateIntervals, interval, evaluateIntervals, repeatIntervals,
    verify,
  };
})();
