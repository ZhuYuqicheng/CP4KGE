/* The code examples, and the check of this page's JavaScript against kgcp. */
"use strict";

(() => {
  const { $, $$, h, number, integer } = UI;
  const REPOSITORY = "https://github.com/ZhuYuqicheng/CP4KGE";
  const CODE = {
    pykeen: [
      `# pip install "kgcp[pykeen] @ git+${REPOSITORY}"`,
      "import kgcp",
      "from pykeen.datasets import Nations",
      "from pykeen.pipeline import pipeline",
      "",
      "dataset = Nations()",
      'result = pipeline(dataset=dataset, model="DistMult", training_kwargs={"num_epochs": 200})',
      "",
      "model = kgcp.PyKEEN(result.model, dataset)",
      "kg = model.kg",
      'queries = kgcp.Queries.from_triples(kg.split("test"))',
      'scores = kgcp.nonconformity(model, queries, kgcp.Softmax(), known=kg.known("train", "valid"))',
      "",
      "cal, test = scores.split(0.5)  # calibrate on half of the test queries",
      "sets = kgcp.SplitConformal(alpha=0.1).calibrate(cal).predict(test)",
      "kgcp.evaluate(sets, alpha=0.1)  # coverage, set size, coverage per relation",
    ],
    libkge: [
      "# in a clone of the repository: uv sync",
      "import kgcp",
      "",
      'model = kgcp.LibKGE("models/umls-distmult/seed1.pt")',
      "kg = model.kg",
      'queries = kgcp.Queries.from_triples(kg.split("test"))',
      'scores = kgcp.nonconformity(model, queries, kgcp.Softmax(), known=kg.known("train", "valid"))',
      "cal, test = scores.split(0.5)",
      "",
      "# CondKGCP: relations merged by the similarity of their embeddings, and rank calibration",
      "groups = kgcp.MergedRelations(model.relation_embeddings(), min_size=20)",
      "sets = kgcp.CondKGCP(alpha=0.1, groups=groups, gamma=1.0).calibrate(cal).predict(test)",
      "kgcp.evaluate(sets, alpha=0.1)",
    ],
    logits: [
      `# pip install "kgcp @ git+${REPOSITORY}"  (needs only PyTorch)`,
      "import kgcp",
      "",
      "# test and train: [m, 3] id triples (head, relation, tail), of E entities and R relations",
      "queries = kgcp.Queries.from_triples(test)  # the tail queries, then the head queries",
      "logits = ...  # [2m, E]: your model's scores of all entities for each of the queries",
      "known = kgcp.KnownAnswers(train, num_entities=E, num_relations=R)",
      "scores = kgcp.nonconformity(logits, queries, kgcp.Softmax(), known=known)",
      "",
      "cal, test = scores.split(0.5)",
      "sets = kgcp.MondrianConformal(alpha=0.1, groups=kgcp.by_relation).calibrate(cal).predict(test)",
      "sets.mask  # [queries, entities] bool",
    ],
    intervals: [
      "# in a clone of the repository, with the models of third_party/unkge",
      "import kgcp",
      "",
      'model = kgcp.LibKGE("models/unkgcp/nl27k-beurre/seed0.pt")',
      'triples = model.kg.uncertain_triples("test")  # (h, r, t) and their confidence c',
      "",
      "# nonconformity |c - M| / H(M) of the predicted confidence M: UnKGCP",
      "residuals = kgcp.residuals(model, triples, kgcp.EntropyResidual())",
      "cal, test = residuals.split(0.5)",
      "intervals = kgcp.SplitConformal(alpha=0.1).calibrate(cal).predict(test)",
      "intervals.lower, intervals.upper  # M ± q H(M), within [0, 1]",
      "kgcp.evaluate_intervals(intervals, alpha=0.1)",
    ],
  };

  const code = $("#use-code");
  let shown = "pykeen";
  function show(tab) {
    shown = tab;
    for (const button of $$("#use-tabs button[data-tab]")) button.setAttribute("aria-pressed", String(button.dataset.tab === tab));
    // comments (a leading "#", or after two spaces) are set off from the code
    code.replaceChildren(
      ...CODE[tab].flatMap((line) => {
        const at = line.startsWith("#") ? 0 : line.indexOf("  # ");
        if (at < 0) return [line, "\n"];
        return [line.slice(0, at), h("span", { class: "c", text: line.slice(at) }), "\n"];
      }),
    );
  }
  for (const button of $$("#use-tabs button[data-tab]")) button.addEventListener("click", () => show(button.dataset.tab));
  $("#use-copy").addEventListener("click", async () => {
    const button = $("#use-copy");
    try {
      await navigator.clipboard.writeText(CODE[shown].join("\n"));
      button.textContent = "Copied";
    } catch {
      const range = document.createRange(); // no clipboard access: select the code instead
      range.selectNodeContents(code);
      const selection = window.getSelection();
      selection.empty();
      selection.addRange(range);
      button.textContent = "Selected: copy with Ctrl+C";
    }
    window.setTimeout(() => (button.textContent = "Copy"), 2200);
  });
  show(shown);

  // ------------------------------------------------------------ the check
  /* A result is identical if every metric agrees with kgcp to six digits. APS and RAPS
     add up probabilities, in 32-bit floats in kgcp and in 64-bit floats here, so that
     candidates tied near a sum of 1 can fall on the other side of a threshold. */
  function grade(raw) {
    const cases = KGCP.verify(raw);
    const exact = (c) => c.worst <= 1e-6 && c.relations;
    const identical = cases.filter(exact).length;
    const close = cases.filter((c) => !exact(c) && c.worst <= 0.05 && ["aps", "raps"].includes(c.score)).length;
    const worst = cases.reduce((a, c) => Math.max(a, c.worst), 0);
    return { cases: cases.length, identical, close, different: cases.length - identical - close, worst };
  }
  function report(grades) {
    const total = (key) => grades.reduce((a, g) => a + g[key], 0);
    const ok = total("different") === 0;
    const close = total("close")
      ? `, ${integer(total("close"))} within 5% (the APS and RAPS scores, which add probabilities in 32-bit floats in kgcp and in 64-bit floats here)`
      : "";
    const end = ok ? `, in ${grades.length} model file${grades.length === 1 ? "" : "s"}.` : `; ${integer(total("different"))} differ.`;
    $("#verify-line").replaceChildren(
      UI.icon(ok ? "check" : "cross"),
      h("span", { text: `${integer(total("identical"))} of ${integer(total("cases"))} results of kgcp are reproduced to six digits${close}${end}` }),
    );
  }

  const preloaded = Object.entries(window.KGCP_DATA).filter(([key]) => key !== "index");
  const idle = window.requestIdleCallback || ((start) => window.setTimeout(start, 600));
  idle(() => report(preloaded.map(([, raw]) => grade(raw))));

  $("#verify-all").addEventListener("click", async () => {
    const button = $("#verify-all");
    button.disabled = true;
    const index = window.KGCP_DATA.index;
    const rows = [];
    const grades = [];
    for (const entry of [...index.sets, ...index.intervals]) {
      button.textContent = `Checking ${entry.key}…`;
      const result = grade(await UI.load(entry.key));
      grades.push(result);
      const worst = result.worst < 1e-6 ? "< 0.000001" : number(result.worst, 4);
      const name = `${UI.label(entry.dataset)} · ${UI.label(entry.model)}`;
      rows.push([name, String(result.cases), String(result.identical), String(result.close), String(result.different), worst]);
      await new Promise((resolve) => window.setTimeout(resolve, 0));
    }
    report(grades);
    const view = $("#verify-table");
    const columns = ["Model file", "Results", "Identical", "Within 5%", "Different", "Largest difference"];
    view.hidden = false;
    view.replaceChildren(
      h(
        "table",
        {},
        h("thead", {}, h("tr", {}, columns.map((c) => h("th", { scope: "col", text: c })))),
        h("tbody", {}, rows.map((row) => h("tr", {}, row.map((cell, column) => h("td", { class: column ? "num" : null, text: cell }))))),
      ),
    );
    button.textContent = "Checked all model files";
  });
})();
