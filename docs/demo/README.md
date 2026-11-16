# The KGCP demo page

An interactive page that shows how the three methods of `kgcp` work, on the outputs of
real models: answer sets (split conformal prediction), guarantees per relation
(Mondrian CP and CondKGCP) and intervals for uncertain knowledge graphs (UnKGCP).

Open `index.html` in a browser: the page needs no server, no build step and no network.
It loads no fonts, scripts or styles from other sites.

## What it computes

No model runs in the browser. The page loads what a model predicted for its test queries
and repeats the calibration of `kgcp` in JavaScript, so that every control (the coverage
target, the score, the predictor, the calibration split) recomputes the result at once.

| File | Content |
|---|---|
| `index.html`, `style.css` | the page |
| `engine.js` | the conformal prediction of `src/kgcp` in JavaScript: scores, split and Mondrian CP, CondKGCP, `MergedRelations`, intervals, metrics |
| `ui.js` | drawing, animation, tooltips, table views, loading of the model files |
| `hero.js`, `how.js`, `playground.js`, `conditional.js`, `intervals.js`, `use.js` | one file per section of the page |
| `data/*.js` | the models' outputs, written by `scripts/export_demo.py` |

## The models

`uv run scripts/export_demo.py` writes `data/`, in a few seconds, from the trained models
in `models/` (see the [README](../../README.md#trained-models)) and `models/unkgcp/` (see
[third_party/unkge](../../third_party/unkge/README.md)):

- **answer sets:** seed 1 of the tuned DistMult, ComplEx and RESCAL on Nations and UMLS,
  and of ComplEx on Kinship. For every test query, the logits of all candidates as 16-bit
  floats, with the known answers of train and valid filtered out; and the L1 distances of
  the relation embeddings.
- **intervals:** seed 0 of UKGE, PASSLEAF and BEUrRE on NL27k, CN15k and PPI5k. The
  predicted and the true confidence of 6,000 of the test triples.
- `index.js` lists the files and holds the benchmark results that the page shows as they
  are in `results/conformal/libkge/` (CondKGCP against split and Mondrian CP on WN18 and
  FB15k).

## Checked against kgcp

Each model file holds metrics that `kgcp` itself computed from the exported numbers, for
fixed calibration splits (even rows calibrate, odd rows are tested), several scores,
predictors and coverage targets: 402 results in all. The page recomputes them with
`engine.js` and reports the outcome at its end ("Check all model files"):

- 376 results are identical to six digits: all those of the Neg, MinMax, Softmax and
  Rank scores, and all intervals.
- 26 results of the APS and RAPS scores differ by up to 5%. These scores add up
  probabilities, which `kgcp` does in 32-bit floats and JavaScript in 64-bit floats, so
  candidates that tie near a sum of 1 can fall on the other side of a threshold. With
  the sums computed in 64-bit floats on the Python side, all of them are identical too.

`tests/test_demo.py` checks that the files in `data/` are what the export script writes
for the current code, so a change of `kgcp` that changes results is noticed here.

## Publishing it

The page is static. To serve it with GitHub Pages, choose "Deploy from a branch", the
branch `main` and the folder `/docs` in the repository's Pages settings: the page is then
at `https://<user>.github.io/<repository>/demo/` (`docs/index.html` forwards there, and
`docs/.nojekyll` serves the files as they are).

## Data

The page shows names of entities and relations and test triples of the datasets: the
test queries of Nations, UMLS and Kinship (Kok and Domingos, 2007), and 400 test triples
each of NL27k (NELL), CN15k (ConceptNet, CC BY-SA 4.0) and PPI5k (STRING, CC BY 4.0), as
released with UKGE (Chen et al., 2019).
