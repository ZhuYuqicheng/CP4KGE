"""The demo page's model files (docs/demo/data/) are what scripts/export_demo.py writes.

The page repeats kgcp's calibration in JavaScript and checks itself against the metrics
of kgcp that the files hold. If a change of kgcp changes these metrics, the files must be
exported again: this test notices it.
"""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
DATA = ROOT / "docs" / "demo" / "data"


@pytest.fixture(scope="module")
def export():
    spec = importlib.util.spec_from_file_location("export_demo", ROOT / "scripts/export_demo.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stored(key: str) -> dict:
    """The value that docs/demo/data/<key>.js registers (a JavaScript file)."""
    text = (DATA / f"{key}.js").read_text().split("] = ", 1)[1].rstrip().removesuffix(";")
    return json.loads(re.sub(r"\bNaN\b", "null", text))


def same(ours, theirs) -> bool:
    """Equal, with NaN (no relation with enough queries) stored as null."""
    return json.loads(re.sub(r"\bNaN\b", "null", json.dumps(ours))) == theirs


@pytest.mark.parametrize("dataset, model", [("nations", "distmult"), ("umls", "complex")])
def test_answer_set_files_are_current(export, dataset, model):
    if not (ROOT / "models" / f"{dataset}-{model}" / "seed1.pt").exists():
        pytest.skip("no trained model (see README.md#trained-models)")
    assert same(export.export_sets(dataset, model), stored(f"sets-{dataset}-{model}"))


def test_interval_files_are_current(export):
    if not (ROOT / "models" / "unkgcp" / "ppi5k-beurre" / "seed0.pt").exists():
        pytest.skip("no converted model (see third_party/unkge/README.md)")
    assert same(export.export_intervals("ppi5k", "beurre"), stored("intervals-ppi5k-beurre"))


def test_index_lists_the_files(export):
    index = stored("index")
    listed = {entry["key"] for kind in ("sets", "intervals") for entry in index[kind]}
    assert listed == {file.stem for file in DATA.glob("*.js")} - {"index"}
    assert same(export.benchmark(), index["benchmark"])
