"""Download and preprocess knowledge graph datasets into libkge's data folder.

    python scripts/download_data.py nations umls kinship

Mirrors third_party/libkge/data/download_all.sh, but fetches only the requested
datasets. Already prepared datasets are skipped.
"""

import argparse
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

LIBKGE_DATA = Path(__file__).resolve().parents[1] / "third_party" / "libkge" / "data"
MIRROR = "https://web.informatik.uni-mannheim.de/pi1/kge-datasets"
CODEX = "https://zenodo.org/record/4281094/files"

# name -> (archive url, raw file name of each split, extra preprocessing arguments)
DATASETS = {
    "toy": (f"{MIRROR}/toy.tar.gz", "{split}.txt", []),
    "nations": (f"{MIRROR}/nations.tar.gz", "{split}.txt", []),
    "umls": (f"{MIRROR}/umls.tar.gz", "{split}.txt", []),
    "kinship": (f"{MIRROR}/kinship.tar.gz", "{split}.txt", []),
    "fb15k": (f"{MIRROR}/fb15k.tar.gz", "freebase_mtr100_mte100-{split}.txt", []),
    "fb15k-237": (f"{MIRROR}/fb15k-237.tar.gz", "{split}.txt", []),
    "wn18": (f"{MIRROR}/wn18.tar.gz", "wordnet-mlj12-{split}.txt", []),
    "wnrr": (f"{MIRROR}/wnrr.tar.gz", "{split}.txt", []),
    "yago3-10": (f"{MIRROR}/yago3-10.tar.gz", "{split}.txt", []),
    "db100k": (f"{MIRROR}/db100k.tar.gz", "_{split}.txt", []),
    "dbpedia50": (f"{MIRROR}/dbpedia50.tar.gz", "{split}.txt", []),
    "dbpedia500": (f"{MIRROR}/dbpedia500.tar.gz", "{split}.txt", ["-S", "0", "-P", "2", "-O", "1"]),
    "wikidata5m": (f"{MIRROR}/wikidata5m.tar.gz", "{split}.txt", []),
    "codex-s": (f"{CODEX}/codex-s.tar.gz?download=1", "{split}.txt", []),
    "codex-m": (f"{CODEX}/codex-m.tar.gz?download=1", "{split}.txt", []),
    "codex-l": (f"{CODEX}/codex-l.tar.gz?download=1", "{split}.txt", []),
}


def download(name: str) -> None:
    folder = LIBKGE_DATA / name
    if (folder / "dataset.yaml").is_file():
        print(f"{name}: already prepared in {folder}")
        return
    url, split_file, preprocess_args = DATASETS[name]

    print(f"{name}: downloading {url}")
    with tempfile.TemporaryFile() as archive:
        with urllib.request.urlopen(url) as response:
            archive.write(response.read())
        archive.seek(0)
        with tarfile.open(fileobj=archive) as tar:
            tar.extractall(LIBKGE_DATA, filter="data")

    # libkge's preprocessing expects the raw splits as {train,valid,test}.txt
    for split in ("train", "valid", "test"):
        raw = folder / split_file.format(split=split)
        if raw.name != f"{split}.txt":
            raw.rename(folder / f"{split}.txt")

    print(f"{name}: preprocessing")
    subprocess.run(
        [sys.executable, "preprocess/preprocess_default.py", name, *preprocess_args],
        cwd=LIBKGE_DATA,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(f"{name}: ready in {folder}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS))
    for name in parser.parse_args().datasets:
        download(name)
