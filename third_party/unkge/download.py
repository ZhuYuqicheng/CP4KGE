"""Downloads the data of UnKGCP's models: the uncertain KGs, and the authors' trained
models.

    python third_party/unkge/download.py data
    python third_party/unkge/download.py models --zip Models.zip --datasets ppi5k --seeds 0

data: CN15k, NL27k and PPI5k as released with UKGE (the authors' data archive has
truncated files, see README.md), checked against their git blob hashes, into LibKGE's
data folder; for CN15k and NL27k also the names of the entities and relations, from
BEUrRE's repository.

models: runs of the authors' Models.zip, a local file or a URL that serves it with range
requests (then only the selected runs are downloaded), except those already downloaded or
converted, into models/unkgcp/raw/:
<dataset>-<model>/seed<N>/ with the run's files and source.json (zip member, size and
sha256 of each file). Models: ukge_logi, ukge_rect, passleaf (PASSLEAF's UKGE_logi_m2),
beurre. Seeds are those of the run names; BEUrRE's CN15k and NL27k runs are named by
wandb ids instead, and get the seeds 0-9 in the order of their names. By default only
seeds 0-4 are downloaded: kgcp evaluates 5 models per model and dataset.
"""

import argparse
import hashlib
import http.client
import io
import json
import re
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/libkge/data"
RAW = ROOT / "models/unkgcp/raw"
CONVERTED = ROOT / "models/unkgcp"
DATASETS = ["cn15k", "nl27k", "ppi5k"]
MODELS = ["ukge_logi", "ukge_rect", "passleaf", "beurre"]
SEEDS = [0, 1, 2, 3, 4]  # the first five of the authors' ten

UKGE = "https://raw.githubusercontent.com/stasl0217/UKGE/5049b84501657abc44c36dd7d2db5f8519e5ee8f"
BEURRE = (
    "https://raw.githubusercontent.com/stasl0217/beurre/374b724008d1a215a0c07006528697bc7f6dcad4"
)
# dataset -> (url, file name) -> git blob sha1
FILES = {
    "cn15k": {
        (UKGE, "train.tsv"): "8f6ffded2c097ef2014cdbd387e10fcaf46ca9ff",
        (UKGE, "val.tsv"): "acc77e6f762142b9d4f569cc17a7bf7da68c8560",
        (UKGE, "test.tsv"): "27c6036baa2a535622c289bf2650be0ffbb5facf",
        (UKGE, "test_with_neg.tsv"): "e30803e69c07133c47769587939f5eb67eec4835",
        (UKGE, "softlogic.tsv"): "cc703b59ae3759548b8601f61fc744354c2ec7b3",
        (BEURRE, "concept_id.csv"): "5b8558c43128637678f554770546a5538f2ba343",
        (BEURRE, "relation_id.csv"): "9863a3ee68a3e9e296da0a38f927b1a82f3fcc2c",
    },
    "nl27k": {
        (UKGE, "train.tsv"): "56a3ba485a1f32363b6bcb91791c1fedd3fa10f0",
        (UKGE, "val.tsv"): "5a320a3da58aa530d0989b26d6a4e8135dbdeaaa",
        (UKGE, "test.tsv"): "56d444ca5b683e569a1d974e49b137fd8c19e5ac",
        (UKGE, "test_with_neg.tsv"): "c6a4e740e2852dd9e97c15a39974665a01173853",
        (UKGE, "softlogic.tsv"): "30357b65ef50e7273127c1629368eae092261df2",
        (BEURRE, "entity_id.csv"): "962b0285d8582159f99d02295190c8228cfe2e27",
        (BEURRE, "relation_id.csv"): "2a3f7baa3c4a39a65b55eaf9ca0537aefad0f3bc",
    },
    "ppi5k": {
        (UKGE, "train.tsv"): "18b88239bcfd133e53fd80872ffdf635340d3cba",
        (UKGE, "val.tsv"): "c4f64909001d9a737ab7c8a576e0d7410c4e7c24",
        (UKGE, "test.tsv"): "795d1cd3fad2148348dafc2a721e4e2afe5991d3",
        (UKGE, "test_with_neg.tsv"): "300dbc2b9162f6194d7fe3f2833e8707b0571d93",
        (UKGE, "softlogic.tsv"): "79d8cd083d5b50e8e51ad51d15c1043e0827468b",
    },
}

# zip member folders of the runs: (folder pattern, model)
RUNS = [
    (r"Models/UKGE/(\w+)/logi_\d+_(\d+)/", "ukge_logi"),
    (r"Models/UKGE/(\w+)/rect_\d+_(\d+)/", "ukge_rect"),
    (r"Models/PASSLEAF/(\w+)/UKGE_logi_m2_\d+_(\d+)/", "passleaf"),
    (r"Models/BeurRE/(\w+)/bigumbelbox-(\w+)\.pt", "beurre"),
]


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def download_data(datasets: list[str]) -> None:
    for dataset in datasets:
        folder = DATA / dataset
        folder.mkdir(parents=True, exist_ok=True)
        for (url, name), sha in FILES[dataset].items():
            file = folder / name
            if file.exists() and git_blob_sha1(file.read_bytes()) == sha:
                continue
            with urllib.request.urlopen(f"{url}/data/{dataset}/{name}", timeout=60) as r:
                data = r.read()
            if git_blob_sha1(data) != sha:
                raise ValueError(f"{dataset}/{name}: unexpected content")
            file.write_bytes(data)
            print(f"{file.relative_to(ROOT)}: {len(data):,} bytes")


class RemoteFile(io.RawIOBase):
    """A file served over HTTP with range requests, readable by zipfile."""

    def __init__(self, url: str):
        self.url, self.position = url, 0
        with urllib.request.urlopen(self._request(0, 0), timeout=120) as response:
            self.size = int(response.headers["Content-Range"].rsplit("/", 1)[1])

    def _request(self, start: int, end: int) -> urllib.request.Request:
        return urllib.request.Request(self.url, headers={"Range": f"bytes={start}-{end}"})

    def _read(self, start: int, end: int) -> bytes:
        """Bytes start to end (inclusive); retries interrupted transfers."""
        for attempt in range(5):
            try:
                with urllib.request.urlopen(self._request(start, end), timeout=120) as response:
                    data = response.read()
                if len(data) == end - start + 1:
                    return data
                error = OSError(f"read {len(data)} of {end - start + 1} bytes")
            except (OSError, http.client.HTTPException) as e:
                error = e
            if attempt == 4:
                raise error
            time.sleep(5 * (attempt + 1))

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        self.position = (offset, self.position + offset, self.size + offset)[whence]
        return self.position

    def readinto(self, buffer) -> int:
        end = min(self.position + len(buffer), self.size)
        if end <= self.position:
            return 0
        data = self._read(self.position, end - 1)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def open_zip(source: str) -> zipfile.ZipFile:
    if re.match(r"https?://", source):
        return zipfile.ZipFile(io.BufferedReader(RemoteFile(source), buffer_size=1 << 18))
    return zipfile.ZipFile(source)


def runs(archive: zipfile.ZipFile) -> dict[tuple[str, str, int], list[zipfile.ZipInfo]]:
    """(dataset, model, seed) -> the zip members of the run: the checkpoint files
    (without TensorFlow's .meta graphs) or BEUrRE's pickled model."""
    found: dict[tuple[str, str, str], list[zipfile.ZipInfo]] = {}
    for info in archive.infolist():
        if info.is_dir() or info.filename.endswith((".meta", ".DS_Store")):
            continue
        for pattern, model in RUNS:
            if match := re.match(pattern, info.filename):
                dataset, name = match.groups()
                found.setdefault((dataset, model, name), []).append(info)
    numbered = {}
    for dataset, model in {(d, m) for d, m, _ in found}:
        names = sorted(n for d, m, n in found if (d, m) == (dataset, model))
        by_rank = not all(n.isdigit() for n in names)  # BEUrRE's wandb ids
        for rank, name in enumerate(names):
            seed = rank if by_rank else int(name)
            numbered[dataset, model, seed] = found[dataset, model, name]
    return numbered


def download_models(source: str, datasets: list[str], models: list[str], seeds) -> None:
    """Runs that are downloaded (source.json) or converted already are skipped."""
    with open_zip(source) as archive:
        selected = {
            (dataset, model, seed): members
            for (dataset, model, seed), members in sorted(runs(archive).items())
            if dataset in datasets
            and model in models
            and seed in seeds
            and not (RAW / f"{dataset}-{model}" / f"seed{seed}" / "source.json").exists()
            and not (CONVERTED / f"{dataset}-{model}" / f"seed{seed}.pt").exists()
        }
        total = sum(info.file_size for members in selected.values() for info in members)
        with tqdm(total=total, unit="B", unit_scale=True, smoothing=0.1) as bar:
            for (dataset, model, seed), members in selected.items():
                bar.set_postfix_str(f"{dataset}-{model} seed{seed}")
                folder = RAW / f"{dataset}-{model}" / f"seed{seed}"
                folder.mkdir(parents=True, exist_ok=True)
                files = {}
                for info in members:
                    target = folder / Path(info.filename).name
                    digest = hashlib.sha256()
                    with archive.open(info) as src, open(target, "wb") as dst:
                        while chunk := src.read(1 << 26):
                            digest.update(chunk)
                            dst.write(chunk)
                            bar.update(len(chunk))
                    files[target.name] = {
                        "member": info.filename,
                        "size": info.file_size,
                        "sha256": digest.hexdigest(),
                    }
                (folder / "source.json").write_text(json.dumps(files, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=["data", "models"])
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    parser.add_argument("--zip", help="Models.zip: a file, or a URL with range requests")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS, help="default: 0-4")
    args = parser.parse_args()
    if args.what == "data":
        download_data(args.datasets)
    elif args.zip is None:
        sys.exit("models: --zip is required")
    else:
        download_models(args.zip, args.datasets, args.models, args.seeds)


if __name__ == "__main__":
    main()
