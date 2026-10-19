"""Makes LibKGE datasets of CN15k, NL27k and PPI5k from the files of download.py data.

    python third_party/unkge/prepare_data.py

Entity and relation ids are the integer ids of the files of CN15k and PPI5k, and those
of BEUrRE's entity_id.csv and relation_id.csv for NL27k, whose files in UKGE's release
name the entities and relations (except test_with_neg.tsv, which holds these ids). The
BEUrRE models use these ids.

Writes, next to the downloaded files: train.del, valid.del, test.del, test_with_neg.del
and softlogic.del, with the ids and the confidence (the 4th column, which LibKGE does not
read) in the order of the original files; entity_ids.del and relation_ids.del (the
identifiers of the original files), entity_strings.del and relation_strings.del (names,
where known); dataset.yaml. Splits: train, valid, test; test_with_neg, the test triples
and as many negatives (confidence 0), with the confidences rounded to 3 digits.
softlogic.del (UKGE's soft-logic triples) is not a split; the conversion of UKGE models
needs it.
"""

import csv
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "third_party/libkge/data"
FILES = {  # file name of the split -> name in LibKGE
    "train": "train",
    "val": "valid",
    "test": "test",
    "test_with_neg": "test_with_neg",
    "softlogic": "softlogic",
}
NAMES = {  # dataset -> (entity names file, relation names file), from BEUrRE
    "cn15k": ("concept_id.csv", "relation_id.csv"),
    "nl27k": ("entity_id.csv", "relation_id.csv"),
}
SIZES = {  # dataset -> entities, relations (BEUrRE's models), train, valid, test triples
    "cn15k": (15000, 36, 204984, 16881, 19293),
    "nl27k": (27221, 417, 149100, 12278, 14034),
    "ppi5k": (4999, 7, 230929, 19017, 21720),
}


def read_names(file: Path) -> dict[int, str]:
    """id -> name, from a csv file with a column of ids (id, vid or rid) and one of names."""
    with open(file, newline="") as f:
        rows = list(csv.reader(f))
    id_column = next(i for i, name in enumerate(rows[0]) if name in ("id", "vid", "rid"))
    return {int(row[id_column]): row[1 - id_column] for row in rows[1:]}


def read_triples(file: Path) -> list[list[str]]:
    with open(file) as f:
        return [line.rstrip("\n").split("\t") for line in f]


def prepare(dataset: str) -> None:
    folder = DATA / dataset
    num_entities, num_relations, *split_sizes = SIZES[dataset]
    entity_names = relation_names = None
    if dataset in NAMES:
        entity_names, relation_names = (read_names(folder / f) for f in NAMES[dataset])
    names_to_ids = None
    if dataset == "nl27k":  # the files name the entities and relations
        identifiers = entity_names, relation_names
        names_to_ids = tuple({name: i for i, name in names.items()} for names in identifiers)
    else:  # the files hold the ids
        identifiers = tuple({i: str(i) for i in range(n)} for n in (num_entities, num_relations))

    sizes = {}
    for source, split in FILES.items():
        rows = read_triples(folder / f"{source}.tsv")
        with open(folder / f"{split}.del", "w") as f:
            for h, r, t, c in rows:
                if names_to_ids is not None and not h.isdigit():
                    h, r, t = names_to_ids[0][h], names_to_ids[1][r], names_to_ids[0][t]
                h, r, t = int(h), int(r), int(t)
                if not (0 <= h < num_entities and 0 <= t < num_entities and 0 <= r < num_relations):
                    raise ValueError(f"{dataset}/{source}.tsv: ids out of range: {h} {r} {t}")
                if not 0 <= float(c) <= 1:
                    raise ValueError(f"{dataset}/{source}.tsv: confidence {c}")
                f.write(f"{h}\t{r}\t{t}\t{c}\n")
        sizes[split] = len(rows)
    if [sizes["train"], sizes["valid"], sizes["test"]] != split_sizes:
        raise ValueError(f"{dataset}: split sizes {sizes}, expected {split_sizes}")

    for kind, ids, names in (
        ("entity", identifiers[0], entity_names),
        ("relation", identifiers[1], relation_names),
    ):
        (folder / f"{kind}_ids.del").write_text("".join(f"{i}\t{ids[i]}\n" for i in sorted(ids)))
        if names is not None:
            (folder / f"{kind}_strings.del").write_text(
                "".join(f"{i}\t{names[i]}\n" for i in sorted(names))
            )

    files = {
        "entity_ids": {"filename": "entity_ids.del", "type": "map"},
        "relation_ids": {"filename": "relation_ids.del", "type": "map"},
    }
    if entity_names is not None:
        files["entity_strings"] = {"filename": "entity_strings.del", "type": "map"}
        files["relation_strings"] = {"filename": "relation_strings.del", "type": "map"}
    for split, split_type in (
        ("train", "train"),
        ("valid", "valid"),
        ("test", "test"),
        ("test_with_neg", "test"),
    ):
        files[split] = {
            "filename": f"{split}.del",
            "size": sizes[split],
            "split_type": split_type,
            "type": "triples",
        }
    options = {
        "name": dataset,
        "num_entities": num_entities,
        "num_relations": num_relations,
        "files": files,
    }
    with open(folder / "dataset.yaml", "w") as f:
        yaml.dump({"dataset": options}, f)
    print(f"{dataset}: {num_entities} entities, {num_relations} relations, splits {sizes}")


def main():
    for dataset in SIZES:
        prepare(dataset)


if __name__ == "__main__":
    main()
