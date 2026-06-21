from __future__ import annotations

"""Deterministic synthetic RAG benchmark generator (synth_rag_v1).

Design for unique gold:
  Each document is a unique combination of (person, location, object, color,
  date, event, organization, numeric_code). We assign every doc a unique
  numeric_code, guaranteeing global uniqueness. Queries are built from a subset
  of a gold doc's fields chosen so that the conjunction of those fields is
  satisfied by exactly one document in the corpus (verified during generation).
  If a sampled field-conjunction is not unique, we add fields (and finally the
  numeric_code) until it is unique, then drop back to the requested type only if
  uniqueness holds. Queries that cannot be made unique within the doc are
  discarded and resampled, so gold-unique rate is ~100%.
"""

import argparse
import random
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import yaml

from ..utils.io import write_jsonl, write_json, content_hash
from ..utils.seeds import seed_everything
from .schema import Doc, Query, DOC_FIELDS

# ---- deterministic vocabularies -------------------------------------------
PERSONS = [
    "Akira Tanaka", "Maria Garcia", "John Smith", "Wei Chen", "Olivia Brown",
    "Liam Johnson", "Emma Wilson", "Noah Davis", "Ava Martinez", "Sophia Lee",
    "Lucas Muller", "Mia Rossi", "Ethan Kim", "Isabella Nguyen", "Mason Clark",
    "Amelia Lopez", "Logan Walker", "Harper Young", "Elijah Hall", "Charlotte King",
    "Yuki Sato", "Hiroshi Ito", "Sven Larsson", "Ingrid Berg", "Diego Torres",
    "Camila Reyes", "Omar Haddad", "Layla Hassan", "Raj Patel", "Priya Sharma",
    "Chloe Dubois", "Hugo Martin", "Anna Novak", "Petr Horak", "Sofia Costa",
    "Tiago Silva", "Mei Wong", "Jin Park", "Ali Khan", "Fatima Ahmed",
]
LOCATIONS = [
    "Kyoto", "Lisbon", "Toronto", "Nairobi", "Reykjavik", "Santiago", "Hanoi",
    "Prague", "Cairo", "Oslo", "Lima", "Seoul", "Athens", "Dublin", "Helsinki",
    "Bogota", "Manila", "Warsaw", "Vienna", "Doha", "Quito", "Riga", "Tbilisi",
    "Accra", "Amman", "Bern", "Cusco", "Davao", "Erfurt", "Faro",
]
OBJECTS = [
    "camera", "bicycle", "guitar", "telescope", "backpack", "umbrella", "lamp",
    "violin", "keyboard", "drone", "watch", "tent", "kettle", "microscope",
    "skateboard", "headphones", "tripod", "speaker", "monitor", "printer",
]
COLORS = [
    "red", "blue", "green", "yellow", "black", "white", "silver", "purple",
    "orange", "teal", "magenta", "gold",
]
EVENTS = [
    "purchase", "return", "repair", "rental", "donation", "auction", "trade",
    "loan", "upgrade", "inspection",
]
EVENT_SYNONYMS = {
    "purchase": "buying", "return": "sending back", "repair": "fixing",
    "rental": "renting", "donation": "donating", "auction": "auctioning",
    "trade": "trading", "loan": "lending", "upgrade": "upgrading",
    "inspection": "inspecting",
}
OBJECT_SYNONYMS = {
    "camera": "photo camera", "bicycle": "bike", "guitar": "acoustic guitar",
    "telescope": "spyglass", "backpack": "rucksack", "umbrella": "parasol",
    "lamp": "desk lamp", "violin": "fiddle", "keyboard": "keypad", "drone": "quadcopter",
    "watch": "wristwatch", "tent": "shelter tent", "kettle": "teakettle",
    "microscope": "scope", "skateboard": "skate deck", "headphones": "earphones",
    "tripod": "camera stand", "speaker": "loudspeaker", "monitor": "display",
    "printer": "laser printer",
}
ORGS = [
    "Nikon Store", "GreenCycle Co", "SoundHouse", "StarOptics", "TrailGear",
    "RainCorp", "BrightLite", "StringWorks", "KeyTech", "SkyView Labs",
    "TimeKeepers", "CampLine", "BrewMakers", "MicroLab Inc", "RollFast",
    "AudioMax", "StandPro", "BoomBox Ltd", "PixelView", "PrintPlus",
]
EVENT_VERB = {
    "purchase": "purchased", "return": "returned", "repair": "repaired",
    "rental": "rented", "donation": "donated", "auction": "auctioned",
    "trade": "traded", "loan": "loaned", "upgrade": "upgraded",
    "inspection": "inspected",
}


def _date(rng: random.Random) -> str:
    y = rng.randint(2018, 2023)
    m = rng.randint(1, 12)
    d = rng.randint(1, 28)
    return f"{y:04d}-{m:02d}-{d:02d}"


def _doc_text(d: dict) -> str:
    verb = EVENT_VERB.get(d["event"], d["event"])
    return (
        f"On {d['date']}, {d['person']} {verb} a {d['color']} {d['object']} "
        f"in {d['location']} at {d['organization']} (ref {d['numeric_code']})."
    )


def gen_docs(n_docs: int, rng: random.Random) -> list[Doc]:
    docs: list[Doc] = []
    for i in range(n_docs):
        rec = {
            "person": rng.choice(PERSONS),
            "location": rng.choice(LOCATIONS),
            "object": rng.choice(OBJECTS),
            "color": rng.choice(COLORS),
            "date": _date(rng),
            "event": rng.choice(EVENTS),
            "organization": rng.choice(ORGS),
            "numeric_code": f"NC-{i:05d}",  # globally unique
        }
        rec["text"] = _doc_text(rec)
        docs.append(
            Doc(
                doc_id=i,
                metadata={"field_values": {k: rec[k] for k in DOC_FIELDS}},
                **rec,
            )
        )
    return docs


def build_field_index(docs: list[Doc]) -> dict[str, dict[str, set[int]]]:
    """field -> value -> set of doc_ids."""
    idx: dict[str, dict[str, set[int]]] = {f: defaultdict(set) for f in DOC_FIELDS}
    for d in docs:
        dd = d.to_dict()
        for f in DOC_FIELDS:
            idx[f][dd[f]].add(d.doc_id)
    return idx


def matching_docs(
    idx: dict[str, dict[str, set[int]]], conditions: dict[str, str]
) -> set[int]:
    sets = [idx[f][v] for f, v in conditions.items()]
    if not sets:
        return set()
    out = set(sets[0])
    for s in sets[1:]:
        out &= s
    return out


# field subsets that define each query type (excluding numeric_code special case)
_TYPE_FIELDS = {
    "two-condition": ["object", "color", "location", "person", "event"],
    "three-condition": ["object", "color", "location", "person", "event", "date"],
    "four-condition": ["object", "color", "location", "person", "event", "date", "organization"],
    "organization+event": ["organization", "event", "location", "object", "color"],
}
_TYPE_NCOND = {
    "two-condition": 2,
    "three-condition": 3,
    "four-condition": 4,
    "organization+event": 2,
}


def _question_text(conditions: dict[str, str], rng: random.Random) -> str:
    obj = conditions.get("object")
    color = conditions.get("color")
    loc = conditions.get("location")
    person = conditions.get("person")
    event = conditions.get("event", "purchase")
    org = conditions.get("organization")
    date = conditions.get("date")
    code = conditions.get("numeric_code")
    verb_map = {
        "purchase": "bought", "return": "returned", "repair": "repaired",
        "rental": "rented", "donation": "donated", "auction": "auctioned",
        "trade": "traded", "loan": "loaned", "upgrade": "upgraded",
        "inspection": "inspected",
    }
    if code is not None:
        extra = ""
        if obj:
            extra = f" involving a {obj}"
        return f"Which record has reference code {code}{extra}?"
    verb = verb_map.get(event, "handled")
    item = ""
    if color and obj:
        item = f"a {color} {obj}"
    elif obj:
        item = f"a {obj}"
    elif color:
        item = f"a {color} item"
    else:
        item = "an item"
    parts = [f"Who {verb} {item}"]
    if loc:
        parts.append(f"in {loc}")
    if org:
        parts.append(f"at {org}")
    if date:
        parts.append(f"on {date}")
    if person:
        # person-conditioned phrasing
        return f"What did {person} {verb}" + (f" in {loc}" if loc else "") + "?"
    return " ".join(parts) + "?"


def gen_queries(
    docs: list[Doc],
    idx: dict[str, dict[str, set[int]]],
    n_queries: int,
    cfg: dict,
    rng: random.Random,
) -> tuple[list[Query], list[int]]:
    """Generate queries with unique gold. Returns (queries, non_unique_count)."""
    weights = cfg["query_type_weights"]
    diff_map = cfg["difficulty_map"]
    adv_frac = cfg.get("adversarial_fraction", 0.15)
    types = list(weights.keys())
    probs = [weights[t] for t in types]

    queries: list[Query] = []
    non_unique = 0
    qid = 0
    attempts = 0
    max_attempts = n_queries * 50
    while len(queries) < n_queries and attempts < max_attempts:
        attempts += 1
        qtype = rng.choices(types, probs)[0]
        gold = rng.choice(docs)
        gdd = gold.to_dict()

        if qtype == "code-based":
            conditions = {"numeric_code": gdd["numeric_code"]}
            if rng.random() < 0.5:
                conditions["object"] = gdd["object"]
        else:
            fields_pool = _TYPE_FIELDS[qtype]
            ncond = _TYPE_NCOND[qtype]
            chosen = rng.sample(fields_pool, min(ncond, len(fields_pool)))
            conditions = {f: gdd[f] for f in chosen}

        match = matching_docs(idx, conditions)
        if len(match) != 1:
            # try to disambiguate by adding fields, then numeric_code
            for extra in ["date", "organization", "person", "numeric_code"]:
                if extra in conditions:
                    continue
                conditions[extra] = gdd[extra]
                match = matching_docs(idx, conditions)
                if len(match) == 1:
                    break
        if len(match) != 1 or next(iter(match)) != gold.doc_id:
            non_unique += 1
            continue

        difficulty = diff_map.get(qtype, "medium")
        if difficulty in ("hard", "medium") and rng.random() < adv_frac:
            difficulty = "adversarial"

        qtext = _question_text(conditions, rng)
        queries.append(
            Query(
                query_id=qid,
                query_text=qtext,
                gold_doc_id=gold.doc_id,
                query_type=qtype,
                difficulty=difficulty,
                conditions=conditions,
                n_conditions=len(conditions),
            )
        )
        qid += 1
    return queries, non_unique


def gen_hard_negatives(
    docs: list[Doc],
    idx: dict[str, dict[str, set[int]]],
    queries: list[Query],
    min_neg: int,
    rng: random.Random,
) -> list[dict]:
    """For each query, build >= min_neg hard negatives of varied types."""
    by_id = {d.doc_id: d for d in docs}
    rows: list[dict] = []
    all_ids = [d.doc_id for d in docs]
    for q in queries:
        gold = by_id[q.gold_doc_id]
        gdd = gold.to_dict()
        negs: dict[int, str] = {}

        # single-field overlap: share exactly one field value
        for f in DOC_FIELDS:
            cand = idx[f].get(gdd[f], set())
            for c in cand:
                if c != gold.doc_id and c not in negs:
                    # check overlap count
                    cdd = by_id[c].to_dict()
                    ov = sum(1 for ff in DOC_FIELDS if cdd[ff] == gdd[ff])
                    if ov == 1:
                        negs[c] = "single-field-overlap"
                if len(negs) >= min_neg * 3:
                    break

        # two-field overlap
        for f1, f2 in combinations([f for f in DOC_FIELDS if f != "numeric_code"], 2):
            inter = idx[f1].get(gdd[f1], set()) & idx[f2].get(gdd[f2], set())
            for c in inter:
                if c != gold.doc_id and c not in negs:
                    cdd = by_id[c].to_dict()
                    ov = sum(1 for ff in DOC_FIELDS if cdd[ff] == gdd[ff])
                    if ov == 2:
                        negs[c] = "two-field-overlap"
            if len(negs) >= min_neg * 4:
                break

        # ensure enough by random fill (still not gold), labelled distractor
        while len(negs) < min_neg + 5:
            c = rng.choice(all_ids)
            if c != gold.doc_id and c not in negs:
                negs[c] = "random-distractor"

        neg_list = list(negs.items())[: max(min_neg + 5, min_neg)]
        rows.append(
            {
                "query_id": q.query_id,
                "gold_doc_id": q.gold_doc_id,
                "hard_negatives": [
                    {"doc_id": c, "neg_type": t} for c, t in neg_list
                ],
                "n_hard_negatives": len(neg_list),
            }
        )
    return rows


def main(config_path: str) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text())
    seed = int(cfg["seed"])
    seed_everything(seed)
    rng = random.Random(seed)

    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    docs = gen_docs(int(cfg["n_docs"]), rng)
    idx = build_field_index(docs)
    queries, non_unique = gen_queries(docs, idx, int(cfg["n_queries"]), cfg, rng)
    hard_negs = gen_hard_negatives(
        docs, idx, queries, int(cfg["min_hard_negatives"]), rng
    )

    docs_path = out_dir / "docs.jsonl"
    q_path = out_dir / "queries.jsonl"
    qrels_path = out_dir / "qrels.tsv"
    hn_path = out_dir / "hard_negatives.jsonl"

    write_jsonl(docs_path, (d.to_dict() for d in docs))
    write_jsonl(q_path, (q.to_dict() for q in queries))
    with qrels_path.open("w") as f:
        for q in queries:
            f.write(f"{q.query_id}\t{q.gold_doc_id}\t1\n")
    write_jsonl(hn_path, hard_negs)

    chash = content_hash(docs_path, q_path, qrels_path)
    # gold_unique_rate = fraction of queries whose condition set matches EXACTLY
    # one doc (the gold). Multiple queries may legitimately target the same doc.
    unique_ok = 0
    for q in queries:
        m = matching_docs(idx, q.conditions)
        if len(m) == 1 and next(iter(m)) == q.gold_doc_id:
            unique_ok += 1
    gold_unique_rate = unique_ok / len(queries) if queries else 0.0
    distinct_gold_ratio = (
        len({q.gold_doc_id for q in queries}) / len(queries) if queries else 0.0
    )
    min_hn = min((r["n_hard_negatives"] for r in hard_negs), default=0)

    stats = {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "non_unique_discarded": non_unique,
        "gold_unique_rate": round(gold_unique_rate, 6),
        "distinct_gold_ratio": round(distinct_gold_ratio, 6),
        "min_hard_negatives": min_hn,
        "content_hash": chash,
        "seed": seed,
        "difficulty_counts": _count_attr(queries, "difficulty"),
        "query_type_counts": _count_attr(queries, "query_type"),
    }
    write_json(out_dir / "stats.json", stats)
    _write_readme(out_dir, stats, cfg)
    return stats


def _count_attr(queries: list[Query], attr: str) -> dict:
    c: dict[str, int] = {}
    for q in queries:
        v = getattr(q, attr)
        c[v] = c.get(v, 0) + 1
    return c


def _write_readme(out_dir: Path, stats: dict, cfg: dict) -> None:
    md = f"""# synth_rag_v1

Deterministic synthetic RAG retrieval benchmark (seed={stats['seed']}).

- docs: {stats['n_docs']}
- queries: {stats['n_queries']}
- gold_unique_rate: {stats['gold_unique_rate']}
- min_hard_negatives per query: {stats['min_hard_negatives']}
- content_hash (docs+queries+qrels sha256): `{stats['content_hash']}`

## Files
- docs.jsonl, queries.jsonl, qrels.tsv, hard_negatives.jsonl

## Query type counts
{stats['query_type_counts']}

## Difficulty counts
{stats['difficulty_counts']}

Re-running gen_synth_rag with the same seed reproduces the same content_hash.
"""
    (out_dir / "README.md").write_text(md)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/r0_synth.yaml")
    args = ap.parse_args()
    s = main(args.config)
    print(s)
