"""Count pairs the frozen negation rule can transform. No embeddings, no K.

argv[1] = repo, argv[2] = directory holding the extracted datasets.
"""
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from norm_k.counterfactuals import toggle_negation  # noqa: E402

D = Path(sys.argv[2])
MAX_WORDS = 25


def eligible(a, b):
    ta, tb = toggle_negation(a), toggle_negation(b)
    if not ta or not tb or ta == a or tb == b:
        return False
    if ta == b and tb == a:  # mirror pair: K carries nothing new
        return False
    return True


def short_single(text):
    if len(text.split()) > MAX_WORDS:
        return False
    return len(re.findall(r"[.!?](\s|$)", text.strip())) <= 1


VERB_WORDS = re.compile(r"\b(own|owns|live|lives|work|works|reside|resides)\b", re.I)
BEFORE_EDIT = re.compile(r",|;|:|\b(which|that|who|whom|whose|what|when|where|while|if|because|although|though"
                         r"|since|until|unless|after|before|as|so|but|and|or|knew|said|say|says|think|thinks"
                         r"|believe|believes|yes|well|uh|um)\b", re.I)


def main_clause(text):
    """Label-blind filter: the edit must land on a main-clause is or was."""
    if VERB_WORDS.search(text):  # the rule would edit these first, and they are often nouns
        return False
    toggled = toggle_negation(text)
    if not toggled:
        return False
    prefix = os.path.commonprefix([text, toggled])
    if BEFORE_EDIT.search(prefix):
        return False
    return not re.fullmatch(r"\s*there\s+(is|was)?\s*", prefix, re.I) and not re.match(r"\s*there\b", prefix, re.I)


def report(name, rows):
    total = len(rows)
    both = [r for r in rows if eligible(r["a"], r["b"])]
    short = [r for r in both if short_single(r["a"]) and short_single(r["b"])]
    strict = [r for r in short if main_clause(r["a"]) and main_clause(r["b"])]
    print(f"{name}: usable {total}, toggles both {len(both)}, short single sentence {len(short)},"
          f" main-clause edit {len(strict)} ({len(strict) / max(total, 1):.1%})")
    print("   labels in strict:", dict(Counter(r["label"] for r in strict)))
    groups = Counter(r.get("group") for r in strict)
    if len(groups) > 1:
        print("   by group:", dict(groups))
    print("   median words a/b:", sorted(len(r["a"].split()) for r in strict)[len(strict) // 2] if strict else None,
          sorted(len(r["b"].split()) for r in strict)[len(strict) // 2] if strict else None)
    print("   unique premises:", len({r["a"] for r in strict}))
    if "--sample" in sys.argv:
        texts = sorted({t for r in strict for t in (r["a"], r["b"])})
        for text in random.Random(2027).sample(texts, min(15, len(texts))):
            print("   *", toggle_negation(text))


def jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


for split in ("matched", "mismatched"):
    path = D / f"multinli_1.0/multinli_1.0_dev_{split}.jsonl"
    if path.exists():
        data = jsonl(path)
        print("MNLI fields:", sorted(data[0]))
        print("annotator label counts:", dict(Counter(len(r.get("annotator_labels", [])) for r in data)))
        rows = [{"a": r["sentence1"], "b": r["sentence2"], "label": r["gold_label"], "group": r["genre"]}
                for r in data if r["gold_label"] != "-"]
        report(f"MNLI dev {split}", rows)
for split in ("dev", "test"):
    path = D / f"snli_1.0/snli_1.0_{split}.jsonl"
    if path.exists():
        data = jsonl(path)
        rows = [{"a": r["sentence1"], "b": r["sentence2"], "label": r["gold_label"]}
                for r in data if r["gold_label"] != "-"]
        report(f"SNLI {split}", rows)
path = D / "wanli_test.jsonl"
if path.exists():
    data = jsonl(path)
    print("WANLI fields:", sorted(data[0]))
    rows = [{"a": r["premise"], "b": r["hypothesis"], "label": r["gold"]} for r in data]
    report("WANLI test", rows)
