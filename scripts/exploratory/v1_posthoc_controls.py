"""Post-hoc, exploratory look at the frozen PASS. Cached vectors only: max_new=0.
argv[1] = repo path. Run with cwd = repo so the default cache path resolves."""
import re
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score as ap

from norm_k.ablation import (BASE_FIELDS, CORNER_FIELDS, K_FIELDS, LEXICAL_FIELDS,
                             _out_of_family_scores, corner_features, counterfactual_features,
                             load_matched_benchmark)
from norm_k.counterfactuals import counterfactual_square
from norm_k.embeddings import EmbeddingCache, OpenAIEncoder
from norm_k.evaluation import token_features
from norm_k.geometry import cosine, k_relative
from norm_k.statistics import grouped_ap_delta_interval

pairs = load_matched_benchmark(Path(sys.argv[1]) / "data/benchmarks/matched-2x2-v1.tsv")
squares = [counterfactual_square(p, "negation") for p in pairs]
texts = sorted({t for s in squares for t in (s.base, s.other, s.transformed_base, s.transformed_other)})
with EmbeddingCache(".norm_k_cache/embeddings.sqlite3") as cache:
    vec = dict(zip(texts, cache.embed(OpenAIEncoder(), texts, max_new=0)))

VERB = re.compile(r"\b(does not|is not|was not|owns|lives|works|resides|is|was)\b")


def edit_gap(p):
    """Tokens between the first A/B difference and the verb negation edits."""
    a, b = p.text_a.lower().split(), p.text_b.lower().split()
    first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    m = VERB.search(p.text_a.lower())
    verb = len(p.text_a.lower()[:m.start()].split())
    return abs(first - verb)


rows = []
for p, s in zip(pairs, squares):
    a, b, ta, tb = (vec[s.base], vec[s.other], vec[s.transformed_base], vec[s.transformed_other])
    r = {**counterfactual_features(s, vec), **corner_features(s, vec),
         **token_features(p.text_a, p.text_b)}
    r["cos_a_ta"], r["cos_b_tb"] = cosine(a, ta), cosine(b, tb)
    r["k_sq"] = r["k_norm"] ** 2
    r["k_rel_rev"] = k_relative(b, tb, a, ta)
    r["k_rel_sym"] = (r["k_relative"] + r["k_rel_rev"]) / 2
    r["edit_gap"] = float(edit_gap(p))
    rows.append(r)
y = np.array([p.label == "contradiction" for p in pairs], dtype=int)
g = np.array([p.relation_family for p in pairs])
high = np.array([p.template_family.startswith("high") for p in pairs])
X = lambda f: np.array([[r[n] for n in f] for r in rows])
S = lambda f: _out_of_family_scores(X(f), y, g, 42)
FULL = BASE_FIELDS + CORNER_FIELDS + LEXICAL_FIELDS
SIX = ("cos_a_ta", "cos_b_tb")
from collections import defaultdict


def rf(fields, seed):
    out = np.zeros(len(y))
    x = X(fields)
    for fam in np.unique(g):
        h = g == fam
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=seed).fit(x[~h], y[~h])
        out[h] = m.predict_proba(x[h])[:, 1]
    return out


sets = {"lexical only": LEXICAL_FIELDS, "base only": BASE_FIELDS,
        "six cosines only": ("cosine", "cos_ta_tb", "cos_a_tb", "cos_b_ta") + SIX,
        "base+lexical (no T sentences)": BASE_FIELDS + LEXICAL_FIELDS,
        "full": FULL, "full+both": FULL + SIX, "full+both+K": FULL + SIX + K_FIELDS,
        "base+lexical+K": BASE_FIELDS + LEXICAL_FIELDS + K_FIELDS}
scores = {}
for name, f in sets.items():
    runs = [rf(f, s) for s in (1, 2, 3)]
    scores[name] = np.mean(runs, axis=0)
    print(f"RF {name:32s} AP mean over 3 seeds={np.mean([ap(y, r) for r in runs]):.3f} "
          f"ensemble={ap(y, scores[name]):.3f}")
for a, b in (("full+both", "full+both+K"), ("base+lexical (no T sentences)", "full+both"),
             ("base+lexical (no T sentences)", "base+lexical+K")):
    ci = grouped_ap_delta_interval(list(y), scores[a], scores[b], list(g), repetitions=2000, seed=42)
    print(f"RF {a} -> {b}: d={ap(y,scores[b])-ap(y,scores[a]):+.4f} CI[{ci['ci_lower']:+.3f},{ci['ci_upper']:+.3f}]")
k = defaultdict(list)
for p, r in zip(pairs, rows):
    k[(p.template_family.split('_')[0], p.label)].append(r["k_norm"])
for key, v in sorted(k.items()):
    print(key, len(v), round(float(np.mean(v)), 3))
