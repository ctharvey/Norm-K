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


def line(name, base, add):
    s0, s1 = S(base), S(base + add)
    ci = grouped_ap_delta_interval(list(y), s0, s1, list(g), repetitions=2000, seed=42)
    print(f"{name:44s} {ap(y,s0):.3f} -> {ap(y,s1):.3f}  d={ap(y,s1)-ap(y,s0):+.4f} "
          f"CI[{ci['ci_lower']:+.3f},{ci['ci_upper']:+.3f}]")


print("k_norm vs reconstruction check: all cached, API calls=0")
line("frozen: full + K", FULL, K_FIELDS)
line("full + k_norm only", FULL, ("k_norm",))
line("full + k_relative only", FULL, ("k_relative",))
line("full + k_sq only", FULL, ("k_sq",))
line("full + symmetric k_relative", FULL, ("k_rel_sym",))
line("full + cos(A,TA),cos(B,TB) [no K]", FULL, SIX)
line("full+both cosines, then + K", FULL + SIX, K_FIELDS)
line("full+both cosines, then + k_norm", FULL + SIX, ("k_norm",))
line("full+both cosines, then + k_relative", FULL + SIX, ("k_relative",))
line("full + edit_gap [no K]", FULL, ("edit_gap",))
line("full+edit_gap, then + K", FULL + ("edit_gap",), K_FIELDS)
line("full+edit_gap+both cosines, then + K", FULL + ("edit_gap",) + SIX, K_FIELDS)

from sklearn.metrics import roc_auc_score
print("edit_gap alone AUC (small gap = contradiction):", round(roc_auc_score(y, -X(("edit_gap",))[:, 0]), 3),
      "high cell:", round(roc_auc_score(y[high], -X(("edit_gap",))[high, 0]), 3))
print("corr(k_norm, edit_gap):", round(float(np.corrcoef(X(("k_norm",))[:, 0], X(("edit_gap",))[:, 0])[0, 1]), 3))

s0, s1 = S(FULL), S(FULL + K_FIELDS)
for name, m in (("high overlap", high), ("low overlap", ~high)):
    print(f"{name}: {ap(y[m],s0[m]):.3f} -> {ap(y[m],s1[m]):.3f} d={ap(y[m],s1[m])-ap(y[m],s0[m]):+.4f}")
print("per-family d (pooled AP dropping that family):")
d_all = ap(y, s1) - ap(y, s0)
for fam in sorted(set(g)):
    m = g != fam
    print(f"  without {fam:12s} d={ap(y[m],s1[m])-ap(y[m],s0[m]):+.4f}")


def rf(fields):
    out = np.zeros(len(y))
    x = X(fields)
    for fam in np.unique(g):
        h = g == fam
        m = RandomForestClassifier(n_estimators=500, min_samples_leaf=3, random_state=42).fit(x[~h], y[~h])
        out[h] = m.predict_proba(x[h])[:, 1]
    return out


r0, r1 = rf(FULL + SIX), rf(FULL + SIX + K_FIELDS)
print(f"random forest, full+both cosines: {ap(y,r0):.3f} -> +K {ap(y,r1):.3f} d={ap(y,r1)-ap(y,r0):+.4f}")
print("top pair flips (full -> full_k), rank gain among contradictions:")
rank0, rank1 = (-s0).argsort().argsort(), (-s1).argsort().argsort()
order = np.argsort(rank1 - rank0)
for i in order[:6]:
    print(f"  y={y[i]} {rank0[i]:3d}->{rank1[i]:3d} {pairs[i].template_family[:12]} | {pairs[i].text_a} | {pairs[i].text_b}")
