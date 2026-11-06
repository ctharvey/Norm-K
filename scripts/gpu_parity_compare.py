"""Compare a frozen result JSON with its GPU rerun: every verdict and every number.

Usage: gpu_parity_compare.py FROZEN_JSON GPU_JSON
"""
import json
import sys


def leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            yield from leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from leaves(value, f"{path}/{index}")
    else:
        yield path, node


frozen, rerun = (dict(leaves(json.load(open(path, encoding="utf-8")))) for path in sys.argv[1:3])
shared = frozen.keys() & rerun.keys()
numbers = [(abs(frozen[k] - rerun[k]), k) for k in shared
           if isinstance(frozen[k], (int, float)) and isinstance(rerun[k], (int, float))
           and not isinstance(frozen[k], bool) and "tokens" not in k]
others = [k for k in shared if not isinstance(frozen[k], (int, float)) or isinstance(frozen[k], bool)]
changed = [(k, frozen[k], rerun[k]) for k in others if frozen[k] != rerun[k]]
verdicts = sorted(k for k in others if "verdict" in k or k.endswith("passes") or k.endswith("gate"))
largest = max(numbers) if numbers else (0, "")
print(sys.argv[1].split("/")[-1], "numbers", len(numbers), "largest gap", largest[0], largest[1],
      "| non-numeric", len(others), "changed", len(changed), "| verdict fields", len(verdicts),
      "| keys only on one side", len(frozen.keys() ^ rerun.keys()))
for item in changed[:12]:
    print("  CHANGED", item)
