#!/usr/bin/env python3
"""两棵上游树在**相同物理输入**下比较 QoTValidator 的 SNR 估计。

用法: python3 cross_snr_compat.py <tree_path> > out.json
"""
import json
import subprocess
import sys

PY = "/home/zsig/miniconda3/envs/rl/bin/python"
SCEN_DIR = "/home/zsig/.hermes/cache/scratch/xsnr/tests_legacy"

CHILD = r'''
import json, sys
tree = sys.argv[1]
sys.path.insert(0, tree + "/src")
sys.path.insert(0, sys.argv[2])

import compat_run as cr

out = {"tree": tree, "results": [], "errors": []}
for sc in cr.SCENARIOS:
    row = {"name": sc.name, "kind": sc.kind}
    try:
        r = cr.run_scenario(sc)
        row["snr_db"] = r.get("snr_db")
        row["accepted"] = bool(r.get("accepted"))
        row["init_params"] = r.get("init_params")
    except Exception as exc:
        row["error"] = "%s: %s" % (type(exc).__name__, exc)
        out["errors"].append(sc.name)
    out["results"].append(row)
print(json.dumps(out))
'''


def probe(tree):
    p = subprocess.run([PY, "-c", CHILD, tree, SCEN_DIR],
                       capture_output=True, text=True, timeout=1800)
    if p.returncode != 0:
        return {"tree": tree, "fatal": p.stderr[-2000:], "results": []}
    for line in reversed(p.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    return {"tree": tree, "fatal": (p.stdout + p.stderr)[-1800:], "results": []}


if __name__ == "__main__":
    print(json.dumps(probe(sys.argv[1])))