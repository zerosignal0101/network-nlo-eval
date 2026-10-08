"""FG-18 §4 门 1：全量门分片执行器。

FG-17 实测 RL 全量串行 ~2h+，本机不能一口气跑完 ⇒ 分片。本脚本把测试文件按
**轮转**方式分成 N 片，每片串行跑完并写自己的 JSON 摘要。

要点（都来自跨 WP 执行规则）：

* ``PYTHONPATH`` 显式带上前缀，否则 editable 的 ``.pth`` 会把测试导到主检出，
  跑的是另一棵树的代码。
* **分片套件会随父进程一起死**（死法：无汇总行、日志停在进度点）。所以每片
  自己写 JSON，读数只看**文件**不看控制台；父进程被杀后已完成的片仍然有效，
  重跑缺失片即可（``--only``）。
* 并发由调用方用 ``--jobs`` 控制，默认 2；本机还有别人的负载，不抢。

用法::

    PYTHONPATH=<rl>/src:<phys>/src python scripts/fg18_shard_suite.py \\
        --repo <repo> --out <dir> --jobs 2 --shards 8
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time


def _loadavg() -> str:
    return pathlib.Path("/proc/loadavg").read_text().split()[:3][0] if pathlib.Path("/proc/loadavg").exists() else "?"


def plan(repo: pathlib.Path, shards: int) -> list[list[str]]:
    """把 ``tests/*.py`` 轮转分片。

    轮转（而非按文件数均分）是为了让**重的**学习/RL 套件均匀摊开——按字母顺序
    切会把 ``test_learn_*`` 全堆进同一片。
    """
    files = sorted(str(p.relative_to(repo)) for p in repo.glob("tests/*.py"))
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for i, f in enumerate(files):
        buckets[i % shards].append(f)
    return buckets


def run_shard(repo: pathlib.Path, files: list[str], outdir: pathlib.Path, index: int, pythonpath: str) -> dict:
    out_json = outdir / f"shard_{index:02d}.json"
    if out_json.exists():
        try:
            return json.loads(out_json.read_text())
        except Exception:  # noqa: BLE001 - 半个文件按缺失处理
            pass

    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / f"shard_{index:02d}.log"
    env = dict(os.environ)
    env["PYTHONPATH"] = pythonpath
    start = time.time()
    with log.open("w") as fh:
        proc = subprocess.run(
            [pythonpath_python(), "-m", "pytest", *files, "-q", "-p", "no:randomly", "--no-header"],
            cwd=repo,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    payload = {
        "shard": index,
        "files": files,
        "returncode": proc.returncode,
        "seconds": round(time.time() - start, 1),
        "loadavg_at_end": _loadavg(),
        "summary_line": _summary(log),
        "log": str(log),
    }
    out_json.write_text(json.dumps(payload, indent=2))
    return payload


def pythonpath_python() -> str:
    return sys.executable


def _summary(log: pathlib.Path) -> str:
    for line in reversed(log.read_text(errors="replace").strip().splitlines()):
        if "passed" in line or "failed" in line or "error" in line:
            return line.strip()
    return "(no summary line -- SHARD MAY HAVE DIED)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--shards", type=int, default=8)
    ap.add_argument("--pythonpath", required=True)
    ap.add_argument("--only", default="", help="逗号分隔的片号；只跑这些片")
    args = ap.parse_args()

    repo = pathlib.Path(args.repo)
    outdir = pathlib.Path(args.out)
    buckets = plan(repo, args.shards)
    only = {int(x) for x in args.only.split(",") if x.strip()} if args.only else None

    todo = [(i, b) for i, b in enumerate(buckets) if b and (only is None or i in only)]
    print(f"repo={repo} shards={len(buckets)} todo={len(todo)} jobs={args.jobs} loadavg={_loadavg()}")
    for i, b in todo:
        print(f"  shard {i}: {len(b)} files")

    running: list[tuple[int, subprocess.Popen]] = []
    queue = list(todo)
    results: dict[int, dict] = {}

    def launch(idx: int, files: list[str]) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        log = outdir / f"shard_{idx:02d}.log"
        env = dict(os.environ)
        env["PYTHONPATH"] = args.pythonpath
        fh = log.open("w")
        p = subprocess.Popen(
            [sys.executable, "-m", "pytest", *files, "-q", "-p", "no:randomly", "--no-header"],
            cwd=repo,
            stdout=fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
        running.append((idx, p))

    while queue or running:
        while queue and len(running) < args.jobs:
            idx, files = queue.pop(0)
            print(f"[start] shard {idx} ({len(files)} files) loadavg={_loadavg()}", flush=True)
            launch(idx, files)
        time.sleep(5)
        still = []
        for idx, p in running:
            if p.poll() is None:
                still.append((idx, p))
            else:
                log = outdir / f"shard_{idx:02d}.log"
                summary = _summary(log)
                print(f"[done ] shard {idx} rc={p.returncode} :: {summary}", flush=True)
        running = still

    # 汇总：只认 JSON 文件
    for idx, files in enumerate(buckets):
        if not files:
            continue
        fj = outdir / f"shard_{idx:02d}.json"
        if fj.exists():
            results[idx] = json.loads(fj.read_text())
        else:
            # 重跑该片（分片可能随父进程一起死了）
            print(f"[retry] shard {idx} missing summary", flush=True)
            results[idx] = run_shard(repo, files, outdir, idx, args.pythonpath)

    summary = {
        "repo": str(repo),
        "shards": {str(k): v for k, v in sorted(results.items())},
        "loadavg_at_end": _loadavg(),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    for k, v in sorted(results.items()):
        print(f"shard {k}: rc={v.get('returncode')} {v.get('seconds')}s :: {v.get('summary_line')}")
    print("loadavg_at_end:", summary["loadavg_at_end"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
