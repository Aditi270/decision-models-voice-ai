"""Run the text benchmark: every turn x every system x N reps, interleaved and resumable.

usage: python run_text.py [--reps 3] [--input ../data/turns.jsonl] [--out ...] [--workers 6]
  --reps      repetitions per (turn, system): 3 lets us measure whether answers are consistent
  --input     a jsonl of turns; for ASR runs the 'caller' field holds the transcript
  --workers   parallel calls; 6 keeps the run quick without hitting gateway rate limits
"""
import argparse, json, random, threading, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import clients

ap = argparse.ArgumentParser()
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--input", default="../data/turns.jsonl")
ap.add_argument("--out", default="../results/text_runs.jsonl")
ap.add_argument("--workers", type=int, default=6)
args = ap.parse_args()

turns = [json.loads(l) for l in open(args.input, encoding="utf-8")]
out = Path(args.out); out.parent.mkdir(exist_ok=True)
done = set()
if out.exists():
    for l in open(out, encoding="utf-8"):
        r = json.loads(l)
        if r["ok"]:
            done.add((r["id"], r["system"], r["rep"]))

jobs = [(t, s, rep) for t in turns for s in clients.SYSTEMS for rep in range(args.reps)
        if (t["id"], s, rep) not in done]
random.Random(0).shuffle(jobs)   # interleave systems so none gets better network conditions; seeded = reproducible
print(f"{len(jobs)} jobs ({len(done)} already done)")

lock = threading.Lock()
n = [0]
f = open(out, "a", encoding="utf-8")


def work(job):
    t, s, rep = job
    r = clients.call(s, t["agent"], t["caller"])
    r.pop("raw", None) if r["ok"] else None
    rec = dict(id=t["id"], system=s, rep=rep, ts=time.time(), **r)
    with lock:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
        n[0] += 1
        if n[0] % 100 == 0 or not r["ok"]:   # progress line every 100 calls, and every failure
            print(n[0], s, t["id"], "OK" if r["ok"] else r["error"], flush=True)


with ThreadPoolExecutor(args.workers) as ex:
    list(ex.map(work, jobs))
print("finished", n[0])
