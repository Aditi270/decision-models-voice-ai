"""Compute metrics + figures from results/*.jsonl.  Writes results/metrics.json and report/fig_*.png"""
import json, collections
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import clients

ROOT = Path(__file__).resolve().parent.parent
RES, FIG = ROOT / "results", ROOT / "report"; FIG.mkdir(exist_ok=True)
SYS = list(clients.SYSTEMS)
KIND = {s: clients.SYSTEMS[s][0] for s in SYS}
SHORT = {s: s.removesuffix(" (LLM)") for s in SYS}                     # chart labels
C = {"decision": "#2a78d6", "llm": "#eb6834"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
LANGS = {"en": "English", "hi": "Hindi", "hx": "Hinglish"}
rng = np.random.default_rng(1)          # seeded: bootstrap CIs and chart jitter are reproducible

THRESHOLD = 0.5      # a yes/no probability >= 0.5 counts as "yes" (the default cut-off for both kinds of system)
BOOTSTRAP_N = 2000   # resamples for 95% confidence intervals
LAT_CLIP_S = 4       # latency chart x-axis limit; the few slower calls (network stalls) are drawn at the edge
CALIB_EDGES = [0, .1, .3, .5, .7, .9, 1.0001]   # probability bins for the calibration chart (finer at the ends)
ACC_AXIS = (80, 101)  # accuracy chart x-range (%): every score and CI is above 80, so this zooms in on the differences
CALIB_MIN_N = 8      # skip calibration bins with fewer points than this; too noisy to plot

plt.rcParams.update({"font.family": "Segoe UI", "font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "figure.dpi": 200, "savefig.bbox": "tight"})


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


turns = {t["id"]: t for f in ("turns.jsonl", "turns_hard.jsonl") for t in load(ROOT / "data" / f)}


def correct(r, t):
    """per-run correctness of the four decisions (threshold 0.5 on predicates)"""
    return dict(intent=r["intent"] in t["intent_ok"], esc=(r["p_esc"] >= THRESHOLD) == bool(t["esc"]),
                done=(r["p_done"] >= THRESHOLD) == bool(t["done"]), reply=r["reply"] == t["reply"])


def auroc(y, p):
    y, p = np.asarray(y), np.asarray(p)
    pos, neg = p[y == 1], p[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def ece(y, p, bins=10):
    y, p = np.asarray(y, float), np.asarray(p, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def boot_ci(per_scen, n=BOOTSTRAP_N):
    """cluster bootstrap over scenarios. per_scen: array (n_scenarios,) of means"""
    a = np.asarray(per_scen)
    m = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def evaluate(runs, label):
    by = collections.defaultdict(list)
    fail = collections.Counter(); total = collections.Counter()
    for r in runs:
        total[r["system"]] += 1
        if not r["ok"]:
            fail[r["system"]] += 1; continue
        by[r["system"]].append(r)
    M = {}
    for s in SYS:
        rs = by[s]
        if not rs:
            continue
        m = dict(n_runs=len(rs), fail_rate=fail[s] / max(total[s], 1))
        cor = [(r, turns[r["id"]], correct(r, turns[r["id"]])) for r in rs]
        for task in ("intent", "esc", "done", "reply"):
            m[f"acc_{task}"] = float(np.mean([c[task] for _, _, c in cor]))
            for lg in LANGS:
                m[f"acc_{task}_{lg}"] = float(np.mean([c[task] for _, t, c in cor if t["lang"] == lg]))
            sc = collections.defaultdict(list)
            for r, t, c in cor:
                sc[t["scenario"]].append(c[task])
            m[f"ci_{task}"] = boot_ci([np.mean(v) for v in sc.values()])
        m["acc_all4"] = float(np.mean([all(c.values()) for _, _, c in cor]))
        for task, key in (("esc", "p_esc"), ("done", "p_done")):
            y = [turns[r["id"]][task] for r in rs]; p = [r[key] for r in rs]
            m[f"auroc_{task}"], m[f"ece_{task}"] = auroc(y, p), ece(y, p)
            m[f"brier_{task}"] = float(np.mean((np.array(p) - np.array(y)) ** 2))
        em = [r for r in rs if turns[r["id"]]["intent"] == "emergency"]
        m["emerg_recall"] = float(np.mean([r["p_esc"] >= THRESHOLD for r in em]))
        m["emerg_missed"] = sorted({r["id"] for r in em if r["p_esc"] < THRESHOLD})
        inc = [r for r in rs if turns[r["id"]]["done"] == 0]
        cmp_ = [r for r in rs if turns[r["id"]]["done"] == 1]
        m["cut_in_rate"] = float(np.mean([r["p_done"] >= THRESHOLD for r in inc]))      # replies while caller still talking
        m["false_wait_rate"] = float(np.mean([r["p_done"] < THRESHOLD for r in cmp_]))  # silence after a finished turn
        nesc = [r for r in rs if turns[r["id"]]["esc"] == 0]
        m["false_escalation_rate"] = float(np.mean([r["p_esc"] >= THRESHOLD for r in nesc]))
        # consistency across reps
        g = collections.defaultdict(list)
        for r in rs:
            g[r["id"]].append(r)
        multi = [v for v in g.values() if len(v) > 1]
        if multi:
            m["consistency"] = float(np.mean([len({(r["intent"], r["p_esc"] >= THRESHOLD, r["p_done"] >= THRESHOLD, r["reply"]) for r in v}) == 1 for v in multi]))
            m["prob_sd"] = float(np.mean([np.std([r["p_esc"] for r in v]) + np.std([r["p_done"] for r in v]) for v in multi]) / 2)
        lat = np.array([r["latency_s"] for r in rs])
        m.update(lat_med=float(np.median(lat)), lat_p90=float(np.percentile(lat, 90)),
                 lat_p99=float(np.percentile(lat, 99)), lat_mean=float(lat.mean()))
        ps = [r["provider_s"] for r in rs if r.get("provider_s")]
        if ps:
            m["provider_med"] = float(np.median(ps))
        m["cost_per_1k"] = float(np.mean([r.get("cost_usd", 0) for r in rs]) * 1000)
        m["in_tokens"] = float(np.mean([r.get("in_tokens") or 0 for r in rs]))
        M[s] = m
    return M


def per_turn_acc(runs, task):
    """mean correctness per (system, turn id)"""
    d = collections.defaultdict(list)
    for r in runs:
        if r["ok"]:
            d[(r["system"], r["id"])].append(correct(r, turns[r["id"]])[task])
    return {k: np.mean(v) for k, v in d.items()}


# ------------------------------------------------------------------ figures
def rows(ax, labels):
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels); ax.invert_yaxis(); ax.grid(axis="y", visible=False)


def fig_latency(runs, path):
    fig, ax = plt.subplots(figsize=(7.2, 2.9))
    for i, s in enumerate(SYS):
        lat = np.array([r["latency_s"] for r in runs if r["ok"] and r["system"] == s])
        jit = rng.uniform(-0.18, 0.18, len(lat))
        ax.scatter(np.clip(lat, 0, LAT_CLIP_S), i + jit, s=4, color=C[KIND[s]], alpha=0.25, linewidths=0)
        q = np.percentile(lat, [50, 90])
        ax.plot([q[0]] * 2, [i - .32, i + .32], color=INK, lw=2, solid_capstyle="round")
        ax.plot([q[1]] * 2, [i - .22, i + .22], color=INK2, lw=1.2)
        ax.text(min(q[1], LAT_CLIP_S - 0.3) + 0.06, i - 0.3, f"median {q[0]*1000:.0f} ms · p90 {q[1]*1000:.0f} ms", fontsize=7.5, color=INK2)
    rows(ax, [SHORT[s] for s in SYS])
    ax.set_xlim(0, LAT_CLIP_S)
    ax.set_xlabel(f"Wall-clock latency per call, all four decisions (s) — values above {LAT_CLIP_S} s shown at {LAT_CLIP_S} s")
    fig.savefig(path); plt.close(fig)


def fig_accuracy(M, path, n_scen):
    tasks = [("intent", "Intent (7-way)"), ("esc", "Escalate to human"), ("done", "Turn finished?"), ("reply", "Reply language")]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.6), sharey=True)
    for ax, (t, name) in zip(axs, tasks):
        for i, s in enumerate(SYS):
            if s not in M: continue
            lo, hi = M[s][f"ci_{t}"]; a = M[s][f"acc_{t}"]
            ax.plot([lo * 100, hi * 100], [i, i], color=C[KIND[s]], lw=1.6, alpha=.45, solid_capstyle="round")
            ax.scatter([a * 100], [i], s=34, color=C[KIND[s]], zorder=3, edgecolors="white", linewidths=1.5)
            ax.text(a * 100, i - 0.32, f"{a*100:.0f}", ha="center", fontsize=7, color=INK2)
        ax.set_title(name, fontsize=8.5, color=INK, loc="left"); ax.set_xlim(*ACC_AXIS)
        ax.set_xticks([80, 85, 90, 95, 100])
    rows(axs[0], [SHORT[s] for s in SYS])
    axs[0].set_ylim(len(SYS) - 0.5, -0.9)   # headroom so the value labels clear the panel titles
    fig.supxlabel(f"Accuracy % (dot) with 95% CI, bootstrapped over the {n_scen} scenarios", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_lang(M, path):
    tasks = [("intent", "Intent"), ("esc", "Escalate"), ("done", "Turn finished?")]
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.6), sharey=True)
    marks = {"en": "o", "hi": "s", "hx": "D"}
    for ax, (t, name) in zip(axs, tasks):
        for i, s in enumerate(SYS):
            vals = [M[s][f"acc_{t}_{lg}"] * 100 for lg in LANGS]
            ax.plot([min(vals), max(vals)], [i, i], color=GRID, lw=3, solid_capstyle="round", zorder=1)
            for (lg, lname), v in zip(LANGS.items(), vals):
                ax.scatter([v], [i], marker=marks[lg], s=26, color=C[KIND[s]], edgecolors="white", linewidths=1, zorder=3,
                           )
        ax.set_title(name, fontsize=8.5, loc="left"); ax.set_xlim(*ACC_AXIS)
    rows(axs[0], [SHORT[s] for s in SYS])
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker=marks[lg], ls="", color=INK2, label=n) for lg, n in LANGS.items()]
    fig.legend(handles=h, loc="upper right", ncol=3, frameon=False, fontsize=7.5, bbox_to_anchor=(1, 1.05))
    fig.supxlabel("Accuracy % per caller language (same 114 scenarios in each language)", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_calib(runs, path):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.1))
    for ax, (task, key, name) in zip(axs, [("esc", "p_esc", "Escalate to human"), ("done", "p_done", "Turn finished?")]):
        ax.plot([0, 1], [0, 1], color=GRID, lw=1)
        for s in SYS:
            rs = [r for r in runs if r["ok"] and r["system"] == s]
            y = np.array([turns[r["id"]][task] for r in rs]); p = np.array([r[key] for r in rs])
            edges = np.array(CALIB_EDGES); idx = np.digitize(p, edges) - 1
            xs, ys = [], []
            for b in range(len(edges) - 1):
                if (idx == b).sum() >= CALIB_MIN_N:
                    xs.append(p[idx == b].mean()); ys.append(y[idx == b].mean())
            ls = "-" if KIND[s] == "decision" else "--"
            ax.plot(xs, ys, ls, marker="o", ms=3.5, lw=1.4, color=C[KIND[s]], label=SHORT[s],
                    alpha=1 if KIND[s] == "decision" else .75)
        ax.set_title(name, fontsize=8.5, loc="left"); ax.set_xlabel("Stated probability (binned)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    axs[0].set_ylabel("Observed frequency")
    axs[1].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_cost(M, path):
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    for s in SYS:
        m = M[s]
        ax.scatter([m["cost_per_1k"]], [m["lat_med"] * 1000], s=46, color=C[KIND[s]], edgecolors="white", linewidths=1.5, zorder=3)
        ax.annotate(f"{SHORT[s]}\n${m['cost_per_1k']:.3f} / 1k turns", (m["cost_per_1k"], m["lat_med"] * 1000),
                    xytext=(7, -2), textcoords="offset points", fontsize=7.2, color=INK2, va="center")
    ax.set_xscale("log"); ax.set_xlim(0.01, 5)
    ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3]); ax.set_xticklabels(["$0.01", "$0.03", "$0.10", "$0.30", "$1", "$3"])
    ax.minorticks_off()
    ax.set_xlabel("Cost per 1,000 caller turns, all four decisions (USD, log scale, billed by gateway)")
    ax.set_ylabel("Median latency (ms)"); ax.set_ylim(0, None)
    fig.savefig(path); plt.close(fig)


def fig_asr(Mt, Ma, path):
    tasks = [("intent", "Intent"), ("esc", "Escalate"), ("done", "Turn finished?"), ("reply", "Reply language")]
    fig, axs = plt.subplots(1, 4, figsize=(7.4, 2.5), sharey=True)
    for ax, (t, name) in zip(axs, tasks):
        for i, s in enumerate(SYS):
            if s not in Ma: continue
            a, b = Mt[s][f"acc_{t}"] * 100, Ma[s][f"acc_{t}"] * 100
            ax.annotate("", (b, i), (a, i), arrowprops=dict(arrowstyle="-|>", color=C[KIND[s]], lw=1.3, shrinkA=0, shrinkB=2))
            ax.scatter([a], [i], s=18, color="white", edgecolors=C[KIND[s]], linewidths=1.2, zorder=3)
        ax.set_title(name, fontsize=8.5, loc="left"); ax.set_xlim(*ACC_AXIS)
    rows(axs[0], [SHORT[s] for s in SYS])
    fig.supxlabel("Accuracy %: typed transcript (hollow) → after TTS + phone line + speech recognition (arrow head)", fontsize=8, color=INK2)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_budget(B, path):
    """stacked horizontal bars: one voice turn, serial pipeline, measured medians.
    B: list of (label, kind, [(segment name, seconds)])"""
    seg_col = {"Speech-to-text": "#c3c2b7", "Decision": None, "Reply LLM first token": "#8f8e88", "TTS first audio": "#d9d8d2"}
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(B) + 0.9))
    for i, (label, kind, segs) in enumerate(B):
        x = 0
        for name, v in segs:
            col = C[kind] if name == "Decision" else seg_col[name]
            ax.barh(i, v, left=x, height=0.55, color=col, edgecolor="white", linewidth=2)
            if v > 0.28:   # only label segments wide enough to hold the number
                ax.text(x + v / 2, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if name == "Decision" else INK)
            x += v
        ax.text(x + 0.05, i, f"{x:.2f} s", va="center", fontsize=8, color=INK)
    rows(ax, [b[0] for b in B])
    ax.set_xlabel("Seconds from end of caller speech to first agent audio (measured medians, serial pipeline)")
    from matplotlib.patches import Patch
    hs = [Patch(color=seg_col["Speech-to-text"], label="Speech-to-text (ElevenLabs Scribe)"),
          Patch(color=C["decision"], label="Decision: decision model"), Patch(color=C["llm"], label="Decision: LLM"),
          Patch(color=seg_col["Reply LLM first token"], label="Reply LLM first token"),
          Patch(color=seg_col["TTS first audio"], label="TTS first audio (ElevenLabs Flash)")]
    ax.legend(handles=hs, frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.45, -0.2))
    ax.set_xlim(0, None); ax.grid(axis="y", visible=False)
    fig.savefig(path); plt.close(fig)


def round3(x):
    """round every float to 3 decimals before saving, so results files stay readable"""
    if isinstance(x, float): return round(x, 3)
    if isinstance(x, dict): return {k: round3(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [round3(v) for v in x]
    return x


def budget_rows(asr_turns, M, budget):
    """one voice turn = speech-to-text + decision + reply-LLM first token + TTS first audio (all measured medians)"""
    stt = float(np.median([t["stt_s"] for t in asr_turns]))
    tts = budget["tts"]["en"]["ttfb"]["median"]
    reply = budget["llm_reply"]["Gemini 3.1 Flash Lite (LLM)"]["ttft"]["median"]   # same reply model in every row
    return [(f"{SHORT[s]} decides", KIND[s], [("Speech-to-text", stt), ("Decision", M[s]["lat_med"]),
                                               ("Reply LLM first token", reply), ("TTS first audio", tts)])
            for s in SYS]


if __name__ == "__main__":
    text, hard, asr = (load(RES / f) for f in ("text_runs.jsonl", "hard_runs.jsonl", "asr_runs.jsonl"))
    both = text + hard
    Mt, Mh, Mall, Ma = evaluate(text, "text"), evaluate(hard, "hard"), evaluate(both, "all"), evaluate(asr, "asr")
    # voice comparison is like with like: the same 342 turns, first text repetition vs their ASR transcripts
    Mt0 = evaluate([r for r in both if r["rep"] == 0], "text-rep0")
    out = {"text": Mt, "hard": Mh, "all": Mall, "asr": Ma, "text_rep0": Mt0}
    fig_latency(both, FIG / "fig_latency.png")
    fig_accuracy(Mt, FIG / "fig_accuracy.png", 80)
    fig_accuracy(Mh, FIG / "fig_accuracy_hard.png", 34)
    fig_lang(Mall, FIG / "fig_lang.png")
    fig_calib(both, FIG / "fig_calib.png")
    fig_cost(Mall, FIG / "fig_cost.png")
    fig_asr(Mt0, Ma, FIG / "fig_asr.png")
    asr_turns = load(ROOT / "data" / "asr_turns.jsonl") + load(ROOT / "data" / "asr_turns_hard.jsonl")
    budget = json.loads((RES / "latency_budget.json").read_text())
    rows_ = budget_rows(asr_turns, Mall, budget)
    out["voice_turn"] = {r[0]: dict(r[2]) | {"total": sum(v for _, v in r[2])} for r in rows_}
    fig_budget(rows_, FIG / "fig_budget.png")
    # turns each system got wrong in most repetitions (used for the report's error examples)
    pt = {t: per_turn_acc(both, t) for t in ("intent", "esc", "done", "reply")}
    out["errors_by_system"] = {t: {s: sorted(i for (ss, i), v in pt[t].items() if ss == s and v < .5) for s in SYS} for t in pt}
    (RES / "metrics.json").write_text(json.dumps(round3(out), indent=1, ensure_ascii=False), encoding="utf-8")
    for s, m in Mt.items():
        print(f"{SHORT[s]:22s} int {m['acc_intent']:.3f} esc {m['acc_esc']:.3f} done {m['acc_done']:.3f} rep {m['acc_reply']:.3f} "
              f"all4 {m['acc_all4']:.3f} | aucE {m['auroc_esc']:.3f} aucD {m['auroc_done']:.3f} eceE {m['ece_esc']:.3f} eceD {m['ece_done']:.3f} "
              f"| emerg {m['emerg_recall']:.2f} cut-in {m['cut_in_rate']:.2f} fEsc {m['false_escalation_rate']:.2f} "
              f"| cons {m.get('consistency', 0):.2f} | lat {m['lat_med']:.2f}/{m['lat_p90']:.2f}/{m['lat_p99']:.2f} "
              f"| ${m['cost_per_1k']:.3f}/1k fail {m['fail_rate']:.3f}")
