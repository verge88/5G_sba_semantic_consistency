"""Exploratory trained Isolation Forest experiment on 5G SBA semantic evidence.

Compare a normal-only trained raw anomaly model, a model trained on
leave-one-trust-origin-out augmented normal samples, and a two-channel fusion.
All decision inputs come from observations; attack and latent state labels are
used ONLY in evaluation-world construction, not feature extraction.

Scientific limits: synthetic event-level TOKEN evidence interventions, no
network-level causal attack simulation, dependent evaluation windows, and no
claimed conditional conformal guarantee or production FPR.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from bisect import bisect_left
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from run_semantic_loo_falsification import (  # noqa: E402
    CONTEXTS, ORIGINS, choose_target, intervene, wilson_interval,
)
from sba_semantic_v8 import (  # noqa: E402
    EvidenceState, OperationalState, SemanticSbaSimulator,
)

ARMS = ("if_raw", "if_min_loo", "if_two_channel", "q3")
SCENARIOS = (0, 1, 2, 3)
N_TREES = 128
MAX_SAMPLES = 256


def features(reports, excluded=None):
    """Fixed-dimensional trust-provenance features, with no simulator oracle.

    The excluded-origin flag is explicit and is present in the LOO model's
    benign training distribution. UNKNOWN and absent reports are separate.
    """
    visible = [r for o, r in reports.items() if o != excluded]
    t0 = min((float(r.observed_at) for r in visible), default=0.0)
    out = []
    bad = unknown = available = 0
    for origin in ORIGINS:
        report = reports.get(origin)
        hidden = origin == excluded
        if hidden:
            out.extend([0., 0., 0., 0., 1., 0., 0.])
            continue
        if report is None:
            out.extend([0., 0., 0., 1., 0., 0., 0.])
            unknown += 1
            continue
        state = report.state
        good = state == EvidenceState.CONSISTENT
        inconsistent = state == EvidenceState.INCONSISTENT
        unk = state == EvidenceState.UNKNOWN
        available += int(not unk)
        bad += int(inconsistent)
        unknown += int(unk)
        # Latency is relative to the first visible report, not simulation
        # time. The data-dependent scaling is frozen and label-free.
        lag = math.log1p(max(0.0, float(report.observed_at) - t0))
        out.extend([
            float(good), float(inconsistent), float(unk), 0., 0.,
            float(max(0., min(1., report.confidence))),
            float(min(lag, 6.0) / 6.0),
        ])
    denominator = max(1, len(ORIGINS) - int(excluded is not None))
    out.extend([
        bad / denominator,
        unknown / denominator,
        available / denominator,
        float(bad >= 2),
    ])
    return out


def rank_p(sorted_calibration, score):
    """Conservative >= tail rank; valid only under relevant exchangeability."""
    n = len(sorted_calibration)
    if n <= 0:
        raise ValueError("rank calibration must be non-empty")
    return (1.0 + n - bisect_left(sorted_calibration, float(score))) / (n + 1.0)


def batch_scores(raw_model, loo_model, reports):
    """Model scores are oriented so larger means more anomalous."""
    if not reports:
        return np.empty(0), np.empty(0)
    full = np.asarray([features(r) for r in reports], dtype=float)
    masked = np.asarray(
        [features(r, excluded=origin) for r in reports for origin in ORIGINS],
        dtype=float,
    )
    raw = -raw_model.score_samples(full)
    masks = (-loo_model.score_samples(masked)).reshape(len(reports), len(ORIGINS))
    return raw, masks.min(axis=1)


def make_data(sim, context, n, start_wid):
    """Take first event with at least 3 consistent reports (label-free)."""
    collected = []
    wid = start_wid
    max_attempts = 20 * n
    while len(collected) < n and wid - start_wid < max_attempts:
        w = sim.generate_window(
            wid, context=context,
            operational_state=OperationalState.NORMAL,
            false_marker=False,
        )
        wid += 1
        candidate = choose_target(w)
        if candidate is not None:
            collected.append(candidate)
    if len(collected) != n:
        raise RuntimeError(
            f"Only {len(collected)} eligible cases in {wid - start_wid} attempts"
        )
    return collected, wid


def fit_normal_only(seed, train_cases):
    normal = [r for r, _ in train_cases]
    if len(normal) < MAX_SAMPLES:
        raise ValueError("insufficient training data")
    raw_x = np.asarray([features(r) for r in normal], dtype=float)
    masked_x = np.asarray([
        features(r, excluded=origin)
        for r in normal for origin in ORIGINS
    ], dtype=float)
    common = dict(n_estimators=N_TREES, max_samples=MAX_SAMPLES,
                  contamination="auto", n_jobs=2)
    raw_model = IsolationForest(random_state=seed + 160_001, **common).fit(raw_x)
    loo_model = IsolationForest(random_state=seed + 160_003, **common).fit(masked_x)
    return raw_model, loo_model


def fused_stat(score_raw, score_loo, ctx, score_reference):
    r_p = rank_p(score_reference[ctx]["raw"], score_raw)
    l_p = rank_p(score_reference[ctx]["loo"], score_loo)
    return -math.log(min(r_p, l_p))


def run_seed(seed, train_n, rank_n, joint_n, eval_n, alpha):
    sim = SemanticSbaSimulator(seed=seed)
    data = {}
    wid = 0
    for ctx in CONTEXTS:
        n = train_n + rank_n + joint_n + eval_n
        cases, wid = make_data(sim, ctx, n, wid)
        data[ctx] = {
            "train": cases[:train_n],
            "rank": cases[train_n:train_n + rank_n],
            "joint": cases[train_n + rank_n:train_n + rank_n + joint_n],
            "eval": cases[train_n + rank_n + joint_n:],
        }
    train = [row for ctx in CONTEXTS for row in data[ctx]["train"]]
    raw_model, loo_model = fit_normal_only(seed, train)

    # The first calibration partition ranks each channel. The second,
    # independent partition calibrates their joint, dependent statistic.
    score_reference = {}
    for ctx in CONTEXTS:
        cal = [r for r, _ in data[ctx]["rank"]]
        r, l = batch_scores(raw_model, loo_model, cal)
        score_reference[ctx] = {
            "raw": sorted(r.tolist()), "loo": sorted(l.tolist()),
        }

    joint_reference = {}
    for ctx in CONTEXTS:
        cal = [r for r, _ in data[ctx]["joint"]]
        r, l = batch_scores(raw_model, loo_model, cal)
        joint_reference[ctx] = sorted([
            fused_stat(ri, li, ctx, score_reference)
            for ri, li in zip(r, l)
        ])

    rows = []
    audits = {"seed": seed, "raw_training_cases": len(train),
              "loo_training_rows": len(train) * len(ORIGINS),
              "rank_calibration_each_context": rank_n,
              "joint_calibration_each_context": joint_n,
              "evaluation_each_context": eval_n,
              "last_generated_window_id": wid}
    # Ground-truth intervention is performed only after all model fitting
    # and all threshold calculations are finalized.
    for ctx in CONTEXTS:
        cases = data[ctx]["eval"]
        worlds = []
        for n_changes in SCENARIOS:
            for reports, consistent in cases:
                worlds.append(intervene(reports, consistent, n_changes))
        r, l = batch_scores(raw_model, loo_model, worlds)
        for i_changes, n_changes in enumerate(SCENARIOS):
            counts = {a: 0 for a in ARMS}
            agree = {"raw_only": 0, "joint_only": 0, "both": 0, "neither": 0}
            for j in range(eval_n):
                idx = i_changes * eval_n + j
                raw_p = rank_p(score_reference[ctx]["raw"], r[idx])
                loo_p = rank_p(score_reference[ctx]["loo"], l[idx])
                joint_score = -math.log(min(raw_p, loo_p))
                joint_p = rank_p(joint_reference[ctx], joint_score)
                n_bad = sum(
                    int(z.state == EvidenceState.INCONSISTENT)
                    for z in worlds[idx].values()
                )
                fired = {
                    "if_raw": raw_p < alpha,
                    "if_min_loo": loo_p < alpha,
                    "if_two_channel": joint_p < alpha,
                    "q3": n_bad >= 3,
                }
                for arm in ARMS:
                    counts[arm] += int(fired[arm])
                key = ("both" if fired["if_raw"] and fired["if_two_channel"]
                       else "raw_only" if fired["if_raw"]
                       else "joint_only" if fired["if_two_channel"] else "neither")
                agree[key] += 1
            for arm in ARMS:
                low, high = wilson_interval(counts[arm], eval_n)
                rows.append({
                    "seed": seed, "context": ctx,
                    "injected_origins": n_changes, "arm": arm,
                    "alerts": counts[arm], "n": eval_n,
                    "alert_rate": counts[arm] / eval_n,
                    "wilson95_lo": low, "wilson95_hi": high,
                    "paired_raw_only": agree["raw_only"],
                    "paired_joint_only": agree["joint_only"],
                })
    return rows, audits


def save_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+",
                   default=[221, 222, 223, 224, 225])
    p.add_argument("--train", type=int, default=400)
    p.add_argument("--rank-calib", type=int, default=300)
    p.add_argument("--joint-calib", type=int, default=300)
    p.add_argument("--eval", type=int, default=250)
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--out-dir",
                   default="runs/sba_if_two_channel/221_225")
    args = p.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("unique seed list required")
    if not (0.0 < args.alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if min(args.train, args.rank_calib, args.joint_calib, args.eval) <= 0:
        raise ValueError("partition sizes must be positive")
    if args.train * len(CONTEXTS) < MAX_SAMPLES:
        raise ValueError("too few training cases")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    plan = {
        "design": "trained IF, raw vs min trust-origin LOO vs calibrated two-channel",
        "status": "preregistered_exploratory_synthetic",
        "seeds": args.seeds, "contexts": list(CONTEXTS),
        "training_only": "eligible benign event observations, no attack labels",
        "feature_origin_order": [o.value for o in ORIGINS],
        "feature_schema": "per-origin onehot state+missing+masked, confidence, lag; aggregate counts",
        "train_per_seed_context": args.train,
        "rank_per_seed_context": args.rank_calib,
        "joint_per_seed_context": args.joint_calib,
        "evaluation_per_seed_context": args.eval,
        "interventions": list(SCENARIOS),
        "alpha": args.alpha,
        "trees": N_TREES, "max_samples": MAX_SAMPLES,
        "models": "two separate IFs: full-benign and masked-benign augmentation",
        "calibration": "context-stratified conservative >= tail rank; independent fusion calibration",
        "fusion": "max of -log per-channel empirical p-values; joint rank calibration",
        "claims": "no exchangeability theorem; no production FPR or 0.1% target",
    }
    raw_plan = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw_plan.encode("utf-8")).hexdigest()
    locked_file = out / "analysis_plan.json"
    if locked_file.exists():
        prior = json.loads(locked_file.read_text(encoding="utf-8"))
        if prior.get("sha256") != digest:
            raise RuntimeError("analysis plan mismatch; choose fresh out dir")
    locked_file.write_text(json.dumps(
        {"sha256": digest, **plan}, ensure_ascii=False, indent=2
    ), encoding="utf-8")

    all_rows, audits = [], []
    for seed in args.seeds:
        rows, audit = run_seed(
            seed, args.train, args.rank_calib, args.joint_calib,
            args.eval, args.alpha,
        )
        all_rows.extend(rows)
        audits.append(audit)
        print(f"SEED {seed} DONE: {len(rows)} rows / {audit['last_generated_window_id']} generated windows", flush=True)
    fields = ["seed", "context", "injected_origins", "arm",
              "alerts", "n", "alert_rate", "wilson95_lo", "wilson95_hi",
              "paired_raw_only", "paired_joint_only"]
    save_csv(out / "per_seed.csv", all_rows, fields)
    pooled_counts = defaultdict(lambda: [0, 0])
    for row in all_rows:
        for scope in (row["context"], "ALL"):
            acc = pooled_counts[(scope, row["injected_origins"], row["arm"])]
            acc[0] += row["alerts"]
            acc[1] += row["n"]
    pooled = []
    for (ctx, n_changes, arm), (hits, n) in sorted(pooled_counts.items()):
        lo, hi = wilson_interval(hits, n)
        pooled.append({
            "context": ctx, "injected_origins": n_changes, "arm": arm,
            "alerts": hits, "n": n, "alert_rate": hits / n,
            "wilson95_lo": lo, "wilson95_hi": hi,
        })
    save_csv(out / "aggregate.csv", pooled,
             ["context", "injected_origins", "arm", "alerts", "n",
              "alert_rate", "wilson95_lo", "wilson95_hi"])
    summary = {
        "analysis_plan_sha256": digest,
        "git_sha": os.getenv("GITHUB_SHA", "unknown"),
        "audits": audits,
        "pooled": [r for r in pooled if r["context"] == "ALL"],
        "interpretation_limits": [
            "Event-level TOKEN-origin interventions; no natural SBA attack-family recall.",
            "Normal states only: operational-transition FPR not evaluated.",
            "Source-exclusion model trained on synthetic source-masked benign data.",
            "Worlds within each seed are correlated; Wilson intervals descriptive only.",
            "Conditional conformal exchangeability not established.",
            "The nominal alpha=1% is not the dissertation target 0.1%.",
            "Two-channel min-p fusion is not an independence test and may lose power.",
            "No theorem or scientific novelty established by this experiment alone.",
        ],
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "analysis_plan_sha256": digest,
        "seeds": args.seeds,
        "pooled": summary["pooled"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
