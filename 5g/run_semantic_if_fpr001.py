"""Held-out 0.1%-nominal FPR / subquorum / suppression 5G SBA experiment.

Research-only validation of three normal-only trained Isolation Forest arms.
Requires 3 disjoint benign time-ordered partitions before a held-out fourth.
The two-channel score is calibrated *as a joint score*, not as two independent
p-values. Target 0.1% is nominal: empirical FPR must be reported separately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "5g"))

from run_semantic_if_two_channel import (
    ARMS, CONTEXTS, ORIGINS, batch_scores, features, fit_normal_only,
    fused_stat, make_data, rank_p, wilson_interval,
)
from run_semantic_loo_falsification import (
    choose_target, event_reports,
)
from sba_semantic_v8 import (
    AttackFamily, EvidenceState, OperationalState, SemanticSbaSimulator,
)

TARGET_FPR = 0.001
SCENARIOS = ("normal", "flip1", "flip2", "flip3", "flip1_suppress1",
             "flip2_suppress1", "flip1_unknown1", "operational_token_refresh",
             "attack_no_token")
RESEARCH_ARMS = ("if_raw", "if_min_loo", "if_two_channel", "q3")


def make_world(reports, consistent, scenario, rotation):
    """Paired report mutations; never fabricate evidence from UNKNOWN.

    Rotate eligible origins deterministically by sample position, to avoid
    concentrating all interventions on a single provenance domain.
    """
    out = dict(reports)
    available = list(consistent)
    offset = rotation % len(available)
    choices = available[offset:] + available[:offset]
    flips = {"normal": 0, "flip1": 1, "flip2": 2, "flip3": 3,
             "flip1_suppress1": 1, "flip2_suppress1": 2,
             "flip1_unknown1": 1}
    if scenario not in flips:
        raise ValueError(scenario)
    count = flips[scenario]
    if len(choices) < max(3, count + int("suppress" in scenario or "unknown" in scenario)):
        raise ValueError("insufficient consistent sources")
    for origin in choices[:count]:
        assert out[origin].state == EvidenceState.CONSISTENT
        out[origin] = replace(out[origin], state=EvidenceState.INCONSISTENT)
    if "suppress" in scenario:
        suppressed = choices[count]
        del out[suppressed]
    if "unknown" in scenario:
        unknown = choices[count]
        out[unknown] = replace(out[unknown], state=EvidenceState.UNKNOWN)
    for origin, obs in reports.items():
        if obs.state == EvidenceState.UNKNOWN:
            assert origin in out and out[origin].state == EvidenceState.UNKNOWN
    return out


def choose_special_window(window, kind):
    """Ground-truth selection is used ONLY to build an explicitly enriched
    validation stratum; *never* to construct the feature vector or thresholds.
    """
    if kind == "operational_token_refresh":
        candidates = sorted(window.operational_event_ids)
        for eid in candidates:
            reports = event_reports(window, eid)
            if sum(o.state == EvidenceState.CONSISTENT for o in reports.values()) >= 3:
                return reports
    elif kind == "attack_no_token":
        for eid in sorted(window.attack_event_ids):
            reports = event_reports(window, eid)
            if len(reports) >= 1:
                return reports
    else:
        raise ValueError(kind)
    return None


def special_cases(seed, kind, n):
    """Fresh generation stream independent of training/calibration/evaluation."""
    sim = SemanticSbaSimulator(seed=seed + (50_000 if kind == "operational_token_refresh" else 90_000))
    found, attempted = [], 0
    while len(found) < n and attempted < 40 * n:
        if kind == "operational_token_refresh":
            w = sim.generate_window(
                1_000_000 + attempted,
                context=CONTEXTS[attempted % len(CONTEXTS)],
                operational_state=OperationalState.TOKEN_REFRESH,
                false_marker=False,
            )
        else:
            w = sim.generate_window(
                2_000_000 + attempted,
                context=CONTEXTS[attempted % len(CONTEXTS)],
                attack_family=AttackFamily.NO_TOKEN,
                hidden_calls=5,
                operational_state=OperationalState.NORMAL,
                false_marker=False,
            )
        attempted += 1
        candidate = choose_special_window(w, kind)
        if candidate is not None:
            # Scoring context is observable; event selection via simulator
            # labels is analysis-only and not a deployable online method.
            found.append((candidate, w.context))
    if len(found) < n:
        raise RuntimeError(f"{kind}: only {len(found)} of {n} eligible in {attempted}")
    return found, attempted


def bools_from_scores(raw, loo, ctx, rank_reference, fusion_reference, alpha):
    p_raw = rank_p(rank_reference[ctx]["raw"], raw)
    p_loo = rank_p(rank_reference[ctx]["loo"], loo)
    fused = -math.log(min(p_raw, p_loo))
    p_fused = rank_p(fusion_reference[ctx], fused)
    return p_raw < alpha, p_loo < alpha, p_fused < alpha


def rate_row(seed, ctx, scenario, arm, hits, n):
    lo, hi = wilson_interval(hits, n)
    return {
        "seed": seed, "context": ctx, "scenario": scenario,
        "arm": arm, "alerts": hits, "n": n,
        "rate": hits / n if n else float("nan"),
        "wilson95_lo": lo, "wilson95_hi": hi,
    }


def run_seed(seed, train_n, rank_n, joint_n, eval_n, op_n, attack_n, alpha):
    sim = SemanticSbaSimulator(seed=seed)
    data = {}
    wid = 0
    for ctx in CONTEXTS:
        n = train_n + rank_n + joint_n + eval_n
        cases, wid = make_data(sim, ctx, n, wid)
        data[ctx] = {
            "train": cases[:train_n],
            "rank": cases[train_n:train_n+rank_n],
            "joint": cases[train_n+rank_n:train_n+rank_n+joint_n],
            "eval": cases[train_n+rank_n+joint_n:],
        }
    normal_training = [case for ctx in CONTEXTS for case in data[ctx]["train"]]
    raw_model, loo_model = fit_normal_only(seed, normal_training)

    rank_reference = {}
    for ctx in CONTEXTS:
        raw, loo = batch_scores(
            raw_model, loo_model, [case for case, _ in data[ctx]["rank"]]
        )
        rank_reference[ctx] = {"raw": sorted(raw.tolist()),
                               "loo": sorted(loo.tolist())}
    fusion_reference = {}
    for ctx in CONTEXTS:
        raw, loo = batch_scores(
            raw_model, loo_model, [case for case, _ in data[ctx]["joint"]]
        )
        fusion_reference[ctx] = sorted([
            fused_stat(s1, s2, ctx, rank_reference)
            for s1, s2 in zip(raw, loo)
        ])

    rows = []
    scenarios = SCENARIOS[:7]
    for ctx in CONTEXTS:
        pairs = data[ctx]["eval"]
        # Evaluate identical counterfactual base events across all arms.
        for scenario in scenarios:
            cases = [
                make_world(reports, consistent, scenario, rotation=k+seed)
                for k, (reports, consistent) in enumerate(pairs)
            ]
            r, l = batch_scores(raw_model, loo_model, cases)
            tally = {arm: 0 for arm in RESEARCH_ARMS}
            for reports, raw, loo in zip(cases, r, l):
                flags = bools_from_scores(
                    raw, loo, ctx, rank_reference, fusion_reference, alpha
                )
                n_bad = sum(o.state == EvidenceState.INCONSISTENT
                            for o in reports.values())
                detections = dict(zip(RESEARCH_ARMS[:3], flags))
                detections["q3"] = n_bad >= 3
                for arm, fired in detections.items():
                    tally[arm] += int(fired)
            rows.extend(rate_row(seed, ctx, scenario, arm, tally[arm], len(cases))
                        for arm in RESEARCH_ARMS)

    audits = {"seed": seed, "generated_normal_windows": wid,
              "training_cases": train_n * len(CONTEXTS),
              "rank_calibration_per_context": rank_n,
              "fusion_calibration_per_context": joint_n,
              "evaluation_per_context": eval_n,
              "special_generation_attempts": {}}
    for scenario, count in [
        ("operational_token_refresh", op_n), ("attack_no_token", attack_n)
    ]:
        cases, attempted = special_cases(seed, scenario, count)
        audits["special_generation_attempts"][scenario] = attempted
        for ctx in CONTEXTS:
            sub = [reports for reports, observable_ctx in cases
                   if observable_ctx == ctx]
            if not sub:
                continue
            r, l = batch_scores(raw_model, loo_model, sub)
            hits = {arm: 0 for arm in RESEARCH_ARMS}
            for reports, s_raw, s_loo in zip(sub, r, l):
                flags = bools_from_scores(
                    s_raw, s_loo, ctx, rank_reference, fusion_reference, alpha
                )
                n_bad = sum(o.state == EvidenceState.INCONSISTENT
                            for o in reports.values())
                decisions = dict(zip(RESEARCH_ARMS[:3], flags))
                decisions["q3"] = n_bad >= 3
                for arm, fired in decisions.items():
                    hits[arm] += int(fired)
            rows.extend(
                rate_row(seed, ctx, scenario, arm, hits[arm], len(sub))
                for arm in RESEARCH_ARMS
            )
    return rows, audits


def csv_write(path, records, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(records)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[301,302,303,304,305])
    p.add_argument("--train", type=int, default=300)
    p.add_argument("--rank-calib", type=int, default=2500)
    p.add_argument("--joint-calib", type=int, default=2500)
    p.add_argument("--eval", type=int, default=2500)
    p.add_argument("--operational", type=int, default=250)
    p.add_argument("--attacks", type=int, default=250)
    p.add_argument("--target-fpr", type=float, default=TARGET_FPR)
    p.add_argument("--out-dir", default="runs/sba_if_fpr001_subquorum/301_305")
    args = p.parse_args(argv)
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("duplicate seeds")
    if min(args.train, args.rank_calib, args.joint_calib,
           args.eval, args.operational, args.attacks) <= 0:
        raise ValueError("all sample sizes must be positive")
    if not (0 < args.target_fpr < 1):
        raise ValueError("target FPR must be strictly between zero and one")
    if min(args.rank_calib, args.joint_calib) * args.target_fpr < 2:
        raise ValueError("calibration too small for the requested tail alpha")
    if args.train * len(CONTEXTS) < 256:
        raise ValueError("insufficient IF training examples")

    plan = {
        "experiment": "prospective nominal FPR 0.1%, source-suppression and enriched context",
        "status": "fixed_exploratory_not_confirmatory",
        "seeds": args.seeds,
        "contexts": list(CONTEXTS),
        "train_per_context": args.train, "rank_calibration_per_context": args.rank_calib,
        "fusion_calibration_per_context": args.joint_calib,
        "independent_evaluation_per_context": args.eval,
        "operational_case_count_per_seed": args.operational,
        "attack_case_count_per_seed": args.attacks,
        "target_fpr": args.target_fpr, "scenarios": list(SCENARIOS),
        "arms": list(RESEARCH_ARMS),
        "features": "trained normal-only IF, provenance one-hot + confidence + relative lag",
        "fusion": "min single-channel rank p; held-out joint score rank p",
        "warning": "rank thresholds do not imply exact empirical FPR",
        "selection": "first event >=3 consistent; operational and attacked event panels truth-selected for evaluation ONLY",
        "inferential_unit": "seed (dependent windows within each seed); no pooled IID claim",
    }
    raw = json.dumps(plan, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    locked = out / "analysis_plan.json"
    if locked.exists() and json.loads(locked.read_text("utf-8")).get("sha256") != digest:
        raise RuntimeError("locked plan changed in same output dir")
    locked.write_text(json.dumps({"sha256": digest, **plan}, indent=2), encoding="utf-8")

    all_rows, audits = [], []
    for seed in args.seeds:
        rows, audit = run_seed(
            seed, args.train, args.rank_calib, args.joint_calib,
            args.eval, args.operational, args.attacks, args.target_fpr,
        )
        all_rows.extend(rows)
        audits.append(audit)
        brief = {
            r["arm"]: round(r["rate"], 5) for r in rows
            if r["context"] == CONTEXTS[0] and r["scenario"] == "flip1"
        }
        print(f"SEED {seed} DONE; {audit['generated_normal_windows']} normal worlds; flip1={brief}", flush=True)
    columns = ["seed","context","scenario","arm","alerts","n","rate",
               "wilson95_lo","wilson95_hi"]
    csv_write(out / "per_seed.csv", all_rows, columns)
    aggregate = defaultdict(lambda: [0,0])
    for row in all_rows:
        for scope in (row["context"], "ALL"):
            agg = aggregate[(scope, row["scenario"], row["arm"])]
            agg[0] += row["alerts"]
            agg[1] += row["n"]
    pooled = []
    for (ctx, scenario, arm), (hits, total) in sorted(aggregate.items()):
        pooled.append(rate_row("ALL", ctx, scenario, arm, hits, total))
    csv_write(out / "aggregate.csv", pooled, columns)
    summary = {
        "analysis_plan_sha256": digest,
        "git_sha": os.environ.get("GITHUB_SHA","unknown"),
        "world_seeds": args.seeds,
        "audits": audits,
        "pooled": [r for r in pooled if r["context"] == "ALL"],
        "limits": [
            "Nominal alpha=0.001; empirical FPR and its seed dependence must be examined.",
            "Only TOKEN fact, synthetic 1/2/3-origin mutations; not real attack family recall.",
            "Operational transient and NO_TOKEN attack event panels ground-truth-selected for evaluation, not deployable end-to-end detector.",
            "Only TOKEN_REFRESH operational state; not full 5G SBA state coverage.",
            "Each seed contains many dependent windows; pooled Wilson CI not an inferential guarantee.",
            "A family of four correlated detector decisions is evaluated; do not infer a novel theorem.",
            "Production 0.1% FPR and novelty cannot be claimed from this experiment.",
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({
        "analysis_plan_sha256": digest,
        "pooled": summary["pooled"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
