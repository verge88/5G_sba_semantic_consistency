# Frozen pre-run protocol: Isolation Forest and two-channel 5G SBA anomaly detection

**Date fixed:** 2026-10-09. **Branch:** experiment/sba-if-two-channel-20261009. This protocol is committed to GitHub *before the corresponding GitHub Actions workflow starts*.

## Motivation and scope

The preceding nonlearned per-event TOKEN evidence counterexample showed that leave-one-trust-origin-out minimum scores can erase a real one-origin anomaly. The next experiment checks whether this problem persists with **actually trained**, normal-only Isolation Forest models and whether a separate **jointly calibrated two-channel decision** recovers sensitivity.

This is a narrowly scoped event-level synthetic falsification trial. It does **not** establish novel Data Mining theory, does **not** validate full 5G attack-family detection, and must **not** be compared numerically to v9's end-to-end event or window recall.

## Exact arms (locked)

- **if_raw:** Isolation Forest trained on full eligible benign TOKEN evidence feature vectors.
- **if_min_loo:** Separate Isolation Forest trained on benign origin-masked augmentations; for a test case score every one-origin exclusion and use the **minimum** anomaly score.
- **if_two_channel:** Take the *larger* of the channel surprisal values -log(p_raw), -log(p_loo), then rank-calibrate that **joint statistic** on a separate set of held-out benign cases. This accounts empirically for dependence between channels; the score is not presumed independent.
- **q3:** Unlearned 3-of-5 inconsistent-origin count reference.

Both trained models use 128 Isolation Forest trees, max_samples=256 and contamination="auto". Deterministic estimator random seeds are derived from the world seed.

## Fixed synthetic data and partitions

- Five fresh world seeds: **221, 222, 223, 224, 225** (not used in prior v9 52–71 or previous min-LOO 111–115 runs).
- Two *observable* context strata: normal_indirect and diurnal_peak.
- In **each world/stratum**, select the first event with at least three CONSISTENT TOKEN reports among the five predefined trust origins. The same eligibility rule applies in every partition. Reject ineligible windows without using attack labels.
- Four **disjoint sequential partitions**: 400 benign training; 300 benign per-channel score-rank calibration; 300 benign fusion-statistic calibration; 250 held-out evaluation events **per seed/context**.
- All source data comes from the corrected simulator, NORMAL operational state, false operational markers disabled. Neither attack labels nor latent operational state appear in the feature extractor.
- One report per origin, by latest observed_at, with state CONSISTENT, INCONSISTENT or UNKNOWN; actual missingness is represented separately from UNKNOWN.
- Feature vector: per-origin one-hot observed state, missing indicator, LOO excluded indicator, reported confidence and relative timing, plus aggregate inconsistent/unknown/available fractions and a binary multi-origin inconsistency feature.
- All raw and LOO training examples are benign. The LOO estimator sees each source excluded once from each training example.
- Models and both calibration partitions are **frozen before any intervention** on evaluation cases.

## Test conditions and metrics

For each held-out evaluation event, construct four *paired counterfactuals*: 0, 1, 2, or 3 originally CONSISTENT reports switched to INCONSISTENT, with no other modifications. The 1-origin case is within the one-compromised-origin trust model. Cases with 2–3 modified origins are stress tests, not covered by that threat model.

Nominal threshold alpha=**0.01 (1%)** for all statistical arms; the q3 comparator uses its 3-of-5 count. Rank p-values use (1 + #calibration_scores >= observed)/(N+1), retaining ties. The two-channel arm undergoes **independent held-out calibration** of its *joint* fusion score. No independence assumption between its components.

Primary descriptive endpoints:
- per-seed and pooled empirical false-alarm rate at 0 changes;
- per-seed and pooled intervention detection rates at 1/2/3 changed origins;
- paired raw-vs-two-channel discordant cases;
- descriptive Wilson 95% intervals and calibration/fitting audit;
- locked plan digest and GitHub run artifact.

No hyperparameter tuning based on the evaluation data; results from this run are exploratory, not preregistered confirmatory inference. The fact that alpha=1% does **not** establish a 0.1% dissertation target.

## Important limitations

- TOKEN-specific synthetic intervention in observations, **not an actual malicious SBA signaling trace**; results are not full 5G SBA attack Recall.
- All evaluations use normal operational state; therefore no claim about token-refresh, policy-sync, recovery, or route-transition FPR.
- Eligibility (>=3 CONSISTENT reports) is an observable but selected subset; results need not generalize to all events.
- Dependence among windows from one simulator and conditional selection may violate exchangeability and reduce effective sample size; Wilson intervals are **descriptive, not inferential**.
- Two-channel fusion can increase thresholds due to multiple testing; no guarantee of improvement over raw IF.
- No proof of scientific novelty, production FPR, or resistance to arbitrary adaptive Byzantine attackers.
