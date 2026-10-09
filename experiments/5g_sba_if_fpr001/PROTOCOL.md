# Fixed research protocol: nominal FPR 0.1%, sub-quorum and suppression

Frozen on 2026-10-09, before this GitHub Actions run.
Target repository: verge88/5G_sba_semantic_consistency.
Branch: experiment/sba-if-fpr001-subquorum-20261009.

## Motivation

Prior five-world trained Isolation Forest (IF) experiment at nominal alpha=1%
returned empirical FPR 0.96% raw IF, 1.24% IF min-LOO, and 0.64%
independently calibrated two-channel IF. The new experiment asks whether any
method retains useful recall in the **0.1% nominal-FPR tail**, whether origin
suppressions defeat it, and how it behaves on a benign operational transition
and actual (synthetic) NO_TOKEN-family attack event.

This experiment is a NEW exploratory study, not a redefinition of old endpoints.
Do not tune models after outcomes. Do not claim novelty, production FPR, or
conditional coverage.

## Frozen train/calibration/evaluation protocol

- Fresh independent RNG roots / worlds: **301, 302, 303, 304, 305**.
- Two observable contexts: normal_indirect and diurnal_peak.
- Per seed AND context: **300** first-eligible benign windows for training;
  **2,500** disjoint benign windows for per-channel rank calibration;
  **2,500** disjoint benign windows for fusion-statistic rank calibration;
  **2,500** held-out benign windows for evaluation.
- Five seeds x two contexts x 2,500 = **25,000 held-out benign
  event samples**. They are dependent within each seed, NOT 25,000 IID
  statistical units. They are drawn as one first-eligible event per window.
- First-eligible event: at least three consistent TOKEN reports among the
  predefined five origins, using only observable evidence.
- Scoring: same fixed feature extraction and the same model configurations
  as previous experiment: raw-benign IF, masked-benign IF with min over
  five leave-one-origin-out scores, and two-channel max(-log individual
  conformal-rank p) calibrated on independent fusion calibration data.
  Deterministic seeds per model. The previous branch is the comparator base.
- All 3 IF-related arms use **nominal alpha = 0.001** on separate, held-out
  benign score ranks, with conservative ties. Empirical FPR is a measured
  outcome, not forced to 0.001; q3of5 has no tunable alpha.
- No hyperparameter, contamination, threshold, or feature retuning after
  looking at the held-out evaluation.
- A finite rank calibration of 2,500 per context has coarse resolution near
  alpha=0.001 (minimum achievable p=1/2501). This and time dependence
  restrict any exact FPR claim.

## Frozen evaluation scenarios

Each normal held-out event is cloned into 7 paired cases; the original
report set is never mutated:
1. normal: no intervention;
2. flip1: change one CONSISTENT source to INCONSISTENT;
3. flip2: change two such distinct sources;
4. flip3: change three such distinct sources (outside single-origin model);
5. flip1_suppress1: change one source, remove a different observed source;
6. flip2_suppress1: change two, remove a different source;
7. flip1_unknown1: change one, mark a different consistent source UNKNOWN.

The target origins are rotated deterministically by event index and seed.
No UNKNOWN observation may ever be promoted to INCONSISTENT. Source
suppression removes the report, **not** creates concrete negative evidence.

Two additional **enriched evaluation-only** cohorts per seed:
- 250 benign TOKEN_REFRESH operational-state windows, selecting the event
  actually disturbed by the generator, eligible only with >=3 consistent
  TOKEN reports after disturbance. This selection uses simulator truth
  ONLY for constructing stress strata and cannot be used to build the
  deployed detector's input. Report its false-alarm rate separately from
  the primary normal FPR.
- 250 NO_TOKEN attacked windows, selecting a known attack event with at
  least one TOKEN report. This uses attack-event truth ONLY to evaluate a
  targeted synthetic attack event, not to construct a deployed alert.
  This is not end-to-end attack-window Recall.

Fixed summary metrics: per-seed and pooled alarm rates and counts, Wilson
descriptive 95% intervals, eligibility/attempt audits, plan SHA256, and
raw/fusion/masked method comparisons.

## Limits and interpretation

- Calibration is stratum-specific per seed, and p-values assume comparable
  calibration and evaluation data; conditional distribution-free coverage is
  NOT proved.
- Only five seed-level independent RNG roots; event-level Wilson intervals
  are **descriptive**, not statistical evidence across networks.
- FPR 0.1% can only be called *empirically observed* if holdout results support
  it; no frequentist guarantee under network drift or temporal dependence.
- NO_TOKEN and TOKEN_REFRESH event selection uses oracle truth for **evaluation
  panel selection**, not feature extraction; cannot imply a deployable detector
  achieves those targeted-event metrics automatically.
- Real 5GC Open5GS trace monitoring and cross-NF provenance remain out of
  scope. Further validation is mandatory before a thesis-grade claim.
