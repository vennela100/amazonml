"""Stage 5: Train CatBoost entity resolution classifier.

Reads:
  - reports/stage4/features.npz  (X, y, s1_ids, cand_ids, feature_names)

Trains:
  - CatBoost binary classifier on the train split
  - Entity-grouped cross-validation (5-fold) to avoid leakage
  - Calibrated probabilities via isotonic regression

Writes:
  - reports/stage5/model.cbm         (CatBoost model)
  - reports/stage5/calibrator.pkl    (isotonic calibrator)
  - reports/stage5/cv_results.json   (per-fold metrics)
  - reports/stage5/report.md

Precision-first design:
  - Optimise F0.5 (precision double-weighted over recall)
  - Use entity-grouped splits so all pairs for one S1 entity
    are always in the same fold (prevents leakage)
  - Report threshold sweep on validation fold

Memory: ~400 MB for the feature matrix (N pairs × 50 features × float32)
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import fbeta_score
from sklearn.model_selection import GroupKFold

try:
    from catboost import CatBoostClassifier, Pool as CatPool
except ImportError:
    raise SystemExit('catboost not installed: pip install catboost')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
log = logging.getLogger(__name__)


# ── Metrics ──────────────────────────────────────────────────────────────────
# The competition metric is set-based per S1 entity over the full queried
# population, computed by the exact Stage 1 scorer. It must NOT be approximated
# on the candidate-pair table: doing so hides singletons (a true singleton
# scores 1.0 for an empty prediction, 0.0 for any match) and cannot see true
# matches lost during retrieval. All scoring here goes through evaluation.py.
import sys
sys.path.insert(0, str(Path(__file__).parent))
import evaluation


def population_sweep(population, truth_map, s1_ids, cand_ids, proba,
                     thresholds=None) -> dict:
    """Population-aware threshold sweep via the exact scorer.

    ``population`` and ``truth_map`` define the full evaluation scope, so missed
    matches and singletons are charged correctly. Ties break toward the higher
    (more precise) threshold, matching the F0.5 precision weighting.
    """
    if thresholds is None:
        # Fine grid: F0.5 is precision-weighted and its optimum often sits between
        # coarse steps (e.g. 0.86, not 0.85). Denser near 1.0 where singletons
        # are decided. A coarse 0.05 grid measurably underscored the model.
        thresholds = np.round(np.arange(0.30, 0.991, 0.01), 4).tolist()
    return evaluation.sweep_thresholds(
        list(population), truth_map, list(s1_ids), list(cand_ids),
        list(proba), thresholds)


# ── Grouped CV split ──────────────────────────────────────────────────────────

def make_entity_groups(s1_ids: np.ndarray) -> np.ndarray:
    """Convert string S1 IDs to integer group codes for GroupKFold."""
    unique_ids, inverse = np.unique(s1_ids, return_inverse=True)
    return inverse  # integer group per sample


# ── Feature loading ───────────────────────────────────────────────────────────

def load_features(npz_path: str | Path) -> dict:
    """Load a Stage 4 .npz, including the queried population and complete truth.

    Returns a dict with the pair arrays (X, y, s1_ids, cand_ids, feature_names)
    and the scoring scope (population, population_roles, truth_map). The last
    three are required to score with the exact competition metric.
    """
    # allow_pickle: the .npz holds string object arrays and is our own Stage 4
    # output (locally generated, trusted), not third-party input.
    data = np.load(npz_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y'].astype(np.int8)
    s1_ids = data['s1_ids'].astype(str)
    cand_ids = data['cand_ids'].astype(str)
    feature_names = list(data['feature_names'].astype(str))
    if 'population' not in data:
        raise ValueError(
            f'{npz_path}: missing population/truth arrays. Rebuild with the '
            f'current Stage 4 (it persists population, population_roles, '
            f'population_truth).')
    population = [str(x) for x in data['population']]
    roles = [str(x) for x in data['population_roles']]
    truth_map = {}
    for row in data['population_truth']:
        s1_id, ids = str(row).split('\t')
        truth_map[s1_id] = set(ids.split(',')) if ids else set()
    log.info(f'Loaded features: X={X.shape}, positives={int(y.sum()):,} '
             f'({100*y.mean():.2f}%), features={len(feature_names)}, '
             f'population={len(population):,}')
    return {
        'X': X, 'y': y, 's1_ids': s1_ids, 'cand_ids': cand_ids,
        'feature_names': feature_names, 'population': population,
        'roles': roles, 'truth_map': truth_map,
    }


# ── CatBoost training ─────────────────────────────────────────────────────────

def build_catboost(n_estimators: int = 2000, depth: int = 6,
                   learning_rate: float = 0.05,
                   class_weights: tuple = (1.0, 5.0)) -> CatBoostClassifier:
    """Build CatBoost classifier tuned for precision-heavy F0.5."""
    return CatBoostClassifier(
        iterations=n_estimators,
        depth=depth,
        learning_rate=learning_rate,
        loss_function='Logloss',
        eval_metric='F:beta=0.5',
        class_weights=list(class_weights),  # [negative_weight, positive_weight]
        early_stopping_rounds=100,
        random_seed=42,
        thread_count=-1,
        verbose=200,
    )


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    s1_ids: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    catboost_kwargs: dict | None = None,
) -> tuple[CatBoostClassifier, list[dict], np.ndarray]:
    """Entity-grouped K-fold CV. Returns best model, fold metrics, OOF probabilities."""
    groups = make_entity_groups(s1_ids)
    gkf = GroupKFold(n_splits=n_folds)

    fold_metrics = []
    oof_proba = np.zeros(len(y), dtype=np.float32)
    kwargs = catboost_kwargs or {}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        log.info(f'\n── Fold {fold_idx + 1}/{n_folds} '
                 f'(train={len(train_idx):,}, val={len(val_idx):,}) ──')

        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        pos_tr = int(y_tr.sum())
        log.info(f'  Train: {pos_tr:,} pos, {len(y_tr)-pos_tr:,} neg '
                 f'({100*pos_tr/len(y_tr):.2f}%)')

        model = build_catboost(**kwargs)
        train_pool = CatPool(X_tr, label=y_tr, feature_names=feature_names)
        val_pool = CatPool(X_val, label=y_val, feature_names=feature_names)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        # Out-of-fold probabilities feed the population-aware sweep in run().
        oof_proba[val_idx] = model.predict_proba(X_val)[:, 1]

        # Pair-level diagnostic only (NOT the decision metric): does the model
        # separate positives from negatives within candidates it was given?
        pair_f05 = fbeta_score(y_val, (oof_proba[val_idx] >= 0.5).astype(int),
                               beta=0.5, zero_division=0)
        fold_metrics.append({
            'fold': fold_idx + 1,
            'n_train': int(len(train_idx)),
            'n_val': int(len(val_idx)),
            'pair_f05_at_0.5_diag': round(float(pair_f05), 4),
            'n_iterations': int(model.best_iteration_),
        })
        log.info(f'  Fold {fold_idx+1}: pair-F0.5@0.5(diag)={pair_f05:.4f}, '
                 f'iters={model.best_iteration_}')

    return fold_metrics, oof_proba


def render_report(fold_metrics, oof_best, val_best, decision, feature_importance,
                  path: Path) -> None:
    src = decision['threshold_source']
    lines = [
        '# Stage 5: CatBoost Classifier', '',
        '**Status**: PASS', '',
        '## Decision threshold', '',
        f'- Selected threshold: **{decision["threshold"]}** (from {src})',
        f'- Scored at inference on the SAME scale (raw model probability); no '
        f'calibrator sits between the threshold and inference.', '',
        '## Held-out validation (full population, exact scorer)', '',
    ]
    if val_best is not None:
        lines += [
            f'- Macro F0.5: **{val_best["macro_f05"]:.4f}** at threshold '
            f'{val_best["threshold"]}',
            f'- False-singleton rate: {val_best["false_singleton_rate"]}',
            f'- Missed non-singleton rate: {val_best["missed_non_singleton_rate"]}',
            '',
        ]
    else:
        lines += ['- (no held-out validation npz supplied; used OOF)', '']
    lines += [
        '## Out-of-fold (training population, exact scorer)', '',
        f'- Macro F0.5: **{oof_best["macro_f05"]:.4f}** at threshold '
        f'{oof_best["threshold"]}', '',
        '## Per-fold diagnostics (pair-level, not the decision metric)', '',
        '| Fold | Train pairs | Val pairs | pair-F0.5@0.5 | Iters |',
        '|---|---:|---:|---:|---:|',
    ]
    for m in fold_metrics:
        lines.append(f'| {m["fold"]} | {m["n_train"]:,} | {m["n_val"]:,} | '
                     f'{m["pair_f05_at_0.5_diag"]} | {m["n_iterations"]} |')
    lines += [
        '', '## Top 20 Feature Importances (by PredValuesChange)', '',
        '| Rank | Feature | Importance |', '|---|---|---:|',
    ]
    for rank, (feat, imp) in enumerate(
            sorted(feature_importance.items(), key=lambda x: -x[1])[:20], 1):
        lines.append(f'| {rank} | {feat} | {imp:.2f} |')
    path.write_text('\n'.join(lines), encoding='utf-8')


def run(args):
    started = time.monotonic()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    train = load_features(args.features_npz)
    X, y = train['X'], train['y']
    s1_ids, cand_ids = train['s1_ids'], train['cand_ids']
    feature_names = train['feature_names']
    log.info(f'Training on {len(y):,} labeled pairs '
             f'over {len(train["population"]):,} S1 entities')

    catboost_kwargs = {
        'n_estimators': args.n_estimators,
        'depth': args.depth,
        'learning_rate': args.learning_rate,
        'class_weights': (1.0, args.pos_weight),
    }

    # Grouped CV produces out-of-fold probabilities without entity leakage.
    fold_metrics, oof_proba = train_with_cv(
        X, y, s1_ids, feature_names,
        n_folds=args.n_folds, catboost_kwargs=catboost_kwargs)

    # OOF sweep over the full training population (diagnostic, no leakage).
    log.info('OOF population sweep (training population)...')
    oof_sweep = population_sweep(
        train['population'], train['truth_map'], s1_ids, cand_ids, oof_proba)
    oof_best = oof_sweep['best']
    log.info(f'OOF best: threshold={oof_best["threshold"]}, '
             f'macro_F0.5={oof_best["macro_f05"]:.4f}')

    # Final model trained on ALL training pairs (used at inference).
    log.info('Fitting final model on all training pairs...')
    final_model = build_catboost(**catboost_kwargs)
    final_model.fit(CatPool(X, label=y, feature_names=feature_names), verbose=False)

    # Select the decision threshold on the HELD-OUT validation population if
    # given. The threshold is chosen on raw model probability, exactly the scale
    # used at inference — no calibrator between selection and application.
    val_best = None
    if args.val_npz:
        val = load_features(args.val_npz)
        val_proba = final_model.predict_proba(val['X'])[:, 1]
        val_sweep = population_sweep(
            val['population'], val['truth_map'], val['s1_ids'], val['cand_ids'],
            val_proba)
        val_best = val_sweep['best']
        log.info(f'Held-out validation best: threshold={val_best["threshold"]}, '
                 f'macro_F0.5={val_best["macro_f05"]:.4f}')
        threshold = val_best['threshold']
        threshold_source = 'held-out validation population'
    else:
        threshold = oof_best['threshold']
        threshold_source = 'out-of-fold training population'

    model_path = report_dir / 'model.cbm'
    final_model.save_model(str(model_path))

    decision = {'threshold': threshold, 'threshold_source': threshold_source,
                'scale': 'raw_model_probability'}
    (report_dir / 'decision.json').write_text(
        json.dumps(decision, indent=2), encoding='utf-8')

    feat_imp = dict(zip(feature_names, final_model.get_feature_importance().tolist()))
    elapsed = time.monotonic() - started
    (report_dir / 'cv_results.json').write_text(json.dumps({
        'fold_metrics': fold_metrics, 'oof_sweep': oof_sweep,
        'val_sweep': (val_sweep if val_best is not None else None),
        'decision': decision, 'feature_importance': feat_imp,
        'elapsed_seconds': elapsed,
    }, indent=2), encoding='utf-8')
    render_report(fold_metrics, oof_best, val_best, decision, feat_imp,
                  report_dir / 'report.md')

    log.info(f'\nSTAGE 5 PASS: {report_dir / "report.md"}  (elapsed {elapsed:.0f}s)')
    print('\nSTAGE 5 PASS')
    print(f'  Decision threshold: {threshold} (from {threshold_source})')
    if val_best is not None:
        print(f'  Held-out macro-F0.5: {val_best["macro_f05"]:.4f}')
    print(f'  OOF macro-F0.5: {oof_best["macro_f05"]:.4f}')
    print(f'  Model: {model_path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features-npz', default='reports/stage4/features.npz',
                        help='Training-role Stage 4 features')
    parser.add_argument('--val-npz', default=None,
                        help='Held-out validation-role Stage 4 features for '
                             'threshold selection (recommended)')
    parser.add_argument('--report-dir', default='reports/stage5')
    parser.add_argument('--n-folds', type=int, default=5)
    parser.add_argument('--n-estimators', type=int, default=2000)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='Positive class weight (higher favors recall)')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
