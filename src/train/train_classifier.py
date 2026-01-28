# src/train/train_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


@dataclass(frozen=True)
class ClassifierTrainResult:
    model_path: str
    best_params: Dict[str, Any]
    class_names: List[str]
    val_acc: float
    val_f1_macro: float
    test_acc: float
    test_f1_macro: float

def _fit_maybe_weighted(clf, X, y, sample_w):
    """
    Fit classifier with sample_weight if supported by this sklearn version.
    Falls back to unweighted fit otherwise.
    """
    try:
        clf.fit(X, y, sample_weight=sample_w)
        return True
    except TypeError:
        clf.fit(X, y)
        return False

def _draw_params(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
    sizes = [(64,), (128,), (256,), (128, 64), (256, 128), (256, 128, 64)]
    batches = [16, 32, 64, 128]
    out = []
    for _ in range(n):
        out.append(
            {
                "hidden_layer_sizes": sizes[rng.randint(len(sizes))],
                "alpha": 10 ** rng.uniform(-5, np.log10(3e-2)),  # ~loguniform(1e-5, 3e-2)
                "learning_rate_init": 10 ** rng.uniform(-4, np.log10(3e-3)),  # ~loguniform(1e-4, 3e-3)
                "batch_size": batches[rng.randint(len(batches))],
                "activation": "relu",
                "solver": "adam",
                "max_iter": 800,
                "early_stopping": False,  # IMPORTANT: use all training samples
                "random_state": 42,
                "verbose": False,
                "tol": 1e-4,
            }
        )
    return out


def _cv_macro_f1(
    *,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    sample_w: np.ndarray,
    params: Dict[str, Any],
    cv: StratifiedGroupKFold,
) -> Tuple[float, float]:
    scores: List[float] = []
    for tr_idx, va_idx in cv.split(X, y, groups):
        clf = MLPClassifier(**params)
        _fit_maybe_weighted(clf, X[tr_idx], y[tr_idx], sample_w[tr_idx])
        pred = clf.predict(X[va_idx])
        scores.append(float(f1_score(y[va_idx], pred, average="macro")))
    return float(np.mean(scores)), float(np.std(scores))


def train_mlp_classifier_with_search(
    *,
    exp_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    sample_w: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Sequence[str],
    out_dir: Path,
    n_iter: int = 50,
    n_splits: int = 5,
    seed: int = 42,
    verbose: bool = True,
    save_name: str | None = None,  # default: clf_{exp_name}.joblib
) -> ClassifierTrainResult:
    """
    Random-search over MLPClassifier configs using:
      - grouped CV (StratifiedGroupKFold) for stability scoring
      - selection by external VAL macro-F1 (tie: VAL acc, then CV mean)
    Then refit best model on FULL TRAIN and evaluate on VAL + TEST.
    Saves classifier artifact to out_dir.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ensure float64
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    sample_w = np.asarray(sample_w, dtype=np.float64)
    groups_train = np.asarray(groups_train).astype(str)

    cn = [str(x) for x in class_names]

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.RandomState(seed)

    params_list = _draw_params(rng, n_iter)

    candidates: List[Dict[str, Any]] = []
    if verbose:
        print(f"\nSearching {n_iter} MLP configs...")

    for i, params in enumerate(params_list, 1):
        cv_mean, cv_std = _cv_macro_f1(
            X=X_train, y=y_train, groups=groups_train, sample_w=sample_w, params=params, cv=cv
        )

        clf_full = MLPClassifier(**params)
        _fit_maybe_weighted(clf_full, X_train, y_train, sample_w)

        val_pred = clf_full.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred))

        candidates.append(
            {
                "params": params,
                "cv_mean": cv_mean,
                "cv_std": cv_std,
                "val_f1": val_f1,
                "val_acc": val_acc,
            }
        )

        if verbose:
            print(
                f"[{i:02d}/{n_iter}] cvF1={cv_mean:.3f}±{cv_std:.3f} | "
                f"VAL F1={val_f1:.3f} acc={val_acc:.3f} | "
                f"{params['hidden_layer_sizes']}, α={params['alpha']:.2e}, "
                f"lr={params['learning_rate_init']:.1e}, bs={params['batch_size']}"
            )

    candidates.sort(key=lambda c: (c["val_f1"], c["val_acc"], c["cv_mean"]), reverse=True)
    best = candidates[0]
    best_params = best["params"]

    if verbose:
        print("\n=== Top candidates (by VAL macro-F1) ===")
        for c in candidates[:5]:
            print(
                f"VAL F1={c['val_f1']:.3f} (acc={c['val_acc']:.3f}) | "
                f"cvF1={c['cv_mean']:.3f}±{c['cv_std']:.3f} | params={c['params']}"
            )
        print("\n🏆 Selected params:")
        print(best_params)

    final_clf = MLPClassifier(**best_params)
    _fit_maybe_weighted(final_clf, X_train, y_train, sample_w)

    # eval VAL
    val_hat = final_clf.predict(X_val)
    val_acc = float(accuracy_score(y_val, val_hat))
    val_f1m = float(f1_score(y_val, val_hat, average="macro"))

    # eval TEST
    test_hat = final_clf.predict(X_test)
    test_acc = float(accuracy_score(y_test, test_hat))
    test_f1m = float(f1_score(y_test, test_hat, average="macro"))

    if verbose:
        for name, ys, yhat in [("VAL", y_val, val_hat), ("TEST", y_test, test_hat)]:
            acc = float(accuracy_score(ys, yhat))
            f1m = float(f1_score(ys, yhat, average="macro"))
            print(f"\n===== {name} =====")
            print(f"{name}: acc={acc:.4f}  f1_macro={f1m:.4f}")
            print(classification_report(ys, yhat, labels=np.arange(len(cn)), target_names=cn, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(ys, yhat))

    model_name = save_name or f"clf_{exp_name}.joblib"
    model_path = out_dir / model_name

    joblib.dump(
        {
            "model": final_clf,
            "class_names": cn,
            "best_params": best_params,
            "search_top5": candidates[:5],
        },
        model_path,
    )

    if verbose:
        print(f"\n✅ Saved classifier to: {model_path}")

    return ClassifierTrainResult(
        model_path=str(model_path),
        best_params=best_params,
        class_names=cn,
        val_acc=val_acc,
        val_f1_macro=val_f1m,
        test_acc=test_acc,
        test_f1_macro=test_f1m,
    )
