# src/train/train_classifier.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier

from imgofup.config.constants import (
    CLS_SEARCH_N_ITER_DEFAULT,
    CLS_SEARCH_N_SPLITS_DEFAULT,
    CLS_SEARCH_SEED_DEFAULT,
    CLS_SEARCH_VERBOSE_DEFAULT,
    CLS_MODEL_NAME_TEMPLATE,
    CLS_SEARCH_TOPK,
)

from imgofup.models.utils._classifier_utils import (
    candidate_sort_key,
    cv_macro_f1,
    draw_mlp_params,
    fit_maybe_weighted,
)


@dataclass(frozen=True)
class ClassifierTrainResult:
    model_path: str
    best_params: Dict[str, Any]
    class_names: List[str]
    val_acc: float
    val_f1_macro: float
    test_acc: float
    test_f1_macro: float


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
    n_iter: int = CLS_SEARCH_N_ITER_DEFAULT,
    n_splits: int = CLS_SEARCH_N_SPLITS_DEFAULT,
    seed: int = CLS_SEARCH_SEED_DEFAULT,
    verbose: bool = CLS_SEARCH_VERBOSE_DEFAULT,
    save_name: str | None = None,
) -> ClassifierTrainResult:
    """
    Random-search over MLPClassifier configs using:
      - grouped CV (StratifiedGroupKFold) for stability scoring
      - selection by external VAL macro-F1 (tie: VAL acc, then CV mean)
    Then refit best model on FULL TRAIN and evaluate on VAL + TEST.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # cast & validate shapes
    X_train = np.asarray(X_train, dtype=np.float64)
    X_val = np.asarray(X_val, dtype=np.float64)
    X_test = np.asarray(X_test, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=int)
    y_val = np.asarray(y_val, dtype=int)
    y_test = np.asarray(y_test, dtype=int)
    sample_w = np.asarray(sample_w, dtype=np.float64)
    groups_train = np.asarray(groups_train).astype(str)

    if not (len(X_train) == len(y_train) == len(sample_w) == len(groups_train)):
        raise ValueError("TRAIN arrays must have matching lengths: X_train, y_train, sample_w, groups_train.")
    if X_val.shape[1] != X_train.shape[1] or X_test.shape[1] != X_train.shape[1]:
        raise ValueError("X_val/X_test must have the same number of columns as X_train.")
    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    cn = [str(x) for x in class_names]
    n_classes = len(cn)
    if n_classes < 2:
        raise ValueError("Need at least 2 classes for classification.")
    if (y_train < 0).any() or (y_val < 0).any() or (y_test < 0).any():
        raise ValueError("Found negative class labels. Check label encoding vs fixed_classes.")
    if y_train.max() >= n_classes or y_val.max() >= n_classes or y_test.max() >= n_classes:
        raise ValueError("Class labels exceed number of class_names.")

    n_groups = len(np.unique(groups_train))
    if n_groups < n_splits:
        raise ValueError(f"Not enough groups for StratifiedGroupKFold: groups={n_groups}, n_splits={n_splits}.")

    cv = StratifiedGroupKFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    rng = np.random.RandomState(int(seed))
    params_list = draw_mlp_params(rng, int(n_iter), random_state=int(seed))

    candidates: List[Dict[str, Any]] = []
    if verbose:
        print(f"\nSearching {n_iter} MLP configs...")

    for i, params in enumerate(params_list, 1):
        cv_mean, cv_std = cv_macro_f1(
            X=X_train, y=y_train, groups=groups_train, sample_w=sample_w, params=params, cv=cv
        )

        clf_full = MLPClassifier(**params)
        fit_maybe_weighted(clf_full, X_train, y_train, sample_w)

        val_pred = clf_full.predict(X_val)
        val_f1 = float(f1_score(y_val, val_pred, average="macro"))
        val_acc = float(accuracy_score(y_val, val_pred))

        candidates.append(
            {"params": params, "cv_mean": cv_mean, "cv_std": cv_std, "val_f1": val_f1, "val_acc": val_acc}
        )

        if verbose:
            print(
                f"[{i:02d}/{n_iter}] cvF1={cv_mean:.3f}±{cv_std:.3f} | "
                f"VAL F1={val_f1:.3f} acc={val_acc:.3f} | "
                f"{params['hidden_layer_sizes']}, α={params['alpha']:.2e}, "
                f"lr={params['learning_rate_init']:.1e}, bs={params['batch_size']}"
            )

    candidates.sort(key=candidate_sort_key, reverse=True)
    best = candidates[0]
    best_params = best["params"]

    if verbose:
        print("\n🏆 Selected params:", best_params)
        print(f"\n=== Top {int(CLS_SEARCH_TOPK)} candidates ===")
        for c in candidates[: int(CLS_SEARCH_TOPK)]:
            print(
                f"VAL F1={c['val_f1']:.3f} acc={c['val_acc']:.3f} | "
                f"cvF1={c['cv_mean']:.3f}±{c['cv_std']:.3f}"
            )

    final_clf = MLPClassifier(**best_params)
    fit_maybe_weighted(final_clf, X_train, y_train, sample_w)

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
            print(f"\n===== {name} =====")
            print(f"{name}: acc={accuracy_score(ys, yhat):.4f}  f1_macro={f1_score(ys, yhat, average='macro'):.4f}")
            print(classification_report(ys, yhat, labels=np.arange(n_classes), target_names=cn, zero_division=0))
            print("Confusion matrix:\n", confusion_matrix(ys, yhat))

    model_name = save_name or CLS_MODEL_NAME_TEMPLATE.format(exp_name=str(exp_name))
    model_path = out_dir / model_name

    joblib.dump(
        {
            "model": final_clf,
            "class_names": cn,
            "best_params": best_params,
            "search_topk": candidates[: int(CLS_SEARCH_TOPK)],
            "metrics": {
                "val_acc": val_acc,
                "val_f1_macro": val_f1m,
                "test_acc": test_acc,
                "test_f1_macro": test_f1m,
            },
            "search": {"n_iter": int(n_iter), "n_splits": int(n_splits), "seed": int(seed)},
        },
        model_path,
    )

    if verbose:
        print(f"\nSaved classifier to: {model_path}")

    return ClassifierTrainResult(
        model_path=str(model_path),
        best_params=best_params,
        class_names=cn,
        val_acc=val_acc,
        val_f1_macro=val_f1m,
        test_acc=test_acc,
        test_f1_macro=test_f1m,
    )
