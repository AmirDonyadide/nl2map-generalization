#src/train/train_regressors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .utils._regressor_utils import fit_reg_maybe_weighted


@dataclass(frozen=True)
class RegressorTrainResult:
    regressors_by_class: Dict[str, Tuple[Any, Any]]  # op -> (regressor, target_scaler)
    cv_summary: Dict[str, Dict[str, Any]]
    use_log1p: bool


def train_regressors_per_operator(
    *,
    X_train_s: np.ndarray,
    df_train: pd.DataFrame,
    y_train_cls: np.ndarray,
    class_names: Sequence[str],
    sample_w: np.ndarray,
    group_col: str = "map_id",
    target_col: str = "param_norm",
    use_log1p: bool = False,
    n_splits: int = 5,
    n_iter: int = 40,
    random_state: int = 42,
    verbose: int = 1,
) -> RegressorTrainResult:
    """
    Train one MLPRegressor per operator to predict target_col (default: param_norm).

    - Optional target transform: log1p (requires non-negative target)
    - Per-operator StandardScaler on the target
    - GroupKFold by group_col to avoid leakage
    - RandomizedSearchCV over MLPRegressor hyperparameters

    Returns:
      regressors_by_class: {operator: (regressor, target_scaler)}
      cv_summary: per-class best params and CV RMSE estimates
    """
    if group_col not in df_train.columns:
        raise KeyError(f"df_train missing group_col '{group_col}'")
    if target_col not in df_train.columns:
        raise KeyError(f"df_train missing target_col '{target_col}'")

    X_train_s = np.asarray(X_train_s, dtype=np.float64)
    y_train_cls = np.asarray(y_train_cls, dtype=int).reshape(-1)
    sample_w = np.asarray(sample_w, dtype=np.float64).reshape(-1)

    if not (len(X_train_s) == len(df_train) == len(y_train_cls) == len(sample_w)):
        raise ValueError("X_train_s, df_train, y_train_cls, sample_w must have the same length.")

    groups_tr = df_train[group_col].astype(str).to_numpy()
    y_train_norm = pd.to_numeric(df_train[target_col], errors="coerce").to_numpy(dtype=float)

    if not np.isfinite(y_train_norm).all():
        raise ValueError(f"Non-finite values found in '{target_col}' in df_train.")

    if use_log1p:
        if (y_train_norm < 0).any():
            raise ValueError("use_log1p=True but target contains negatives.")
        ytr_t = np.log1p(y_train_norm)
    else:
        ytr_t = y_train_norm.copy()

    if n_iter < 1:
        raise ValueError("n_iter must be >= 1.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")

    base_reg = MLPRegressor(
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        early_stopping=False,
        max_iter=2000,
        tol=1e-3,
        random_state=random_state,
        verbose=False,
        batch_size="auto",
    )

    param_dist_reg = {
        "hidden_layer_sizes": [(64,), (128,), (256,), (128, 64), (256, 128)],
        "alpha": loguniform(1e-6, 3e-2),
        "learning_rate_init": loguniform(1e-4, 3e-3),
    }

    regressors: Dict[str, Tuple[Any, Any]] = {}
    cv_summary: Dict[str, Dict[str, Any]] = {}

    for cls_idx, cls_name in enumerate(class_names):
        op = str(cls_name).strip().lower()
        m_tr = (y_train_cls == cls_idx)

        Xk = X_train_s[m_tr]
        yk = ytr_t[m_tr]
        gk_tr = groups_tr[m_tr]
        wk = sample_w[m_tr]

        n_samples = int(Xk.shape[0])
        n_groups = int(len(np.unique(gk_tr)))

        if n_samples < 10:
            if verbose:
                print(f"⚠️ Skipping class '{op}' (too few samples: {n_samples}).")
            continue
        if n_groups < 2:
            if verbose:
                print(f"⚠️ Skipping class '{op}' (too few groups for GroupKFold: groups={n_groups}).")
            continue

        # adapt splits if needed
        n_splits_eff = min(int(n_splits), n_groups)
        gk = GroupKFold(n_splits=n_splits_eff)

        t_scaler = StandardScaler()
        yk_s = t_scaler.fit_transform(yk.reshape(-1, 1)).ravel()

        splits = list(gk.split(Xk, yk_s, groups=gk_tr))

        search = RandomizedSearchCV(
            estimator=base_reg,
            param_distributions=param_dist_reg,
            n_iter=int(n_iter),
            scoring="neg_root_mean_squared_error",
            cv=splits,
            n_jobs=-1,
            refit=True,
            random_state=int(random_state),
            verbose=int(verbose),
        )

        # fit search (with weights if supported)
        try:
            search.fit(Xk, yk_s, sample_weight=wk)
            used_w_search = True
        except TypeError:
            search.fit(Xk, yk_s)
            used_w_search = False

        rmse_scaled = float(-search.best_score_)
        scale0 = float(getattr(t_scaler, "scale_", [1.0])[0] or 1.0)
        rmse_norm_units = float(rmse_scaled * scale0)

        if verbose:
            print(f"\n=== Regressor for class '{op}' (predicting {target_col}) ===")
            print(f"samples={n_samples}, groups={n_groups}, cv_splits={n_splits_eff}, used_sample_weight={used_w_search}")
            print("best CV RMSE (scaled):", rmse_scaled)
            print("best CV RMSE (param_norm units):", rmse_norm_units)
            print("best params:", search.best_params_)

        cv_summary[op] = {
            "n_samples": n_samples,
            "n_groups": n_groups,
            "cv_splits": n_splits_eff,
            "used_sample_weight": bool(used_w_search),
            "rmse_scaled": rmse_scaled,
            "rmse_param_norm": rmse_norm_units,
            "params": search.best_params_,
        }

        reg_full = MLPRegressor(
            **{
                **search.best_estimator_.get_params(),
                "early_stopping": False,
                "max_iter": 2000,
                "random_state": int(random_state),
            }
        )
        fit_reg_maybe_weighted(reg_full, Xk, yk_s, wk)
        regressors[op] = (reg_full, t_scaler)

    return RegressorTrainResult(
        regressors_by_class=regressors,
        cv_summary=cv_summary,
        use_log1p=bool(use_log1p),
    )
