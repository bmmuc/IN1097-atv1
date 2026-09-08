from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata, wilcoxon
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ALPHA = 0.05
RANDOM_STATE = 0
N_ESTIMATORS = 100


# helpers

def ranks_from_P(P: pd.DataFrame) -> pd.DataFrame:
    """Rank matrix R from P: per dataset, rank 1 = best (rank -P.row, average ties)."""
    arr = P.values
    ranked = np.apply_along_axis(lambda row: rankdata(-row, method="average"), 1, arr)
    return pd.DataFrame(ranked, index=P.index, columns=P.columns)


def _order_and_scores(rank_scores: pd.Series):
    """rank_scores: lower = better. Returns (order, rank_scores) with a stable sort."""
    order = rank_scores.sort_values(kind="mergesort").index.tolist()
    return order, rank_scores


def make_regressor_pipeline(random_state: int = RANDOM_STATE) -> Pipeline:
    """StandardScaler (fit on train split only, via Pipeline) + multi-output RF."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=N_ESTIMATORS, random_state=random_state)),
    ])


# base class

class BaseApproach:
    name = "base"
    needs_X = False

    def fit(self, X_train: pd.DataFrame, P_train: pd.DataFrame, R_train: pd.DataFrame,
            folds_train: pd.DataFrame | None = None):
        raise NotImplementedError

    def recommend(self, X_query=None):
        raise NotImplementedError


# 1. AverageRank

class AverageRank(BaseApproach):
    """Mean of R over training rows per algorithm; re-rank those means."""
    name = "AR"
    needs_X = False

    def fit(self, X_train, P_train, R_train, folds_train=None):
        mean_r = R_train.mean(axis=0)
        self.rank_scores_ = pd.Series(
            rankdata(mean_r.values, method="average"), index=mean_r.index)
        return self

    def recommend(self, X_query=None):
        return _order_and_scores(self.rank_scores_)


# 2. MedianRank

class MedianRank(BaseApproach):
    """Median of R over training rows per algorithm; re-rank those medians."""
    name = "MR"
    needs_X = False

    def fit(self, X_train, P_train, R_train, folds_train=None):
        med_r = R_train.median(axis=0)
        self.rank_scores_ = pd.Series(
            rankdata(med_r.values, method="average"), index=med_r.index)
        return self

    def recommend(self, X_query=None):
        return _order_and_scores(self.rank_scores_)


# 3. SignificantWins

def compute_win_tensor(P_folds: pd.DataFrame, dataset_ids, algorithms, alpha: float = ALPHA):
    """Pairwise significant-win outcome per dataset, computed ONCE up front
    (each entry depends only on that dataset's own per-fold values, so
    slicing this dict by a training-id subset later introduces no leakage).
    For each dataset and ordered pair (a, b), runs a paired Wilcoxon
    signed-rank test on the per-fold accuracies and records a win for a over
    b when significant at `alpha` and the mean difference favours a.
    Returns dict: dataset_id -> (m, m) array, win[i, j] == 1 iff i beat j.
    """
    algorithms = list(algorithms)
    m = len(algorithms)
    wins = {}
    for did in dataset_ids:
        sub = P_folds[P_folds["dataset_id"] == did]
        piv = sub.pivot_table(index="fold", columns="algorithm", values="value")
        piv = piv[algorithms]
        W = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            a = piv.iloc[:, i].values.astype(float)
            for j in range(m):
                if i == j:
                    continue
                b = piv.iloc[:, j].values.astype(float)
                diff = a - b
                if np.allclose(diff, 0.0):
                    # all-tie case: scipy.wilcoxon raises/warns -- not significant, no win
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # exact method: correct small-sample null and ~100x faster
                        # than "auto" here, given n=10 paired folds with rare ties
                        stat, p = wilcoxon(a, b, zero_method="wilcox", method="exact")
                    except ValueError:
                        continue  # e.g. all-zero diff under wilcox/pratt, or n too small
                if p < alpha and diff.mean() > 0:
                    W[i, j] = 1
        wins[did] = W
    return wins


class SignificantWins(BaseApproach):
    """Wilcoxon win-counting approach; sums the precomputed win tensor
    (see `compute_win_tensor`) over the training dataset ids -- no tests
    are re-run inside LOO folds."""
    name = "SW"
    needs_X = False

    def __init__(self, win_tensor: dict, algorithms):
        self.win_tensor = win_tensor
        self.algorithms = list(algorithms)

    def fit(self, X_train, P_train, R_train, folds_train=None):
        train_ids = list(P_train.index)
        m = len(self.algorithms)
        total = np.zeros(m, dtype=float)
        for did in train_ids:
            total += self.win_tensor[did].sum(axis=1)
        self.win_counts_ = pd.Series(total, index=self.algorithms)
        # more wins = better = rank 1; ties get the average rank
        self.rank_scores_ = pd.Series(
            rankdata(-self.win_counts_.values, method="average"), index=self.algorithms)
        return self

    def recommend(self, X_query=None):
        return _order_and_scores(self.rank_scores_)


# 4. RegressorOnP  (Abordagem 1)

class RegressorOnP(BaseApproach):
    """Multi-output regressor X -> P. Highest predicted performance = rank 1."""
    name = "RegressorOnP"
    needs_X = True

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X_train, P_train, R_train, folds_train=None):
        self.algorithms = list(P_train.columns)
        self.model_ = make_regressor_pipeline(self.random_state)
        self.model_.fit(np.asarray(X_train.values, dtype=float), P_train.values)
        return self

    def recommend(self, X_query):
        Xq = np.asarray(X_query, dtype=float).reshape(1, -1)
        pred = self.model_.predict(Xq)[0]
        rank_scores = pd.Series(rankdata(-pred, method="average"), index=self.algorithms)
        return _order_and_scores(rank_scores)


# 5. RegressorOnR  (Abordagem 2)

class RegressorOnR(BaseApproach):
    """Multi-output regressor X -> R. Lowest predicted rank value = rank 1."""
    name = "RegressorOnR"
    needs_X = True

    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state

    def fit(self, X_train, P_train, R_train, folds_train=None):
        self.algorithms = list(R_train.columns)
        self.model_ = make_regressor_pipeline(self.random_state)
        self.model_.fit(np.asarray(X_train.values, dtype=float), R_train.values)
        return self

    def recommend(self, X_query):
        Xq = np.asarray(X_query, dtype=float).reshape(1, -1)
        pred = self.model_.predict(Xq)[0]
        rank_scores = pd.Series(pred, index=self.algorithms)  # already lower = better
        return _order_and_scores(rank_scores)


# 6. Harris

def _get_harris_forest_cls():
    from harris import HarrisForest  # lazy import: keep other approaches usable standalone
    return HarrisForest


class Harris(BaseApproach):
    """Wraps HarrisForest for a fixed lam. Rank from predict_rank (lower=better)."""
    needs_X = True

    def __init__(self, lam: float = 0.5, n_estimators: int = N_ESTIMATORS,
                 max_features="sqrt", min_samples_leaf: int = 3, max_depth=None,
                 bootstrap: bool = True, random_state: int = RANDOM_STATE, n_jobs: int = 1):
        self.lam = lam
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.name = f"Harris(lam={lam})"

    def fit(self, X_train, P_train, R_train, folds_train=None):
        HarrisForest = _get_harris_forest_cls()
        self.algorithms = list(P_train.columns)
        self.model_ = HarrisForest(
            lam=self.lam, n_estimators=self.n_estimators, max_features=self.max_features,
            min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth,
            bootstrap=self.bootstrap, random_state=self.random_state, n_jobs=self.n_jobs)
        self.model_.fit(X_train.values, P_train.values, R_train.values)
        return self

    def recommend(self, X_query):
        Xq = np.asarray(X_query, dtype=float).reshape(1, -1)
        scores = np.asarray(self.model_.predict_rank(Xq))[0]
        rank_scores = pd.Series(scores, index=self.algorithms)
        return _order_and_scores(rank_scores)


ALL_APPROACH_NAMES = ["AR", "MR", "SW", "RegressorOnP", "RegressorOnR", "Harris"]
