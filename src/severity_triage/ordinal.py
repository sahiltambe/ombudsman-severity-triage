"""Ordinal models not in scikit-learn: ordered logit, Frank-Hall, regression cuts."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.utils.validation import check_is_fitted

__all__ = [
    "FrankHallClassifier",
    "ProportionalOddsClassifier",
    "RegressionThresholdClassifier",
    "balanced_sample_weight",
]

_EPS = 1e-12


def balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    """Inverse-frequency weights so each class contributes equally."""
    return compute_sample_weight("balanced", y)


def _as_array(X: object) -> np.ndarray:
    return np.asarray(X, dtype=float)


# ---------------------------------------------------------------------------
# Proportional odds (cumulative logit) with ridge penalty
# ---------------------------------------------------------------------------


class ProportionalOddsClassifier(BaseEstimator, ClassifierMixin):
    """Ordered logistic regression: ``P(y <= k | x) = sigmoid(theta_k - x.beta)``.

    Parameters
    ----------
    alpha:
        L2 penalty on the slope coefficients (not on the cut-points).
    class_weight:
        ``"balanced"`` reweights samples by inverse class frequency; ``None``
        leaves them unweighted.
    max_iter:
        L-BFGS iteration cap.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        class_weight: str | None = "balanced",
        max_iter: int = 500,
        tol: float = 1e-6,
    ) -> None:
        self.alpha = alpha
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol

    # -- parameterisation ---------------------------------------------------
    # theta_0 = a_0 ; theta_k = theta_{k-1} + exp(a_k)  guarantees ordering.

    @staticmethod
    def _thresholds(a: np.ndarray) -> np.ndarray:
        theta = np.empty_like(a)
        theta[0] = a[0]
        theta[1:] = a[0] + np.cumsum(np.exp(a[1:]))
        return theta

    def _unpack(self, params: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
        beta = params[:n_features]
        theta = self._thresholds(params[n_features:])
        return beta, theta

    def _objective(
        self, params: np.ndarray, X: np.ndarray, codes: np.ndarray, w: np.ndarray
    ) -> tuple[float, np.ndarray]:
        d = X.shape[1]
        k_classes = len(self.classes_)
        beta, theta = self._unpack(params, d)
        eta = X @ beta

        # Upper and lower cumulative logits for each sample's observed class.
        upper = np.where(
            codes < k_classes - 1, theta[np.minimum(codes, k_classes - 2)] - eta, np.inf
        )
        lower = np.where(codes > 0, theta[np.maximum(codes - 1, 0)] - eta, -np.inf)
        s_u, s_l = expit(upper), expit(lower)
        p = np.clip(s_u - s_l, _EPS, None)

        nll = -np.sum(w * np.log(p)) + 0.5 * self.alpha * beta @ beta

        # Gradients. sigma'(z) = sigma(z)(1 - sigma(z)); zero at +/- inf.
        d_u = np.where(np.isfinite(upper), s_u * (1 - s_u), 0.0)
        d_l = np.where(np.isfinite(lower), s_l * (1 - s_l), 0.0)
        inv_p = w / p

        grad_eta = -inv_p * (-d_u + d_l)  # d(-log p)/d eta
        grad_beta = X.T @ grad_eta + self.alpha * beta

        grad_theta = np.zeros(k_classes - 1)
        # d(-log p)/d theta_k :  -inv_p * d_u  when theta_k is the upper bound
        #                        +inv_p * d_l  when theta_k is the lower bound
        np.add.at(
            grad_theta,
            np.minimum(codes, k_classes - 2),
            np.where(codes < k_classes - 1, -inv_p * d_u, 0.0),
        )
        np.add.at(grad_theta, np.maximum(codes - 1, 0), np.where(codes > 0, inv_p * d_l, 0.0))

        # Chain rule through the monotone reparameterisation.
        a = params[d:]
        grad_a = np.empty_like(a)
        grad_a[0] = grad_theta.sum()
        # theta_k depends on a_m (m >= 1) for every k >= m.
        tail = np.cumsum(grad_theta[::-1])[::-1]  # tail[m] = sum_{k>=m} grad_theta[k]
        grad_a[1:] = np.exp(a[1:]) * tail[1:]

        return float(nll), np.concatenate([grad_beta, grad_a])

    # -- sklearn API ----------------------------------------------------------

    def fit(self, X: object, y: object) -> ProportionalOddsClassifier:
        X_arr = _as_array(X)
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        codes = np.searchsorted(self.classes_, y_arr)
        w = (
            balanced_sample_weight(y_arr)
            if self.class_weight == "balanced"
            else np.ones(len(y_arr))
        )

        n_features = X_arr.shape[1]
        k = len(self.classes_)
        # Initialise cut-points at the empirical cumulative logits so the
        # optimiser starts from the intercept-only solution.
        cum = np.cumsum(np.bincount(codes, weights=w, minlength=k))[:-1] / w.sum()
        theta0 = np.log(cum / (1 - cum))
        a0 = np.concatenate([[theta0[0]], np.log(np.maximum(np.diff(theta0), 1e-3))])
        x0 = np.concatenate([np.zeros(n_features), a0])

        result = minimize(
            self._objective,
            x0,
            args=(X_arr, codes, w),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "gtol": self.tol},
        )
        self.coef_, self.cutpoints_ = self._unpack(result.x, n_features)
        self.n_iter_ = int(result.nit)
        self.converged_ = bool(result.success)
        self.n_features_in_ = n_features
        return self

    def latent_score(self, X: object) -> np.ndarray:
        """The latent score ``x.beta``; higher means more severe."""
        check_is_fitted(self, "coef_")
        return _as_array(X) @ self.coef_

    def predict_proba(self, X: object) -> np.ndarray:
        eta = self.latent_score(X)
        cum = expit(self.cutpoints_[None, :] - eta[:, None])  # P(y <= k)
        cum = np.hstack([np.zeros((len(eta), 1)), cum, np.ones((len(eta), 1))])
        return np.clip(np.diff(cum, axis=1), _EPS, None)

    def predict(self, X: object) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Frank & Hall decomposition
# ---------------------------------------------------------------------------


class FrankHallClassifier(BaseEstimator, ClassifierMixin):
    """K-1 binary classifiers for ``P(y > k)`` recombined into class probabilities."""

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def fit(self, X: object, y: object) -> FrankHallClassifier:
        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.estimators_ = []
        for threshold in self.classes_[:-1]:
            target = (y_arr > threshold).astype(int)
            self.estimators_.append(clone(self.estimator).fit(X, target))
        self.n_features_in_ = _as_array(X).shape[1] if not hasattr(X, "shape") else X.shape[1]
        return self

    def predict_greater_than(self, X: object) -> np.ndarray:
        """Matrix of ``P(y > k)`` for each threshold, monotone non-increasing."""
        check_is_fitted(self, "estimators_")
        columns = []
        for est in self.estimators_:
            proba = est.predict_proba(X)
            if proba.shape[1] == 1:  # degenerate fold with a single class
                columns.append(np.full(len(proba), float(est.classes_[0])))
            else:
                columns.append(proba[:, list(est.classes_).index(1)])
        gt = np.column_stack(columns)
        # Enforce the logical constraint P(y > k) >= P(y > k+1).
        return np.minimum.accumulate(gt, axis=1)

    def predict_proba(self, X: object) -> np.ndarray:
        gt = self.predict_greater_than(X)
        first = 1.0 - gt[:, :1]
        middle = gt[:, :-1] - gt[:, 1:]
        last = gt[:, -1:]
        proba = np.hstack([first, middle, last]).clip(_EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)

    def predict(self, X: object) -> np.ndarray:
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


# ---------------------------------------------------------------------------
# Regression with learned cut-points
# ---------------------------------------------------------------------------


class RegressionThresholdClassifier(BaseEstimator, ClassifierMixin):
    """Regress the ordinal target, then cut at thresholds tuned for QWK."""

    def __init__(
        self,
        regressor: BaseEstimator,
        optimise_thresholds: bool = True,
        n_inner_splits: int = 5,
        random_state: int | None = None,
    ) -> None:
        self.regressor = regressor
        self.optimise_thresholds = optimise_thresholds
        self.n_inner_splits = n_inner_splits
        self.random_state = random_state

    def _cut(self, scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        return self.classes_[np.searchsorted(thresholds, scores)]

    def _optimise(self, scores: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        """Coordinate descent on each cut-point, maximising QWK."""
        best = thresholds.copy()
        best_score = cohen_kappa_score(y, self._cut(scores, best), weights="quadratic")
        for _ in range(3):  # a few sweeps are plenty for 5 thresholds
            for i in range(len(best)):
                lo = best[i - 1] if i > 0 else scores.min()
                hi = best[i + 1] if i + 1 < len(best) else scores.max()
                for candidate in np.linspace(lo, hi, 41)[1:-1]:
                    trial = best.copy()
                    trial[i] = candidate
                    score = cohen_kappa_score(y, self._cut(scores, trial), weights="quadratic")
                    if score > best_score:
                        best, best_score = trial, score
        return best

    def fit(self, X: object, y: object) -> RegressionThresholdClassifier:
        y_arr = np.asarray(y, dtype=float)
        self.classes_ = np.unique(y_arr).astype(int)
        w = balanced_sample_weight(y_arr.astype(int))

        cv = KFold(n_splits=self.n_inner_splits, shuffle=True, random_state=self.random_state)
        oof = cross_val_predict(clone(self.regressor), X, y_arr, cv=cv, params={"sample_weight": w})

        thresholds = (self.classes_[:-1] + self.classes_[1:]) / 2.0
        if self.optimise_thresholds:
            thresholds = self._optimise(oof, y_arr.astype(int), thresholds)
        self.thresholds_ = thresholds
        self.sigma_ = float(np.std(y_arr - oof)) or 1.0

        self.regressor_ = clone(self.regressor).fit(X, y_arr, sample_weight=w)
        self.n_features_in_ = X.shape[1] if hasattr(X, "shape") else _as_array(X).shape[1]
        return self

    def latent_score(self, X: object) -> np.ndarray:
        check_is_fitted(self, "regressor_")
        return np.asarray(self.regressor_.predict(X), dtype=float)

    def predict(self, X: object) -> np.ndarray:
        return self._cut(self.latent_score(X), self.thresholds_)

    def predict_proba(self, X: object) -> np.ndarray:
        score = self.latent_score(X)
        edges = np.concatenate([[-np.inf], self.thresholds_, [np.inf]])
        cdf = norm.cdf((edges[None, :] - score[:, None]) / self.sigma_)
        proba = np.diff(cdf, axis=1).clip(_EPS, None)
        return proba / proba.sum(axis=1, keepdims=True)
