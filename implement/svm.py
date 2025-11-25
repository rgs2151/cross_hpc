from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class eunjiSVC(BaseEstimator, ClassifierMixin):
    """A minimal linear SVM (hinge loss) compatible with sklearn API.

    Parameters
    ----------
    C : float, default=1.0
        Regularization parameter (larger C => less regularization).
    max_iter : int, default=1000
        Maximum number of passes (epochs) over the training data.
    tol : float, default=1e-4
        Stopping tolerance on the change of the objective (not strict).
    learning_rate : float or 'auto', default='auto'
        Initial learning rate for SGD. If 'auto', a simple schedule is used.
    random_state : int or None
        Random seed for data shuffling.
    verbose : int, default=0
        Verbosity level.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        tol: float = 1e-4,
        learning_rate: float | str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
    ) -> None:
        self.C = float(C)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.verbose = int(verbose)

    def _init_params(self, n_features: int) -> None:
        rng = np.random.RandomState(self.random_state)
        # initialize small random weights to break symmetry
        self._w = rng.normal(scale=1e-4, size=n_features)
        self._b = 0.0

    def fit(self, X, y):
        """
        Only binary classification is supported.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError("X must be 2D array-like")

        n_samples, n_features = X.shape

        classes = np.unique(y)
        if classes.shape[0] != 2:
            raise NotImplementedError("Only binary classification is implemented")

        # store classes similar to sklearn
        self.classes_ = classes

        # map labels to {-1, +1}
        self._class_to_sign = {classes[0]: -1, classes[1]: 1}
        y_signed = np.vectorize(self._class_to_sign.get)(y)

        # initialize parameters
        self._init_params(n_features)

        # learning rate handling
        if self.learning_rate == "auto":
            eta0 = 0.1
        else:
            eta0 = float(self.learning_rate)

        rng = np.random.RandomState(self.random_state)

        prev_obj = np.inf
        for epoch in range(1, self.max_iter + 1):
            # shuffle
            idx = rng.permutation(n_samples)
            X_shuf = X[idx]
            y_shuf = y_signed[idx]

            # simple learning rate schedule
            eta = eta0 / np.sqrt(epoch)

            for i in range(n_samples):
                xi = X_shuf[i]
                yi = y_shuf[i]
                margin = yi * (np.dot(self._w, xi) + self._b)
                # L2 regularization applied as weight decay
                if margin < 1.0:
                    # subgradient for hinge + L2: w - C * y_i * x_i
                    # SGD update: w <- w - eta * (w - C * y_i * x_i)
                    self._w = (1.0 - eta) * self._w + eta * self.C * yi * xi
                    # update bias (no regularization on bias)
                    self._b += eta * self.C * yi
                else:
                    # only regularization
                    self._w = (1.0 - eta) * self._w

            # compute (approx) objective to check convergence
            margins = y_signed * (X.dot(self._w) + self._b)
            hinge = np.maximum(0.0, 1.0 - margins)
            obj = 0.5 * np.dot(self._w, self._w) + self.C * np.sum(hinge)

            if self.verbose:
                print(f"Epoch {epoch}: obj={obj:.6f}")

            if abs(prev_obj - obj) < self.tol:
                self.n_iter_ = epoch
                break
            prev_obj = obj

        else:
            self.n_iter_ = self.max_iter

        # expose sklearn-like attributes
        self.coef_ = self._w.reshape(1, -1).copy()
        self.intercept_ = np.array([float(self._b)])

        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return X.dot(self.coef_.T).ravel() + float(self.intercept_)

    def predict(self, X):
        scores = self.decision_function(X)
        # map positive -> classes_[1], negative/zero -> classes_[0]
        return np.where(scores > 0, self.classes_[1], self.classes_[0])

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        return np.mean(preds == y)
