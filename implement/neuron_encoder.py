from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
import statsmodels.api as sm


class eunjiLinearEncoder(BaseEstimator, RegressorMixin):

    def __init__(
        self,
        include_laser: bool = False,
        verbose: int = 0,
    ) -> None:
        self.include_laser = include_laser
        self.verbose = int(verbose)

    def fit(self, X, y):

        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be 2D array-like")
            
            # Expected column names based on paper
            n_features = X.shape[1]
            if self.include_laser:
                if n_features != 4:
                    raise ValueError(
                        "X must have 4 columns: prev_outcome, evidence, choice, laser"
                    )
                cols = ['prev_outcome', 'evidence', 'choice', 'laser']
            else:
                if n_features != 3:
                    raise ValueError(
                        "X must have 3 columns: prev_outcome, evidence, choice"
                    )
                cols = ['prev_outcome', 'evidence', 'choice']
            
            X_df = pd.DataFrame(X, columns=cols)
        
        y = np.asarray(y, dtype=float).ravel()
        
        if X_df.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
        
        # Validate expected columns
        required_cols = ['prev_outcome', 'evidence', 'choice']
        if self.include_laser:
            required_cols.append('laser')
        
        missing_cols = set(required_cols) - set(X_df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Add constant (intercept) term
        X_with_const = sm.add_constant(X_df[required_cols])
        
        # Fit Generalized Linear Model (GLM) with Gaussian family
        # This is equivalent to ordinary least squares (OLS) but uses GLM framework
        model = sm.GLM(y, X_with_const, family=sm.families.Gaussian())
        self.model_result_ = model.fit()
        
        # Extract coefficients
        self.coef_ = {
            'intercept': self.model_result_.params['const'],
            'prev_outcome': self.model_result_.params['prev_outcome'],
            'evidence': self.model_result_.params['evidence'],
            'choice': self.model_result_.params['choice'],
        }
        if self.include_laser:
            self.coef_['laser'] = self.model_result_.params['laser']
        
        # Extract t-statistics (for significance testing)
        self.tvalues_ = {
            'intercept': self.model_result_.tvalues['const'],
            'prev_outcome': self.model_result_.tvalues['prev_outcome'],
            'evidence': self.model_result_.tvalues['evidence'],
            'choice': self.model_result_.tvalues['choice'],
        }
        if self.include_laser:
            self.tvalues_['laser'] = self.model_result_.tvalues['laser']
        
        # Extract p-values
        self.pvalues_ = {
            'intercept': self.model_result_.pvalues['const'],
            'prev_outcome': self.model_result_.pvalues['prev_outcome'],
            'evidence': self.model_result_.pvalues['evidence'],
            'choice': self.model_result_.pvalues['choice'],
        }
        if self.include_laser:
            self.pvalues_['laser'] = self.model_result_.pvalues['laser']
        
        if self.verbose:
            print("=" * 60)
            print("Linear Encoding Model Results")
            print("=" * 60)
            print(self.model_result_.summary())
        
        return self

    def predict(self, X):
        """
        Predict firing rates for new trials.
        
        Parameters
        ----------
        X : array-like or DataFrame, shape (n_trials, n_features)
            Task variables (same format as fit).
            
        Returns
        -------
        y_pred : array, shape (n_trials,)
            Predicted firing rates.
        """
        # Convert to DataFrame if needed
        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
        else:
            X = np.asarray(X, dtype=float)
            if X.ndim != 2:
                raise ValueError("X must be 2D array-like")
            
            n_features = X.shape[1]
            if self.include_laser:
                if n_features != 4:
                    raise ValueError(
                        "X must have 4 columns: prev_outcome, evidence, choice, laser"
                    )
                cols = ['prev_outcome', 'evidence', 'choice', 'laser']
            else:
                if n_features != 3:
                    raise ValueError(
                        "X must have 3 columns: prev_outcome, evidence, choice"
                    )
                cols = ['prev_outcome', 'evidence', 'choice']
            
            X_df = pd.DataFrame(X, columns=cols)
        
        # Add constant term
        required_cols = ['prev_outcome', 'evidence', 'choice']
        if self.include_laser:
            required_cols.append('laser')
        
        X_with_const = sm.add_constant(X_df[required_cols], has_constant='add')
        
        # Predict using the fitted model
        y_pred = self.model_result_.predict(X_with_const)
        
        return y_pred.values

    def score(self, X, y):
        """
        Return the coefficient of determination R² of the prediction.
        
        Parameters
        ----------
        X : array-like, shape (n_trials, n_features)
            Task variables.
        y : array-like, shape (n_trials,)
            True firing rates.
            
        Returns
        -------
        r2_score : float
            R² score.
        """
        y_pred = self.predict(X)
        y = np.asarray(y, dtype=float).ravel()
        
        # Compute R²
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        
        return r2

    def get_tuning_preference(self, variable: str = 'evidence') -> str:
        """
        Determine neuron's preference for a task variable.
        
        Parameters
        ----------
        variable : str, default='evidence'
            Which variable to check ('evidence' or 'choice').
            
        Returns
        -------
        preference : str
            'ipsilateral' if coefficient > 0,
            'contralateral' if coefficient < 0,
            'none' if coefficient is 0.
        """
        if variable not in self.coef_:
            raise ValueError(f"Variable '{variable}' not in fitted model")
        
        coef_value = self.coef_[variable]
        
        if coef_value > 0:
            return 'ipsilateral'
        elif coef_value < 0:
            return 'contralateral'
        else:
            return 'none'

    def is_significant(self, variable: str, alpha: float = 0.05) -> bool:
        """
        Check if a coefficient is statistically significant.
        
        Parameters
        ----------
        variable : str
            Which variable to check (e.g., 'evidence', 'choice', 'laser').
        alpha : float, default=0.05
            Significance level.
            
        Returns
        -------
        is_sig : bool
            True if p-value < alpha.
        """
        if variable not in self.pvalues_:
            raise ValueError(f"Variable '{variable}' not in fitted model")
        
        return self.pvalues_[variable] < alpha

