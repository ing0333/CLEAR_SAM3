"""(4) Part composition (few-shot): scene score = sum_j w_cj * logit(presence of phrase j).

Every phrase in the bank is treated as a "part". A multinomial logistic regression learns,
per scene, which parts should be present (w > 0) or absent (w < 0) from the val split
(20 labelled images per class). Always report it next to the control that uses only the 30
original class-name presences: on AID most of the gain came from learning the classifier
(36.8% -> 77.1%), the parts added +8.3%p on top (85.4%), and several learned parts are
context (e.g. Church <- 'crowded rooftops') rather than the object itself.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from .balance import logit


class PartComposer:
    def __init__(self, features: list[str], Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0), cv: int = 5,
                 is_logit: bool = False):
        """is_logit=False: H holds probabilities (SAM 3 presence) and is mapped through logit;
        is_logit=True: H already holds logits / similarities (CLIP)."""
        self.features = list(features)
        self.is_logit = is_logit
        self.clf = LogisticRegressionCV(Cs=list(Cs), cv=cv, max_iter=5000)

    def _X(self, H: dict) -> np.ndarray:
        X = np.stack([H[p] for p in self.features], 1)
        return X if self.is_logit else logit(X)

    def fit(self, H: dict, y: np.ndarray) -> "PartComposer":
        X = self._X(H)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.clf.fit((X - self.mu) / self.sd, y)
        return self

    def predict_proba(self, H: dict) -> np.ndarray:
        return self.clf.predict_proba((self._X(H) - self.mu) / self.sd)

    def top_parts(self, class_index: int, k: int = 3):
        w = self.clf.coef_[class_index]
        o = np.argsort(w)
        return ([(self.features[j], float(w[j])) for j in o[::-1][:k]],
                [(self.features[j], float(w[j])) for j in o[:k]])
