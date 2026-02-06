"""
For some of our tests we need custom sklearn models which are importable via module name.
This doesn't work if defined within the tests/ directory.
So we define them here.
"""

import sklearn.base

import ituna


class CustomConsistencyTransform(ituna.metrics.ConsistencyTransform):
    def _fit(self, X, **kwargs):
        # X list of dicts with keys model and data
        models = [x["model"] for x in X]
        _ = [x["data"] for x in X]
        self.model_ = models[0]
        return self

    def _transform(self, X, **kwargs):
        return self.model_.transform(X[0])

    def _score(self, X, **kwargs):
        return self.transform(X).mean()


class CustomConsistencyEnsemble(ituna.ConsistencyEnsemble):
    def fit(self, X, **kwargs):
        """
        Fit all ensemble models.

        Returns
        -------
        self : object
            Returns the instance itself.
        """

        self.estimators_ = self._backend.fit_models(
            self._init_estimators(),
            X,
            **kwargs,
        )
        # self.consistency_transform_ = sklearn.base.clone(self.consistency_transform).fit(self.estimators_)
        consistency_X = [{"model": model, "data": X} for model in self.estimators_]
        self.consistency_transform_ = self._backend.fit_models(
            [sklearn.base.clone(self.consistency_transform)],
            consistency_X,
        )[0]
