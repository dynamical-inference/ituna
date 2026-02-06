from typing import List

import sklearn.base

from ituna._backends import base


class InMemoryBackend(base.Backend):
    def fit_models(
        self,
        models: List[sklearn.base.BaseEstimator],
        *data_args,
        **fit_params,
    ):
        """
        Fit models in the queue.
        """
        trained_models = []
        for model in models:
            trained_models.append(
                model.fit(
                    *data_args,
                    **fit_params,
                )
            )

        return trained_models
