from typing import Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric, Metric


class ClassificationMetric(LambdaMetric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        y_pred = np.concatenate(y_pred['logits'])
        y_true = np.concatenate(y_true['label'])

        method_args = self.method_args if self.method_args is not None else {}
        metric_value = self.method(y_pred=y_pred, y_true=y_true, **method_args)
        return metric_value if not as_dict else {self.name: metric_value}


class MemoryUsage(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ):
        """
        y_pred: rounded memory scores of shape [#samples, M]
        """

        y_pred = np.maximum(y_pred.sum(axis=-1), 1.0)
        usage = y_pred.sum() / y_pred.shape[0]
        return usage if not as_dict else {self.name: usage}


class MemoryCoverage(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ):
        """
        y_pred: rounded memory scores of shape [#samples, M]
        y_true: memory targets as binary mask of shape [#samples, M]
        """

        hits = np.maximum((y_pred * y_true).sum(axis=-1), 1.0)
        coverage = hits.sum() / y_pred.shape[0]
        return coverage if not as_dict else {self.name: coverage}


