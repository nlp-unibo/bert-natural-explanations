from typing import Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric


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
