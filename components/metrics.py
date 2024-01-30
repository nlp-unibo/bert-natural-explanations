from typing import Any

import numpy as np

from cinnamon_generic.components.metrics import LambdaMetric, Metric
from utility.numpy_utility import topk


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
        Percentage of examples for which memory is used.

        y_pred: raw memory scores of shape [#samples, M]
        """

        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        selections = np.minimum(th_memory_scores.sum(axis=-1), 1.0)
        usage = selections.mean()

        return usage if not as_dict else {self.name: usage}


class MemoryCoverage(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ):
        """
        Percentage of positive examples for which (at least one) memory slot selection is correct.

        y_pred: raw memory scores of shape [#samples, M]
        y_true: memory targets as binary mask of shape [#samples, M]
        """

        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        hits = np.minimum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
        coverage = hits.sum() / th_memory_scores.shape[0]

        return coverage if not as_dict else {self.name: coverage}


class MemoryCoveragePrecision(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ):
        """
        Percentage of positive examples for which (at least one) memory slot selection is correct.

        y_pred: rounded memory scores of shape [#samples, M]
        y_true: memory targets as binary mask of shape [#samples, M]
        """

        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        hits = np.minimum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
        usage = np.minimum(th_memory_scores.sum(axis=-1), 1.0).sum()
        cp = (hits.sum() / usage) if usage > 0 else 0.0

        return cp if not as_dict else {self.name: cp}


class MemoryPrecision(Metric):

    def in_top_k(
            self,
            memory_scores,
            memory_targets,
            threshold,
            k
    ):
        top_k_indexes, top_k_values = topk(memory_scores, k=k, ascending=False)
        target_indexes = np.argwhere(memory_targets).ravel()
        hits = set(top_k_indexes).intersection(target_indexes)

        if not hits:
            return 0.0

        th_hits = [memory_scores[hit_index] for hit_index in hits if memory_scores[hit_index] >= threshold]

        if len(th_hits):
            return 1.0

        return 0.0

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:

        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]

        th_top_k_hits = [self.in_top_k(memory_scores=s_mem_scores,
                                       memory_targets=s_mem_targets,
                                       k=self.k,
                                       threshold=self.threshold)
                         for s_mem_scores, s_mem_targets in zip(memory_scores, memory_targets)]
        mp = np.sum(th_top_k_hits) / memory_scores.shape[0]

        return mp if not as_dict else {self.name: mp}


class MemoryMRR(Metric):

    def best_target_rank(
            self,
            memory_scores,
            memory_targets,
            threshold,
    ):
        target_indexes = np.argwhere(memory_targets).ravel()
        sorted_score_indexes = np.argsort(memory_scores)[::-1]
        best_target_index = np.where(np.in1d(sorted_score_indexes, target_indexes))[0][0]
        return 0.0 if memory_scores[best_target_index] < threshold else 1 / (best_target_index + 1)

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]

        mrr = [self.best_target_rank(memory_scores=s_mem_scores,
                                     memory_targets=s_mem_targets,
                                     threshold=self.threshold)
               for s_mem_scores, s_mem_targets in zip(memory_scores, memory_targets)]
        mrr = np.sum(mrr) / memory_scores.shape[0]

        return mrr if not as_dict else {self.name: mrr}


class MemoryClassificationPrecision(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        predictions = np.concatenate(y_pred['logits'], axis=0)
        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]
        predictions = predictions[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        hits = np.minimum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
        correct = predictions == 1
        correct_and_hit = (hits * correct).sum() / predictions.shape[0]

        return correct_and_hit if not as_dict else {self.name: correct_and_hit}


class NonMemoryClassificationPrecision(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        memory_targets = np.concatenate(y_true['memory_targets'], axis=0)
        memory_targets = np.take_along_axis(memory_targets, memory_indices, axis=1)

        predictions = np.concatenate(y_pred['logits'], axis=0)
        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]
        memory_targets = memory_targets[label == 1]
        predictions = predictions[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        no_hits = 1.0 - np.minimum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
        correct = predictions == 1
        correct_and_no_hit = (no_hits * correct).sum() / predictions.shape[0]

        return correct_and_no_hit if not as_dict else {self.name: correct_and_no_hit}


class MemoryAPM(Metric):

    def run(
            self,
            y_pred: Any,
            y_true: Any,
            as_dict: bool = False
    ) -> Any:
        memory_scores = np.concatenate(y_pred['memory_scores'], axis=0)
        memory_indices = np.concatenate(y_pred['sampled_indices'], axis=0)
        memory_scores = np.take_along_axis(memory_scores, memory_indices, axis=1)
        label = np.concatenate(y_true['label'], axis=0)

        # Only positive examples
        memory_scores = memory_scores[label == 1]

        th_memory_scores = np.where(memory_scores >= self.threshold, 1.0, 0.0)
        apm = th_memory_scores.sum(axis=-1) / memory_scores.shape[1]
        apm = apm.mean()

        return apm if not as_dict else {self.name: apm}