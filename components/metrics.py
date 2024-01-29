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

        memory_scores = y_pred['memory_scores']

        usages = []
        for threshold in self.thresholds:
            th_memory_scores = np.where(memory_scores >= threshold, 1.0, 0.0)
            selections = np.maximum(th_memory_scores.sum(axis=-1), 1.0)
            usage = selections.sum() / th_memory_scores.shape[0]
            usages.append(usage)

        return usages if not as_dict else {self.name: usages}


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

        memory_scores = y_pred['memory_scores']
        memory_targets = y_true['memory_targets']

        coverages = []
        for threshold in self.thresholds:
            th_memory_scores = np.where(memory_scores >= threshold, 1.0, 0.0)
            hits = np.maximum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
            coverage = hits.sum() / th_memory_scores.shape[0]
            coverages.append(coverage)

        return coverages if not as_dict else {self.name: coverages}


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

        memory_scores = y_pred['memory_scores']
        memory_targets = y_true['memory_targets']

        cps = []
        for threshold in self.thresholds:
            th_memory_scores = np.where(memory_scores >= threshold, 1.0, 0.0)
            hits = np.maximum((th_memory_scores * memory_targets).sum(axis=-1), 1.0)
            usage = np.maximum(th_memory_scores.sum(axis=-1), 1.0).sum() / th_memory_scores.shape[0]
            cp = (hits.sum() / usage) if usage > 0 else 0.0
            cps.append(cp)

        return cps if not as_dict else {self.name: cps}


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

        memory_scores = y_pred['memory_scores']
        memory_targets = y_true['memory_targets']

        precisions = []
        for threshold in self.thresholds:
            for k in self.K:
                th_top_k_hits = [self.in_top_k(memory_scores=s_mem_scores,
                                               memory_targets=s_mem_targets,
                                               k=k,
                                               threshold=threshold)
                                 for s_mem_scores, s_mem_targets in zip(memory_scores, memory_targets)]
                precisions.append(th_top_k_hits)

        return precisions if not as_dict else {self.name: precisions}


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
        memory_scores = y_pred['memory_scores']
        memory_targets = y_true['memory_targets']

        mrrs = []
        for threshold in self.thresholds:
            mrr = [self.best_target_rank(memory_scores=s_mem_scores,
                                         memory_targets=s_mem_targets,
                                         threshold=threshold)
                   for s_mem_scores, s_mem_targets in zip(memory_scores, memory_targets)]
            mrrs.append(mrr)

        return mrrs if not as_dict else {self.name: mrrs}
