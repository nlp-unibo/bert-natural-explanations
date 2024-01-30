from typing import List

from sklearn.metrics import f1_score

from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.configurations.metrics import LambdaMetricConfig, MetricConfig
from components.metrics import ClassificationMetric


class ClassificatioMetricConfig(LambdaMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()
        config.name = 'clf_f1'
        config.method = f1_score
        config.method_args = {'average': 'binary'}
        return config


class MemoryMetricConfig(MetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.add(name='threshold',
                   type_hint=float,
                   is_required=True,
                   description='Rounding threshold for raw memory scores.')

        return config


@register
def register_metrics_configurations():
    Registry.add_and_bind(config_class=ClassificatioMetricConfig,
                          component_class=ClassificationMetric,
                          name='metrics',
                          tags={'clf_f1'},
                          namespace='nle')
