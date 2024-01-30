from typing import List

from sklearn.metrics import f1_score

from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.components.metrics import MetricPipeline
from cinnamon_generic.configurations.metrics import LambdaMetricConfig, MetricConfig
from cinnamon_generic.configurations.pipeline import PipelineConfig
from components.metrics import MemoryUsage, MemoryCoverage, MemoryCoveragePrecision, MemoryPrecision, MemoryMRR, \
    MemoryClassificationPrecision, NonMemoryClassificationPrecision, MemoryAPM
from configurations.metrics import MemoryMetricConfig


class ToSMemoryUsageConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_U'
        config.threshold = 0.5

        return config


class ToSMemoryCoverageConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_C'
        config.threshold = 0.5

        return config


class ToSMemoryCoveragePrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_CP'
        config.threshold = 0.5

        return config


class ToSMemoryPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_P@K'
        config.threshold = 0.5

        config.add(name='k',
                   is_required=True,
                   type_hint=int,
                   description='Top-K value for computing memory precision @K.')

        return config

    @classmethod
    def get_precision_at_1_config(
            cls
    ):
        config = cls.get_default()

        config.name = 'M_P@1'
        config.k = 1

        return config

    @classmethod
    def get_precision_at_3_config(
            cls
    ):
        config = cls.get_default()

        config.name = 'M_P@3'
        config.k = 3

        return config


class ToSMemoryMRRConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_MRR'
        config.threshold = 0.5

        return config


class ToSMemoryClassificationPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_MP'
        config.threshold = 0.5

        return config


class ToSNonMemoryClassificationPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_NMP'
        config.threshold = 0.5

        return config


class ToSMemoryAPM(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_APM'
        config.threshold = 0.5

        return config


@register
def register_metrics():
    Registry.add_and_bind(config_class=ToSMemoryUsageConfig,
                          component_class=MemoryUsage,
                          name='metrics',
                          tags={'memory', 'usage'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryCoverageConfig,
                          component_class=MemoryCoverage,
                          name='metrics',
                          tags={'memory', 'coverage'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryCoveragePrecisionConfig,
                          component_class=MemoryCoveragePrecision,
                          name='metrics',
                          tags={'memory', 'coverage', 'precision'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryPrecisionConfig,
                          config_constructor=ToSMemoryPrecisionConfig.get_precision_at_1_config,
                          component_class=MemoryPrecision,
                          name='metrics',
                          tags={'memory', 'precision', 'k=1'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryPrecisionConfig,
                          config_constructor=ToSMemoryPrecisionConfig.get_precision_at_3_config,
                          component_class=MemoryPrecision,
                          name='metrics',
                          tags={'memory', 'precision', 'k=3'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryMRRConfig,
                          component_class=MemoryMRR,
                          name='metrics',
                          tags={'memory', 'mrr'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryClassificationPrecisionConfig,
                          component_class=MemoryClassificationPrecision,
                          name='metrics',
                          tags={'memory', 'classification', 'precision'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSNonMemoryClassificationPrecisionConfig,
                          component_class=NonMemoryClassificationPrecision,
                          name='metrics',
                          tags={'non-memory', 'classification', 'precision'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSMemoryAPM,
                          component_class=MemoryAPM,
                          name='metrics',
                          tags={'memory', 'apm'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='metrics', tags={'clf_f1'}, namespace='nle'),
                                  RegistrationKey(name='metrics', tags={'memory', 'usage'}, namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'coverage'}, namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'coverage', 'precision'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'mrr'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'precision', 'k=1'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'precision', 'k=3'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'classification', 'precision'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'non-memory', 'classification', 'precision'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='metrics', tags={'memory', 'apm'},
                                                  namespace='nle/tos'),
                              ],
                              'names': [
                                  'clf_f1',
                                  'memory_usage',
                                  'memory_coverage',
                                  'memory_coverage_precision',
                                  'memory_mrr',
                                  'memory_precision_@1',
                                  'memory_precision_@3',
                                  'memory_classification_precision',
                                  'non-memory_classification_precision',
                                  'memory_apm'
                              ]
                          },
                          component_class=MetricPipeline,
                          name='metrics',
                          tags={'memory'},
                          namespace='nle/tos')
