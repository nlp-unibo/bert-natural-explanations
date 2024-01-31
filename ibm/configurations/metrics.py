from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.components.metrics import MetricPipeline
from cinnamon_generic.configurations.pipeline import PipelineConfig
from components.metrics import MemoryUsage, MemoryCoverage, MemoryCoveragePrecision, MemoryPrecision, MemoryMRR, \
    MemoryClassificationPrecision, NonMemoryClassificationPrecision, MemoryAPM
from configurations.metrics import MemoryMetricConfig


class IBMMemoryUsageConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_U'
        config.threshold = 0.1

        return config


class IBMMemoryCoverageConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_C'
        config.threshold = 0.1

        return config


class IBMMemoryCoveragePrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_CP'
        config.threshold = 0.1

        return config


class IBMMemoryPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_P@K'
        config.threshold = 0.1

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


class IBMMemoryMRRConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_MRR'
        config.threshold = 0.1

        return config


class IBMMemoryClassificationPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_MP'
        config.threshold = 0.1

        return config


class IBMNonMemoryClassificationPrecisionConfig(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_NMP'
        config.threshold = 0.1

        return config


class IBMMemoryAPM(MemoryMetricConfig):

    @classmethod
    def get_default(
            cls
    ):
        config = super().get_default()

        config.name = 'M_APM'
        config.threshold = 0.1

        return config


@register
def register_metrics():
    Registry.add_and_bind(config_class=IBMMemoryUsageConfig,
                          component_class=MemoryUsage,
                          name='metrics',
                          tags={'memory', 'usage'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryCoverageConfig,
                          component_class=MemoryCoverage,
                          name='metrics',
                          tags={'memory', 'coverage'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryCoveragePrecisionConfig,
                          component_class=MemoryCoveragePrecision,
                          name='metrics',
                          tags={'memory', 'coverage', 'precision'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryPrecisionConfig,
                          config_constructor=IBMMemoryPrecisionConfig.get_precision_at_1_config,
                          component_class=MemoryPrecision,
                          name='metrics',
                          tags={'memory', 'precision', 'k=1'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryPrecisionConfig,
                          config_constructor=IBMMemoryPrecisionConfig.get_precision_at_3_config,
                          component_class=MemoryPrecision,
                          name='metrics',
                          tags={'memory', 'precision', 'k=3'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryMRRConfig,
                          component_class=MemoryMRR,
                          name='metrics',
                          tags={'memory', 'mrr'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryClassificationPrecisionConfig,
                          component_class=MemoryClassificationPrecision,
                          name='metrics',
                          tags={'memory', 'classification', 'precision'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMNonMemoryClassificationPrecisionConfig,
                          component_class=NonMemoryClassificationPrecision,
                          name='metrics',
                          tags={'non-memory', 'classification', 'precision'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMMemoryAPM,
                          component_class=MemoryAPM,
                          name='metrics',
                          tags={'memory', 'apm'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=PipelineConfig,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='metrics', tags={'clf_f1'}, namespace='nle'),
                                  RegistrationKey(name='metrics', tags={'memory', 'usage'}, namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'coverage'}, namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'coverage', 'precision'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'mrr'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'precision', 'k=1'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'precision', 'k=3'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'classification', 'precision'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'non-memory', 'classification', 'precision'},
                                                  namespace='nle/ibm'),
                                  RegistrationKey(name='metrics', tags={'memory', 'apm'},
                                                  namespace='nle/ibm'),
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
                          namespace='nle/ibm')
