from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import RegistrationKey, Registry, register
from components.pipeline import ResultsPipeline
from configurations.pipeline import ResultsPipelineConfig


class IBMResultsPipelineConfig(ResultsPipelineConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.get('routine').variants = [
            RegistrationKey(name='routine',
                            tags={'hf', 'baseline'},
                            namespace='nle/ibm'),
            RegistrationKey(name='routine',
                            tags={'kb', 'hf'},
                            namespace='nle/ibm'),
        ]

        return config


@register
def register_pipelines():
    Registry.add_and_bind_variants(config_class=IBMResultsPipelineConfig,
                                   component_class=ResultsPipeline,
                                   name='pipeline',
                                   namespace='nle/ibm')
