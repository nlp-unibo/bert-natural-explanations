from typing import Type, List

from cinnamon_core.core.configuration import C, Configuration
from cinnamon_core.core.registry import RegistrationKey, Registry, register
from components.pipeline import ResultsPipeline
from configurations.pipeline import ResultsPipelineConfig


class ToSResultsPipelineConfig(ResultsPipelineConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.get('routine').variants = [
            RegistrationKey(name='routine',
                            tags={'hf', 'baseline'},
                            namespace='nle/tos'),
            RegistrationKey(name='routine',
                            tags={'kb', 'hf'},
                            namespace='nle/tos'),
            RegistrationKey(name='routine',
                            tags={'kb', 'baseline'},
                            namespace='nle/tos'),
            RegistrationKey(name='routine',
                            tags={'lstm', 'baseline'},
                            namespace='nle/tos')
        ]

        return config


@register
def register_pipelines():
    Registry.add_and_bind_variants(config_class=ToSResultsPipelineConfig,
                                   component_class=ResultsPipeline,
                                   name='pipeline',
                                   namespace='nle/tos')
