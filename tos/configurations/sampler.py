from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, register
from components.sampler import KBSampler, AttentionKBSampler, LossGainKBSampler
from configurations.sampler import KBSamplerConfig


class ToSKBSamplerConfig(KBSamplerConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.epsilon = 0.01
        config.alpha = 0.9
        # config.sampling_size = 5
        config.get('sampling_size').variants = [1, 5, 10, 15]

        return config

    @classmethod
    def get_full_config(
            cls
    ):
        config = super().get_default()

        config.epsilon = 0.01
        config.alpha = 0.9
        config.sampling_size = -1

        return config


@register
def register_samplers():
    Registry.add_and_bind(config_class=ToSKBSamplerConfig,
                          config_constructor=ToSKBSamplerConfig.get_full_config,
                          component_class=KBSampler,
                          name='sampler',
                          tags={'full'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSKBSamplerConfig,
                          component_class=KBSampler,
                          name='sampler',
                          tags={'uniform'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSKBSamplerConfig,
                          component_class=AttentionKBSampler,
                          name='sampler',
                          tags={'attention'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSKBSamplerConfig,
                          component_class=LossGainKBSampler,
                          name='sampler',
                          tags={'gain'},
                          namespace='nle/tos')
