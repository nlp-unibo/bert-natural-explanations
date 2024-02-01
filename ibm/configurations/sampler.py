from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, register
from components.sampler import KBSampler, AttentionKBSampler, LossGainKBSampler
from configurations.sampler import KBSamplerConfig


class IBMKBSamplerConfig(KBSamplerConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.epsilon = 0.01
        config.alpha = 0.9
        config.sampling_size = 10

        return config

    @classmethod
    def get_full_config(
            cls
    ):
        config = cls.get_default()

        config.sampling_size = -1

        return config


@register
def register_samplers():
    Registry.add_and_bind(config_class=IBMKBSamplerConfig,
                          component_class=KBSampler,
                          name='sampler',
                          tags={'uniform'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMKBSamplerConfig,
                          component_class=AttentionKBSampler,
                          name='sampler',
                          tags={'attention'},
                          namespace='nle/ibm')

    Registry.add_and_bind(config_class=IBMKBSamplerConfig,
                          component_class=LossGainKBSampler,
                          name='sampler',
                          tags={'gain'},
                          namespace='nle/ibm')
