from typing import Type, Dict

import torch as th

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, register, RegistrationKey
from cinnamon_generic.configurations.model import NetworkConfig
from components.model import HFBaseline, HFMem
from configurations.model import MemoryNetworkConfig


class HFBaselineConfig(NetworkConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.epochs = 30
        config.add(name='hf_model_name',
                   value='prajjwal1/bert-small',
                   is_required=True,
                   description="HugginFace's model card.")
        config.add(name='freeze_hf',
                   value=False,
                   allowed_range=lambda value: value in [False, True],
                   type_hint=bool,
                   description='If enabled, the HF model weights are freezed.')
        config.add(name='num_classes',
                   value=1,
                   type_hint=int,
                   description='Number of classification classes.',
                   tags={'model'},
                   is_required=True)
        config.add(name='optimizer_class',
                   value=th.optim.Adam,
                   is_required=True,
                   tags={'model'},
                   description='Optimizer to use for network weights update')
        config.add(name='optimizer_args',
                   value={
                       "lr": 5e-06,
                       "weight_decay": 1e-05
                   },
                   type_hint=Dict,
                   tags={'model'},
                   description="Arguments for creating the network optimizer")
        config.add(name='dropout_rate',
                   value=0.20,
                   type_hint=float,
                   description='Dropout rate for dropout layer')

        return config


class HFMemConfig(MemoryNetworkConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        for key, value in HFBaselineConfig.get_default().items():
            config[key] = value

        config.lookup_weights = [
            32
        ]
        config.get('kb_sampler').variants = [
            RegistrationKey(name='sampler',
                            tags={'full'},
                            namespace='nle/tos'),
            RegistrationKey(name='sampler',
                            tags={'uniform'},
                            namespace='nle/tos'),
            RegistrationKey(name='sampler',
                            tags={'attention'},
                            namespace='nle/tos'),
            RegistrationKey(name='sampler',
                            tags={'gain'},
                            namespace='nle/tos'),
        ]

        return config


@register
def register_models():
    Registry.add_and_bind_variants(config_class=HFBaselineConfig,
                                   component_class=HFBaseline,
                                   name='model',
                                   tags={'hf', 'baseline'},
                                   namespace='nle/tos')

    Registry.add_and_bind_variants(config_class=HFMemConfig,
                                   component_class=HFMem,
                                   name='model',
                                   tags={'hf', 'memory'},
                                   namespace='nle/tos')
