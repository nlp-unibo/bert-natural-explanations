from typing import Type, Dict

import torch as th

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, register
from cinnamon_generic.configurations.model import NetworkConfig
from components.model import HFBaseline


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
                   value=2,
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


@register
def register_models():
    Registry.add_and_bind_variants(config_class=HFBaselineConfig,
                                   component_class=HFBaseline,
                                   name='model',
                                   tags={'hf', 'baseline'},
                                   namespace='nle/tos')
