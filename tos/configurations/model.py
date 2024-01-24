from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, register
from components.model import HFBaseline
from cinnamon_generic.configurations.model import NetworkConfig


class HFBaselineConfig(NetworkConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.epochs = 30
        config.hf_model_name = 'prajjwal1/bert-small'
        config.freeze_hf = False
        config.num_classes = 2
        config.optimizer_args = {
            'lr': 1e-03,
            'weight_decay': 1e-05
        }
        config.dropout_rate = 0.2

        return config


@register
def register_models():
    Registry.add_and_bind_variants(config_class=HFBaselineConfig,
                                   component_class=HFBaseline,
                                   name='model',
                                   tags={'hf', 'baseline'},
                                   namespace='nle/tos')
