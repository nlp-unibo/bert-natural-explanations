from cinnamon_generic.configurations.model import NetworkConfig
from typing import Type
from cinnamon_core.core.configuration import C


class MemoryNetworkConfig(NetworkConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='kb_sampler',
                   is_child=True,
                   is_required=True,
                   description='KB sampler component.')
        config.add(name='lookup_weights',
                   is_required=True,
                   description='MLP weights for memory lookup module')
        config.add(name='ss_margin',
                   value=0.5,
                   type_hint=float,
                   description='Margin for strong supervision loss.')
        config.add(name='ss_coefficient',
                   is_required=True,
                   type_hint=float,
                   description='Strong supervision loss coefficient.')

        return config
