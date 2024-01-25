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

        return config
