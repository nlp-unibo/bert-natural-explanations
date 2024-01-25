from typing import Type

from cinnamon_core.core.configuration import Configuration, C


class KBSamplerConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='sampling_size',
                   is_required=True,
                   description='Maximum number of memory slots to sample.')
        config.add(name='epsilon',
                   value=0.01,
                   description='Epsilon hyper-parameter of PER.')
        config.add(name='alpha',
                   value=0.7,
                   description='Alpha hyper-parameter of PER.')

        return config



