from typing import Type

from cinnamon_core.core.configuration import Configuration, C


class WandDBConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='entity',
                   value='federico-ruggeri6',
                   is_required=True,
                   type_hint=str,
                   description='Profile name on wandb for login.')
        config.add(name='project',
                   value='nsf',
                   is_required=True,
                   type_hint=str,
                   description='Project name on wandb.')
        config.add(name='disabled',
                   value=True,
                   type_hint=bool,
                   description='If True, the callback is disabled.')

        return config
