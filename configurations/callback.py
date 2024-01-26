from typing import Type

from cinnamon_core.core.registry import Registry, register
from cinnamon_core.core.configuration import Configuration, C
from components.callback import SamplerPriorityUpdater


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


class SamplerPriorityUpdaterConfig(Configuration):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.add(name='update_rate',
                   value=1,
                   is_required=True,
                   allowed_range=lambda value: value > 0,
                   description='Period (in number of epochs) to update kb sampler priority weights.')

        return config


@register
def register_callbacks():
    Registry.add_and_bind(config_class=SamplerPriorityUpdaterConfig,
                          component_class=SamplerPriorityUpdater,
                          name='callback',
                          tags={'sampler', 'updater'},
                          namespace='nle')