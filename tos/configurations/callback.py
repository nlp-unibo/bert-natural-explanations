from typing import Type

from cinnamon_core.core.configuration import C
from cinnamon_core.core.registry import Registry, RegistrationKey, register
from cinnamon_generic.components.callback import CallbackPipeline
from cinnamon_generic.configurations.pipeline import PipelineConfig
from cinnamon_th.components.callback import THEarlyStopping
from cinnamon_th.configurations.callback import THEarlyStoppingConfig
from components.callback import WandDB
from configurations.callback import WandDBConfig


class ToSEarlyStoppingConfig(THEarlyStoppingConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.patience = 5
        config.monitor = 'val_clf_f1'

        return config


class ToSWandDBConfig(WandDBConfig):

    @classmethod
    def get_default(
            cls: Type[C]
    ) -> C:
        config = super().get_default()

        config.project = 'nle-tos'
        config.disabled = False

        return config


@register
def register_callback_configurations():
    Registry.add_and_bind(config_class=ToSEarlyStoppingConfig,
                          component_class=THEarlyStopping,
                          name='callback',
                          tags={'early_stopping', 'th'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=ToSWandDBConfig,
                          component_class=WandDB,
                          name='callback',
                          tags={'wandb'},
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=PipelineConfig,
                          component_class=CallbackPipeline,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='callback',
                                                  tags={'early_stopping', 'th'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='callback',
                                                  tags={'wandb'},
                                                  namespace='nle/tos')

                              ],
                              'names': [
                                  'early_stopping',
                                  'wandb_logger'
                              ]
                          },
                          name='callback',
                          namespace='nle/tos')

    Registry.add_and_bind(config_class=PipelineConfig,
                          component_class=CallbackPipeline,
                          config_constructor=PipelineConfig.from_keys,
                          config_kwargs={
                              'keys': [
                                  RegistrationKey(name='callback',
                                                  tags={'early_stopping', 'th'},
                                                  namespace='nle/tos'),
                                  RegistrationKey(name='callback',
                                                  tags={'sampler', 'updater'},
                                                  namespace='nle'),
                                  RegistrationKey(name='callback',
                                                  tags={'wandb'},
                                                  namespace='nle/tos')

                              ],
                              'names': [
                                  'early_stopping',
                                  'sampler_updater',
                                  'wandb_logger'
                              ]
                          },
                          name='callback',
                          tags={'memory'},
                          namespace='nle/tos')