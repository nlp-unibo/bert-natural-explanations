from pathlib import Path

import wandb

from cinnamon_core.core.component import Component
from cinnamon_core.utility import logging_utility
from cinnamon_core.utility.pickle_utility import save_pickle
from cinnamon_generic.components.callback import Callback


class WandDB(Callback):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.step_info = None

    def on_routine_step_begin(
            self,
            logs=None
    ):
        self.step_info = {key: value for key, value in logs.items() if key not in ['is_training', 'serialization_path']}

    def setup(
            self,
            component: Component,
            save_path: Path
    ):
        super().setup(component=component, save_path=save_path)

        if self.disabled:
            return

        if save_path is None:
            self.disabled = True
            return

        config = {**component.config.to_value_dict(), **self.step_info}
        step_info_suffix = '_'.join([f'{key}-{value}' for key, value in self.step_info.items()])
        name = f'{save_path.stem}_{step_info_suffix}'

        try:
            wandb.init(
                entity=self.entity,
                project=self.project,
                config=config,
                name=name
            )
        except Exception as e:
            logging_utility.logger.warning(f'Could not log in to wandb account. '
                                           f'Reason {e}.'
                                           f'Ignoring wandb uploader...')
            self.disabled = True

    def on_epoch_end(
            self,
            logs=None
    ):
        if self.disabled:
            return

        if logs is not None:
            wandb.log({key: value for key, value in logs.items() if key != 'epoch'})

    def on_routine_step_end(
            self,
            logs=None
    ):
        if self.disabled:
            return

        wandb.finish()


class SamplerPriorityUpdater(Callback):

    def on_epoch_end(
            self,
            logs=None
    ):
        epoch = logs['epoch']
        if epoch > 0 and epoch % self.update_rate == 0:
            self.component.kb_sampler.update_priority()
            logging_utility.logger.info('Updating sampler priority...')