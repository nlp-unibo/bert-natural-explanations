from pathlib import Path

import pandas as pd
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


class MemoryInfoRetriever(Callback):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.lookup = {}
        self.routine_metrics = {}

    def on_batch_evaluate_end(
            self,
            logs=None
    ):
        if 'model_additional_info' not in logs:
            return

        model_additional_info = logs['model_additional_info']
        memory_scores = model_additional_info['memory_scores'].detach().cpu().numpy()
        memory_targets = model_additional_info['memory_targets'].detach().cpu().numpy()
        sample_id = model_additional_info['sample_id'].detach().cpu().numpy()
        predictions = logs['batch_predictions']['logits'].detach().cpu().numpy()
        label = logs['batch_y']['label'].detach().cpu().numpy()
        suffixes = logs['suffixes']

        self.lookup.setdefault('memory_scores', []).append(memory_scores)
        self.lookup.setdefault('sample_id', []).append(sample_id)
        self.lookup.setdefault('memory_targets', []).append(memory_targets)
        self.lookup.setdefault('predictions', []).append(predictions)
        self.lookup.setdefault('label', []).append(label)
        for suffix_name, suffix_value in suffixes.items():
            self.lookup.setdefault(suffix_name, []).append([suffix_value] * memory_scores.shape[0])

    def on_batch_predict_end(
            self,
            logs=None
    ):
        return self.on_batch_evaluate_end(logs=logs)

    def on_routine_step_end(
            self,
            logs=None
    ):
        lookup_df = pd.DataFrame.from_dict(self.lookup)
        step_df = lookup_df[(lookup_df.seed == logs['seed']) & (lookup_df.fold == logs['fold'])]

        # Validation
        step_val_df = step_df[step_df.split == 'val']
        step_val_y_pred = step_val_df[['memory_scores', 'predictions']].to_dict()
        step_val_y_true = step_val_df[['memory_targets', 'label']].to_dict()
        step_val_metrics = self.memory_metrics.run(y_pred=step_val_y_pred,
                                                   y_true=step_val_y_true,
                                                   as_dict=True)
        for metric_name, metric_value in step_val_metrics.items():
            self.routine_metrics.setdefault(metric_name, []).append(metric_value)
            self.routine_metrics.setdefault('seed', []).append(logs['seed'])
            self.routine_metrics.setdefault('fold', []).append(logs['fold'])
            self.routine_metrics.setdefault('split', []).append('val')

        step_test_df = step_df[step_df.split == 'test']
        step_test_y_pred = step_test_df[['memory_scores', 'predictions']].to_dict()
        step_test_y_true = step_test_df[['memory_targets', 'label']].to_dict()
        step_test_metrics = self.memory_metrics.run(y_pred=step_test_y_pred,
                                                    y_true=step_test_y_true,
                                                    as_dict=True)
        for metric_name, metric_value in step_test_metrics.items():
            self.routine_metrics.setdefault(metric_name, []).append(metric_value)
            self.routine_metrics.setdefault('seed', []).append(logs['seed'])
            self.routine_metrics.setdefault('fold', []).append(logs['fold'])
            self.routine_metrics.setdefault('split', []).append('test')

        self.fragments_dict.clear()

    def on_routine_end(
            self,
            logs=None
    ):
        if 'serialization_path' not in logs:
            return

        serialization_path: Path = logs['serialization_path']

        if serialization_path is None:
            return

        # TODO: compute best folds per seed according to clf metric
        # TODO: compute mean/std memory metrics values (avg all folds and seeds and over best folds)

        self.routine_metrics.clear()