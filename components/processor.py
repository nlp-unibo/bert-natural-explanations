import math
import os
from functools import partial
from typing import Any, Dict, Optional, Iterable

import numpy as np
import pandas as pd
import torch as th
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper
from transformers import AutoTokenizer

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.processor import Processor
from sklearn.utils.class_weight import compute_class_weight

from torchtext.vocab import build_vocab_from_iterator
from torchtext.data import get_tokenizer


# Input/Output

class LabelProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.encoder = LabelEncoder()

    def prepare_save_data(
            self
    ) -> Dict:
        data = super().prepare_save_data()

        data['encoder'] = self.encoder
        return data

    def clear(
            self
    ):
        super().clear()
        self.encoder = LabelEncoder()

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ):
        label = data.label
        data['label'] = self.encoder.fit_transform(label) if is_training_data else self.encoder.transform(label)
        return data


# Tokenizer


class HFTokenizer(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = None
        self.vocabulary = None
        self.vocab_size = None

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        # Efficiency
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.hf_model_name)
            self.vocabulary = self.tokenizer.get_vocab()
            self.vocab_size = len(self.vocabulary)

        data.add(name='pad_token_id', value=self.tokenizer.pad_token_id)
        data.add(name='input_ids', value=[])
        data.add(name='attention_mask', value=[])
        data.add(name='id', value=[])

        text = data['text']
        tok_info = self.tokenizer(text, **self.tokenization_args)

        data.input_ids.extend(tok_info['input_ids'])
        data.attention_mask.extend(tok_info['attention_mask'])
        data.id.extend(np.arange(len(text)).tolist())

        return data


class HFKBTokenizer(HFTokenizer):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kb = None

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        data = super().run(data=data, is_training_data=is_training_data)

        kb = data['kb']
        tok_info = self.tokenizer(kb, **self.tokenization_args)
        self.kb = FieldDict({
            'input_ids': tok_info['input_ids'],
            'attention_mask': tok_info['attention_mask'],
            'pad_token_id': data.pad_token_id
        })

        return data


class THTokenizer(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = get_tokenizer(**self.tokenization_args)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

    def fit(
            self,
            data: FieldDict,
    ):
        self.vocabulary = build_vocab_from_iterator(iterator=[self.tokenizer(text) for text in data.text],
                                                    specials=[self.pad_token, self.unk_token],
                                                    special_first=True)
        self.vocab_size = len(self.vocabulary)

    def tokenize(
            self,
            text: Iterable[str]
    ) -> Any:
        return [[self.vocabulary[token] if token in self.vocabulary else self.vocabulary[self.unk_token]
                 for token in self.tokenizer(seq)] for seq in text]

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        data.add(name='pad_token_id', value=0)
        data.add(name='input_ids', value=[])
        data.add(name='attention_mask', value=[])
        data.add(name='id', value=[])

        if is_training_data:
            self.fit(data=data)

        text = data.text
        input_ids = self.tokenize(text=text)
        attention_mask = [[1] * len(seq) for seq in input_ids]

        data.input_ids.extend(input_ids)
        data.attention_mask.extend(attention_mask)
        data.id.extend(np.arange(len(text)).tolist())

        return data


class THKBTokenizer(THTokenizer):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.kb = None

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        data = super().run(data=data, is_training_data=is_training_data)

        kb = data['kb']
        input_ids = self.tokenize(text=kb)
        attention_mask = [[1] * len(seq) for seq in input_ids]
        self.kb = FieldDict({
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'pad_token_id': data.pad_token_id
        })

        return data


# Model

class PosWeightProcessor(Processor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.pos_weight = None

    def clear(
            self
    ):
        self.pos_weight = None

    def run(
            self,
            data: Optional[FieldDict] = None,
            is_training_data: bool = False
    ) -> Optional[FieldDict]:
        if is_training_data:
            self.pos_weight = compute_class_weight(y=data.label,
                                                   class_weight='balanced',
                                                   classes=[0, 1])[1]

        return data


class ModelProcessor(Processor):

    def batch_data(
            self,
            input_batch,
            device,
            pad_token_id
    ):
        input_ids, attention_mask, sample_id = [], [], []
        y = []

        # input_x is:
        # 0 -> input_ids
        # 1 -> attention_mask
        # 2 -> sample_id
        for input_x, input_y in input_batch:
            input_ids.append(th.tensor(input_x[0], dtype=th.int32))
            attention_mask.append(th.tensor(input_x[1], dtype=th.int32))
            sample_id.append(input_x[2])

            y.append(input_y)

        # input
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token_id)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)
        sample_id = th.tensor(sample_id, dtype=th.int32)

        # input
        x = {
            'input_ids': input_ids.to(device),
            'attention_mask': attention_mask.to(device),
            'sample_id': sample_id.to(device)
        }

        # output
        y = {
            'label': th.tensor(y, dtype=th.float32).to(device),
        }

        return x, y

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        x_input_ids = SequenceWrapper(data.input_ids)
        x_attention_mask = SequenceWrapper(data.attention_mask)
        x_id = SequenceWrapper(data.id)

        x_data = x_input_ids.zip(x_attention_mask, x_id)

        y_data = SequenceWrapper(data.label)

        th_data = x_data.zip(y_data)

        if is_training_data:
            th_data = th_data.shuffle()

        device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        th_data = DataLoader(th_data,
                             shuffle=is_training_data,
                             batch_size=self.batch_size,
                             num_workers=self.num_workers,
                             collate_fn=partial(self.batch_data,
                                                device=device,
                                                pad_token_id=data.pad_token_id))

        steps = math.ceil(len(data.input_ids) / self.batch_size)

        return FieldDict({'iterator': lambda: iter(th_data),
                          'input_iterator': lambda: iter(th_data.map(lambda x, y: x)),
                          'output_iterator': lambda: iter(th_data.map(lambda x, y: y.detach().cpu().numpy())),
                          'steps': steps})


class THClassifierProcessor(Processor):

    def process(
            self,
            data: Any,
            is_training_data: bool = False
    ) -> Any:
        data['logits'] = th.round(th.sigmoid(data['logits']))
        return data


# Routine

class ResultsProcessor(Processor):

    def compute_best_per_fold(
            self,
            steps
    ):
        """
        Computes metric values (mean and std) only considering the best seed per fold.
        """
        metric_dict = {}
        for step_info in steps:
            metric_dict.setdefault('seed', []).append(step_info.seed)
            metric_dict.setdefault('fold', []).append(step_info.fold)

            for metric_name, metric_value in step_info.val_info.metrics.items():
                metric_dict.setdefault(f'val_{metric_name}', []).append(metric_value)

            for metric_name, metric_value in step_info.test_info.metrics.items():
                metric_dict.setdefault(f'test_{metric_name}', []).append(metric_value)

        metric_df = pd.DataFrame.from_dict(metric_dict)
        metric_names = [col for col in metric_df.columns if col.startswith('test')]
        best_series = []
        for _, fold_df in metric_df.groupby('fold'):
            best_idx = fold_df.reset_index().idxmax(axis=0)['val_clf_f1']
            best_series.append(fold_df.iloc[best_idx][metric_names])

        best_df = pd.DataFrame(best_series, columns=best_series[0].keys().values.tolist())
        best_dict = {col: f'{np.mean(best_df[col].values)} +/- {np.std(best_df[col].values)}' for col in best_df.columns}
        return best_dict

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ):
        avg_info = data['average']['all']

        results = {}
        for key, value in avg_info['average'].items():
            results[key] = f'{value:.4f} +/- {avg_info["std"][key]:.4f}'

        data.add(name='parsed', value=results)
        logging_utility.logger.info(f'Model performance: {os.linesep}{results}')

        best_per_fold = self.compute_best_per_fold(steps=data.steps)
        data.add(name='best_per_fold', value=best_per_fold)
        logging_utility.logger.info(f'Best per fold: {os.linesep}{best_per_fold}')

        return data
