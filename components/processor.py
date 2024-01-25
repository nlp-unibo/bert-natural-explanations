import math
import os
from functools import partial
from typing import Any, Dict, Optional

import numpy as np
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

        return data
