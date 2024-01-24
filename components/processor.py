import math
import os
from functools import partial
from typing import Iterable, Any, Dict

import torch as th
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchdata.datapipes.map import SequenceWrapper
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from cinnamon_core.core.data import FieldDict
from cinnamon_core.utility import logging_utility
from cinnamon_generic.components.processor import Processor
from cinnamon_generic.nlp.components.processor import TokenizerProcessor


# Input/Output

class TextProcessor(Processor):

    def clean_text(
            self,
            text: str
    ):
        return text.lower().strip()

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        if 'text' not in data:
            raise RuntimeError(f'Expected a text field in data...')

        data.text = [self.clean_text(text) for text in data.text]
        return data


class KBProcessor(TextProcessor):

    def process(
            self,
            data: FieldDict,
            is_training_data: bool = False
    ) -> FieldDict:
        if 'kb' not in data:
            raise RuntimeError(f'Expected a kb field in data...')

        data.kb = {key: [self.clean_text(kb_text) for kb_text in value] for key, value in data.kb.items()}
        return data


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

class THTextTokenizer(TokenizerProcessor):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = get_tokenizer(**self.tokenizer_args)
        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'

    def fit(
            self,
            data: FieldDict,
    ):
        if 'text_sequence' not in data:
            raise RuntimeError(f'Expected a text_sequence field in data...')

        self.vocabulary = build_vocab_from_iterator(iterator=[self.tokenizer(text) for text in data.text_sequence],
                                                    specials=[self.pad_token, self.unk_token],
                                                    special_first=True)
        self.vocab_size = len(self.vocabulary)

    def finalize(
            self
    ):
        """
        If specified, it loads embedding model and computes pre-trained embedding matrix.
        """

        if self.embedding_type is not None:
            self.load_embedding_model()

            if self.embedding_model is None:
                raise RuntimeError(f'Expected a pre-trained embedding model. Got {self.embedding_model}')

            vocabulary, added_tokens = self.build_embeddings_matrix(vocabulary=self.vocabulary.get_stoi(),
                                                                    embedding_model=self.embedding_model,
                                                                    embedding_dimension=self.embedding_dimension)
            for token in added_tokens:
                self.vocabulary.append_token(token)

    def tokenize(
            self,
            text: Iterable[str],
            remove_special_tokens: bool = False
    ) -> Any:
        return [[self.vocabulary[token] if token in self.vocabulary else self.vocabulary[self.unk_token]
                 for token in self.tokenizer(seq)] for seq in text]

    def detokenize(
            self,
            ids: Iterable[int],
            remove_special_tokens: bool = False
    ) -> Iterable[str]:
        return [' '.join([self.vocabulary[idx] for idx in seq]) for seq in ids]


# Model

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
            'label': th.tensor(y, dtype=th.long).to(device),
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
        data['logits'] = th.argmax(data['logits'], dim=-1)
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
