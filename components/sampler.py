from typing import Dict, List

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence

from cinnamon_core.core.component import Component
from cinnamon_core.core.data import FieldDict


class KBSampler(Component):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.rng = np.random.default_rng()
        self.accumulated_scores = None
        self.accumulated_counts = None
        self.sampling_priority = None

    def prepare_save_data(
            self
    ) -> Dict:
        save_data = super().prepare_save_data()
        save_data['sampling_priority'] = self.sampling_priority
        return save_data

    def clear(
            self
    ):
        self.accumulated_scores = None
        self.accumulated_counts = None
        self.sampling_priority = None

    def sample(
            self,
            memory_size
    ):
        sampling_size = self.sampling_size if self.sampling_size > 0 else memory_size
        sampling_size = np.minimum(sampling_size, memory_size)
        return self.rng.choice(memory_size,
                               size=sampling_size,
                               p=self.sampling_priority,
                               replace=False)

    # Note: this is invoked by an external callback to control update rate
    def update_priority(
            self,
    ):
        # Un-normalized
        priority = self.accumulated_scores / np.maximum(self.accumulated_counts, 1.0)
        priority = (priority + self.epsilon) ** self.alpha

        # Normalized
        priority = priority / np.sum(priority)

        # Reset since we are changing the underlying sampling distribution
        self.accumulated_scores *= 0
        self.accumulated_counts *= 0

        self.sampling_priority = priority

    # Note: this is invoked internally by the model after computing the loss
    def update(
            self,
            model_info,
            memory_indices
    ):
        pass

    def run(
            self,
            kb: FieldDict,
    ) -> Dict[str, th.Tensor]:
        memory_size = len(kb.input_ids)

        if self.sampling_priority is None:
            self.sampling_priority = np.ones((memory_size,), dtype=np.float) / memory_size
            self.accumulated_scores = np.zeros_like(self.sampling_priority)
            self.accumulated_counts = np.zeros_like(self.sampling_priority)

        # [M]
        sampled_indices = self.sample(memory_size=memory_size)
        input_ids, attention_mask = [], []
        for idx in sampled_indices:
            input_ids.append(th.tensor(kb.input_ids[idx], dtype=th.int32))
            attention_mask.append(th.tensor(kb.attention_mask[idx], dtype=th.float32))

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=kb.pad_token_id).view(len(sampled_indices), -1)
        attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0).view(len(sampled_indices), -1)
        sampled_indices = th.tensor(sampled_indices, dtype=th.int32)

        return {
            'kb_input_ids': input_ids,
            'kb_attention_mask': attention_mask,
            'memory_indices': sampled_indices,
        }


class AttentionKBSampler(KBSampler):

    def update(
            self,
            model_info,
            memory_indices
    ):
        # [bs, M]
        memory_scores = model_info['memory_scores'].detach().cpu().numpy()

        # [bs,]
        positive_mask = model_info['positive_mask'].detach().cpu().numpy()

        # [M]
        memory_scores *= positive_mask[:, np.newaxis]
        memory_scores = np.sum(memory_scores, axis=0)

        memory_indices = memory_indices.detach().cpu().numpy()

        update_mask = np.minimum(np.sum(positive_mask), 1.0)

        self.accumulated_scores[memory_indices] += memory_scores * update_mask
        self.accumulated_counts[memory_indices] += 1 * update_mask


class LossGainKBSampler(KBSampler):

    def update(
            self,
            model_info,
            memory_indices
    ):
        # [bs,]
        input_only_bce = model_info['input_only_bce'].detach().cpu().numpy()
        mem_bce = model_info['mem_bce'].detach().cpu().numpy()

        # [bs,]
        gain = np.clip(input_only_bce - mem_bce, a_min=-3.0, a_max=3.0)
        gain = np.exp(gain)

        # [bs, M]
        memory_scores = model_info['memory_scores'].detach().cpu().numpy()

        # [bs,]
        positive_mask = model_info['positive_mask'].detach().cpu().numpy()

        # [M]
        gain = memory_scores * positive_mask[:, np.newaxis] * gain[:, np.newaxis]
        gain = np.sum(gain, axis=0)

        memory_indices = memory_indices.detach().cpu().numpy()

        update_mask = np.minimum(np.sum(positive_mask), 1.0)

        self.accumulated_scores[memory_indices] += gain * update_mask
        self.accumulated_counts[memory_indices] += 1 * update_mask
