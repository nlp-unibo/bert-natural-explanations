from typing import Optional, Any, Dict, Tuple

import torch as th

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.processor import Processor
from cinnamon_th.components.model import THNetwork
from modeling.baseline import M_HFBaseline
from modeling.memory import M_HFMANN, M_MANN
from modeling.losses import strong_supervision


# HF Models

class HFBaseline(THNetwork):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_HFBaseline(num_classes=self.num_classes,
                                  hf_model_name=self.hf_model_name,
                                  freeze_hf=self.freeze_hf,
                                  dropout_rate=self.dropout_rate).to(self.get_device())

        self.optimizer = self.optimizer_class(**self.optimizer_args,
                                              params=self.model.parameters())
        self.bce = th.nn.BCELoss(reduction='none').to(self.get_device())

        self.pos_weight = processor.find('pos_weight')

    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {},
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        (output,
         model_additional_info) = self.model(batch_x,
                                             input_additional_info=input_additional_info)
        total_loss = 0
        true_loss = 0

        # Downstream task loss
        # [bs]
        pos_weights = th.where(batch_y['label'] == 1, self.pos_weight, 1.0)
        self.bce.weight = pos_weights
        ce_loss = self.bce(th.sigmoid(output['logits']), batch_y['label'])
        ce_loss = ce_loss.sum(dim=-1) / pos_weights.sum(dim=-1)

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        return total_loss, true_loss, loss_info, output, model_additional_info


class HFMANN(THNetwork):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_HFMANN(num_classes=self.num_classes,
                              hf_model_name=self.hf_model_name,
                              freeze_hf=self.freeze_hf,
                              lookup_weights=self.lookup_weights,
                              dropout_rate=self.dropout_rate).to(self.get_device())

        self.optimizer = self.optimizer_class(**self.optimizer_args,
                                              params=self.model.parameters())
        self.bce = th.nn.BCELoss(reduction='none').to(self.get_device())

        self.pos_weight = processor.find('pos_weight')

        self.kb = processor.find('kb')

    def input_additional_info(
            self,
            batch_x: Any,
            batch_y: Any
    ) -> Dict:
        batch_size = batch_x['input_ids'].shape[0]
        sampler_info = self.kb_sampler.run(kb=self.kb,
                                           batch_size=batch_size)
        return {key: value.to(self.get_device()) for key, value in sampler_info.items()}

    def batch_loss(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {},
    ) -> Tuple[Any, Any, Dict, Any, Dict]:
        (output,
         model_additional_info) = self.model(batch_x,
                                             input_additional_info=input_additional_info)
        total_loss = 0
        true_loss = 0

        # Downstream task loss
        # [bs]
        pos_weights = th.where(batch_y['label'] == 1, self.pos_weight, 1.0)
        self.bce.weight = pos_weights
        mem_ce_loss = self.bce(th.sigmoid(output['logits']), batch_y['label'])

        # []
        ce_loss = mem_ce_loss.sum(dim=-1) / pos_weights.sum(dim=-1)

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        # Strong supervision (SS)

        # [M]
        memory_indices = input_additional_info['memory_indices'].to(th.long)

        # [bs, M]
        memory_targets = th.take_along_dim(batch_y['memory_targets'], memory_indices, dim=1)
        output['sampled_indices'] = memory_indices

        ss_loss = strong_supervision(memory_scores=output['memory_scores'],
                                     memory_targets=memory_targets,
                                     margin=self.ss_margin)
        total_loss += ss_loss * self.ss_coefficient
        true_loss += ss_loss
        loss_info['SS'] = ss_loss

        # Input only downstream task loss
        # [bs]
        input_ce_loss = self.bce(th.sigmoid(model_additional_info['input_only_logits']), batch_y['label'])
        input_ce_loss *= pos_weights

        # Update sampler
        model_info = {
            'memory_scores': output['memory_scores'],
            'positive_mask': batch_y['label'],
            'input_only_bce': input_ce_loss,
            'mem_bce': mem_ce_loss
        }
        self.kb_sampler.update(model_info=model_info,
                               memory_indices=memory_indices)

        return total_loss, true_loss, loss_info, output, model_additional_info


class MANN(HFMANN):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_MANN(num_classes=self.num_classes,
                            embedding_dimension=self.embedding_dimension,
                            embedding_matrix=processor.find('embedding_matrix'),
                            vocab_size=processor.find('vocab_size'),
                            lookup_weights=self.lookup_weights,
                            dropout_rate=self.dropout_rate,
                            pre_classifier_weight=self.pre_classifier_weight).to(self.get_device())

        self.optimizer = self.optimizer_class(**self.optimizer_args,
                                              params=self.model.parameters())
        self.bce = th.nn.BCELoss(reduction='none').to(self.get_device())

        self.pos_weight = processor.find('pos_weight')

        self.kb = processor.find('kb')
