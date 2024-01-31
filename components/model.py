from typing import Optional, Any, Dict, Tuple

import torch as th

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.processor import Processor
from cinnamon_th.components.model import THNetwork
from modeling.baseline import M_HFBaseline, M_LSTMBaseline
from modeling.losses import strong_supervision
from modeling.memory import M_HFMANN, M_MANN


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
        self.class_weights = th.tensor(processor.find('class_weights'), dtype=th.float32).to(self.get_device())
        self.ce = th.nn.CrossEntropyLoss(reduction='none', weight=self.class_weights).to(self.get_device())

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
        ce_loss = self.ce(output['logits'], batch_y['label']).mean()

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
        self.class_weights = th.tensor(processor.find('class_weights'), dtype=th.float32).to(self.get_device())
        self.ce = th.nn.CrossEntropyLoss(reduction='none', weight=self.class_weights).to(self.get_device())

        self.kb = processor.find('kb')

    def input_additional_info(
            self,
            batch_x: Any,
            batch_y: Any
    ) -> Dict:
        sampler_info = self.kb_sampler.run(kb=self.kb)
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
        mem_ce_loss = self.ce(output['logits'], batch_y['label'])

        # []
        ce_loss = mem_ce_loss.mean()

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        # Strong supervision (SS)

        # [M]
        memory_indices = input_additional_info['memory_indices'].to(th.long)

        # [bs, M]
        memory_targets = batch_y['memory_targets'][:, memory_indices]
        output['sampled_indices'] = memory_indices[None, :].expand(memory_targets.shape[0], -1)

        ss_loss = strong_supervision(memory_scores=output['memory_scores'],
                                     memory_targets=memory_targets,
                                     margin=self.ss_margin)
        total_loss += ss_loss * self.ss_coefficient
        true_loss += ss_loss
        loss_info['SS'] = ss_loss

        # Input only downstream task loss
        # [bs]
        input_ce_loss = self.ce(model_additional_info['input_only_logits'], batch_y['label'])

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
        self.class_weights = th.tensor(processor.find('class_weights'), dtype=th.float32).to(self.get_device())
        self.ce = th.nn.CrossEntropyLoss(reduction='none', weight=self.class_weights).to(self.get_device())

        self.kb = processor.find('kb')

    def batch_fit(
            self,
            batch_x: Any,
            batch_y: Any,
            input_additional_info: Dict = {}
    ) -> Tuple[Dict, Any, Dict]:
        loss, \
            true_loss, \
            loss_info, \
            predictions, \
            model_additional_info, = self.batch_train(batch_x=batch_x,
                                                      batch_y=batch_y,
                                                      input_additional_info=input_additional_info)

        # Gradient clipping
        th.nn.utils.clip_grad_norm_(self.model.parameters(), 40)

        self.optimizer.step()

        loss_info['loss'] = true_loss
        return loss_info, predictions, model_additional_info


class LSTMBaseline(THNetwork):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_LSTMBaseline(num_classes=self.num_classes,
                                    embedding_dimension=self.embedding_dimension,
                                    embedding_matrix=processor.find('embedding_matrix'),
                                    vocab_size=processor.find('vocab_size'),
                                    lstm_weights=self.lstm_weights,
                                    pre_classifier_weight=self.pre_classifier_weight,
                                    dropout_rate=self.dropout_rate).to(self.get_device())

        self.optimizer = self.optimizer_class(**self.optimizer_args,
                                              params=self.model.parameters())
        self.class_weights = th.tensor(processor.find('class_weights'), dtype=th.float32).to(self.get_device())
        self.ce = th.nn.CrossEntropyLoss(reduction='none', weight=self.class_weights).to(self.get_device())

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
        ce_loss = self.ce(output['logits'], batch_y['label']).mean()

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        return total_loss, true_loss, loss_info, output, model_additional_info
