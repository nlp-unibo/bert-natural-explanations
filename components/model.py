from typing import Optional, Any, Dict, Tuple

import torch as th

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.processor import Processor
from cinnamon_th.components.model import THNetwork
from modeling.baseline import M_HFBaseline
from modeling.memory import M_HFMem


# HF Models

class HFBaseline(THNetwork):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_HFBaseline(num_classes=self.num_classes,
                                  hf_model_name=self.hf_model_name,
                                  freeze_hf=self.freeze_hf).to(self.get_device())

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
        ce_loss = self.bce(th.sigmoid(output['logits']), batch_y['label'].unsqueeze(-1)).squeeze(-1)
        ce_loss *= pos_weights
        ce_loss = ce_loss.sum(dim=-1) / pos_weights.sum(dim=-1)

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        return total_loss, true_loss, loss_info, output, model_additional_info


class HFMem(THNetwork):

    def build(
            self,
            processor: Processor,
            callbacks: Optional[Callback] = None
    ):
        self.model = M_HFMem(num_classes=self.num_classes,
                             hf_model_name=self.hf_model_name,
                             freeze_hf=self.freeze_hf,
                             lookup_weights=self.lookup_weights).to(self.get_device())

        self.optimizer = self.optimizer_class(**self.optimizer_args,
                                              params=self.model.parameters())
        self.bce = th.nn.BCELoss(reduction='none').to(self.get_device())

        self.pos_weight = processor.find('pos_weight')

        self.kb = processor.find('kb')

    def input_additional_info(
            self
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
        pos_weights = th.where(batch_y['label'] == 1, self.pos_weight, 1.0)
        mem_ce_loss = self.bce(th.sigmoid(output['logits']), batch_y['label'].unsqueeze(-1)).squeeze(-1)
        mem_ce_loss *= pos_weights

        # []
        ce_loss = mem_ce_loss.sum(dim=-1) / pos_weights.sum(dim=-1)

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        # TODO: add SS

        # Input only downstream task loss
        # [bs]
        input_ce_loss = self.bce(th.sigmoid(model_additional_info['input_only_logits']),
                                 batch_y['label'].unsqueeze(-1)).squeeze(-1)
        input_ce_loss *= pos_weights

        # Update sampler
        model_info = {
            'memory_scores': model_additional_info['memory_scores'],
            'positive_mask': batch_y['label'],
            'input_only_bce': input_ce_loss,
            'mem_bce': mem_ce_loss
        }
        self.kb_sampler.update(model_info=model_info,
                               memory_indices=input_additional_info['memory_indices'])

        return total_loss, true_loss, loss_info, output, model_additional_info
