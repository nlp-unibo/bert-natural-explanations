from typing import Optional, Any, Dict, Tuple

import torch as th

from cinnamon_generic.components.callback import Callback
from cinnamon_generic.components.processor import Processor
from cinnamon_th.components.model import THNetwork
from modeling.baseline import M_HFBaseline


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
        ce_loss = self.ce(th.sigmoid(output['logits']), batch_y['label'])

        total_loss += ce_loss
        true_loss += ce_loss
        loss_info = {
            'CE': ce_loss
        }

        return total_loss, true_loss, loss_info, output, model_additional_info
