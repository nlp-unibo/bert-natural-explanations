import torch as th
from transformers import AutoConfig, AutoModel


class M_HFBaseline(th.nn.Module):

    def __init__(
            self,
            hf_model_name,
            num_classes,
            freeze_hf=False,
            dropout_rate=0.0
    ):
        super().__init__()

        # Input
        self.embedding_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=hf_model_name)
        self.embedding = AutoModel.from_pretrained(pretrained_model_name_or_path=hf_model_name)

        if freeze_hf:
            for module in self.embedding.modules():
                for param in module.parameters():
                    param.requires_grad = False

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.pre_classifier = th.nn.Linear(in_features=self.embedding_config.hidden_size,
                                           out_features=self.embedding_config.hidden_size)
        self.pre_activation = th.nn.ReLU()
        self.classifier = th.nn.Linear(out_features=num_classes,
                                       in_features=self.embedding_config.hidden_size)

    def forward(
            self,
            inputs,
            input_additional_info={}
    ):
        # Step 0: Inputs

        # [bs, N]
        input_ids = inputs['input_ids']

        # [bs, N]
        attention_mask = inputs['attention_mask']

        # [bs, N, d]
        embeddings = self.embedding(input_ids=input_ids,
                                    attention_mask=attention_mask).last_hidden_state
        # [bs, d]
        input_embedding = th.mean(embeddings, dim=1)

        pre_logits = self.pre_classifier(input_embedding)
        pre_logits = self.pre_activation(pre_logits)
        if self.training:
            pre_logits = self.dropout(pre_logits)
        logits = self.classifier(pre_logits).squeeze(-1)

        return {
            'logits': logits
        }, None
