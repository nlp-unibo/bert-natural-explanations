import torch as th
from transformers import AutoConfig, AutoModel
from modeling.layers import MemoryLookup, MemoryExtraction, MemoryReasoning


class M_HFMem(th.nn.Module):

    def __init__(
            self,
            hf_model_name,
            num_classes,
            lookup_weights,
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

        # Memory
        self.memory_lookup = MemoryLookup(embedding_dim=self.embedding_config.hidden_size,
                                          lookup_weights=lookup_weights)
        self.memory_extraction = MemoryExtraction()
        self.memory_reasoning = MemoryReasoning()

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.pre_classifier = th.nn.Linear(in_features=self.embedding_config.hidden_size * 2,
                                           out_features=self.embedding_config.hidden_size)
        self.pre_activation = th.nn.ReLU()
        self.classifier = th.nn.Linear(out_features=num_classes,
                                       in_features=self.embedding_config.hidden_size)

    def compute_logits(
            self,
            classification_emb
    ):
        # classification_emb: [bs, 2 * d]

        pre_logits = self.pre_classifier(classification_emb)
        pre_logits = self.pre_activation(pre_logits)
        if self.training:
            pre_logits = self.dropout(pre_logits)
        logits = self.classifier(pre_logits)
        return logits

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

        # [M, N]
        kb_input_ids = input_additional_info['kb_input_ids']

        # [M, N]
        kb_attention_mask = input_additional_info['kb_attention_mask']

        # Input
        # [bs, N, d]
        input_embeddings = self.embedding(input_ids=input_ids,
                                          attention_mask=attention_mask).last_hidden_state
        # [bs, d]
        input_embedding = th.mean(input_embeddings, dim=1)

        # Memory
        # [M, N, d]
        memory_embeddings = self.embedding(input_ids=kb_input_ids,
                                           attention_mask=kb_attention_mask).last_hidden_state
        # [M, d]
        memory_embedding = th.mean(memory_embeddings, dim=1)

        # [bs, M]
        memory_scores = self.memory_lookup(input_embedding=input_embedding,
                                           memory_embedding=memory_embedding)

        # [bs, d]
        memory_extraction = self.memory_extraction(memory_scores=memory_scores,
                                                   memory_embedding=memory_embedding)

        # [bs, 2*d]
        mem_classification_emb = self.memory_reasoning(memory_extraction=memory_extraction,
                                                       input_embedding=input_embedding)
        mem_logits = self.compute_logits(classification_emb=mem_classification_emb)

        # [bs, 2*d]
        input_classification_emb = th.concat((input_embedding, th.zeros_like(input_embedding)), dim=-1)
        input_only_logits = self.compute_logits(classification_emb=input_classification_emb.detach())

        return {
            'logits': mem_logits,
        }, {
            'memory_scores': memory_scores,
            'input_only_logits': input_only_logits
        }
