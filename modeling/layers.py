import torch as th


class MemoryLookup(th.nn.Module):

    def __init__(self,
                 embedding_dim,
                 lookup_weights
                 ):
        super(MemoryLookup, self).__init__()

        self.lookup_mlp = th.nn.Sequential()
        in_features = embedding_dim * 2
        for idx, out_features in enumerate(lookup_weights):
            self.lookup_mlp.add_module(name=f'lookup_{idx}',
                                       module=th.nn.Linear(in_features=in_features,
                                                           out_features=out_features))
            self.lookup_mlp.add_module(name=f'lookup_{idx}_activation',
                                       module=th.nn.LeakyReLU())
            in_features = out_features

        self.lookup_mlp.add_module(name='last_lookup',
                                   module=th.nn.Linear(in_features=in_features,
                                                       out_features=1))

    def forward(
            self,
            input_embedding,
            memory_embedding,
    ):
        # input_embedding:  [bs, d]
        # memory_embedding: [bs, M, d]

        M = memory_embedding.shape[1]
        batch_size = input_embedding.shape[0]

        # [bs * M, 2 * d]
        mlp_input = th.concat((input_embedding[:, None, :].expand(-1, M, -1), memory_embedding), dim=-1)
        mlp_input = mlp_input.view(batch_size * M, -1)

        # [bs, M]
        memory_scores = self.lookup_mlp(mlp_input).squeeze(-1).view(batch_size, M)
        memory_scores = th.sigmoid(memory_scores)

        return memory_scores


class MemoryExtraction(th.nn.Module):

    def forward(
            self,
            memory_embedding,
            memory_scores
    ):
        # memory_embedding: [bs, M, d]
        # memory_scores:    [bs, M]

        # [bs, d]
        return th.mean(memory_embedding * memory_scores[:, :, None], dim=1)


class MemoryReasoning(th.nn.Module):

    def forward(
            self,
            input_embedding,
            memory_extraction
    ):
        # input_embedding:      [bs, d]
        # memory_extraction:    [bs, d]

        # [bs, 2*d]
        return th.concat((input_embedding, memory_extraction), dim=-1)