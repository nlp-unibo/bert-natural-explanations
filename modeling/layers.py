import tensorflow as tf
import tensorflow_addons as tfa

# TODO: re-write in torch
class MemorySentenceEmbedder(tf.keras.layers.Layer):

    def __init__(
            self,

            **kwargs
    ):
        super(MemorySentenceEmbedder, self).__init__(**kwargs)
        self.mode = mode

        if self.mode not in ['sum', 'mean']:
            raise RuntimeError(f'Invalid sentence embedding mode! Got: {self.mode}')

    def call(self,
             inputs,
             training=None,
             **kwargs):
        # [bs, T, d]

        if self.mode == 'sum':
            return tf.reduce_sum(inputs, axis=1)
        elif self.mode == 'mean':
            return tf.reduce_mean(inputs, axis=1)


class HeadAttention(tf.keras.layers.Layer):

    def __init__(self, segments_amount, head_attention, embedding_dimension, attention_dimension,
                 use_masking=False, **kwargs):
        super(HeadAttention, self).__init__(**kwargs)
        self.head_attention = head_attention
        self.embedding_dimension = embedding_dimension
        self.attention_dimension = attention_dimension
        self.segments_amount = segments_amount
        self.use_masking = use_masking

        self.reduction_layers = [tf.keras.layers.Dense(units=self.attention_dimension, activation=tf.nn.leaky_relu),
                                 tf.keras.layers.Dense(units=self.embedding_dimension)]

    def call(self, x, **kwargs):

        # [batch_size, seq_length, embedding_dim]

        # [batch_size, seq_length, embedding_dim]
        reduction_input = x
        for reduction_layer in self.reduction_layers:
            reduction_input = reduction_layer(reduction_input)

        # [batch_size, embedding_dim]
        if self.use_masking:
            mask_width = x.shape[1] // self.segments_amount
            indexes = tf.range(0, x.shape[1])
            desired_shape = [x.shape[0], x.shape[1]]
            mask_lower = tf.where(indexes >= mask_width * self.head_attention,
                                  tf.ones(desired_shape), tf.zeros(desired_shape))
            mask_upper = tf.where(indexes < (mask_width * self.head_attention) + mask_width,
                                  tf.ones(desired_shape), tf.zeros(desired_shape))
            mask = mask_lower * mask_upper
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, multiples=[1, 1, self.embedding_dimension])
            reduction_weights = sparse_softmax(tf.transpose(reduction_input, [0, 2, 1]),
                                               tf.transpose(mask, [0, 2, 1]))
            reduction_weights = tf.transpose(reduction_weights, [0, 2, 1])
        else:
            reduction_weights = tf.nn.softmax(reduction_input, axis=1)
        pooled_embeddings = tf.reduce_sum(x * reduction_weights, axis=1)

        return pooled_embeddings


class GeneralizedPooling(tf.keras.layers.Layer):
    def __init__(self, segments_amount, attention_dimension, embedding_dimension, use_masking=False, **kwargs):
        super(GeneralizedPooling, self).__init__(*kwargs)

        self.segments_amount = segments_amount
        self.attention_dimension = attention_dimension
        self.embedding_dimension = embedding_dimension
        self.use_masking = use_masking

    def _compute_disagreement_penalization(self, segment_embeddings):
        segment_embeddings = tf.math.l2_normalize(segment_embeddings, axis=2)

        # [batch_size, segments_amount, segments_amount]
        cosine_distance = tf.matmul(segment_embeddings, segment_embeddings, transpose_b=True)

        # [batch_size, ]
        cosine_distance = tf.reduce_sum(cosine_distance, axis=2)
        cosine_distance = tf.reduce_sum(cosine_distance, axis=1)

        # [batch_size, ]
        cosine_distance = cosine_distance / (self.segments_amount * self.segments_amount)

        return cosine_distance

    def call(self, x, **kwargs):
        # [batch_size, seq_length, embedding_dim]

        def condition(index, stack, segments_amount, head_attention,
                      embedding_dimension, attention_dimension, use_masking=False):
            return tf.less(index, self.segments_amount)

        def body(index, stack, segments_amount, head_attention,
                 embedding_dimension, attention_dimension, use_masking=False):
            pooled_embeddings = HeadAttention(segments_amount=segments_amount,
                                              head_attention=head_attention,
                                              embedding_dimension=embedding_dimension,
                                              attention_dimension=attention_dimension,
                                              use_masking=use_masking)(x)
            stack = stack.write(index, pooled_embeddings)
            index = tf.add(index, 1)
            head_attention = head_attention + 1
            return index, stack, segments_amount, head_attention, embedding_dimension, attention_dimension, use_masking

        temp_stack = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
        result = tf.while_loop(condition,
                               body,
                               [tf.constant(0),
                                temp_stack,
                                self.segments_amount,
                                tf.constant(0),
                                self.embedding_dimension,
                                self.attention_dimension,
                                self.use_masking])
        segment_embeddings = result[1].stack()

        # [batch_size, segments_amount, embedding_dim]
        segment_embeddings = tf.transpose(segment_embeddings, [1, 0, 2])

        if self.segments_amount > 1:
            self.disagreement_penalization = self._compute_disagreement_penalization(segment_embeddings)

        return segment_embeddings


class MemorySentenceLookup(tf.keras.layers.Layer):
    """
    Basic Memory Lookup layer. Query to memory cells similarity is computed either via doc product or via
    a FNN. The content vector is computed by weighting memory cells according to their corresponding computed
    similarities.

    Moreover, the layer is sensitive to strong supervision loss, since the latter is implemented via a max-margin loss
    on computed similarity distribution (generally a softmax).
    """

    def __init__(self,
                 reasoning_info,
                 memory_lookup_info,
                 dropout_rate=0.2,
                 l2_regularization=0.,
                 **kwargs):
        super(MemorySentenceLookup, self).__init__(**kwargs)
        self.memory_lookup_info = memory_lookup_info
        self.reasoning_info = reasoning_info
        self.dropout_rate = dropout_rate

        if self.memory_lookup_info['mode'] == 'mlp':
            self.mlp_weights = []
            for weight in self.memory_lookup_info['weights']:
                self.mlp_weights.append(tf.keras.layers.Dense(units=weight,
                                                              activation=tf.tanh,
                                                              kernel_regularizer=tf.keras.regularizers.l2(
                                                                  l2_regularization)))
                self.mlp_weights.append(tf.keras.layers.Dropout(rate=dropout_rate))

            self.mlp_weights.append(tf.keras.layers.Dense(units=1,
                                                          activation=None,
                                                          kernel_regularizer=tf.keras.regularizers.l2(
                                                              l2_regularization)))

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, inputs, training=False, state='training', mask=None, **kwargs):
        # [batch_size, embedding_dim]
        query = inputs['query']

        # [batch_size, mem_size, embedding_dim]
        memory = inputs['memory']

        if self.memory_lookup_info['mode'] == 'dot_product':
            q_temp = tf.expand_dims(query, axis=1)
            memory = tf.tile(input=memory,
                             multiples=[1, 1, tf.cast(query.shape[-1] / memory.shape[-1], tf.int32)])
            dotted = tf.reduce_sum(memory * q_temp, axis=2)
        elif self.memory_lookup_info['mode'] == 'scaled_dot_product':
            # [batch_size, mem_size, embedding_dim]
            q_temp = tf.expand_dims(query, axis=1)
            memory = tf.tile(input=memory,
                             multiples=[1, 1, tf.cast(query.shape[-1].value / memory.shape[-1].value, tf.int32)])
            dotted = tf.reduce_sum(memory * q_temp, axis=2) * self.embedding_dimension ** 0.5
        elif self.memory_lookup_info['mode'] == 'cosine_similarity':
            q_norm = tf.nn.l2_normalize(query, axis=-1)
            q_norm = tf.expand_dims(q_norm, axis=1)
            memory_norm = tf.nn.l2_normalize(memory, axis=-1)
            memory_norm = tf.tile(input=memory_norm,
                                  multiples=[1, 1, tf.cast(query.shape[-1] / memory.shape[-1], tf.int32)])
            dotted = tf.reduce_sum(q_norm * memory_norm, axis=-1)
        elif self.memory_lookup_info['mode'] == 'mlp':

            # Memory efficient scenario
            if memory.shape[0] == 1:
                dotted = []
                for mem_id in range(memory.shape[1]):
                    # [batch_size, embedding_dim * 2]
                    mem_slice = tf.tile(memory[:, mem_id, :], [query.shape[0], 1])
                    mem_att_input = tf.concat((mem_slice, query), axis=-1)

                    mem_dotted = mem_att_input
                    for block in self.mlp_weights:
                        mem_dotted = block(mem_dotted, training=training)

                    dotted.append(mem_dotted)

                # [batch_size, mem_size]
                dotted = tf.stack(dotted, axis=-1)
            else:
                if self.reasoning_info['mode'] == 'concat':
                    query_dimension = query.shape[-1]
                    repeat_amount = query_dimension // memory.shape[-1]
                    memory = tf.tile(memory, multiples=[1, 1, repeat_amount])

                # [batch_size, mem_size, embedding_dim]
                repeated_query = tf.expand_dims(query, axis=1)
                repeated_query = tf.tile(repeated_query, [1, memory.shape[1], 1])

                # [batch_size, mem_size, embedding_dim * 2]
                att_input = tf.concat((memory, repeated_query), axis=-1)
                att_input = tf.reshape(att_input, [-1, att_input.shape[-1]])

                dotted = att_input
                for block in self.mlp_weights:
                    dotted = block(dotted, training=training)

            dotted = tf.reshape(dotted, [-1, memory.shape[1]])
        else:
            raise RuntimeError('Invalid similarity operation! Got: {}'.format(self.memory_lookup_info['mode']))

        # [batch_size, memory_size]
        return dotted


class SentenceMemoryExtraction(tf.keras.layers.Layer):

    def __init__(self, extraction_info, partial_supervision_info, padding_amount=None, **kwargs):
        super(SentenceMemoryExtraction, self).__init__(**kwargs)
        self.extraction_info = extraction_info
        self.partial_supervision_info = partial_supervision_info
        self.padding_amount = padding_amount

        self.supervision_loss = None

    def _add_supervision_loss(self, prob_dist, positive_idxs, negative_idxs, mask_idxs):

        padding_amount = self.padding_amount

        # Repeat mask for each positive element in each sample memory
        # Mask_idxs shape: [batch_size, padding_amount]
        # Mask res shape: [batch_size * padding_amount, padding_amount]
        mask_res = tf.tile(mask_idxs, multiples=[1, padding_amount])
        mask_res = tf.reshape(mask_res, [-1, padding_amount, padding_amount])
        mask_res = tf.transpose(mask_res, [0, 2, 1])
        mask_res = tf.reshape(mask_res, [-1, padding_amount])

        # Split each similarity score for a target into a separate sample
        # similarities shape: [batch_size, memory_max_length]
        # positive_idxs shape: [batch_size, padding_amount]
        # gather_nd shape: [batch_size, padding_amount]
        # pos_scores shape: [batch_size * padding_amount, 1]
        pos_scores = tf.gather(prob_dist, positive_idxs, batch_dims=1)
        pos_scores = tf.reshape(pos_scores, [-1, 1])

        # Repeat similarity scores for non-target memories for each positive score
        # similarities shape: [batch_size, memory_max_length]
        # negative_idxs shape: [batch_size, padding_amount]
        # neg_scores shape: [batch_size * padding_amount, padding_amount]
        neg_scores = tf.gather(prob_dist, negative_idxs, batch_dims=1)
        neg_scores = tf.tile(neg_scores, multiples=[1, padding_amount])
        neg_scores = tf.reshape(neg_scores, [-1, padding_amount])

        # Compare each single positive score with all corresponding negative scores
        # [batch_size * padding_amount, padding_amount]
        # [batch_size, padding_amount]
        # [batch_size, 1]
        # Samples without supervision are ignored by applying a zero mask (mask_res)
        hop_supervision_loss = tf.maximum(0., self.partial_supervision_info['margin'] - pos_scores + neg_scores)
        hop_supervision_loss = hop_supervision_loss * tf.cast(mask_res, dtype=hop_supervision_loss.dtype)
        hop_supervision_loss = tf.reshape(hop_supervision_loss, [-1, padding_amount, padding_amount])

        hop_supervision_loss = tf.reduce_sum(hop_supervision_loss, axis=[1, 2])
        # hop_supervision_loss = tf.reduce_max(hop_supervision_loss, axis=1)
        normalization_factor = tf.cast(tf.reshape(mask_res, [-1, padding_amount, padding_amount]),
                                       hop_supervision_loss.dtype)
        normalization_factor = tf.reduce_sum(normalization_factor, axis=[1, 2])
        normalization_factor = tf.maximum(normalization_factor, tf.ones_like(normalization_factor))
        hop_supervision_loss = tf.reduce_sum(hop_supervision_loss / normalization_factor)

        # Normalize by number of unfair examples
        valid_examples = tf.reduce_sum(mask_idxs, axis=1)
        valid_examples = tf.cast(valid_examples, tf.float32)
        valid_examples = tf.minimum(valid_examples, 1.0)
        valid_examples = tf.reduce_sum(valid_examples)
        valid_examples = tf.maximum(valid_examples, 1.0)
        hop_supervision_loss = hop_supervision_loss / valid_examples

        return hop_supervision_loss

    def call(self, inputs, state='training', training=False, mask=None, **kwargs):

        similarities = inputs['similarities']
        context_mask = inputs['context_mask']

        context_mask = tf.cast(context_mask, similarities.dtype)
        similarities += (context_mask * -1e9)

        if self.extraction_info['mode'] == 'softmax':
            probs = tf.nn.softmax(similarities, axis=1)
        elif self.extraction_info['mode'] == 'sparsemax':
            probs = tfa.activations.sparsemax(similarities, axis=1)
        elif self.extraction_info['mode'] == 'sigmoid':
            probs = tf.nn.sigmoid(similarities)
        else:
            raise RuntimeError('Invalid extraction mode! Got: {}'.format(self.extraction_info['mode']))

        supervision_loss = None
        if self.partial_supervision_info['flag'] and state != 'prediction':
            supervision_loss = self._add_supervision_loss(prob_dist=probs,
                                                          positive_idxs=inputs['positive_indexes'],
                                                          negative_idxs=inputs['negative_indexes'],
                                                          mask_idxs=inputs['mask_indexes'])

        return probs, supervision_loss


class SentenceMemoryReasoning(tf.keras.layers.Layer):
    """
    Basic Memory Reasoning layer. The new query is computed simply by summing the content vector to current query
    """

    def __init__(self, reasoning_info, **kwargs):
        super(SentenceMemoryReasoning, self).__init__(**kwargs)
        self.reasoning_info = reasoning_info

    def call(self, inputs, training=False, state='training', **kwargs):
        query = inputs['query']
        memory_search = inputs['memory_search']

        if self.reasoning_info['mode'] == 'sum':
            upd_query = query + memory_search
        elif self.reasoning_info['mode'] == 'concat':
            upd_query = tf.concat((query, memory_search), axis=1)
        elif self.reasoning_info['mode'] == 'rnn':
            cell = tf.keras.layers.GRUCell(query.shape[-1])
            upq_query, _ = cell(memory_search, [query])
        elif self.reasoning_info['mode'] == 'mlp':
            upd_query = tf.keras.layers.Dense(query.shape[-1],
                                              activation=tf.nn.relu)(tf.concat((query, memory_search), axis=1))
        else:
            raise RuntimeError(
                'Invalid aggregation mode! Got: {} -- Supported: [sum, concat]'.format(self.reasoning_info['mode']))

        return upd_query
