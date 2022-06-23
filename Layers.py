"""
## Setup
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations, initializers, regularizers
import tensorflow.keras.backend as K


class TimeWiseGCN(layers.Layer):
    # Input [bacth_size, timesteps, node_num, feature_dim]
    # Output [batch_size, timesteps, node_num, new_feature_dim]
    def __init__(self, model_dim,
                 timewise=True,
                 adj_matrix=None,
                 activation=None,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)

        super(TimeWiseGCN, self).__init__(**kwargs)
        self._model_dim = model_dim
        self.timewise = timewise
        self.adj_matrix = tf.cast(adj_matrix, dtype=tf.float32)
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        self.kernel_regularizer = regularizers.get(kernel_regularizer)

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        timesteps = input_shapes[1]
        if self.timewise:
            self.kernel = self.add_weight(
                shape=(timesteps, input_dim, self._model_dim),
                initializer=self.kernel_initializer,
                name='kernel',
                regularizer=self.kernel_regularizer
                )
            self.bias = self.add_weight(
                shape=(timesteps, 1, self._model_dim,),
                initializer=self.bias_initializer,
                name='bias'
                )
        # kenel_shape T*C*C'
        else:
            self.kernel = self.add_weight(
                shape=(input_dim, self._model_dim),
                initializer=self.kernel_initializer,
                name='kernel',
                regularizer=self.kernel_regularizer
                )
        # kenel_shape C*C'
            self.bias = self.add_weight(
                shape=(self._model_dim,),
                initializer=self.bias_initializer,
                name='bias'
                )
            # bias_shape C'
        self.built = True

    # core code
    def call(self, inputs, mask=None):
        output = tf.matmul(inputs, self.kernel)
        output = tf.matmul(self.adj_matrix, output)
        output += self.bias
        return self.activation(output)


class S2TATBlock(layers.Layer):
    def __init__(self, adj, model_dim, num_heads, d_head, name='block'):
        super(S2TATBlock, self).__init__()
        self.model_dim = model_dim
        self.num_heads = num_heads

        self.projection_dim = d_head
        self.query_dense = layers.Dense(num_heads*d_head)
        self.key_dense = layers.Dense(num_heads*d_head)
        self.value_gcn = TimeWiseGCN(model_dim=num_heads*d_head, adj_matrix=adj, activation=None)

        self.state_dense = layers.Dense(1, activation='relu')
        self.signal_dense = layers.Dense(model_dim, activation='relu')

        self.state_layernorm = layers.LayerNormalization(axis=[1, 2])
        self.signal_layernorm = layers.LayerNormalization(axis=[1, 2, 3])

        # self.state_layernorm = layers.LayerNormalization(axis=-1)
        # self.signal_layernorm = layers.LayerNormalization(axis=-1)

    def attention(self, query, key, value, mask=None):
        score = tf.matmul(query, key, transpose_b=True)  # Bh*T_o*T
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        if mask is not None:
            scaled_score = scaled_score + (mask * -1e9)
        weights = K.softmax(scaled_score, axis=-1)  # Bh*T_o*T
        output = K.batch_dot(weights, value, [2, 1])  # Bh*T_o*N*p
        return output, weights

    def call(self, inputs, mask=None):
        graph_state, graph_signal = inputs    # Q B*T_o*N*C, K B*T*N*C, V B*T*N*C

        query = self.query_dense(graph_state)  # B*T_o*N*h
        key = self.key_dense(graph_state)  # B*T*N*h
        value = self.value_gcn(graph_signal)  # B*T*N*C'

        query = tf.concat(tf.split(query, self.num_heads, axis=-1), axis=0)  # Bh*T_o*N
        key = tf.concat(tf.split(key, self.num_heads, axis=-1), axis=0)  # Bh*T*N
        value = tf.concat(tf.split(value, self.num_heads, axis=-1), axis=0)  # Bh*T*N*p

        attention, weights = self.attention(query, key, value, mask)
        graph_update = tf.concat(tf.split(attention, self.num_heads, axis=0), axis=-1)  # B*T_o*N*C'

        graph_state_ = tf.squeeze(self.state_dense(graph_update), axis=-1)
        graph_signal_ = self.signal_dense(graph_update)

        graph_state = self.state_layernorm(graph_state + graph_state_)
        graph_signal = self.signal_layernorm(graph_signal + graph_signal_)

        return graph_state, graph_signal


class FeatureEmbedding(layers.Layer):
    def __init__(self, seq_time, embed_dim):
        super(FeatureEmbedding, self).__init__()
        self._seq_time = seq_time
        self._embed_dim = embed_dim
        self.state_emb = layers.Dense(1, activation='relu')
        self.siganl_emb = layers.Dense(embed_dim, activation='relu')

    def build(self, input_shapes):
        N = input_shapes[-2]
        self.order_emb1 = self.add_weight(shape=(1, self._seq_time, 1, self._embed_dim), initializer='glorot_uniform', name='tebd1')
        self.order_emb2 = self.add_weight(shape=(1, self._seq_time, N), initializer='glorot_uniform', name='tebd2')

    def call(self, inputs):
        graph_state = self.state_emb(inputs)
        graph_state = tf.squeeze(graph_state, axis=-1) + self.order_emb2
        graph_signal = self.siganl_emb(inputs) + self.order_emb1
        # graph_signal = self.siganl_emb(inputs)
        return graph_state, graph_signal


class TimeWisePredict(layers.Layer):
    def __init__(self, timesteps, model_dim):
        super(TimeWisePredict, self).__init__()
        self._timesteps = timesteps
        self._model_dim = model_dim
        self._pred_dense = layers.Dense(1)

    def build(self, input_shapes):
        input_dim = input_shapes[-1]
        self.kernel = self.add_weight(shape=(self._timesteps, input_dim, self._model_dim),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(self._timesteps, self._model_dim,), initializer='zeros', name='bias')

    def call(self, inputs):
        assert len(inputs.shape) == 3
        x = tf.tensordot(inputs, self.kernel, [2, 1])
        x = K.relu(x + self.bias)
        y = self._pred_dense(x)
        return tf.squeeze(y, axis=-1)


class HistoryStateLayer(layers.Layer):
    def __init__(self, model_dim, d_his, seq_time, n_nodes):
        super(HistoryStateLayer, self).__init__()
        self.d_his = d_his
        self.model_dim = model_dim
        self.seq_time = seq_time
        self.n_nodes = n_nodes
        self.his_weights = layers.Dense(1, activation='relu')
        self.his_dense = layers.Dense(d_his, activation='relu')

    def call(self, inputs):
        graph_state, graph_signal = inputs
        his_states = tf.reshape(graph_signal, [K.shape(graph_state)[0], self.seq_time, self.n_nodes*self.model_dim])
        weights = K.softmax(self.his_weights(graph_state), axis=-1)
        his_state = tf.matmul(weights, his_states, transpose_a=True)
        his_signal = tf.reshape(his_state, [K.shape(graph_state)[0], self.n_nodes, self.model_dim])
        return self.his_dense(his_signal)


class OutputLayer(layers.Layer):
    def __init__(self, model_dim, d_his, d_future, seq_time, pre_time, n_nodes, agg='att', name='output'):
        super(OutputLayer, self).__init__()
        self.agg = agg
        if agg == 'att':
            self.historystate = HistoryStateLayer(model_dim, d_his, seq_time, n_nodes)
        self.prediction = TimeWisePredict(pre_time, d_future)

    def call(self, inputs):
        if self.agg == 'att':
            his_state = self.historystate(inputs)
        elif self.agg == 'ave':
            his_state = tf.reduce_mean(inputs[1], axis=1)
        return tf.transpose(self.prediction(his_state), [0, 2, 1])
