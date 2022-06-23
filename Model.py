
import tensorflow as tf
from tensorflow.keras.models import Model
from Layers import FeatureEmbedding, S2TATBlock, OutputLayer


class S2TAT(Model):
    def __init__(self, adj, model_dim, n_nodes, n_layers, n_heads, d_head, d_his, d_future, pre_time, seq_time, receptive_field=None):
        super(S2TAT, self).__init__()
        self.model_dim = model_dim
        self.n_nodes = n_nodes
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_head = d_head
        self.d_his = d_his
        self.d_future = d_future
        self.seq_time = seq_time
        self.pre_time = pre_time
        self.receptive_field = receptive_field

        self.Embedding = FeatureEmbedding(seq_time, model_dim)
        self.TAGCLs = [S2TATBlock(adj, model_dim, n_heads, d_head) for i in range(n_layers)]
        self.Output = OutputLayer(model_dim, d_his, d_future, seq_time, pre_time, n_nodes)

    def call(self, inputs):
        graph_state, graph_signal = self.Embedding(inputs)  # B*T*N*C
        if self.receptive_field is not None:
            mask = tf.linalg.band_part(tf.ones((self.seq_time, self.pre_time)), self.receptive_field[0], self.receptive_field[1])
        else:
            mask = None

        for i in range(self.n_layers):
            graph_state, graph_signal = self.TAGCLs[i]([graph_state, graph_signal], mask)

        return self.Output([graph_state, graph_signal])
