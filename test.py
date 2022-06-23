from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model import S2TAT
from utils import *
import argparse
import json


parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, help='configuration file')
args = parser.parse_args()
config_filename = args.config
with open(config_filename, 'r') as f:
    config = json.loads(f.read())

batch_size = config['batch_size']
epochs = config['epochs']

adj_filename = config['adj_filename']
id_filename = config['id_filename']
feature_filename = config['graph_signal_matrix_filename']

n_nodes = config['num_of_vertices']
seq_time = config['points_per_hour']
pre_time = config['num_for_predict']
pre_time = config['num_for_predict']

model_dim = config['model_dim']
n_blocks = config['n_blocks']
n_heads = config['n_heads']
d_head = config['d_head']
d_his = config['d_his']
d_future = config['d_future']
receptive_field = config['receptive_field']


adj = get_adjacency_matrix(adj_filename, n_nodes, id_filename=id_filename)
adj_norm = normalize_adj(adj)
print("The shape of localized adjacency matrix: {}".format(adj_norm.shape), flush=True)

# Construct Model
print('\nConstruct model and Load data...')
loaders = []
for idx, (x, y) in enumerate(generate_data(feature_filename, (seq_time, pre_time))):
    y = y.squeeze(axis=-1)
    print(f'Input shape:{x.shape}, Prediction shape:{y.shape}')
    loaders.append(
        tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    )
    if idx == 2:
        X_test = x
        Y_test = y

train_loader, val_loader, test_loader = loaders
train_loader = train_loader.shuffle(200*batch_size, reshuffle_each_iteration=True)

model = S2TAT(adj_norm, model_dim, n_nodes, n_blocks, n_heads, d_head, d_his, d_future, pre_time, seq_time, receptive_field)
optimizer = Adam(learning_rate=0.001)
model.compile(
    optimizer,
    loss=tf.keras.losses.Huber(delta=1),
    metrics=['mae'])

checkpoint_path = "./checkpoints/"+config['dataset']+'_'+config['model_type']+"/S2TAT.ckpt"

print('\nEvaluating ...')
model.load_weights(checkpoint_path)
model.evaluate(x=test_loader, verbose=2)

model.summary()

Y_pre = model.predict(x=X_test, batch_size=batch_size, verbose=2)
mae = masked_mae_np(Y_test, Y_pre, 0)
mape = masked_mape_np(Y_test, Y_pre, 0)
rmse = masked_mse_np(Y_test, Y_pre, 0) ** 0.5
print('Evaluating compelete.')
print(f'MAE:{mae}, MAPE:{mape}, RMSE:{rmse}')
