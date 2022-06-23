from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model import TAGCN
from utils import get_adjacency_matrix, normalize_adj
from utils import masked_mae_np, masked_mape_np, masked_mse_np
from utils import generate_data
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt


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
r_f = config['receptive_field']

model_dim = config['model_dim']
n_blocks = config['n_blocks']
n_heads = config['n_heads']
d_head = config['d_head']
d_his = config['d_his']
d_future = config['d_future']


loss_object = tf.keras.losses.Huber()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_mae = tf.keras.metrics.MeanAbsoluteError(name='train_mae')


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)


def scheduler(epoch, lr):
    if epoch < 20:
        return 5e-5 * epoch
    else:
        return lr



adj = get_adjacency_matrix(adj_filename, n_nodes, id_filename=id_filename)
adj_norm = normalize_adj(adj)
print("The shape of localized adjacency matrix: {}".format(adj_norm.shape), flush=True)

# Construct Model

loaders = []
for idx, (x, y) in enumerate(generate_data(feature_filename, (12, 12))):
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    loaders.append(
        tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    )
    if idx == 2:
        X_test = x
        Y_test = y

train_loader, val_loader, test_loader = loaders

model = TAGCN(adj_norm, model_dim, n_nodes, n_blocks, n_heads, d_head, d_his, d_future, pre_time, seq_time, r_f)
optimizer = Adam(learning_rate=1e-4)

model.compile(optimizer,
              loss=loss_function,
              metrics=['mae'])

model.load_weights('./checkpoints/PEMS08base/S2TAT.ckpt').expect_partial()

# Model fit

# model.evaluate(x=test_loader)

# model.summary()
# model.fit(x=train_loader,
#           epochs=3,
#           validation_data=val_loader,
#           verbose=2,
#           workers=5,
#           use_multiprocessing=True)
Y_pre = model.predict(x=X_test, batch_size=batch_size)
print(Y_pre.shape)
np.save('./results/S2TAT.npy', Y_pre)
np.save('./results/GT.npy', Y_test)

mae = masked_mae_np(Y_test, Y_pre, 0)
mape = masked_mape_np(Y_test, Y_pre, 0)
rmse = masked_mse_np(Y_test, Y_pre, 0) ** 0.5
print('MAE:', mae, '\nMAPE:', mape, '\nRMSE:', rmse)

n = np.random.randint(0, X_test.shape[2])
n = 29
y_true = np.zeros((Y_test.shape[0]+Y_test.shape[1]-4,))
y_pred = np.zeros((Y_test.shape[0]+Y_test.shape[1]-4,))

for i in range(0, Y_pre.shape[0], 12):
    y_true[i: i+12] = Y_test[i, :, n]
    y_pred[i: i+12] = Y_pre[i, :, n]

plt.figure(figsize=(6.5, 4.5))
plt.plot(y_true[:500], color='goldenrod', linewidth=1.4)
plt.plot(y_pred[:500], color='steelblue', linewidth=1.8)

plt.yticks(fontproperties='Times New Roman', size=12)
plt.xticks(fontproperties='Times New Roman', size=12)

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 14,}
plt.legend(['Ground Truth', 'S$^2$TAT Prediction'], loc='lower right', prop=font1)
plt.xlabel('Time steps', font2)
plt.ylabel('Traffic flow data', font2)
plt.savefig('test.png')
# plt.savefig('test.pdf')
