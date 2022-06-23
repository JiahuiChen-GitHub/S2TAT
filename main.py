from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
import sys
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from Model import S2TAT
from utils import *
import argparse
import json
import time
import logging


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


def seed_tensorflow(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)


adj = get_adjacency_matrix(adj_filename, n_nodes, id_filename=id_filename)
adj_norm = normalize_adj(adj)
print("The shape of localized adjacency matrix: {}".format(adj_norm.shape), flush=True)


# Log
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)
log_path = 'log/'+config['dataset']+'_'+config['model_type']+"/"
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_file = time.strftime('%Y%m%d_%H%M',time.localtime())+'.log'
handler = logging.FileHandler(log_path+log_file, mode='a+')
handler.setLevel(logging.DEBUG)
logger.addHandler(handler)
logger.addHandler(logging.StreamHandler(sys.stdout))

# t = time.localtime()
# log_file = time.strftime('%Y%m%d_%H%M',t)+'.txt'
# log = open(log_path+log_file, 'w+')
for key, value in config.items():
    logger.info('{}: {}'.format(key, value))
# print('\n', file=log)


# Construct Model
loaders = []
for idx, (x, y) in enumerate(generate_data(feature_filename, (seq_time, pre_time))):
    y = y.squeeze(axis=-1)
    print(x.shape, y.shape)
    loaders.append(
        tf.data.Dataset.from_tensor_slices((x, y)).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    )
    if idx == 2:
        X_test = x
        Y_test = y

seed_tensorflow()
train_loader, val_loader, test_loader = loaders
train_loader = train_loader.shuffle(200*batch_size, reshuffle_each_iteration=True)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = S2TAT(adj_norm, model_dim, n_nodes, n_blocks, n_heads, d_head, d_his, d_future, pre_time, seq_time, receptive_field)
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer,
        loss=tf.keras.losses.Huber(delta=1),
        metrics=['mae'])

lr = tf.keras.callbacks.ReduceLROnPlateau(factor=0.3, patience=20, verbose=2, min_lr=9e-5)
es = tf.keras.callbacks.EarlyStopping(patience=30)
checkpoint_path = "./checkpoints/"+config['dataset']+config['model_type']+"/S2TAT.ckpt"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True)

# Model fit
model.fit(x=train_loader,
          epochs=epochs,
          validation_data=val_loader,
          callbacks=[lr, es, checkpoint],
          verbose=2,
          workers=5,
          use_multiprocessing=True)

model.summary()

model.load_weights(checkpoint_path)
model.evaluate(x=test_loader)
Y_pre = model.predict(x=X_test, batch_size=batch_size)
mae = masked_mae_np(Y_test, Y_pre, 0)
mape = masked_mape_np(Y_test, Y_pre, 0)
rmse = masked_mse_np(Y_test, Y_pre, 0) ** 0.5
logger.info(f'MAE:{mae}, MAPE:{mape}, RMSE:{rmse}')
