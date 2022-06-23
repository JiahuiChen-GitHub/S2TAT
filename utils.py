import numpy as np


def get_adjacency_matrix(distance_df_filename, num_of_vertices, type_='connectivity', id_filename=None):
    """
    Parameters
    ----------
    id_filename
    distance_df_filename: str, path of the csv file contains edges information
    num_of_vertices: int, the number of vertices
    type_: str, {connectivity, distance}
    Returns
    ----------
    A: np.ndarray, adjacency matrix
    """
    import csv

    A = np.zeros((int(num_of_vertices), int(num_of_vertices)), dtype=np.float32)

    if id_filename:
        with open(id_filename, 'r') as f:
            id_dict = {int(i): idx
                       for idx, i in enumerate(f.read().strip().split('\n'))}
        with open(distance_df_filename, 'r') as f:
            f.readline()
            reader = csv.reader(f)
            for row in reader:
                if len(row) != 3:
                    continue
                i, j, distance = int(row[0]), int(row[1]), float(row[2])
                A[id_dict[i], id_dict[j]] = 1
                A[id_dict[j], id_dict[i]] = 1
        return A
    # Fills cells in the matrix with distances.
    with open(distance_df_filename, 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            if len(row) != 3:
                continue
            i, j, distance = int(row[0]), int(row[1]), float(row[2])
            if type_ == 'connectivity':
                A[i, j] = 1
                A[j, i] = 1
            elif type_ == 'distance':
                A[i, j] = 1 / distance
                A[j, i] = 1 / distance
            else:
                raise ValueError("type_ error, must be "
                                 "connectivity or distance!")
    return A


def normalize_adj(adj):
    """
    Add self-connection and normalized adj
    Parameters
    ----------
    adj: np.ndarray, adjacency matrix, shape is (N, N)
    ----------
    new adjacency matrix:shape is (N, N)
    """
    for i in range(len(adj)):
        adj[i, i] = 1

    d = np.diag(np.power(adj.sum(1), -0.5).flatten())
    adj_norm = adj.dot(d).transpose().dot(d)

    return adj_norm


def load_data(feature_filename, train_timesteps, Seq_time, Pre_time, test_size=0.2):
    """
    load data

    Returns
    ----------
    X: shape is (B, T, N)
    """
    from sklearn.preprocessing import normalize
    from sklearn.model_selection import train_test_split

    data = np.load(feature_filename)['data']
    data = data[:, :, 0]  # TT*N

    x_data = normalize(data)

    step = 1
    x = list()
    y = list()

    for i in range(0, data.shape[0] - Seq_time - Pre_time - 1, step):
        x.append(x_data[i:i + Seq_time])
        y.append(data[i + Seq_time:i + Seq_time + Pre_time])
    X = np.stack(x, axis=0)
    Y = np.stack(y, axis=0)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size)

    return X_train, Y_train, X_test, Y_test


def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def generate_from_train_val_test(data, slice, transformer):
    mean = None
    std = None
    seq_time, pre_time = slice
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], seq_time, pre_time)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_from_data(data, length, slice, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    seq_time, pre_time = slice
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], seq_time, pre_time)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, slice, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, slice, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, slice, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, [train_length], axis=1)


class Logger(object):
    def __init__(self, log_file_path, stream):
        self.stream = stream
        print('Logger: ', log_file_path)
        self.log_file = log_file_path
    
    def write(self, message):
        with open(self.log_file, 'a') as log:
            self.stream.write(message)
            log.write(message)
            self.stream.flush()
            log.flush()
    
    def flush(self):
        pass