from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Lambda
from keras.layers import TimeDistributed, Bidirectional, Masking, Dropout

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, multi_gpu_model

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils

from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from tqdm import tqdm
from collections import OrderedDict
import pickle
import argparse

import numpy as np
np.set_printoptions(precision=3, linewidth=1000, edgeitems=100, suppress=True)
seed = 7
np.random.seed(seed)

MAX_LEN = 6
fixed_param = dict(hidden_size=512, seq_len=MAX_LEN, batch_size=10, dropout=0.9)

paramsearch = [
    dict(c_weights=True, joint=True, regularizer=None, recurrent_dropout=0),
    dict(c_weights=False, joint=True, regularizer=None, recurrent_dropout=0),
    dict(c_weights=True, joint=True, regularizer='l2', recurrent_dropout=0),
    dict(c_weights=True, joint=True, regularizer=None, recurrent_dropout=0.9),
]


def create_model(seq_len, hidden_size, dropout, recurrent_dropout, regularizer='l1', joint=False, n_gpu=None):
    inp = Input(shape=(seq_len, 2640), name='input')

    mask = Masking(mask_value=0)(inp)
    dropped = Dropout(dropout)(mask)
    fc = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer=regularizer), name='FC_input')(dropped)

    encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder', recurrent_dropout=recurrent_dropout, dropout=dropout))(fc)
    decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=recurrent_dropout, dropout=dropout)(encoder)

    if joint:
        typ = TimeDistributed(Dense(2, use_bias=True, kernel_regularizer=regularizer, activation='softmax'), name='type')(encoder)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer=regularizer), name='E')(encoder)
    D = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer=regularizer), name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    # glorot_uniform initializer:
    # for hidden=512: uniform(-0,108, +0,108)
    vt = Dense(1, use_bias=False, kernel_regularizer=regularizer, name='vT')
    softmax = Activation('softmax', name='link')

    attention = add([E,DD])
    attention = tanh(attention)
    attention = vt(attention)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = softmax(attention)

    if joint:
        model = Model(inputs=inp, outputs=[attention, typ])
    else:
        model = Model(inputs=inp, outputs=attention)

    if n_gpu:
        parallel_model = multi_gpu_model(model, gpus=n_gpu)
        return parallel_model
    else:
        return model


def stringify(param):
    return '-'.join(['{}_{}'.format(k,v) for k, v in param.items() if k != 'tboard'])


def get_sample_weights(Ys):
    sample_weights = []

    for Y in Ys:
        class_num = Y.shape[-1]
        class_weights = get_class_weights(Y)
        sample_weights.append(np.zeros(Y.shape[:2]))
        for i in range(class_num):
            sample_weights[-1][Y.argmax(2) == i] = class_weights[i]
        sample_weights[-1][Y.sum(2) == 0] = 0

    return sample_weights


def get_class_weights(Y):
    class_num = Y.shape[-1]
    class_weights = dict(zip(np.arange(class_num),np.zeros(class_num)))
    labels = Y.argmax(2).flatten()
    un_labels = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', un_labels, labels)
    class_weights.update(zip(un_labels, weights))
    return class_weights


def preprocess(docs, links, types=None):
    docs_filtered = np.array([x for x in docs if len(x) <= MAX_LEN])
    links_filtered = np.array([x for x in links if len(x) <= MAX_LEN])
    types_filtered = np.array([x for x in types if len(x) <= MAX_LEN])

    # pad with zeros, truncate longer than 6
    X = pad_sequences(docs_filtered, dtype=float, truncating='post',padding='post')

    # convert links 1,1,2 -> [1,0,0]
    #                        [1,0,0]
    #                        [0,1,0]
    Yl = [to_categorical([np.array(ys)-1 for ys in y],
                         num_classes=X.shape[1]) for y in links_filtered]
    Yl = pad_sequences(Yl, dtype=int, truncating='post', padding='post')
    if types is not None:
        Yt = [pad_sequences(to_categorical(np.array(y)), truncating='post',
                            padding='post') for y in types_filtered]
        Yt = pad_sequences(Yt, dtype=int, truncating='post', padding='post')
    else:
        Yt = None

    return X, Yl, Yt, docs_filtered, links_filtered, types_filtered


def crossvalidation(X, Yl, Yt, epochs, paramsearch, n_gpu):
    NUM_TRIALS = 2
    metrics = []
    test_metrics = []
    metric_keys = ['iteration', 'outer', 'inner', 'param', 'score', 'epoch']
    paramset = []
    score_keys = []

    inner_seed = np.random.RandomState(0)

    for i in range(NUM_TRIALS):
        inner = KFold(n_splits=5, shuffle=True, random_state=inner_seed)

        metrics.append([])
        test_metrics.append([])
        models = []

        # for (o, (training, test)) in tqdm(enumerate(outer.split(X))):
        X_training = X[:100]
        Yl_training = Yl[:100]
        Yt_training = Yt[:100]
        metrics[-1].append([])
        models.append([])

        for (k, (train, val)) in tqdm(enumerate(inner.split(X_training))):
            print('data lengths', len(train),len(val),len(X[100:]))
            metrics[-1][-1].append([])
            for param in paramsearch:

                params = {}
                params.update(fixed_param)
                params.update(param)
                params.update({'cv_iter': i})
                params.update({'tboard': {0:i,
                                          1:k,
                                          2:stringify(params)}})
                if param not in paramset:
                    paramset.append(params)

                X_train, X_val = X_training[train], X_training[val]
                Yl_train, Yl_val = Yl_training[train], Yl_training[val]
                Yt_train, Yt_val = Yt_training[train], Yt_training[val]

                metric, model = train_model(X_train, X_val, Yl_train, Yl_val, Yt_train, Yt_val, epochs, params, n_gpu)
                models[-1].append(model)
                score_keys = list(OrderedDict(sorted(metric.items())).keys())
                metrics[-1][-1][-1].append(list(OrderedDict(sorted(metric.items())).values()))

            # print(np.shape(metrics[-1][-1]))
            # best_param_ind = np.argmax(np.max(np.mean(metrics[-1][-1], 0)[:,list(score_keys).index('link_macro_f1'),:],1))
            # print(best_param_ind)
            # best_params = paramsearch[best_param_ind]
            # best_params.update({'tboard': {0:i,
                                           # 1:'test_'+stringify(params)}})
            # test_metric, _ = train_model(X_training, X_test, Yl_training,
                                         # Yl_test, Yt_training, Yt_test, epochs,
                                         # best_params, n_gpu)
            # test_metrics[-1].append(test_metric)

            # print('CV iteration {} has testing score: {}\nfor params: {}'.format(i, {k:v[-1] for k, v in metric.items()}, paramsearch[best_param_ind]))

        # last validation accuracy
        # for i, metric in enumerate(metric_keys):
            # val_acc = [h[m][-1] for h in metrics[-1]]
            # mean, var = np.mean(val_acc, axis=0), np.var(val_acc, axis=0)
            # print("{}:{} (+/- {})".format(metric, mean, var))

    with open('cross_validation/train.pl', 'wb') as f:
        metrics = dict(metrics=metrics, metric_keys=metric_keys, score_keys=score_keys, params=paramset)
        pickle.dump(metrics, f)

    with open('cross_validation/test.pl', 'wb') as f:
        test_metrics = dict(metrics=test_metrics, metric_keys=metric_keys, score_keys=score_keys, params=paramset)
        pickle.dump(test_metrics, f)

    print(metric_keys)
    print(score_keys)
    return metrics, metric_keys


def train_model(X_train, X_val, Yl_train, Yl_val, Yt_train, Yt_val, epochs, param,
                n_gpu=0, model=None):
    params = {}
    params.update(fixed_param)
    params.update(param)
    print(params)
    adam = Adam()
    metric = utils.JointMetrics() if params['joint'] else utils.Metrics()
    callbacks = [metric]
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50)
    # checkpoint = ModelCheckpoint('checkpoints/' + stringify(params), monitor='val_loss', save_best_only=True)
    if params.get('tboard'):
        tboard_desc = params['tboard']
        tboard_run = '/'.join([str(v) for k, v in sorted(tboard_desc.items())])
        tensorboard = TensorBoard(log_dir='/cache/tensorboard-logdir/'+tboard_run,
                                  histogram_freq=100, batch_size=32, write_grads=True)
        callbacks.append(tensorboard)

    loss_weight = [0.5, 0.5] if params['joint'] else None

    if model is None:
        model = create_model(seq_len=params['seq_len'], hidden_size=params['hidden_size'], dropout=params['dropout'], recurrent_dropout=params['recurrent_dropout'], regularizer=params['regularizer'], joint=params['joint'], n_gpu=n_gpu)

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'],
                      sample_weight_mode='temporal',
                      loss_weights=loss_weight)

    Ys = [Yl_train, Yt_train] if params['joint'] else [Yl_train]
    Ys_val = [Yl_val, Yt_val] if params['joint'] else [Yl_val]
    sample_weights = None
    if params['c_weights']:
        sample_weights = get_sample_weights(Ys)

    model.fit(X_train,
              Ys,
              validation_data=(X_val, Ys_val),
              callbacks=callbacks,
              epochs=epochs,
              batch_size=params['batch_size'],
              verbose=0,
              sample_weight=sample_weights)

    return metric.metrics, model


def main(docs, links, types=None, epochs=1000, paramsearch=paramsearch, n_gpu=None):
    X, Yl, Yt, _, _, _ = preprocess(docs, links, types)
    metrics = crossvalidation(X, Yl, Yt, epochs, paramsearch, n_gpu)
    return metrics


def load_vec(docs, links, types=None):
    docs, links = np.load(docs), np.load(links)
    types = np.load(types) if types else None
    return docs, links, types


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--docs', help='Documents', type=str)
    parser.add_argument('-l', '--links', help='Links', type=str)
    parser.add_argument('-t', '--types', help='Types', type=str, default=None)
    parser.add_argument('-i', '--ifold', help='Types', type=int, default=3)
    parser.add_argument('-e', '--epochs', help='Epoch', type=int, default=2000)
    parser.add_argument('-g', '--n_gpu', help='Multi GPU', type=int, default=None)
    args = parser.parse_args()

    docs, links, types = load_vec(args.docs, args.links, args.types)

    print(docs.shape, links.shape, args.epochs)
    main(docs, links, types, args.epochs, paramsearch, args.n_gpu)
