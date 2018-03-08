from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Reshape, Lambda, Concatenate, \
                         TimeDistributed, Bidirectional, Masking

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

from keras.models import load_model

import utils

from sklearn.model_selection import ShuffleSplit, train_test_split, KFold
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
    dict(c_weights=True, joint=True, regularizer=None),
    dict(c_weights=False, joint=True, regularizer=None),
]


def create_model(seq_len, hidden_size, dropout, regularizer='l1', joint=False):
    inp = Input(shape=(seq_len, 2640), name='input')

    mask = Masking(mask_value=0)(inp)
    fc = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer=regularizer), name='FC_input')(mask)

    encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder', recurrent_dropout=dropout, dropout=dropout))(fc)
    decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=dropout, dropout=dropout)(encoder)

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
    return model


def stringify(param):
    return '-'.join(['{}_{}'.format(k,v) for k, v in param.items()])


def get_sample_weights(Ys):
    sample_weights = []

    for Y in Ys:
        class_num = Y.shape[-1]
        class_weights = dict(zip(np.arange(class_num),np.zeros(class_num)))
        labels = Y.argmax(2).flatten()
        un_labels = np.unique(labels)
        weights = class_weight.compute_class_weight('balanced', un_labels, labels)
        class_weights.update(zip(un_labels, weights))
        sample_weights.append(np.zeros(Y.shape[:2]))
        for i in range(class_num):
            sample_weights[-1][Y.argmax(2) == i] = class_weights[i]

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
    x_masked = np.array([x for x in docs if len(x) <= MAX_LEN])
    yl_masked = np.array([x for x in links if len(x) <= MAX_LEN])
    yt_masked = np.array([x for x in types if len(x) <= MAX_LEN])

    # pad with zeros, truncate longer than 6
    X = pad_sequences(x_masked, dtype=float, truncating='post',padding='post')

    # convert links 1,1,2 -> [1,0,0]
    #                        [1,0,0]
    #                        [0,1,0]
    Yl = [pad_sequences(to_categorical(np.array(y)), truncating='post', padding='post', maxlen=X.shape[1]) for y in yl_masked]
    Yl = pad_sequences(Yl, dtype=int, truncating='post', padding='post')
    if types is not None:
        Yt = [pad_sequences(to_categorical(np.array(y)), truncating='post', padding='post') for y in yt_masked]
        Yt = pad_sequences(Yt, dtype=int, truncating='post', padding='post')
    else:
        Yt = None

    return X, Yl, Yt


def crossvalidation(X, Yl, Yt, epochs, paramsearch=paramsearch):
    NUM_TRIALS = 2
    metrics = []
    test_metrics = []
    metric_keys = ['iteration', 'outer', 'inner', 'param', 'score', 'epoch']
    paramset = []
    score_keys = []

    for i in range(NUM_TRIALS):
        inner = KFold(n_splits=5, shuffle=True, random_state=0)
        outer = KFold(n_splits=6, shuffle=True, random_state=0)

        metrics.append([])
        test_metrics.append([])
        models = []

        for (o, (training, test)) in enumerate(tqdm(outer.split(X))):
            X_training, X_test = X[training], X[test]
            Yl_training, Yl_test = Yl[training], Yl[test]
            Yt_training, Yt_test = Yt[training], Yt[test]
            metrics[-1].append([])
            models.append([])

            for (k, (train, val)) in enumerate(tqdm(inner.split(training))):
                print('data lengths', len(train),len(val),len(test))
                metrics[-1][-1].append([])
                for param in paramsearch:

                    params = {}
                    params.update(fixed_param)
                    params.update(param)
                    params.update({'cv_iter': i})
                    if param not in paramset:
                        paramset.append(param)

                    X_train, X_val = X_training[train], X_training[val]
                    Yl_train, Yl_val = Yl_training[train], Yl_training[val]
                    Yt_train, Yt_val = Yt_training[train], Yt_training[val]

                    metric, model = train_model(X_train, X_val, Yl_train, Yl_val, Yt_train, Yt_val, epochs, param)
                    models[-1].append(model)
                    score_keys = list(OrderedDict(sorted(metric.items())).keys())
                    metrics[-1][-1][-1].append(list(OrderedDict(sorted(metric.items())).values()))

            print(np.shape(metrics[-1][-1]))
            best_param_ind = np.argmax(np.max(np.mean(metrics[-1][-1], 0)[:,list(score_keys).index('link_macro_f1'),:],1))
            print(best_param_ind)
            test_metric, _ = train_model(X_training, X_test, Yl_training, Yl_test, Yt_training, Yt_test, epochs, paramsearch[best_param_ind])
            test_metrics[-1].append(test_metric)

            print('CV iteration {} has testing score: {}\nfor params: {}'.format(i, {k:v[-1] for k, v in metric.items()}, paramsearch[best_param_ind]))
            break

        # last validation accuracy
        # for i, metric in enumerate(metric_keys):
            # val_acc = [h[m][-1] for h in metrics[-1]]
            # mean, var = np.mean(val_acc, axis=0), np.var(val_acc, axis=0)
            # print("{}:{} (+/- {})".format(metric, mean, var))

    with open('cross_validation/' + stringify(param) + '.pl', 'wb') as f:
        metrics = dict(metrics=metrics, metric_keys=metric_keys, score_keys=score_keys, params=paramset)
        pickle.dump(metrics, f)

    with open('cross_validation/' + stringify(param) + '_test.pl', 'wb') as f:
        test_metrics = dict(metrics=test_metrics, metric_keys=metric_keys, score_keys=score_keys, params=paramset)
        pickle.dump(test_metrics, f)

    print(metric_keys)
    print(score_keys)
    return metrics, metric_keys


def train_model(X_train, X_val, Yl_train, Yl_val, Yt_train, Yt_val, epochs, params, model=None):
    param = {}
    param.update(fixed_param)
    param.update(params)
    print(param)
    adam = Adam()
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50)
    checkpoint = ModelCheckpoint('checkpoints/' + stringify(param), monitor='val_loss', save_best_only=True)
    metric = utils.JointMetrics() if param['joint'] else utils.Metrics()

    loss_weight = [0.5,0.5] if param['joint'] else None

    if model is None:
        model = create_model(seq_len=param['seq_len'], hidden_size=param['hidden_size'], dropout=param['dropout'], regularizer=param['regularizer'], joint=param['joint'])

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy'],
                      sample_weight_mode='temporal',
                      loss_weights=loss_weight)

    Ys = [Yl_train, Yt_train] if param['joint'] else [Yl_train]
    Ys_val = [Yl_train, Yt_train] if param['joint'] else [Yl_train]
    sample_weights = None
    if param['c_weights']:
        sample_weights = get_sample_weights(Ys)

    model.fit(X_train, Ys, validation_data=(X_val, Ys_val),
              callbacks=[metric,
                         # tensorboard,
                         # earlystopping,
                         checkpoint],
              epochs=epochs, batch_size=param['batch_size'], verbose=2, sample_weight=sample_weights)

    return metric.metrics, model


def main(docs, links, types=None, epochs=1000, paramsearch=paramsearch):
    X, Yl, Yt = preprocess(docs, links, types)
    metrics = crossvalidation(X, Yl, Yt, epochs, paramsearch)
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
    args = parser.parse_args()

    docs, links, types = load_vec(args.docs, args.links, args.types)

    print(docs.shape, links.shape, args.epochs)
    main(docs, links, types, args.epochs)
