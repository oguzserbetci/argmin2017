from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Lambda
from keras.layers import TimeDistributed, Bidirectional, Masking, Dropout

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, multi_gpu_model

from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils
import os

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

MAX_LEN = 7
fixed_param = dict(c_weights=True, joint=True, regularizer=None, hidden_size=512, seq_len=MAX_LEN, batch_size=10)

paramsearch = [
    dict(recurrent_dropout=0.9, dropout=0),
    dict(recurrent_dropout=0.9, dropout=0, regularizer='l2'),
]


def create_model(seq_len=10, hidden_size=512,
                 dropout=0.9, recurrent_dropout=0,
                 regularizer=None,
                 activity_regularizer=None,
                 joint=False, n_gpu=None,
                 drop_encoder=0,drop_decoder=0,
                 drop_input=0, drop_fc=0):

    inp = Input(shape=(seq_len, 2640), name='input')

    mask = Masking(mask_value=0)(inp)
    dropped = Dropout(drop_input)(mask)
    fc = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer=regularizer), name='FC_input')(dropped)
    dropped = Dropout(drop_fc)(fc)

    encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder',
                                 recurrent_dropout=recurrent_dropout,
                                 dropout=dropout))(dropped)
    decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=recurrent_dropout, dropout=dropout)(encoder)

    if joint:
        typ = TimeDistributed(Dense(2, use_bias=True, kernel_regularizer=regularizer,
                                    activity_regularizer=activity_regularizer,
                                    activation='softmax'), name='type')(encoder)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer=regularizer), name='E')(encoder)
    D = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer=regularizer), name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    attention = add([E,DD])
    attention = tanh(attention)

    vt = Dense(1, use_bias=False, kernel_regularizer=regularizer, name='vT')
    attention = vt(attention)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = Activation('softmax', name='link')(attention)

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
        class_weights = get_class_weights(Y)
        sample_weight = np.zeros(Y.shape[:2])

        class_num = len(class_weights)
        for i in range(class_num):
            sample_weight[Y.argmax(2) == i] = class_weights[i]

        sample_weight[Y.sum(2) == 0] = 0
        sample_weights.append(sample_weight)

    return sample_weights


def get_class_weights(Y):
    class_num = Y.shape[-1]
    class_weights = dict(zip(np.arange(class_num),np.zeros(class_num)))
    labels = Y.argmax(2).flatten()
    unique_labels = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', unique_labels, labels)
    class_weights.update(zip(unique_labels, weights))
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
    NUM_TRIALS = 10
    metrics = []
    metric_keys = ['outer', 'inner', 'param', 'score', 'epoch']
    paramset = []
    score_keys = []

    inner_seed = np.random.RandomState(0)
    outer = KFold(n_splits=NUM_TRIALS, shuffle=True, random_state=1)
    training_idx = []
    test_idx = []

    for (i, (training, test)) in tqdm(enumerate(outer.split(X)), desc='outer'):
        training_idx.append(training)
        test_idx.append(test)
        inner = KFold(n_splits=5, shuffle=True, random_state=inner_seed)

        metrics.append([])

        X_training = X[training]
        Yl_training = Yl[training]
        Yt_training = Yt[training]

        for (k, (train, val)) in tqdm(enumerate(inner.split(X_training)), desc='inner'):
            print('data lengths', len(train),len(val),len(test))
            metrics[-1].append([])
            for param in paramsearch:
                param.update({'cv_iter': i})
                param.update({'tboard': {0:i,
                                         1:k,
                                         2:stringify(param)}})

                if param not in paramset:
                    paramset.append(param)

                X_train, X_val = X_training[train], X_training[val]
                Yl_train, Yl_val = Yl_training[train], Yl_training[val]
                Yt_train, Yt_val = Yt_training[train], Yt_training[val]

                metric, model = train_model(X_train, X_val,
                                            Yl_train, Yl_val,
                                            Yt_train, Yt_val,
                                            epochs, param, n_gpu)

                score_keys = list(OrderedDict(sorted(metric.items())).keys())
                metrics[-1][-1].append(list(OrderedDict(sorted(metric.items())).values()))

        fn = 'cross_validation/{iter}'.format(iter=i)
        os.makedirs(fn, exist_ok=True)
        with open(fn+'/train.pl', 'wb') as f:
            training_data = dict(metrics=metrics, metric_keys=metric_keys,
                                 score_keys=score_keys, params=paramset,
                                 training_idx=training, test_idx=test)
            pickle.dump(training_data, f)
        break

    with open('cross_validation/train.pl', 'wb') as f:
        training_data = dict(metrics=metrics, metric_keys=metric_keys,
                             score_keys=score_keys, params=paramset,
                             training_idx=training_idx, test_idx=test_idx)
        pickle.dump(training_data, f)

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
    callbacks = []
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50)
    # checkpoint = ModelCheckpoint('checkpoints/' + stringify(params), monitor='val_loss', save_best_only=True)
    if params.get('tboard'):
        tboard_desc = params['tboard']
        tboard_run = '/'.join([str(v) for k, v in sorted(tboard_desc.items())])
        tensorboard = TensorBoard(log_dir='/cache/tensorboard-logdir/'+tboard_run,
                                  write_graph=False)
        callbacks.append(tensorboard)

    loss_weight = [0.5, 0.5] if params['joint'] else None

    if model is None:
        model = create_model(seq_len=params['seq_len'],
                             hidden_size=params['hidden_size'],
                             dropout=params['dropout'],
                             recurrent_dropout=params['recurrent_dropout'],
                             regularizer=params['regularizer'], joint=params['joint'],
                             n_gpu=n_gpu,
                             drop_input=params.get('drop_input', 0),
                             activity_regularizer=params.get('activity_regularizer'),
                             drop_fc=params.get('drop_fc', 0))

        model.compile(optimizer=adam,
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', utils.f1],
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
