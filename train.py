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

from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm
import argparse
import pickle

import numpy as np
np.set_printoptions(precision=3, linewidth=1000, edgeitems=100, suppress=True)
seed = 7
np.random.seed(seed)

MAX_LEN = 6
fixed_params = dict(hidden_size=512, seq_len=MAX_LEN, batch_size=10, dropout=0)

paramsearch = [
    dict(c_weights=True, joint=True),
]


def create_model(seq_len, hidden_size, dropout, joint=False):
    print('create')
    inp = Input(shape=(seq_len, 2641), name='input')

    mask = Masking(mask_value=0)(inp)
    fc = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer='l2'), name='FC_input')(mask)

    encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder', recurrent_dropout=dropout, dropout=dropout))(fc)
    decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=dropout, dropout=dropout)(encoder)

    if joint:
        typ = TimeDistributed(Dense(2, use_bias=True, kernel_regularizer='l2', activation='softmax'), name='type')(encoder)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer='l2'), name='E')(encoder)
    D = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer='l2'), name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    # glorot_uniform initializer:
    # for hidden=512: uniform(-0,108, +0,108)
    vt = Dense(1, use_bias=False, kernel_regularizer='l2', name='vT')
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


def stringify(params):
    return '-'.join(['{}_{}'.format(k,v) for k, v in params.items()])


def get_class_weights(Ys):
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


def crossvalidation(docs, links, epochs, fold=3, types=None):
    print('cv')
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100)

    metrics = []

    docs_masked = np.array([x for x in docs if len(x) <= MAX_LEN])
    links_masked = np.array([x for x in links if len(x) <= MAX_LEN])
    types_masked = np.array([x for x in types if len(x) <= MAX_LEN])

    # pad with zeros, truncate longer than 6
    X = pad_sequences(docs_masked, dtype=float, truncating='post',padding='post')

    # convert links 1,1,2 -> [1,0,0]
    #                        [1,0,0]
    #                        [0,1,0]
    Y = [pad_sequences(to_categorical(np.array(y)-1), truncating='post', padding='post', maxlen=X.shape[1]) for y in links_masked]
    Y = pad_sequences(Y, dtype=int, truncating='post', padding='post')
    if types is not None:
        Y_t = [pad_sequences(to_categorical(np.array(y)), truncating='post', padding='post') for y in types_masked]
        Y_t = pad_sequences(Y_t, dtype=int, truncating='post', padding='post')

    chooser = np.random.RandomState(20)
    rest = chooser.choice(np.arange(len(X)), 100)
    X_train = X[rest]
    Y_train = Y[rest]
    Y_t_train = Y_t[rest]
    kfold = ShuffleSplit(n_splits=fold, test_size=20, random_state=0)
    splits = kfold.split(X_train)

    for param in paramsearch:
        param.update(fixed_params)
        metrics.append([])

        for (i, (train, val)) in enumerate(tqdm(splits)):
            checkpoint = ModelCheckpoint('checkpoints/' + stringify(param) + str(i), monitor='val_loss', save_best_only=True)

            loss_weight = [0.5,0.5] if param['joint'] else None

            model = create_model(param['seq_len'], param['hidden_size'], param['dropout'], param['joint'])
            adam = Adam()
            metric = utils.JointMetrics() if param['joint'] else utils.Metrics()
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'],
                          sample_weight_mode='temporal',
                          loss_weights=loss_weight)

            Ys = [Y_train[train], Y_t_train[train]] if param['joint'] else [Y_train[train]]
            Ys_val = [Y_train[val], Y_t_train[val]] if param['joint'] else [Y_train[val]]
            sample_weights = get_class_weights(Ys) if param['c_weights'] else None
            model.fit(X_train[train], Ys, validation_data=(X_train[val], Ys_val),
                      callbacks=[
                          # tensorboard,
                          metric,
                          earlystopping,
                          checkpoint
                      ],
                      epochs=epochs, batch_size=param['batch_size'], verbose=2, sample_weight=sample_weights)

            metrics[-1].append(metric.metrics)
            yield model, metric.metrics

        # last validation accuracy
        for m in metrics[-1][-1].keys():
            val_acc = [h[m][-1] for h in metrics[-1]]
            mean, var = np.mean(val_acc, axis=0), np.var(val_acc, axis=0)
            print("{}:{} (+/- {})".format(m, mean, var))

        with open('cross_validation/' + stringify(param), 'wb') as f:
            pickle.dump(metrics, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--docs', help='Documents', type=str)
    parser.add_argument('-l', '--links', help='Links', type=str)
    parser.add_argument('-t', '--types', help='Types', type=str, default=None)
    parser.add_argument('-e', '--epochs', help='Epoch', type=int, default=2000)
    args = parser.parse_args()

    docs, links = np.load(args.docs), np.load(args.links)
    types = np.load(args.types) if args.types else None

    crossvalidation(docs, links, args.epochs, fold=3, types=types)
    print('FINISH')
