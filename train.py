from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Reshape, Lambda, Concatenate, \
                         TimeDistributed, Bidirectional, Masking, Embedding

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model

from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping

from keras.models import load_model

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import ShuffleSplit, train_test_split
from sklearn.utils import class_weight
from tqdm import tqdm
import argparse
import dill

import numpy as np
np.set_printoptions(precision=3,linewidth=1000,edgeitems=100,suppress=True)
seed = 7
np.random.seed(seed)


paramsearch = [
    dict(c_weights=True,  embedding=False),
    dict(c_weights=False, embedding=False),
]


def create_model(seq_len, hidden_size, dropout, embedding):
    inp = Input(shape=(seq_len, 2641), name='input')

    if not embedding:
        mask = Masking(mask_value=0)(inp)
        inputt = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer='l2'), name='FC_input')(mask)
        encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder', recurrent_dropout=dropout, dropout=dropout))(inputt)
        decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=dropout, dropout=dropout)(encoder)
    else:
        embedding = Embedding(mask_zeros=True)(inp)
        inputt = TimeDistributed(Dense(hidden_size, activation='sigmoid', kernel_regularizer='l2'), name='FC_input')(embedding)
        encoder = Bidirectional(LSTM(hidden_size//2, return_sequences=True, name='encoder', recurrent_dropout=dropout, dropout=dropout))(inputt)
        decoder = LSTM(hidden_size, return_sequences=True, name='decoder', recurrent_dropout=dropout, dropout=dropout)(encoder)

    # typ = TimeDistributed(Dense(hidden_size, use_bias=True, kernel_regularizer='l2'), name='type')(encoder)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer='l2'), name='E')(encoder)
    D = TimeDistributed(Dense(hidden_size, use_bias=False, kernel_regularizer='l2'), name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0,108, +0,108)
    vt = Dense(1, use_bias=False, kernel_regularizer='l2', name='vT')
    softmax = Activation('softmax', name='softmax')

    attention = add([E,DD])
    attention = tanh(attention)
    attention = vt(attention)
    attention = Lambda(lambda x: K.squeeze(x, -1))(attention)
    attention = softmax(attention)

    model = Model(inputs=inp, outputs=attention)
    return model


def stringify(params):
    return '-'.join(['{}_{}'.format(k,v) for k, v in params.items()])


def get_class_weights(Y):
    class_num = Y.shape[-1]
    class_weights = dict(zip(np.arange(class_num),np.zeros(class_num)))
    labels = Y[train].argmax(2).flatten()
    un_labels = np.unique(labels)
    weights = class_weight.compute_class_weight('balanced', un_labels, labels)
    class_weights.update(zip(un_labels, weights))
    sample_weights = np.zeros(Y[train].shape[:2])
    for i in range(class_num):
        sample_weights[Y[train].argmax(2) == i] = class_weights[i]

    return sample_weights


def cv(docs, links, epochs):
    tensorboard = TensorBoard(write_graph=False, histogram_freq=100)
    earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=100)

    histories, metrics = [], []

    # pad with zeros, truncate longer than 6
    X = pad_sequences(docs, dtype=float, truncating='post',padding='post', value=0)

    # convert links 1,1,2 -> [1,0,0]
    #                        [1,0,0]
    #                        [0,1,0]
    Y = [pad_sequences(to_categorical(np.array(y)-1), truncating='post', padding='post', value=0, maxlen=X.shape[1]) for y in links]
    Y = pad_sequences(Y, dtype=int, truncating='post', padding='post', value=0)

    fixed_params = dict(hidden_size=512, seq_len=X[0].shape[0], nb_epoch=epochs, batch_size=10, dropout=.9)

    kfold = ShuffleSplit(n_splits=5, test_size=20, random_state=seed)
    chooser = np.random.RandomState(20)
    rest = chooser.choice(np.arange(len(X), 100))
    splits = kfold.split(X[rest])

    for param in paramsearch:
        param.update(fixed_params)
        histories.append([])

        for train, val in tqdm(splits):
            # checkpoint = ModelCheckpoint('checkpoints/', monitor='val_loss' save_best_only=True)
            model = create_model(param['seq_len'], param['hidden_size'], param['dropout'], embedding=param['embedding'])
            adam = Adam()
            model.compile(optimizer=adam,
                          loss='categorical_crossentropy',
                          metrics=['categorical_accuracy'],
                          sample_weight_mode="temporal")

            sample_weights = get_class_weights(Y) if param['c_weights'] else None
            history = model.fit(X[train], Y[train], validation_data=(X[val], Y[val]),
                                callbacks=[tensorboard, earlystopping],
                                epochs=param['nb_epoch'], batch_size=param['batch_size'], verbose=2, sample_weight=sample_weights)

            histories[-1].append(history)

        # last validation accuracy
        val_acc = [h.history['val_categorical_accuracy'][-1] for h in histories]
        metrics.append({'mean':np.mean(val_acc), 'var':np.var(val_acc)})
        print("{} (+/- {})".format(metrics[-1].values()))

        with open('cross_validation/'+stringify(param)) as f:
            dill.dump(histories, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--docs', help='Documents', type=str)
    parser.add_argument('-l', '--links', help='Links', type=str)
    parser.add_argument('-e', '--epochs', help='Epoch', type=int, default=2000)

    args = parser.parse_args()

    docs, links = np.load(args.docs), np.load(args.links)

    cv(docs, links, args.epochs)
