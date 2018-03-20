from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Lambda, Concatenate
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
REPR_SIZE = 2927
fixed_param = dict(c_weights=True, joint=True, regularizer=None, hidden_size=512,
                   seq_len=MAX_LEN, batch_size=10)

paramsearch = [
    dict(di=True, dropout=0, recurrent_dropout=0.9, regularizer='l2'),
    dict(di=True, dropout=0, recurrent_dropout=0.9),
]


def create_model(seq_len=10, hidden_size=512,
                 dropout=0.9, recurrent_dropout=0,
                 regularizer=None,
                 activity_regularizer=None,
                 joint=False, n_gpu=None,
                 drop_encoder=0,drop_decoder=0,
                 drop_input=0, drop_fc=0):

    encoder_inputs = Input(shape=(None, REPR_SIZE), name='input')

    masked = Masking(mask_value=0)(encoder_inputs)
    fc_enc = TimeDistributed(Dense(hidden_size, activation='sigmoid',
                                   kernel_regularizer=regularizer), name='FC_input')(masked)

    encoder, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(hidden_size//2, name='encoder',
                                                                               return_state=True,
                                                                               return_sequences=True,
                                                                               recurrent_dropout=recurrent_dropout,
                                                                               dropout=dropout))(fc_enc)

    if joint:
        typ = TimeDistributed(Dense(2, use_bias=True, kernel_regularizer=regularizer,
                                    activity_regularizer=activity_regularizer,
                                    activation='softmax'), name='type')(encoder)

    forward_h = Lambda(lambda x: x[-1:])(forward_h)
    forward_c = Lambda(lambda x: x[-1:])(forward_c)
    backward_h = Lambda(lambda x: x[-1:])(backward_h)
    backward_c = Lambda(lambda x: x[-1:])(backward_c)

    encoder_h = Concatenate()([forward_h, backward_h])
    encoder_c = Concatenate()([forward_c, backward_c])
    last_encoder_state = [encoder_h, encoder_c]

    decoder_inputs = Input(shape=(None, REPR_SIZE))
    fc_dec = TimeDistributed(Dense(hidden_size, activation='sigmoid',
                                   kernel_regularizer=regularizer),
                             name='FC_input')(decoder_inputs)

    decoder = LSTM(hidden_size, name='decoder',
                   return_sequences=True,
                   recurrent_dropout=recurrent_dropout,
                   dropout=dropout)(fc_dec, initial_state=last_encoder_state)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False,
                              kernel_regularizer=regularizer), name='E')(encoder)

    D = TimeDistributed(Dense(hidden_size, use_bias=False,
                              kernel_regularizer=regularizer), name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    pointer = add([E,DD])
    pointer = tanh(pointer)

    vt = Dense(1, use_bias=False, kernel_regularizer=regularizer, name='vT')
    pointer = vt(pointer)
    pointer = Lambda(lambda x: K.squeeze(x, -1))(pointer)
    pointer = Activation('softmax', name='link')(pointer)

    if joint:
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=[pointer, typ])
    else:
        model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=pointer)

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


def preprocess(enc_input, dec_input, links, types=None):
    enc_input_filtered = np.array([x for x in enc_input if len(x) <= MAX_LEN])
    dec_input_filtered = np.array([x for x in dec_input if len(x) <= MAX_LEN])
    links_filtered = np.array([x for x in links if len(x) <= MAX_LEN])
    types_filtered = np.array([x for x in types if len(x) <= MAX_LEN])

    # pad with zeros, truncate longer than 6
    Xe = pad_sequences(enc_input_filtered, dtype=float, truncating='post',padding='post')
    Xd = pad_sequences(dec_input_filtered, dtype=float, truncating='post',padding='post')

    # convert links 1,1,2 -> [1,0,0]
    #                        [1,0,0]
    #                        [0,1,0]
    Yl = [to_categorical([np.array(ys) for ys in y],
                         num_classes=Xe.shape[1]) for y in links_filtered]
    Yl = pad_sequences(Yl, dtype=int, truncating='post', padding='post')
    if types is not None:
        Yt = [pad_sequences(to_categorical(np.array(y)), truncating='post',
                            padding='post') for y in types_filtered]
        Yt = pad_sequences(Yt, dtype=int, truncating='post', padding='post')
    else:
        Yt = None

    return Xe, Xd, Yl, Yt, enc_input_filtered, dec_input_filtered, links_filtered, types_filtered


def crossvalidation(Xe, Xd, Yl, Yt, epochs, paramsearch, n_gpu):
    NUM_TRIALS = 10
    metrics = []
    metric_keys = ['outer', 'inner', 'param', 'score', 'epoch']
    paramset = []
    score_keys = []

    inner_seed = np.random.RandomState(0)
    outer = KFold(n_splits=NUM_TRIALS, shuffle=True, random_state=1)
    training_idx = []
    test_idx = []

    for (i, (training, test)) in tqdm(enumerate(outer.split(Xe)), desc='outer'):
        training_idx.append(training)
        test_idx.append(test)
        inner = KFold(n_splits=5, shuffle=True, random_state=inner_seed)

        metrics.append([])

        Xe_training = Xe[training]
        Xd_training = Xd[training]
        Yl_training = Yl[training]
        Yt_training = Yt[training]

        for (k, (train, val)) in tqdm(enumerate(inner.split(Xe_training)), desc='inner'):
            print('data lengths', len(train),len(val),len(test))
            metrics[-1].append([])
            for param in paramsearch:
                param.update({'cv_iter': i})
                param.update({'tboard': {0:i,
                                         1:k,
                                         2:stringify(param)}})

                if param not in paramset:
                    paramset.append(param)

                Xe_train, Xe_val = Xe_training[train], Xe_training[val]
                Xd_train, Xd_val = Xd_training[train], Xd_training[val]
                Yl_train, Yl_val = Yl_training[train], Yl_training[val]
                Yt_train, Yt_val = Yt_training[train], Yt_training[val]

                model, history = train_model(Xe_train, Xe_val,
                                             Xd_train, Xd_val,
                                             Yl_train, Yl_val,
                                             Yt_train, Yt_val,
                                             epochs, param, n_gpu)

                score_keys = list(OrderedDict(sorted(history.items())).keys())
                metrics[-1][-1].append(list(OrderedDict(sorted(history.items())).values()))

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


def train_model(Xe_train, Xe_val, Xd_train, Xd_val, Yl_train, Yl_val, Yt_train, Yt_val, epochs, param,
                n_gpu=0, model=None):
    params = {}
    params.update(fixed_param)
    params.update(param)
    print(params)
    adam = Adam()
    callbacks = []
    # callbacks = [metric]
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

    history = model.fit([Xe_train, Xd_train],
                        Ys,
                        validation_data=([Xe_val, Xd_val], Ys_val),
                        callbacks=callbacks,
                        epochs=epochs,
                        batch_size=params['batch_size'],
                        verbose=0,
                        sample_weight=sample_weights)

    return model, history.history


def main(enc_input, dec_input, links, types=None, epochs=1000, paramsearch=paramsearch, n_gpu=None):
    Xe, Xd, Yl, Yt, _, _, _, _ = preprocess(enc_input, dec_input, links, types)
    metrics = crossvalidation(Xe, Xd, Yl, Yt, epochs, paramsearch, n_gpu)
    return metrics


def load_vec(enc_input, dec_input, links, types=None):
    enc_input, dec_input, links = np.load(enc_input), np.load(dec_input), np.load(links)
    types = np.load(types) if types else None
    return enc_input, dec_input, links, types


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ei', '--enc_input', help='Encoder Input', type=str)
    parser.add_argument('-di', '--dec_input', help='Decoder Input', type=str)
    parser.add_argument('-l', '--links', help='Links', type=str)
    parser.add_argument('-t', '--types', help='Types', type=str, default=None)
    parser.add_argument('-i', '--ifold', help='Types', type=int, default=3)
    parser.add_argument('-e', '--epochs', help='Epoch', type=int, default=2000)
    parser.add_argument('-g', '--n_gpu', help='Multi GPU', type=int, default=None)
    args = parser.parse_args()

    enc_input, dec_input, links, types = load_vec(args.enc_input, args.dec_input, args.links, args.types)

    print(enc_input.shape, dec_input.shape, links.shape, args.epochs)
    main(enc_input, dec_input, links, types, args.epochs, paramsearch, args.n_gpu)
