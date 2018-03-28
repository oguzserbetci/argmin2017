from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

import utils
from model import create_model
import os

from sklearn.model_selection import KFold
from sklearn.utils import class_weight
from tqdm import tqdm
from collections import OrderedDict
import pickle
import argparse
from collections import defaultdict

import numpy as np

np.set_printoptions(precision=3, linewidth=1000, edgeitems=100, suppress=True)
seed = 7
np.random.seed(seed)

MAX_LEN = 10
REPR_SIZE = 2927
fixed_param = dict(regularizer=None, hidden_size=512,
                   seq_len=MAX_LEN, batch_size=16, dropout=0.9, recurrent_dropout=0.9)

paramsearch = [
    dict(di=True, c_weights=True, joint=True),
    dict(di=True, c_weights=False, joint=True),
    dict(di=True, c_weights=True, joint=False),
    dict(di=True, c_weights=False, joint=False),
]


def stringify(param):
    return '-'.join(sorted(['{}_{}'.format(k,v) for k, v in param.items() if k != 'tboard']))


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
    tests = defaultdict(lambda: defaultdict(list))
    metric_keys = ['outer', 'inner', 'param', 'score', 'epoch']
    paramset = []
    score_keys = []

    # inner_seed = np.random.RandomState(0)
    outer = KFold(n_splits=NUM_TRIALS, shuffle=True, random_state=1)
    training_idx = []
    test_idx = []

    for (i, (training, test)) in tqdm(enumerate(outer.split(Xe)), desc='outer'):
        training_idx.append(training)
        test_idx.append(test)
        # inner = KFold(n_splits=5, shuffle=True, random_state=inner_seed)

        metrics.append([])

        Xe_training = Xe[training]
        Xd_training = Xd[training]
        Yl_training = Yl[training]
        Yt_training = Yt[training]

        Xe_test = Xe[test]
        Xd_test = Xd[test]
        Yl_test = Yl[test]
        Yt_test = Yt[test]

        # for (k, (train, val)) in tqdm(enumerate(inner.split(Xe_training)), desc='inner'):
            # print('data lengths', len(train),len(val),len(test))
            # metrics[-1].append([])
            # for param in paramsearch:
                # param.update({'cv_iter': i})
                # param.update({'tboard': {0:i,
                                         # 1:k,
                                         # 2:stringify(param)}})

                # if param not in paramset:
                    # paramset.append(param)

                # inputs = [Xe_training[train], Xd_training[train]]
                # targets = [Yl_training[train], Yt_training[train]] if param['joint'] else [Yl_training[train]]
                # if param['joint']:
                    # validation = ([Xe_training[val], Xd_training[val]], [Yl_training[val], Yt_training[val]])
                # else:
                    # validation = ([Xe_training[val], Xd_training[val]], Yl_training[val])

                # # TRAIN & VALIDATE
                # model, history = train_model(inputs, targets, validation, epochs, param, n_gpu)

                # score_keys = list(OrderedDict(sorted(history.items())).keys())
                # metrics[-1][-1].append(list(OrderedDict(sorted(history.items())).values()))

        # TEST
        for param in tqdm(paramsearch, desc='params'):
            param_str = stringify(param)
            inputs = [Xe_training, Xd_training]
            targets = [Yl_training, Yt_training] if param['joint'] else [Yl_training]
            model, history = train_model(inputs, targets, validation=None,
                                         epochs=epochs, param=param, n_gpu=n_gpu)
            model.save('../cross_validation/{}_{}.h5'.format(stringify(param), i))
            score_keys = list(OrderedDict(sorted(history.items())).keys())
            metrics[-1].append(list(OrderedDict(sorted(history.items())).values()))

            Y_preds = model.predict([Xe_test, Xd_test])
            Y_preds = [Y_preds] if not isinstance(Y_preds, list) else Y_preds
            Y_trues = [Yl_test, Yt_test] if param['joint'] else [Yl_test]
            outputs = ['links', 'types'] if param['joint'] else ['links']
            f1_functions = [utils.flat_f1, utils.binary_f1] if param['joint'] else [utils.flat_f1]

            for Y_pred, Y_true, output, func in zip(Y_preds, Y_trues, outputs, f1_functions):
                print(output.upper())
                macro = func(Y_true, Y_pred, average='macro')
                macro_cat = utils.binary_f1(Y_true, Y_pred, average='macro')
                weighted = func(Y_true, Y_pred, average='weighted')
                singles = func(Y_true, Y_pred, average=None)
                print('macro:', macro)
                print('macro_cat:', macro_cat)
                print('weighted:', weighted)
                print('categories:', singles)
                tests[param_str].setdefault('macro', []).append(macro)
                tests[param_str].setdefault('macro_cat', []).append(macro_cat)
                tests[param_str].setdefault('weighted', []).append(weighted)
                tests[param_str].setdefault('cat0', []).append(singles[0])
                tests[param_str].setdefault('cat1', []).append(singles[1])
                print(tests[param_str])

        fn = '../cross_validation/{iter}'.format(iter=i)
        os.makedirs(fn, exist_ok=True)
        with open(fn+'/train.pl', 'wb') as f:
            training_data = dict(metrics=metrics, metric_keys=metric_keys,
                                 score_keys=score_keys, params=paramset,
                                 training_idx=training, test_idx=test,
                                 tests=dict(tests))
            pickle.dump(training_data, f)

    with open('../cross_validation/train.pl', 'wb') as f:
        training_data = dict(metrics=metrics, metric_keys=metric_keys,
                             score_keys=score_keys, params=paramset,
                             training_idx=training_idx, test_idx=test_idx,
                             tests=tests)
        pickle.dump(training_data, f)

    print(metric_keys)
    print(score_keys)
    return metrics, metric_keys


def train_model(inputs, targets, validation, epochs, param, n_gpu=0, model=None):
    params = {}
    params.update(fixed_param)
    params.update(param)
    print(params)
    callbacks = []
    # earlystopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50)
    # checkpoint = ModelCheckpoint('checkpoints/' + stringify(params), monitor='val_loss', save_best_only=True)

    if params.get('tboard'):
        tboard_desc = params['tboard']
        tboard_run = '/'.join([str(v) for k, v in sorted(tboard_desc.items())])
        tensorboard = TensorBoard(log_dir='/cache/tensorboard-logdir/'+tboard_run,
                                  write_graph=False)
        callbacks.append(tensorboard)

    loss_weights = [0.5, 0.5] if params['joint'] else None

    if model is None:
        model = create_model(**params)

        model.compile(optimizer=params.get('optimizer', 'adam'),
                      loss='categorical_crossentropy',
                      metrics=['categorical_accuracy', utils.f1_metric],
                      sample_weight_mode='temporal',
                      loss_weights=loss_weights)

    sample_weights = None
    if params['c_weights']:
        sample_weights = get_sample_weights(targets)

    history = model.fit(inputs,
                        targets,
                        validation_data=validation,
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
