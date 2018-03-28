from keras.callbacks import Callback
from sklearn.metrics import f1_score
from collections import defaultdict
import numpy as np
import keras.backend as K


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.metrics = defaultdict(list)

    def on_epoch_end(self, batch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        mask_ind = self.validation_data[2].any(axis=2).flatten()
        link_predict = predict.argmax(2).flatten()[mask_ind]
        link_target = self.validation_data[1].argmax(2).flatten()[mask_ind]
        self.metrics['link_macro_f1'].append(f1_score(link_target, link_predict, average='macro'))
        for k, v in logs.items():
            self.metrics[k].append(v)


class JointMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.metrics = defaultdict(list)

    def on_epoch_end(self, batch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        val_link_f1 = flat_f1(self.validation_data[1], predict[0], average='macro')
        val_type_f1 = flat_f1(self.validation_data[2], predict[1], average='macro')
        self.metrics['val_link_macro_f1'].append(val_link_f1)
        self.metrics['val_type_macro_f1'].append(val_type_f1)
        for k, v in logs.items():
            self.metrics[k].append(v)


def f1(y_true, y_pred, **kwargs):
    mask_ind = _mask(y_true)
    y_pred = y_pred.argmax(2).flatten()[mask_ind]
    y_true = y_true.argmax(2).flatten()[mask_ind]
    return f1_score(y_true, y_pred, **kwargs, pos_label=None)


def flat_f1(y_true, y_pred, **kwargs):
    n_labels = y_true.shape[-1]

    mask_ind = _mask(y_true)
    y_true = np.concatenate(y_true, 0)[mask_ind].flatten()

    y_pred = np.concatenate(y_pred, 0)
    y_pred = y_pred[mask_ind].argmax(-1).flatten()
    y_pred = np.eye(n_labels)[y_pred].flatten()
    return f1_score(y_true, y_pred, **kwargs, pos_label=None)


def binary_f1(y_true, y_pred, **kwargs):
    mask_ind = _mask(y_true)
    y_true = np.concatenate(y_true, 0)[mask_ind].argmax(-1).flatten()

    y_pred = np.concatenate(y_pred, 0)
    y_pred = y_pred[mask_ind].argmax(-1).flatten()
    return f1_score(y_true, y_pred, **kwargs, pos_label=None)


def _mask(y_true):
    return y_true.any(2).flatten()


def f1_metric(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))
