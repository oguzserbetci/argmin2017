from keras.callbacks import Callback
from sklearn.metrics import f1_score
from collections import defaultdict
import numpy as np


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
        val_link_f1 = _flat_f1(self.validation_data[1], predict[0])
        val_type_f1 = _flat_f1(self.validation_data[2], predict[1])
        self.metrics['val_link_macro_f1'].append(val_link_f1)
        self.metrics['val_type_macro_f1'].append(val_type_f1)
        for k, v in logs.items():
            self.metrics[k].append(v)


def _flat_f1(y_true, y_pred):
    mask_ind = _mask(y_true)
    y_pred = y_pred.argmax(2).flatten()[mask_ind]
    y_true = y_true.argmax(2).flatten()[mask_ind]
    return f1_score(y_true, y_pred, average='macro')


def _mask(y_true):
    return y_true.any(2).flatten()

