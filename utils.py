from keras.callbacks import Callback
from sklearn.metrics import f1_score
from collections import defaultdict


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.metrics = defaultdict(list)

    def on_epoch_end(self, batch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        link_predict = predict.argmax(2).flatten()
        link_target = self.validation_data[1].argmax(2).flatten()
        self.metrics['link_macro_f1'].append(f1_score(link_target, link_predict, average='macro'))
        for k, v in logs.items():
            self.metrics[k].append(v)


class JointMetrics(Callback):
    def on_train_begin(self, logs={}):
        self.metrics = defaultdict(list)

    def on_epoch_end(self, batch, logs={}):
        predict = self.model.predict(self.validation_data[0])
        link_predict = predict[0].argmax(2).flatten()
        type_predict = predict[0].argmax(2).flatten()
        link_target = self.validation_data[1].argmax(2).flatten()
        type_target = self.validation_data[2].argmax(2).flatten()
        self.metrics['link_macro_f1'].append(f1_score(link_target, link_predict, average='macro'))
        self.metrics['type_macro_f1'].append(f1_score(type_target, type_predict, average='macro'))
        self.metrics['type_ind_f1'].append(f1_score(type_target, type_predict, labels=[0,1], average=None))
        for k, v in logs.items():
            self.metrics[k].append(v)
