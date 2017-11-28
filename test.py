from keras.models import Model
from keras.layers import LSTM, Input
from keras.callbacks import LearningRateScheduler
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import sys
sys.path.append('../pointer-networks')
from PointerLSTM import PointerLSTM

from corpus import *
from sklearn.utils import shuffle


corpus = MTCorpus()


def scheduler(epoch):
    if epoch < nb_epochs/4:
        return learning_rate
    elif epoch < nb_epochs/2:
        return learning_rate*0.5
    return learning_rate*0.1


print("preparing dataset...")
X, Y = shuffle(corpus.documents, corpus.links, random_state=0)
X, Y = pad_sequences(X), pad_sequences(Y)
X_train, Y_train = X[:100], Y[:100]
X_test, Y_test = X[100:], Y[100:]

Y_train, Y_test = np.array([to_categorical(y, nb_classes=X.shape[1]) for y in Y_train]), np.array([to_categorical(y, nb_classes=Y.shape[1]) for y in Y_test])

print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

hidden_size = 128
seq_len = X.shape[1]
nb_epochs = 100
learning_rate = 0.1
batch_size = 5

print("building model...")
main_input = Input(shape=(seq_len, 300), name='main_input')

encoder = LSTM(output_dim=hidden_size, return_sequences=True, name="encoder")(main_input)
decoder = PointerLSTM(hidden_size, output_dim=hidden_size, name="decoder")(encoder)

model = Model(input=main_input, output=decoder)
model.compile(optimizer='adadelta',
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.fit(X_train, Y_train, nb_epoch=nb_epochs, batch_size=batch_size,callbacks=[LearningRateScheduler(scheduler),])
prediction = model.predict(X_test)
print(prediction.shape)
print("------")
print(Y_test.shape)
print("------")
print(model.evaluate(X_test,Y_test))
