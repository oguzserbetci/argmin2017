from keras.models import Model
from keras import backend as K
from keras.layers import LSTM, Input, Dense, Activation, Add, Lambda, Concatenate
from keras.layers import TimeDistributed, Bidirectional, Masking, Dropout
from keras.utils import multi_gpu_model


def create_model(seq_len=10,
                 embedding_size=3946,
                 hidden_size=512,
                 n_gpu=0, joint=False,
                 # PARAMS
                 dropout=0.9, recurrent_dropout=0.9,
                 regularizer=None, activity_regularizer=None,
                 drop_encoder=0,drop_decoder=0,
                 drop_input=0, drop_fc=0, **kwargs):

    encoder_inputs = Input(shape=(None, embedding_size), name='input')

    masked = Masking()(encoder_inputs)
    fc_enc = TimeDistributed(Dense(hidden_size, activation='sigmoid',
                                   kernel_regularizer=kwargs.get('regularizer', None)),
                             name='FC_enc')(masked)

    encoder, forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(hidden_size//2, name='encoder',
                                                                               return_state=True,
                                                                               return_sequences=True,
                                                                               recurrent_dropout=kwargs.get('recurrent_dropout', 0),
                                                                               dropout=kwargs.get('dropout', 0)))(fc_enc)

    if joint:
        typ = TimeDistributed(Dense(2, activation='softmax',
                                    kernel_regularizer=kwargs.get('regularizer', None),
                                    activity_regularizer=kwargs.get('activity_regularizer', None)),
                              name='type')(encoder)

    forward_h = Lambda(lambda x: x[-1:])(forward_h)
    forward_c = Lambda(lambda x: x[-1:])(forward_c)
    backward_h = Lambda(lambda x: x[-1:])(backward_h)
    backward_c = Lambda(lambda x: x[-1:])(backward_c)

    encoder_h = Concatenate()([forward_h, backward_h])
    encoder_c = Concatenate()([forward_c, backward_c])
    last_encoder_state = [encoder_h, encoder_c]

    decoder_inputs = Input(shape=(None, embedding_size))
    fc_dec = TimeDistributed(Dense(hidden_size, activation='sigmoid',
                                   kernel_regularizer=kwargs.get('regularizer', None)),
                             name='FC_dec')(decoder_inputs)

    decoder = LSTM(hidden_size, name='decoder',
                   return_sequences=True,
                   recurrent_dropout=kwargs.get('recurrent_dropout', 0),
                   dropout=kwargs.get('dropout', 0))(fc_dec, initial_state=last_encoder_state)

    # glorot_uniform initializer:
    # uniform([-limit,limit]) where limit = sqrt(6/(in+out))
    # for hidden=512: uniform(-0.07, +0.07)
    E = TimeDistributed(Dense(hidden_size, use_bias=False,
                              kernel_regularizer=kwargs.get('regularizer', None)),
                        name='E')(encoder)

    D = TimeDistributed(Dense(hidden_size, use_bias=False,
                              kernel_regularizer=kwargs.get('regularizer', None)),
                        name='D')(decoder)

    DD = Lambda(lambda x: K.repeat_elements(K.expand_dims(x, 2), seq_len, 2))(D)

    add = Add(name='W1E_W2Di')
    tanh = Activation('tanh', name='tanh')

    pointer = add([E,DD])
    pointer = tanh(pointer)

    vt = Dense(1, use_bias=False, kernel_regularizer=kwargs.get('regularizer', None), name='vT')
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
