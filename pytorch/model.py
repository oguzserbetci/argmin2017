import torch
import torch.nn as nn
from torch.autograd import Variable

import numpy as np


class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.9):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size  # bidirectional
        # Add fully connected layer to inputs
        # self.fc = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=self.hidden_size, num_layers=1,
                            dropout=dropout)

    def forward(self, inputs, states):
        # (seq_len, batch, input_size)
        inputs = inputs.permute(1, 0, 2)
        output, states = self.lstm(inputs, states)
        # (batch, seq_len, input_size)
        output = output.permute(1, 0, 2)
        return output, states

    def init_states(self):
        state_h = torch.Tensor(1, 1, self.hidden_size)
        state_c = torch.Tensor(1, 1, self.hidden_size)
        state_h = nn.init.xavier_uniform(state_h, gain=nn.init.calculate_gain('relu'))
        state_c = nn.init.xavier_uniform(state_c, gain=nn.init.calculate_gain('relu'))
        state_h = Variable(state_h)
        state_c = Variable(state_c)
        return state_h, state_c


class Decoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0.9):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        # Add fully connected layer to inputs
        # self.fc = nn.Linear(input_size, hidden_size)
        self.lstm = nn.LSTM(embedding_size, self.hidden_size, 1,
                            dropout=dropout)

    def forward(self, inputs, states):
        # (seq_len, batch, input_size)
        inputs = inputs.permute(1, 0, 2)
        output, states = self.lstm(inputs, states)
        # (batch, seq_len, input_size)
        output = output.permute(1, 0, 2)
        return output, states

    def init_hidden(self):
        state_h = torch.Tensor(1, 1, self.hidden_states)
        state_c = torch.Tensor(1, 1, self.hidden_states)
        state_h = nn.init.xavier_uniform(state_h, gain=nn.init.calculate_gain('relu'))
        state_c = nn.init.xavier_uniform(state_c, gain=nn.init.calculate_gain('relu'))
        state_h = Variable(state_h)
        state_c = Variable(state_c)
        return state_h, state_c


class Attention(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0):
        super(Attention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout

        # Last decoder output is the input
        self.input_linear = nn.Linear(self.input_size, self.hidden_size, bias=False)
        # Encoder outputs is the context
        self.context_linear = nn.Conv1d(self.input_size, self.hidden_size, 1, 1, bias=False)
        self.v = nn.Linear(self.input_size, 1, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.LogSoftmax(-1)

    def forward(self, input, context):
        '''
        context: Attention context ~ Encoder outputs (t, (batch, seq_len, hidden_size))
        input: Attention input ~ Decoder output (batch, seq_len, hidden_size)
        '''
        input = self.input_linear(input)
        # Repeat input.
        # # print('attn input', input.shape)
        input = input.expand(-1, context.size(1), -1)
        # print('attn input', input.shape)

        # (batch, hidden_size, seq_len)
        context = context.permute(0, 2, 1)
        # print('attn context', context.shape)
        # (batch, hidden_size, seq_len)
        context = self.context_linear(context)
        # print('attn context', context.shape)
        # (batch, seq_len, hidden_size)
        context = context.permute(0, 2, 1)
        # print('attn context', context.shape)

        # (batch, seq_len, hidden_size)
        hidden = self.v(self.tanh(input + context))
        # print('V', hidden.shape)
        # (batch, seq_len)
        hidden = hidden.squeeze(-1)
        # print('V', hidden.shape)

        output = self.softmax(hidden)
        return output


class PointerNetwork(nn.Module):
    def __init__(self, embedding_size, hidden_size, max_length, dropout):
        super(PointerNetwork, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.encoder = Encoder(embedding_size, hidden_size, dropout)
        self.decoder = Decoder(embedding_size, hidden_size, dropout)
        self.attention = Attention(hidden_size, hidden_size, dropout)
        self.training = True
        self.teacher_forcing = 0.5

    def set_training(self, training):
        self.training = training

    def forward(self, encoder_inputs, decoder_inputs):
        '''
        encoder_inputs:
        decoder_inputs:
        '''
        # (batch, 1, hidden)
        encoder_states = self.encoder.init_states()
        # print('es', encoder_states[0].size())

        seq_length = encoder_inputs.size()[1]
        # # # print('e', encoder_inputs.size())
        # encoder_outputs = []
        encoder_outputs = Variable(torch.zeros(1, seq_length, self.hidden_size))
        for ei in range(seq_length):
            # (batch, seq_len, hidden)
            encoder_input = encoder_inputs[:,[ei]]
            # print('ei', encoder_input.size())
            encoder_output, encoder_states = self.encoder(encoder_input, encoder_states)
            # print('eo', encoder_output.size())
            encoder_outputs[:, ei] = encoder_output[:, 0]

        # (batch, seq_len, hidden)
        # encoder_outputs = torch.cat(encoder_outputs, 1)
        # print('eos', encoder_outputs.size())
        # (batch, 1, hidden)
        decoder_states = encoder_states
        # print('ds', decoder_states[0].size())

        decoder_outputs = Variable(torch.zeros(1, seq_length, self.hidden_size))

        outputs = []

        use_teacher_forcing = np.random.rand() < self.teacher_forcing if self.training else False
        if use_teacher_forcing:
            for di in range(seq_length):
                decoder_input = decoder_inputs[:, [di]]
                # print('di', decoder_input.size())
                decoder_output, decoder_states = self.decoder(decoder_input,
                                                              decoder_states)
                decoder_outputs[:, ei] = decoder_output[:, 0]
                # print('do', decoder_output.size())
                link = self.attention(decoder_output, encoder_outputs)
                # print('link', link.shape)
                outputs.append(link)
        else:
            decoder_input = decoder_inputs[:, [0]]
            for di in range(seq_length):
                # print('di', decoder_input.size())
                decoder_output, decoder_states = self.decoder(decoder_input,
                                                              decoder_states)
                decoder_outputs[:, ei] = decoder_output[:, 0]
                # print('do', decoder_output.size())
                link = self.attention(decoder_output, encoder_outputs)
                decoder_input = encoder_inputs[:, torch.max(link, -1)[1]]
                # print('link', link.shape)
                outputs.append(link)

        # outputs = torch.cat(outputs, 0)
        # print('pno', outputs.size())
        return outputs
