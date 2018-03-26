from model import PointerNetwork
from MTG import MTCDataset

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch

from tqdm import tqdm
import numpy as np
import time


def timer(func):
    def timed_func(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        # print(start-time.time())
    return timed_func


BATCH_SIZE = 1


@timer
def train(dataloader, n_epochs, print_every=1000, plot_every=100, **params):
    model = PointerNetwork(**params)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.2)
    print([str(p) for p in list(model.parameters())])
    return

    losses = []

    for epoch in range(1, n_epochs + 1):
        batchloss = []
        iterator = tqdm(dataloader, unit='Batch')

        for i_batch, sample_batched in enumerate(iterator):
            optimizer.zero_grad()

            encoder_batch = Variable(sample_batched['Encoder'])
            decoder_batch = Variable(sample_batched['Decoder'])
            links_batch = Variable(sample_batched['Links']).squeeze()
            print('encoder inputs:', encoder_batch.shape)
            print('decoder inputs:', decoder_batch.shape)
            print('links:', links_batch.shape)

            outputs = model(encoder_batch, decoder_batch)[:len(links_batch)]
            output_batch = torch.max(torch.cat(outputs, 0), -1)[1]
            print(output_batch, links_batch)

            loss = 0
            for i, output in enumerate(outputs):
                loss += criterion(output, links_batch[i])

            loss.backward()
            optimizer.step()

            losses.append(loss.data[0])
            batchloss.append(loss.data[0])

            iterator.set_postfix(loss=str(loss.data[0]))
            break
        iterator.set_postfix(loss=np.mean(batchloss))


if __name__ == "__main__":
    dataset = MTCDataset('../corpus/encoder_input.npy','../corpus/decoder_input.npy',
                         '../corpus/Y_links_1.npy', '../corpus/Y_types_1.npy',
                         maxlen=7)
    dataloader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            num_workers=4)

    train(dataloader, n_epochs=4000, embedding_size=2927, hidden_size=56, max_length=7,
          dropout=0)
