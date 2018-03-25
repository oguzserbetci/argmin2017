from torch import from_numpy
from torch.utils.data import Dataset
import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences


class MTCDataset(Dataset):
    def __init__(self, enc_input, dec_input, links, types, maxlen=7):
        enc_input, dec_input, links, types = load_vec(enc_input, dec_input, links, types)
        self.Xe, self.Xd, self.Yl, self.Yt, _, _, _, _ = preprocess(enc_input, dec_input, links, types,
                                                                    maxlen)
        self.maxlen = maxlen

    def __len__(self):
        return len(self.Xe)

    def __getitem__(self, idx):
        encoder_inputs = from_numpy(self.Xe[idx]).float()
        decoder_inputs = from_numpy(self.Xd[idx]).float()
        links = from_numpy(self.Yl[idx]).long()
        types = from_numpy(self.Yt[idx]).long()

        sample = {'Encoder':encoder_inputs, 'Decoder':decoder_inputs, 'Links': links, 'Types':types}

        return sample


def load_vec(enc_input, dec_input, links, types=None):
    enc_input, dec_input, links = np.load(enc_input), np.load(dec_input), np.load(links)
    types = np.load(types) if types else None
    return enc_input, dec_input, links, types


def preprocess(enc_input, dec_input, links, types, MAX_LEN):
    enc_input_filtered = np.array([x for x in enc_input if len(x) <= MAX_LEN])
    dec_input_filtered = np.array([x for x in dec_input if len(x) <= MAX_LEN])
    links_filtered = np.array([x for x in links if len(x) <= MAX_LEN])
    types_filtered = np.array([x for x in types if len(x) <= MAX_LEN])

    return enc_input_filtered, dec_input_filtered, links_filtered, types_filtered
