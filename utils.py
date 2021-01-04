import re
import torch
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset

from nltk import word_tokenize


DAYS = '\d\d?'
YEARS = '\d{4}'
MONTHS = '(jan(uary)?|feb(ruary)?|mar(ch)?|apr(il)?|may|jun(e)?|jul(y)?|aug(ust)?|sep(tember)?|oct(ober)?|nov(ember)?|dec(ember)?)'
SHORT_DATES = '\d\d?[/.-]\d\d?[/.-]\d\d\d?\d?'

RE_NUMBER = re.compile('\d')
RE_DATE = re.compile(r'{0}\s{1},?\s+{2}|{1}\s{0},?\s{2}|{1}\s{0}|{3}|{2}|{1}'.format(DAYS, MONTHS, YEARS, SHORT_DATES), re.I)
RE_CURRENCY = re.compile(r'[£$€¥₽]')


def encode_sentence(sentence, word_to_index, do_lowercase=True):
    unknown  = word_to_index.get('<UNK>', 1)
    date     = word_to_index.get('<DATE>', 2)
    currency = word_to_index.get('<CURR>', 3)
    number   = word_to_index.get('<NUMB>', 4)

    if isinstance(sentence, str):
        sentence = sentence.split()

    encoded = []
    for token in sentence:
        if   RE_DATE.search(token):     encoded.append(date)
        elif RE_CURRENCY.search(token): encoded.append(currency)
        elif RE_NUMBER.search(token):   encoded.append(number)
        else: encoded.append(word_to_index.get(token.lower() if do_lowercase else token, unknown))

    return encoded


def pad_encodings(inputs):
    pad_ix = 0
    seqlen = len(inputs)
    maxlen = max(map(len, inputs))

    mask = torch.zeros(seqlen, maxlen).long()
    outs = []

    for i in range(seqlen):
        mask[i, :len(inputs[i])] = 1
        outs.append(inputs[i] + [pad_ix] * max(0, maxlen - len(inputs[i])))

    return torch.tensor(outs).long(), mask


def load_model(path, model):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state['model'])


def predict(sentences, vocab, model, batch_size=3):
    index_to_label = {
        0 : 'subjective', 
        1 : 'objective'
    }

    model.eval()
    encodings = [encode_sentence(s, vocab) for s in sentences]
    batches = len(encodings) // batch_size + int(len(encodings) % batch_size != 0)

    probs, attns, labels = [], [], []
    for i in range(batches):
        batch = encodings[i * batch_size: (i + 1) * batch_size]
        batch, mask = pad_encodings(batch)
        
        batch_probs, _, batch_attns = model(batch, mask)
        
        probs.extend(batch_probs.tolist())
        labels.extend([index_to_label[l] for l in batch_probs.argmax(dim=1).tolist()])
        
        attns.extend([batch_attns[i][:mask[i].sum()].tolist() for i in range(len(batch_attns))])
        
    return labels, probs, attns


def plot_attention_bars(words, attention, label, word_axis=4.8, value_axis=10, horbars=False):
    indexes = np.arange(len(words))
    word_axis_figsize = round(len(indexes) / word_axis)

    if horbars:
        matplotlib.rc('figure', figsize=(value_axis, word_axis_figsize))
        pltbar = plt.barh
    else:
        matplotlib.rc('figure', figsize=(word_axis_figsize, value_axis))
        pltbar = plt.bar

    width = 1
    pltbar(indexes, attention, width, alpha=0.5)
    
    if horbars:
        plt.xlabel('\nProbability', fontsize=16)
        # plt.ylabel('Sentence', fontsize=16)
        
        plt.ylim(-1, len(indexes))
        plt.yticks(indexes, words)
        plt.xticks(fontsize=14)
        plt.tick_params(labeltop=True, labelsize=14)
        ax = plt.gca()                   # get the axis
        ax.set_ylim(ax.get_ylim()[::-1]) # invert the axis
    else:
        # plt.xlabel('Sentence', fontsize=16)
        plt.ylabel('Probability', fontsize=16)
        
        plt.tick_params(labelright=True)
        plt.xticks(indexes, words, rotation=90, fontsize=16)
        plt.xlim(left=-1, right=len(indexes))
        
    plt.title(f'{label.upper()}\n', fontsize=24)
    plt.grid(True)
    plt.show()



class ObjSubDataset(Dataset):
    def __init__(self, sentences, labels=None, companies=None, filings=None, sections=None, indexes=None):
        self.sentences = sentences
        self.labels = labels
        self.encodings = []
        self.is_encoded = False

        self.companies = companies
        self.filings = filings
        self.sections = sections
        self.indexes = indexes

    def __getitem__(self, index):
        assert len(self.encodings) == len(self.sentences), f"{len(self.encodings)}, {len(self.sentences)}"
        if self.labels is not None:
            if self.is_encoded:
                assert index < len(self.encodings)
                assert index < len(self.labels), f"{index}, {len(self.labels)}"
                return self.encodings[index], self.labels[index]
            else:
                assert index < len(self.sentences)
                return self.sentences[index], self.labels[index]
        else:
            if self.is_encoded:
                assert index < len(self.encodings)
                return self.encodings[index], None
            else:
                assert index < len(self.sentences)
                return self.sentences[index], None

    def __len__(self):
        return len(self.sentences)

    def encode_labels(self, label_to_index):
        self.labels = [label_to_index[l] for l in self.labels]

    def decode_labels(self, index_to_label):
        self.labels = [index_to_label[l] for l in self.labels]

    def encode(self, word_to_index):
        self.encodings = []
        for sentence in self.sentences:
            self.encodings.append(encode_sentence(sentence, word_to_index, do_lowercase=True))
        self.is_encoded = True
        assert len(self.encodings) == len(self.sentences)

    def decode(self, index_to_word):
        for i in range(len(self.encodings)):
            self.encodings[i] = decode_sentence(self.encodings[i], index_to_word)
        self.is_encoded = False
        assert len(self.encodings) == len(self.sentences)

    def collate(self, batch):
        encoded, labels = zip(*batch)
        encoded_t = [e for e in encoded]

        pad_ix = 0
        maxlen = max([len(e) for e in encoded_t])
        encoded_mask_t = torch.zeros(len(encoded_t), maxlen).long()

        # Padding and masking for the sentences
        for i in range(len(encoded_t)):
            encoded_mask_t[i, :len(encoded_t[i])] = 1
            encoded_t[i] += [pad_ix] * max(0, (maxlen - len(encoded_t[i])))

        encoded_t = torch.tensor(encoded_t).long()
        if not any(label is None for label in labels):
            labels_t = torch.tensor(labels).long()
            return encoded_t, encoded_mask_t, labels_t
        else:
            return encoded_t, encoded_mask_t


embedded_data = [
    ("Our oil production was 9% lower in 2008 than 2007.", 1),
    ("The impact of inflation on the Company's operations has not been significant to date.", 0),
    ("We believe that we can continue to operate efficiently in challenging economic and industry environments.", 0),
    ("The number of commercial sites in the U.S. is now approximately 400.", 1)
]            