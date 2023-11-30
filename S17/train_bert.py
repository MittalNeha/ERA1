from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import Counter
from os.path import exists
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch
import math
import re

from BERT import build_BERT

# =============================================================================
# Dataset
# =============================================================================
class SentencesDataset(Dataset):
    # Init dataset
    def __init__(self, sentences, vocab, seq_len):
        dataset = self

        dataset.sentences = sentences
        dataset.vocab = vocab + ['<ignore>', '<oov>', '<mask>']
        dataset.vocab = {e: i for i, e in enumerate(dataset.vocab)}
        dataset.rvocab = {v: k for k, v in dataset.vocab.items()}
        dataset.seq_len = seq_len

        # special tags
        dataset.IGNORE_IDX = dataset.vocab['<ignore>']  # replacement tag for tokens to ignore
        dataset.OUT_OF_VOCAB_IDX = dataset.vocab['<oov>']  # replacement tag for unknown words
        dataset.MASK_IDX = dataset.vocab['<mask>']  # replacement tag for the masked word prediction task

    # fetch data
    def __getitem__(self, index, p_random_mask=0.15):
        dataset = self

        # while we don't have enough word to fill the sentence for a batch
        s = []
        while len(s) < dataset.seq_len:
            s.extend(dataset.get_sentence_idx(index % len(dataset)))
            index += 1

        # ensure that the sequence is of length seq_len
        s = s[:dataset.seq_len]
        [s.append(dataset.IGNORE_IDX) for i in range(dataset.seq_len - len(s))]  # PAD ok

        # apply random mask
        s = [(dataset.MASK_IDX, w) if random.random() < p_random_mask else (w, dataset.IGNORE_IDX) for w in s]

        return {'input': torch.Tensor([w[0] for w in s]).long(),
                'target': torch.Tensor([w[1] for w in s]).long()}

    # return length
    def __len__(self):
        return len(self.sentences)

    # get words id
    def get_sentence_idx(self, index):
        dataset = self
        s = dataset.sentences[index]
        s = [dataset.vocab[w] if w in dataset.vocab else dataset.OUT_OF_VOCAB_IDX for w in s]
        return s


# =============================================================================
# Methods / Class
# =============================================================================
def get_batch(loader, loader_iter):
    try:
        batch = next(loader_iter)
    except StopIteration:
        loader_iter = iter(loader)
        batch = next(loader_iter)
    return batch, loader_iter


# =============================================================================
# #Init
# =============================================================================
print('initializing..')
batch_size = 1024
seq_len = 20
embed_size = 128
inner_ff_size = embed_size * 4
n_heads = 8
n_code = 8
n_vocab = 40000
dropout = 0.1
# n_workers = 12

# optimizer
optim_kwargs = {'lr': 1e-4, 'weight_decay': 1e-4, 'betas': (.9, .999)}

# =============================================================================
# Input
# =============================================================================
# 1) load text
print('loading text...')
pth = 'training.txt'
sentences = open(pth).read().lower().split('\n')

# 2) tokenize sentences (can be done during training, you can also use spacy udpipe)
print('tokenizing sentences...')
special_chars = ',?;.:/*!+-()[]{}"\'&'
sentences = [re.sub(f'[{re.escape(special_chars)}]', ' \g<0> ', s).split(' ') for s in sentences]
sentences = [[w for w in s if len(w)] for s in sentences]

# 3) create vocab if not already created
print('creating/loading vocab...')
pth = 'vocab.txt'
if not exists(pth):
    words = [w for s in sentences for w in s]
    vocab = Counter(words).most_common(n_vocab)  # keep the N most frequent words
    vocab = [w[0] for w in vocab]
    open(pth, 'w+').write('\n'.join(vocab))
else:
    vocab = open(pth).read().split('\n')

# 4) create dataset
print('creating dataset...')
dataset = SentencesDataset(sentences, vocab, seq_len)
# kwargs = {'num_workers':n_workers, 'shuffle':True,  'drop_last':True, 'pin_memory':True, 'batch_size':batch_size}
kwargs = {'shuffle': True, 'drop_last': True, 'pin_memory': True, 'batch_size': batch_size}
data_loader = torch.utils.data.DataLoader(dataset, **kwargs)

# =============================================================================
# Model
# =============================================================================
# init model
print('initializing model...')
model = build_BERT(N=n_code, n_heads=n_heads, d_model=embed_size, d_ff=inner_ff_size, src_vocab_size=len(dataset.vocab), src_seq_len=seq_len, dropout=dropout, tgt_vocab_size=len(dataset.vocab))
model = model.cuda()

# =============================================================================
# Optimizer
# =============================================================================
print('initializing optimizer and loss...')
optimizer = optim.Adam(model.parameters(), **optim_kwargs)
loss_model = nn.CrossEntropyLoss(ignore_index=dataset.IGNORE_IDX)

# =============================================================================
# Train
# =============================================================================
print('training...')
print_each = 10
model.train()
batch_iter = iter(data_loader)
n_iteration = 10000
for it in range(n_iteration):

    # get batch
    batch, batch_iter = get_batch(data_loader, batch_iter)

    # infer
    masked_input = batch['input']
    masked_target = batch['target']

    masked_input = masked_input.cuda(non_blocking=True)
    masked_target = masked_target.cuda(non_blocking=True)
    output = model(masked_input)

    # compute the cross entropy loss
    output_v = output.view(-1, output.shape[-1])
    target_v = masked_target.view(-1, 1).squeeze()
    loss = loss_model(output_v, target_v)

    # compute gradients
    loss.backward()

    # apply gradients
    optimizer.step()

    # print step
    if it % print_each == 0:
        print('it:', it,
              ' | loss', np.round(loss.item(), 2),
              ' | Δw:', round(model.src_embed.embedding.weight.grad.abs().sum().item(), 3))

    # reset gradients
    optimizer.zero_grad()

# =============================================================================
# Results analysis
# =============================================================================
print('saving embeddings...')
N = 3000
np.savetxt('values.tsv', np.round(model.embeddings.weight.detach().cpu().numpy()[0:N], 2), delimiter='\t', fmt='%1.2f')
s = [dataset.rvocab[i] for i in range(N)]
open('names.tsv', 'w+').write('\n'.join(s))

print('end')