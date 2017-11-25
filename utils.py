import torch
from torch.autograd import Variable


def to_onehot(index, vocab_size):
    batch_size, seq_len = index.size(0), index.size(1)
    onehot = torch.FloatTensor(batch_size, seq_len, vocab_size).zero_()
    onehot.scatter_(2, index.data.cpu().unsqueeze(2), 1)
    return onehot


def sample(G, TEXT, batch_size, seq_len, vocab_size, use_cuda=True):
    noise = torch.randn(batch_size, 128)
    if use_cuda:
        noise = noise.cuda()
    noisev = Variable(noise, volatile=True)
    samples = G(noisev)
    samples = samples.view(-1, seq_len, vocab_size)
    _, argmax = torch.max(samples, 2)
    argmax = argmax.cpu().data
    decoded_samples = []
    for i in range(len(argmax)):
        decoded = "".join([TEXT.vocab.itos[s] for s in argmax[i]])
        decoded_samples.append(decoded)
    return decoded_samples
