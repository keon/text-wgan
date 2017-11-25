import os
import argparse
import torch
from torch.autograd import Variable, grad
from torch import optim
from torchtext.data import Field, BucketIterator
from torchtext.datasets import IMDB
from model import Generator, Discriminator
from utils import to_onehot, sample


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-batchs', type=int, default=500000,
                   help='number of epochs for train')
    p.add_argument('-critic_iters', type=int, default=5,
                   help='critic iterations')
    p.add_argument('-batch_size', type=int, default=8,
                   help='number of epochs for train')
    p.add_argument('-seq_len', type=int, default=32,
                   help='sequence length')
    p.add_argument('-lamb', type=int, default=10,
                   help='lambda')
    p.add_argument('-lr', type=float, default=0.00001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=0.25,
                   help='initial learning rate')
    return p.parse_args()


def penalize_grad(D, real, fake, batch_size, lamb, use_cuda=True):
    """
    lamb: lambda
    """
    alpha = torch.rand(batch_size, 1, 1).expand(real.size())
    if use_cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real + ((1 - alpha) * fake)
    if use_cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    d_interpolates = D(interpolates)
    ones = torch.ones(d_interpolates.size())
    if use_cuda:
        ones = ones.cuda()
    gradients = grad(outputs=d_interpolates, inputs=interpolates,
                     grad_outputs=ones, create_graph=True,
                     retain_graph=True, only_inputs=True)[0]
    grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lamb
    return grad_penalty


def train_discriminator(D, G, optim_D, real, lamb, batch_size, use_cuda=True):
    D.zero_grad()

    # train with real
    d_real = D(real)
    d_real = d_real.mean()
    d_real.backward(mone)

    # train with fake
    noise = torch.randn(batch_size, 128)
    if use_cuda:
        noise = noise.cuda()
    noise = Variable(noise, volatile=True)  # freeze G
    fake = G(noise)
    fake = Variable(fake.data)
    inputv = fake
    d_fake = D(inputv)
    d_fake = d_fake.mean()
    d_fake.backward(one)

    grad_penalty = penalize_grad(D, real.data, fake.data,
                                 batch_size, lamb, use_cuda)
    grad_penalty.backward()

    d_loss = d_fake - d_real + grad_penalty
    wasserstein = d_real - d_fake
    optim_D.step()
    return d_loss, wasserstein


def train_generator(D, G, optim_G, batch_size, use_cuda=True):
    G.zero_grad()
    noise = torch.randn(batch_size, 128)
    if use_cuda:
        noise = noise.cuda()
    noisev = Variable(noise)
    fake = G(noisev)
    g = D(fake)
    g = g.mean()
    g.backward(mone)
    g_loss = -g
    optim_G.step()
    return g_loss


def main():
    args = parse_arguments()
    use_cuda = torch.cuda.is_available()

    print("[!] preparing dataset...")
    TEXT = Field(lower=True, fix_length=args.seq_len,
                 tokenize=list, batch_first=True)
    LABEL = Field(sequential=False)
    train_data, test_data = IMDB.splits(TEXT, LABEL)
    TEXT.build_vocab(train_data)
    LABEL.build_vocab(train_data)
    train_iter, test_iter = BucketIterator.splits(
            (train_data, test_data), batch_size=args.batch_size, repeat=True)
    vocab_size = len(TEXT.vocab)
    print("[TRAIN]:%d (dataset:%d)\t[TEST]:%d (dataset:%d)\t[VOCAB]:%d"
          % (len(train_iter), len(train_iter.dataset),
             len(test_iter), len(test_iter.dataset), vocab_size))

    # instantiate models
    G = Generator(dim=512, seq_len=args.seq_len, vocab_size=vocab_size)
    D = Discriminator(dim=512, seq_len=args.seq_len, vocab_size=vocab_size)
    optim_G = optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.9))
    optim_D = optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.9))

    global one, mone
    one = torch.FloatTensor([1])
    mone = one * -1
    if use_cuda:
        G, D = G.cuda(), D.cuda()
        one, mone = one.cuda(), mone.cuda()

    train_iter = iter(train_iter)
    batch_size = args.batch_size
    for b in range(1, args.batchs+1):
        # (1) Update D network
        for p in D.parameters():  # reset requires_grad
            p.requires_grad = True
        for iter_d in range(args.critic_iters):  # CRITIC_ITERS
            batch = next(train_iter)
            text, label = batch.text, batch.label
            text = to_onehot(text, vocab_size)
            if use_cuda:
                text = text.cuda()
            real = Variable(text)
            d_loss, wasserstein = train_discriminator(
                    D, G, optim_D, real, args.lamb, batch_size, use_cuda)
        # (2) Update G network
        for p in D.parameters():
            p.requires_grad = False  # to avoid computation
        g_loss = train_generator(D, G, optim_G, batch_size, use_cuda)

        if b % 500 == 0 and b > 1:
            samples = sample(G, TEXT, 1, args.seq_len, vocab_size, use_cuda)
            print("D:%5.2f G:%5.2f W:%5.2f \nsample:%s \t [%d]" %
                  (d_loss.data[0], g_loss.data[0], wasserstein.data[0],
                   samples[0], label.data[0]))
        if b % 5000 == 0 and b > 1:
            print("[!] saving model")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(G.state_dict(), './.save/wgan_g_%d.pt' % (b))
            torch.save(D.state_dict(), './.save/wgan_d_%d.pt' % (b))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
