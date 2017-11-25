import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, dim):
        super(ResBlock, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Conv1d(dim, dim, 5, padding=2),  # nn.Linear(DIM, DIM),
        )

    def forward(self, input):
        output = self.res_block(input)
        return input + (0.3 * output)


class Generator(nn.Module):
    def __init__(self, dim, seq_len, vocab_size):
        super(Generator, self).__init__()
        self.dim = dim
        self.seq_len = seq_len

        self.fc1 = nn.Linear(128, dim * seq_len)
        self.block = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1 = nn.Conv1d(dim, vocab_size, 1)
        self.softmax = nn.Softmax()

    def forward(self, noise):
        batch_size = noise.size(0)
        output = self.fc1(noise)
        # (BATCH_SIZE, DIM, SEQ_LEN)
        output = output.view(-1, self.dim, self.seq_len)
        output = self.block(output)
        output = self.conv1(output)
        output = output.transpose(1, 2)
        shape = output.size()
        output = output.contiguous()
        output = output.view(batch_size * self.seq_len, -1)
        output = self.softmax(output)
        # (BATCH_SIZE, SEQ_LEN, len(charmap))
        return output.view(shape)


class Discriminator(nn.Module):
    def __init__(self, dim, seq_len, vocab_size):
        super(Discriminator, self).__init__()
        self.dim = dim
        self.seq_len = seq_len

        self.block = nn.Sequential(
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
            ResBlock(dim),
        )
        self.conv1d = nn.Conv1d(vocab_size, dim, 1)
        self.linear = nn.Linear(seq_len * dim, 1)

    def forward(self, input):
        # (BATCH_SIZE, VOCAB_SIZE, SEQ_LEN)
        output = input.transpose(1, 2)
        output = self.conv1d(output)
        output = self.block(output)
        output = output.view(-1, self.seq_len * self.dim)
        output = self.linear(output)
        return output
