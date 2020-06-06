#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
from torch.autograd import Variable


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, filters, features, shape):
        super(ConvLSTM, self).__init__()
        # self.shape = shape
        self.input_channels = input_channels
        self.filters = filters
        self.features = features
        self.padding = (filters - 1) // 2  # make output same size
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=self.input_channels + self.features,
                      out_channels=4 * self.features,
                      kernel_size=self.filters,
                      stride=1,
                      padding=self.padding),
            nn.GroupNorm(self.features // 4, 4 * self.features)
        )

        self.ci = Variable(torch.randn(self.features, shape[0], shape[1])).cuda()
        self.cf = Variable(torch.randn(self.features, shape[0], shape[1])).cuda()
        self.co = Variable(torch.randn(self.features, shape[0], shape[1])).cuda()

    def calc(self, x, hx, cx):
        grs = self.conv(torch.cat((x, hx), 1))  # return: B, features*4, H, W
        i, f, c, o = torch.split(grs, self.features, dim=1)

        # calc[2]
        i = torch.sigmoid(i + self.ci * cx)  # input
        f = torch.sigmoid(f + self.cf * cx)  # forget
        c = torch.tanh(c)
        o = torch.sigmoid(o + self.co * c)

        # calc[3]
        cy = (f * cx) + (i * c)
        hy = o * torch.tanh(cy)
        return cy, hy

    def forward(self, inputs, hidden, seq):
        # input: Batch, Channels, Seq, H*, W*
        if hidden is None:
            hx = torch.randn(inputs.size(0), self.features, inputs.size(-2), inputs.size(-1)).cuda()
            cx = torch.randn(inputs.size(0), self.features, inputs.size(-2), inputs.size(-1)).cuda()
            # print('s1: -> ', inputs.size(), hx.size())
        else:
            hx, cx = hidden

        inner = []
        if inputs is None:
            inputs = torch.randn(hx.size(0), self.input_channels, seq, hx.size(-2), hx.size(-1)).cuda()
            # print('s2: -> ', inputs.size(), hx.size())

        for index in range(inputs.size(2)):
            x = inputs.select(dim=2, index=index)
            hx, cx = self.calc(x, hx, cx)
            inner.append(hx)

        return torch.stack(inner, dim=2), (hx, cx)


class ConvGRU(nn.Module):
    def __init__(self, input_channels, filters, features):
        super(ConvGRU, self).__init__()
        self.input_channels = input_channels
        self.filters = filters
        self.features = features
        self.padding = (filters - 1) // 2  # make output same size
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.features,
                      2 * self.features, self.filters, 1,
                      self.padding),
            nn.GroupNorm(2 * self.features // 16, 2 * self.features)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.input_channels + self.features,
                      self.features, self.filters, 1, self.padding),
            nn.GroupNorm(self.features // 16, self.features)
        )

    def calc(self, x, hx):
        # data spilt
        combined_1 = torch.cat((x, hx), 1)
        gates = self.conv1(combined_1)

        # calc[1]
        z, r = torch.split(gates, self.features, dim=1)
        z = torch.sigmoid(z)
        r = torch.sigmoid(r)

        # calc[2]
        combined_2 = torch.cat((x, r * hx), 1)
        ht = self.conv2(combined_2)
        ht = torch.tanh(ht)
        hy = (torch.tensor(1) - z) * hx + z * ht
        return hy

    def forward(self, inputs, hidden_state, seq):
        if hidden_state is None:
            hx = torch.randn(inputs.size(0), self.features, inputs.size(-2), inputs.size(-1)).cuda()
        else:
            hx = hidden_state
        inner = []
        if inputs is None:
            inputs = torch.randn(hx.size(0), self.input_channels, seq, hx.size(-2), hx.size(-1)).cuda()

        for index in range(inputs.size(2)):
            x = inputs.select(dim=2, index=index)
            hx = self.calc(x, hx)
            inner.append(hx)

        return torch.stack(inner, dim=2), hx


def cnn_block(inputs, cnn):
    batch, channels, seq, height, width = inputs.size()
    inputs = torch.reshape(inputs, (-1, channels, height, width))
    inputs = cnn(inputs)
    inputs = torch.reshape(inputs, (batch, inputs.size(1), seq, inputs.size(2), inputs.size(3)))
    return inputs


class Encoder(nn.Module):
    def __init__(self, rnn):
        super(Encoder, self).__init__()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 16, 3, 1, 1),  # 512 -> 512
                MishActivation(),

                nn.Conv2d(16, 16, 6, 4, 1),  # 512 -> 128
                MishActivation(),
            ),
            nn.Sequential(
                nn.Conv2d(64, 96, 4, 2, 1),  # 128 -> 64
                MishActivation(),
            ),
            nn.Sequential(
                nn.Conv2d(96, 128, 2, 2, 0),  # 64 -> 32
                MishActivation(),
            ),
            nn.Sequential(
                nn.Conv2d(128, 192, 1, 1, 0),
                MishActivation(),
            ),
        ])
        self.rnn = nn.ModuleList(rnn)

    def forward(self, inputs):
        # input: Batch, Seq, Channels, Height ,Width
        # to Batch, Channels, Seq, Height ,Width
        inputs = inputs.transpose(1, 2)
        hidden_states = []
        for cnn, rnn in zip(self.cnn, self.rnn):
            # to Batch, Channels, Seq, H*, W*
            # print(inputs.size())
            inputs = cnn_block(inputs, cnn)
            # print(inputs.size())
            inputs, state = rnn(inputs, None, inputs.size(2))
            # print(inputs.size())
            hidden_states.append(state)

        return tuple(hidden_states)


class Decoder(nn.Module):
    def __init__(self, rnn, seq):
        super(Decoder, self).__init__()
        self.cnn = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(192, 128, 1, 1, 0),
                MishActivation()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 96, 2, 2, 0),
                MishActivation()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(96, 96, 4, 2, 1),
                MishActivation()
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 16, 6, 4, 1),
                MishActivation(),

                nn.ConvTranspose2d(16, 16, 3, 1, 1),
                MishActivation(),
            )
        ])
        self.fin = nn.Sequential(
            nn.ConvTranspose3d(16, 1, 1, 1, 0),
            MishActivation()
        )
        self.rnn = nn.ModuleList(rnn)
        self.seq = seq

    def calc(self, inputs, state, cnn, rnn):
        inputs, state_stage = rnn(inputs, state, self.seq)
        # print(inputs.size())
        inputs = cnn_block(inputs, cnn)
        # print(inputs.size())
        # out: Batch, Channels, Seq, H*, W*
        return inputs

    def forward(self, hidden_states):
        inputs = self.calc(None, hidden_states[-1], self.cnn[0], self.rnn[0])
        for (cnn, rnn, hidden) in list(zip(self.cnn, self.rnn, hidden_states[::-1]))[1:]:
            inputs = self.calc(inputs, hidden, cnn, rnn)
        inputs = self.fin(inputs)
        inputs = inputs.transpose(1, 2)  # to Batch, Seq, Channels,H*, W*
        return inputs


class ConvLSTMNetwork(nn.Module):
    def __init__(self, seq):
        super(ConvLSTMNetwork, self).__init__()
        # Encoder
        # in_channels, out_channels, kernel_size, stride, padding

        # [shape], input_channels, filters, features
        self.encode_rnn = [
            ConvLSTM(16, 3, 64, (128, 128)),
            ConvLSTM(96, 3, 96, (64, 64)),
            ConvLSTM(128, 3, 128, (32, 32)),
            ConvLSTM(192, 3, 192, (32, 32)),
        ]
        self.encoder = Encoder(self.encode_rnn)

        # Decoder
        # in_channels, out_channels, kernel_size, stride, padding
        self.decode_rnn = [
            ConvLSTM(192, 3, 192, (32, 32)),
            ConvLSTM(128, 3, 128, (32, 32)),
            ConvLSTM(96, 3, 96, (64, 64)),
            ConvLSTM(96, 3, 64, (128, 128)),
        ]
        # [shape], input_channels, filters, features

        self.decoder = Decoder(self.decode_rnn, seq)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class ConvGRUNetwork(nn.Module):
    def __init__(self, seq):
        super(ConvGRUNetwork, self).__init__()
        # Encoder
        # in_channels, out_channels, kernel_size, stride, padding, dilation=1
        self.encode_rnn = [
            ConvGRU(16, 3, 64),
            ConvGRU(96, 3, 96),
            ConvGRU(128, 3, 128),
            ConvGRU(192, 3, 192),
        ]
        self.encoder = Encoder(self.encode_rnn)

        # Decoder
        self.decode_rnn = [
            ConvGRU(192, 3, 192),
            ConvGRU(128, 3, 128),
            ConvGRU(96, 3, 96),
            ConvGRU(96, 3, 64),
        ]
        # in_channels, out_channels, kernel_size, stride, padding, dilation=1

        self.decoder = Decoder(self.decode_rnn, seq)

    def forward(self, inputs):
        return self.decoder(self.encoder(inputs))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature = nn.Sequential(
            nn.Conv3d(1, 16, 4, 8, 1),  # 128
            MishActivation(),
            nn.Conv3d(16, 64, 4, 8, 1),  # 32
            MishActivation(),
        )
        self.classify = nn.Sequential(
            nn.Linear(4096, 1024),
            MishActivation(),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            MishActivation(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def forward(self, inputs):
        inputs = inputs.transpose(1, 2)
        # Batch, Seq, Channels,H*, W*
        # print(inputs.size())
        x = self.feature(inputs)
        x = x.view(-1)
        # print(x.size())
        return self.classify(x)


class MishActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (torch.tanh(nn.functional.softplus(x)))


# class MishActivation(nn.Module):
#     def __init__(self):
#         super().__init__()
#
#     def forward(self, x):
#         return nn.functional.leaky_relu(x, 0.2, True)


class weightedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.thresholds = [0.5, 2, 5, 10, 30]
        self.weights = [1, 1, 2, 5, 10, 30]

    def forward(self, pred, label):
        # Batch, Channels, Seq, H*, W*
        weights = torch.ones_like(pred) * 3
        for i, threshold in enumerate(self.thresholds):
            weights = weights + (self.weights[i + 1] - self.weights[i]) * (label >= threshold).float()
            # weights = weights + (self.weights[i + 1] - self.weights[i]) * (pred*255 >= threshold).float()

        mse = torch.sum(weights * ((pred - label) ** 2), (1, 3, 4))
        mae = torch.sum(weights * (torch.abs((pred - label))), (1, 3, 4))
        return (torch.mean(mse) + torch.mean(mae)) * 5e-6
