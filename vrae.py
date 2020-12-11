import numpy as np
import torch
from torch import nn, optim
from torch import distributions
from base import BaseEstimator
from torch.utils.data import DataLoader
from torch.autograd import Variable
import os


class Encoder(nn.Module):
    def __init__(self, number_of_features, hidden_size, hidden_layer_depth, latent_length, dropout, block = 'LSTM'):

        super(Encoder, self).__init__()

        self.number_of_features = number_of_features
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length

        if block == 'LSTM':
            self.model = nn.LSTM(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        elif block == 'GRU':
            self.model = nn.GRU(self.number_of_features, self.hidden_size, self.hidden_layer_depth, dropout = dropout)
        else:
            raise NotImplementedError

    def forward(self, x):
        _, (h_end, c_end) = self.model(x)

        h_end = h_end[-1, :, :]
        return h_end


class Lambda(nn.Module):
    def __init__(self, hidden_size, latent_length):
        super(Lambda, self).__init__()

        self.hidden_size = hidden_size
        self.latent_length = latent_length

        self.hidden_to_mean = nn.Linear(self.hidden_size, self.latent_length)
        self.hidden_to_logvar = nn.Linear(self.hidden_size, self.latent_length)

        nn.init.xavier_uniform_(self.hidden_to_mean.weight)
        nn.init.xavier_uniform_(self.hidden_to_logvar.weight)

    def forward(self, cell_output):
        self.latent_mean = self.hidden_to_mean(cell_output)
        self.latent_logvar = self.hidden_to_logvar(cell_output)

        if self.training:
            std = torch.exp(0.5 * self.latent_logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(self.latent_mean)
        else:
            return self.latent_mean

class Decoder(nn.Module):
    def __init__(self, sequence_length, batch_size, hidden_size, hidden_layer_depth, latent_length, output_size, dtype, block='LSTM'):

        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.output_size = output_size
        self.dtype = dtype

        if block == 'LSTM':
            self.c_0 = torch.zeros(self.hidden_layer_depth, self.batch_size, self.hidden_size, requires_grad=True).type(self.dtype)
            self.model = nn.LSTM(1, self.hidden_size, self.hidden_layer_depth)
        elif block == 'GRU':
            self.model = nn.GRU(1, self.hidden_size, self.hidden_layer_depth)
        else:
            raise NotImplementedError

        self.latent_to_hidden = nn.Linear(self.latent_length, self.hidden_size)
        self.hidden_to_output = nn.Linear(self.hidden_size, self.output_size)

        self.decoder_inputs = torch.zeros(self.sequence_length, self.batch_size, 1, requires_grad=True).type(self.dtype)
        

        nn.init.xavier_uniform_(self.latent_to_hidden.weight)
        nn.init.xavier_uniform_(self.hidden_to_output.weight)

    def forward(self, latent):
        """Converts latent to hidden to output

        :param latent: latent vector
        :return: outputs consisting of mean and std dev of vector
        """
        h_state = self.latent_to_hidden(latent)

        if isinstance(self.model, nn.LSTM):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, (h_0, self.c_0))
        elif isinstance(self.model, nn.GRU):
            h_0 = torch.stack([h_state for _ in range(self.hidden_layer_depth)])
            decoder_output, _ = self.model(self.decoder_inputs, h_0)
        else:
            raise NotImplementedError

        out = self.hidden_to_output(decoder_output)
        return out

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

class VRAE(BaseEstimator, nn.Module):
    def __init__(self, sequence_length, number_of_features, hidden_size=90, hidden_layer_depth=2, latent_length=20,
                 batch_size=32, learning_rate=0.005, block='LSTM',
                 n_epochs=5, dropout_rate=0., optimizer='Adam', loss='MSELoss',
                 cuda=False, print_every=100, clip=True, max_grad_norm=5, dload='.'):

        super(VRAE, self).__init__()


        self.dtype = torch.FloatTensor
        self.use_cuda = cuda

        if not torch.cuda.is_available() and self.use_cuda:
            self.use_cuda = False


        if self.use_cuda:
            self.dtype = torch.cuda.FloatTensor


        self.encoder = Encoder(number_of_features = number_of_features,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               dropout=dropout_rate,
                               block=block)

        self.lmbd = Lambda(hidden_size=hidden_size,
                           latent_length=latent_length)

        self.decoder = Decoder(sequence_length=sequence_length,
                               batch_size = batch_size,
                               hidden_size=hidden_size,
                               hidden_layer_depth=hidden_layer_depth,
                               latent_length=latent_length,
                               output_size=number_of_features,
                               block=block,
                               dtype=self.dtype)

        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.hidden_layer_depth = hidden_layer_depth
        self.latent_length = latent_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs

        self.print_every = print_every
        self.clip = clip
        self.max_grad_norm = max_grad_norm
        self.is_fitted = False
        self.dload = dload

        if self.use_cuda:
            self.cuda()

        if optimizer == 'Adam':
            self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == 'SGD':
            self.optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        else:
            raise ValueError('Not a recognized optimizer')

        if loss == 'SmoothL1Loss':
            self.loss_fn = nn.SmoothL1Loss(size_average=False)
        elif loss == 'MSELoss':
            self.loss_fn = nn.MSELoss(size_average=False)

    def __repr__(self):
        return """VRAE(n_epochs={n_epochs},batch_size={batch_size},cuda={cuda})""".format(
                n_epochs=self.n_epochs,
                batch_size=self.batch_size,
                cuda=self.use_cuda)

    def forward(self, x):

        cell_output = self.encoder(x)
        latent = self.lmbd(cell_output)
        x_decoded = self.decoder(latent)

        return x_decoded, latent

    def _rec(self, x_decoded, x, loss_fn):
        latent_mean, latent_logvar = self.lmbd.latent_mean, self.lmbd.latent_logvar

        kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
        recon_loss = loss_fn(x_decoded, x)

        return kl_loss + recon_loss, recon_loss, kl_loss

    def compute_loss(self, X):
        x = Variable(X[:,:,:].type(self.dtype), requires_grad = True)

        x_decoded, _ = self(x)
        loss, recon_loss, kl_loss = self._rec(x_decoded, x.detach(), self.loss_fn)

        return loss, recon_loss, kl_loss, x


    def _train(self, train_loader):
        self.train()

        epoch_loss = 0
        t = 0

        for t, X in enumerate(train_loader):
            X = X.permute(1,0,2)

            self.optimizer.zero_grad()
            loss, recon_loss, kl_loss, _ = self.compute_loss(X)
            loss.backward()

            if self.clip:
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm = self.max_grad_norm)

            # accumulator
            epoch_loss += loss.item()

            self.optimizer.step()

            if (t + 1) % self.print_every == 0:
                print('Batch %d, loss = %.4f, recon_loss = %.4f, kl_loss = %.4f' % (t + 1, loss.item(),
                                                                                    recon_loss.item(), kl_loss.item()))

        print('Average loss: {:.4f}'.format(epoch_loss / t))

    def decode(self , x):
        return self.decoder(x)
