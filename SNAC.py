"""
Hello World !

This is SNAC - Spiking Neral Autoencoders for Compression,
or in other words, a library for spiking neural network compression
on microcontrollers to reduce wireless data comunication and power
consumption on stuff like IoT nodes.

by Neuronova, bringing intelligence to the Edge!

Version: 1.0.0-pre-alpha
Authors (Professional Edgers): Marco Rasetto, Michele Mastella, Alessandro Milozzi

"""

import torch, os
import torch.nn as nn
import snntorch as snn
import numpy as np


class time_series_net(nn.Module):
    """
    Pytorch network class for the autoencoder
    """

    def __init__(self, net_parameters):
        super().__init__()

        n_inputs_en = net_parameters["n_inputs_en"]
        n_hidden_en = net_parameters["n_hidden_en"]

        # Initialize layers
        # Encoder
        self.lin_1_en = nn.Linear(n_inputs_en, n_hidden_en, bias=False)
        self.lin_1_en.weight.data = torch.ones(self.lin_1_en.weight.size())
        self.lif_1_en = snn.Leaky(
            beta=hid_beta_en,
            threshold=hid_thresh_en,
            spike_grad=spike_grad,
            reset_mechanism="zero",
        )
        self.lin_2_en = nn.Linear(n_hidden_en, n_latent_en, bias=False)
        self.lin_2_en.weight.data = self.lin_2_en.weight.data + 0.1
        self.lif_2_en = snn.Leaky(
            beta=latent_beta_en,
            threshold=latent_thresh,
            spike_grad=spike_grad,
            reset_mechanism="zero",
        )

        # Decoder
        # self.lin_1_de = nn.Linear(n_latent_en, n_hidden_1_de, bias=False)
        # self.lif_1_de = snn.Leaky(
        #     beta=hid_beta_en,
        #     threshold=100000,
        #     spike_grad=spike_grad,
        #     reset_mechanism="zero",
        # )
        self.rnn_1_de = nn.Linear(n_latent_en + n_hidden_1_de, n_hidden_1_de)
        self.sig_1_de = nn.LeakyReLU()
        self.rnn_2_de = nn.Linear(n_hidden_1_de + n_hidden_2_de, n_hidden_2_de)
        self.sig_2_de = nn.LeakyReLU()
        self.lin_output = nn.Linear(n_hidden_2_de, n_outputs_de)
        self.relu_output = nn.LeakyReLU()


class time_series_compressor:
    def __init__(self) -> None:
        self.parameters = {
            # Encoder parameters
            "n_hidden_en": 30,
            "hid_freq": None,
            "hid_thresh_en": None,
            "n_latent_en": 4,
            "latent_beta_en": 0.99,
            "latent_thresh": 0.5,
            # Decoder parameters
            "n_hidden_1_de": 30,
            "n_hidden_2_de": 30,
            "n_outputs_de": 1,
        }


def bitMSE(MSE, res=32, min=0, max=2):
    """
    This function is used to calculate the reconstruction error in
    bits. It is bounded between [res (maximum error) 0(perfect reconstruction)]

    Parameters:

        res : (int) number of bits used to encode our signal.
        min : (float) minimum value of our encoded signal
        max : (float) maximum value of our encoded signal

    Return :

        bit_error : (int) number of bits needed to represent our error

    """
    abs_error = np.sqrt(MSE)

    # Min prec is the minimum number of bits needed to represent
    # this error
    min_prec = int(np.floor(np.log2((max - min) / abs_error)))

    if min_prec <= res:
        bit_error = res - min_prec
    else:
        bit_error = 0

    return bit_error


def freq_to_beta(sampl_freq, freqs):
    """
    This function takes a list of frequencies in Hz for the hidden layer
    of our autoencoder and calculate neuron betas.

    Ref: https://www.frontiersin.org/articles/10.3389/fncom.2010.00149/full

    Parameters:

        sampl_freq : (float) sampling frequency in Hz used to sample a time series.
        freqs: (list) of frequences to transform to beta.

    Return:

        betas: (list) of betas for the encoder hidden layer.

    """

    taus = 1 / (np.pi * 2 * freqs)
    sample_period = 1 / sampl_freq
    betas = np.exp(-sample_period / taus)

    return betas
