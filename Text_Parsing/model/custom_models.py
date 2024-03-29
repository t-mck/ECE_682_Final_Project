from Text_Parsing.model import model_utilities as mu

import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            hidden_dim: int,
            output_dim: int,
            n_layers: int,
            dropout_rate: float,
            pad_index: int,
            bidirectional: bool,
            **kwargs):
        """
        Create a LSTM model for classification.
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of embeddings
        :param hidden_dim: dimension of hidden features
        :param output_dim: dimension of the output layer which equals to the number of labels.
        :param n_layers: number of layers.
        :param dropout_rate: dropout rate.
        :param pad_index: index of the padding token.we
        """
        super().__init__()
        self.model_utils = mu.ModelUtilities()
        # Add your code here. Initializing each layer by the given arguments.

        # embedding layer
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
                                        embedding_dim=embedding_dim)
        # LSTM cell, dropout layer
        self.lstm_layer = nn.LSTM(input_size=embedding_dim,
                                  hidden_size=hidden_dim,
                                  num_layers=n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True,
                                  dropout=dropout_rate)  # dropout may need to be applied here dropout=dropout_rate

        self.dropout_layer = nn.Dropout(p=dropout_rate)
        # linear layer
        fc_input_size = hidden_dim
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=output_dim)

        # Weight initialization.
        if "weight_init_fn" not in kwargs:
            self.apply(self.model_utils.init_weights)
        else:
            self.apply(kwargs["weight_init_fn"])

    def forward(self, ids: torch.Tensor, length: torch.Tensor):
        """
        Feed the given token ids to the model.
        :param ids: [batch size, seq len] batch of token ids.
        :param length: [batch size] batch of length of the token ids.
        :return: prediction of size [batch size, output dim].
        """

        embeds_b = self.embed_layer(ids)

        drop_out_b = self.dropout_layer(embeds_b)
        lstm_out_b, (ht_b, ct_b) = self.lstm_layer(
            input=torch.nn.utils.rnn.pack_padded_sequence(input=drop_out_b,
                                                          lengths=length,
                                                          batch_first=True,
                                                          enforce_sorted=False))
        drop_out_b2 = self.dropout_layer(ht_b[-1])
        prediction_b = self.fc1(drop_out_b2)

        return prediction_b
