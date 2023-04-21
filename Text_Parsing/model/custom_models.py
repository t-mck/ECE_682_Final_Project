from Text_Parsing.model import model_utilities as mu

import torch.nn as nn
import torch
from transformers import BertModel

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


class GRU(nn.Module):
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
        self.embed_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim,)
        # GRU cell, dropout layer
        self.gru_layer = nn.GRU(input_size=embedding_dim,
                                hidden_size=hidden_dim,
                                num_layers=n_layers,
                                bidirectional=bidirectional,
                                batch_first=True,
                                dropout=dropout_rate) # dropout may need to be applied here dropout=dropout_rate
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        # linear layer
        fc_input_size = hidden_dim
        # if bidirectional:
        #     fc_input_size = hidden_dim * 2
        #self.fc1 = nn.Linear(in_features=fc_input_size*256, out_features=output_dim)
        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=output_dim)

        # Weight initialization. DO NOT CHANGE!
        if "weight_init_fn" not in kwargs:
            self.apply(self.model_utils.init_weights)
        else:
            self.apply(kwargs["weight_init_fn"])


    def forward(self, ids:torch.Tensor, length:torch.Tensor):
        """
        Feed the given token ids to the model.
        :param ids: [batch size, seq len] batch of token ids.
        :param length: [batch size] batch of length of the token ids.
        :return: prediction of size [batch size, output dim].
        """
        # Add your code here.
        embeds_b = self.embed_layer(ids)

        # gru_out_b, ht_b= self.gru_layer(input=embeds_b) #h_0 and c_0 not provided, so the initial hidden state and cell state default to zero.
        # # functorch.vmap vectorizes a supplied function for faster computation.
        # dropout_out_b = functorch.vmap(self.dropout_layer)(gru_out_b)
        # flatten_out_b = functorch.vmap(torch.flatten)(dropout_out_b)
        # prediction_b = functorch.vmap(self.fc1)(flatten_out_b)

        gru_out_b, ht_b= self.gru_layer(input=torch.nn.utils.rnn.pack_padded_sequence(input=embeds_b, lengths=length, batch_first=True, enforce_sorted=False))
        dropout_out_b = self.dropout_layer(ht_b[-1])
        prediction_b = self.fc1(dropout_out_b)

        return prediction_b


class CustomPretrainedBERT(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embedding_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 n_layers: int,
                 dropout_rate: float,
                 pad_index: int,
                 bidirectional: bool,
                 **kwargs):
        #super().__init__()
        super(CustomPretrainedBERT, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")

        # # embedding layer
        # self.embed_layer = nn.Embedding(num_embeddings=vocab_size,
        #                                 embedding_dim=embedding_dim)

        self.lstm_layer = nn.LSTM(input_size=768,
                                  hidden_size=256,
                                  num_layers=n_layers,
                                  bidirectional=bidirectional,
                                  batch_first=True,
                                  dropout=dropout_rate)  # dropout may need to be applied here dropout=dropout_rate
        self.dropout_layer = nn.Dropout(p=dropout_rate)
        # linear layer
        fc_input_size = 256*2

        self.fc1 = nn.Linear(in_features=fc_input_size, out_features=output_dim)

    def forward(self, ids: torch.Tensor, length: torch.Tensor):
        """
        Feed the given token ids to the model.
        :param ids: [batch size, seq len] batch of token ids.
        :param length: [batch size] batch of length of the token ids.
        :return: prediction of size [batch size, output dim].
        """

        sequence_output, pooled_output = self.bert(ids) #,
                                                   #attention_mask=mask)

        # sequence_output has the following shape: (batch_size, sequence_length, 768)
        lstm_output, (h, c) = self.lstm_layer(sequence_output)  ## extract the 1st token's embeddings
        hidden = torch.cat((lstm_output[:, -1, :256], lstm_output[:, 0, 256:]), dim=-1)
        prediction = self.fc1(hidden.view(-1,
                                                256 * 2))  ### assuming that you are only using the output of the last LSTM cell to perform classification

        return prediction


        # embeds_b = self.embed_layer(ids)
        #
        # drop_out_b = self.dropout_layer(embeds_b)
        # lstm_out_b, (ht_b, ct_b) = self.lstm_layer(
        #     input=torch.nn.utils.rnn.pack_padded_sequence(input=drop_out_b,
        #                                                   lengths=length,
        #                                                   batch_first=True,
        #                                                   enforce_sorted=False))
        # drop_out_b2 = self.dropout_layer(ht_b[-1])
        # prediction_b = self.fc1(drop_out_b2)
        #
        # return prediction_b