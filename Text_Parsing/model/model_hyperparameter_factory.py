from nltk.corpus import stopwords


class HyperParameterFactory:
    """
    This class encapsulates the optimized NN hyperparameter settings for different model types.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_lstm_imdb_hyperparams(batch_size: int = 96):
        return LSTMHyperParams().get_imdb(batch_size=batch_size)


class HyperParams:
    def __init__(self,
                 batch_size: int,  # 96-256 are standard values. Reduce if you run out of CUDA memory
                 dropout_rate: float,  # when optimizing start at 0.5, then bracket it. 0.65 seems to be a good value
                 lr: float,  # 0.01 is the highest you should go. 0.001-0.00001 are good starting values
                 n_epochs: int,  # highly dependent on underlying algorithm. 5 (NLP) - 200 (vision)
                 optim: str  # varies widely, SGD is a good starting place in general
                 ):
        """
        Encapsulates hyper parameters common to all NN models

        :param batch_size: (int)
        :param dropout_rate: (float)
        :param lr: (float)
        :param n_epochs: (int)
        :param optim: (str)
        """
        self.BATCH_SIZE = batch_size
        self.DROPOUT_RATE = dropout_rate
        self.LR = lr
        self.N_EPOCHS = n_epochs
        self.OPTIM = optim


class LSTMHyperParams(HyperParams):
    # Constance hyperparameters. They have been tested and don't need to be tuned.
    def __init__(self,
                 pad_index: int = 0,
                 unk_index: int = 1,
                 pad_token: str = '<pad>',
                 unk_token: str = '<unk>',
                 stop_words=set(stopwords.words('english')),
                 max_length: int = 256,
                 batch_size: int = 96,
                 embedding_dim: int = 48,
                 hidden_dim: int = 200,
                 output_dim: int = 2,
                 n_layers: int = 2,
                 dropout_rate: float = 0.65,  # 0.65 is best so far, fin acc 0.8868
                 lr: float = 0.00001,
                 n_epochs: int = 5,
                 wd: int = 0,
                 optim: str = "rmsprop",
                 bidirectional: bool = True,
                 seed: int = 2
                 ):
        """
        Hyperparameter settings for training a LSTM model. These settings are just a starting point, and were chosen
        using the IMDB data set included with this package, and should be optimized if a different data set is used.

        Specifically, focus on:
        -embedding_dim, try increments of 10
        -hidden_dim , try increments of 50
        -n_layers, try increments of 1
        -dropout_rate, start at 0.5, try increments of 0.1 and 0.05
        -lr, start at 0.01, and reduce by factors of 10
        -bidirectional, start with False,
        -optim, sgd is a good starting place, but rmsprop tends to work well

        The following are set based on the data set, and should not be optimized (they only have one setting):
        -output_dim, pad_index, unk_index, pad_token, unk_token.

        Other possible optimizations (less likely to be worth your time):
        -stop_words
        -max_len
        -batch_size (you should really only care about this if you are running out of GPU memory when training)

        Seed should NOT be optimized! It just sets the random seed, and ensures you get the same results each time if
        your other settings have not changed.

        :param pad_index: (int)
        :param unk_index: (int)
        :param pad_token: (str)
        :param unk_token: (str)
        :param stop_words:
        :param max_length: (int)
        :param batch_size: (int)
        :param embedding_dim: (int)
        :param hidden_dim: (int)
        :param output_dim: (int)
        :param n_layers: (int)
        :param dropout_rate: (float)
        :param lr: (float)
        :param n_epochs: (int)
        :param wd: (int)
        :param optim: (str)
        :param bidirectional: (bool)
        :param seed: (int)
        """
        super().__init__(batch_size=batch_size,
                         dropout_rate=dropout_rate,
                         lr=lr,
                         n_epochs=n_epochs,
                         optim=optim)
        self.PAD_INDEX = pad_index
        self.UNK_INDEX = unk_index
        self.PAD_TOKEN = pad_token
        self.UNK_TOKEN = unk_token
        self.STOP_WORDS = stop_words
        self.MAX_LENGTH = max_length
        self.EMBEDDING_DIM = embedding_dim
        self.HIDDEN_DIM = hidden_dim
        self.OUTPUT_DIM = output_dim
        self.N_LAYERS = n_layers
        self.WD = wd
        self.BIDIRECTIONAL = bidirectional
        self.SEED = seed

    def get_imdb(self, batch_size: int = 96):
        self.BATCH_SIZE = batch_size
        self.DROPOUT_RATE = 0.65
        self.LR = 0.00001
        self.N_EPOCHS = 5
        self.OPTIM = "rmsprop"
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.STOP_WORDS = set(stopwords.words('english'))
        self.MAX_LENGTH = 256
        self.EMBEDDING_DIM = 48
        self.HIDDEN_DIM = 200
        self.OUTPUT_DIM = 2
        self.N_LAYERS = 2
        self.WD = 0
        self.BIDIRECTIONAL = True
        self.SEED = 2
