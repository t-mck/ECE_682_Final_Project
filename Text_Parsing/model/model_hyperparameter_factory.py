from nltk.corpus import stopwords


class HyperParameterFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_language_hyperparams(batch_size: int = 96):
        return LanguageHyperParams(batch_size=batch_size)


class HyperParams:
    def __init__(self,
                 batch_size: int,
                 dropout_rate: float,
                 lr: float,
                 n_epochs: int,
                 optim: str
                 ):
        self.BATCH_SIZE = batch_size
        self.DROPOUT_RATE = dropout_rate
        self.LR = lr
        self.N_EPOCHS = n_epochs
        self.OPTIM = optim


class LanguageHyperParams(HyperParams):
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
                 n_layers: int = 1,
                 dropout_rate: float = 0.5,
                 lr: float = 0.00001,
                 n_epochs: int = 5,
                 wd: int = 0,
                 optim: str = "rmsprop",
                 bidirectional: bool = True,
                 seed: int = 2
                 ):
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