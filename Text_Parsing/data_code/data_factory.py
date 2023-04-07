from torch.utils.data import Dataset
from Text_Parsing.data_code import data_utilities as du
from Text_Parsing.data_code import data_builder as db
from Text_Parsing.model import model_hyperparameter_factory as mhf


class AbstractDatasetFactory:
    """
    This class builds Dataset objects for use in Neural Network Model construction
    """
    def __init__(self):
        self.data_utils = du.DataUtilities()

    def get_datasets(self,
                     data_builder: db.DataBuilder,
                     hparams: mhf.HyperParams) -> tuple:
        """

        :param data_builder: (db.DataBuilder)
        :param hparams: (mhf.HyperParams)
        :return: At a minimum a tuple with x_train, x_valid, y_train, y_valid AND possibly x_test & y_test. Additional
        data structures as required.
        """
        pass


class LanguageDatasetFactory(AbstractDatasetFactory):
    """
    This class builds Dataset objects for use in Neural Network Model construction
    """
    def __init__(self):
        super().__init__()

    def get_datasets(self,
                     data_builder: db.DataBuilder,
                     hparams: mhf.HyperParams) -> tuple:
        x_train, x_valid, x_test, y_train, y_valid, y_test = data_builder.load_and_build_data()
        vocab = self.data_utils.build_vocab(x_train, hparams=hparams)
        vocab_size = len(vocab)
        print(f'Length of vocabulary is {vocab_size}')

        return x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, vocab_size

    def get_yelp_datasets(self,
                                     hparams: mhf.LanguageHyperParams) -> tuple:
        data_builder = db.YelpDataBuilder()
        x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, vocab_size = self.get_datasets(data_builder, hparams)

        train_data = YelpDataset(x_train, y_train, vocab, hparams.MAX_LENGTH)
        valid_data = YelpDataset(x_valid, y_valid, vocab, hparams.MAX_LENGTH)
        test_data = YelpDataset(x_test, y_test, vocab, hparams.MAX_LENGTH)

        return train_data, valid_data, test_data, vocab_size

    def get_imdb_datasets(self,
                          hparams: mhf.LanguageHyperParams) -> tuple:
        data_builder = db.IMDBDataBuilder()
        x_train, x_valid, x_test, y_train, y_valid, y_test, vocab, vocab_size = self.get_datasets(data_builder, hparams)

        train_data = IMDB(x_train, y_train, vocab, hparams.MAX_LENGTH)
        valid_data = IMDB(x_valid, y_valid, vocab, hparams.MAX_LENGTH)
        test_data = IMDB(x_test, y_test, vocab, hparams.MAX_LENGTH)

        return train_data, valid_data, test_data, vocab_size


class AbstractDataset(Dataset):
    def __init__(self,
                 x: list,
                 y: list,
                 vocab: dict,
                 max_length: int = 256):
        """
        :param x: (list) list of yelp comments
        :param y: (list) list of health code ratings
        :param vocab: (dict) vocabulary dictionary {word:index}.
        :param max_length: (int) the maximum sequence length.
        """
        self.x = x
        self.y = y
        self.vocab = vocab
        self.max_length = max_length
        self.data_utilities = du.DataUtilities()

    @staticmethod
    def get_final_label(review_sentiment):
        pass

    def __getitem__(self, idx: int):
        """
        Return the tokenized review and label by the given index.
        :param idx: (int) index of the sample.
        :return: a dictionary containing three keys: 'ids', 'length', 'label' which represent the list of token ids,
        the length of the sequence, the binary label.
        """
        review_string = self.x[idx]
        review_sentiment = self.y[idx]
        review_tokens = self.data_utilities.tokenize(self.vocab, review_string)

        final_tokens = []
        if len(review_tokens) > self.max_length:
            final_tokens = review_tokens[0:self.max_length]
        else:
            final_tokens = review_tokens

        final_length = len(final_tokens)

        final_label = self.get_final_label(review_sentiment=review_sentiment)

        rtr_dict = {'ids': final_tokens,     # indexes of the review's words in the vocabulary dictionary
                    'length': final_length,  # total number of words in the review (0 up to max_length)
                    'label': final_label}    # label of the review (positive or negative)

        return rtr_dict

    def __len__(self) -> int:
        return len(self.x)


class YelpHCDataset(AbstractDataset):
    def __init__(self, x, y, vocab, max_length=256):
        """
        :param x: list of yelp comments
        :param y: list of health code ratings
        :param vocab: vocabulary dictionary {word:index}.
        :param max_length: the maximum sequence length.
        """
        super().__init__(x, y, vocab, max_length)

    @staticmethod
    def get_final_label(review_sentiment):
        final_label = ...
        return final_label


class IMDB(AbstractDataset):
    def __init__(self, x, y, vocab, max_length=256):
        """
        :param x: list of reviews
        :param y: list of labels
        :param vocab: vocabulary dictionary {word:index}.
        :param max_length: the maximum sequence length.
        """
        super().__init__(x, y, vocab, max_length)

    @staticmethod
    def get_final_label(review_sentiment):
        final_label = 0
        if review_sentiment == "positive":
            final_label = 1
        return final_label


class YelpDataset(AbstractDataset):
    def __init__(self, x, y, vocab, max_length=256):
        """
        :param x: list of reviews
        :param y: list of labels
        :param vocab: vocabulary dictionary {word:index}.
        :param max_length: the maximum sequence length.
        """
        super().__init__(x, y, vocab, max_length)

    @staticmethod
    def get_final_label(review_sentiment):
        final_label = 0
        if review_sentiment == "0":
            final_label = 0
        elif review_sentiment == "1":
            final_label = 1
        elif review_sentiment == "2":
            final_label = 2
        elif review_sentiment == "3":
            final_label = 3
        elif review_sentiment == "4":
            final_label = 4

        elif review_sentiment == 0:
            final_label = 0
        elif review_sentiment == 1:
            final_label = 1
        elif review_sentiment == 2:
            final_label = 2
        elif review_sentiment == 3:
            final_label = 3
        elif review_sentiment == 4:
            final_label = 4

        return final_label
