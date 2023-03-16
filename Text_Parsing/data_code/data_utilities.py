import torch
import torch.nn as nn
from nltk.corpus import stopwords


class DataUtilities:

    def __init__(self):
        pass

    @staticmethod
    def tokenize(vocab: dict, example: str) -> list:
        """
        Tokenize the give example string into a list of token indices.
        :param vocab: dict, the vocabulary.
        :param example: a string of text.
        :return: a list of token indices.
        """
        # Your code here.
        example_index_list = []
        example_words = example.split()
        example_words_processed = []
        for word in example_words:
            wd = word  # nltk.stem.PorterStemmer().stem(word=word, to_lowercase=True)
            example_words_processed.append(wd)

        for word in example_words_processed:
            if word in vocab.keys():
                example_index_list.append(vocab[word])
            else:
                # if the word is not part of the vocab dict, treat it as an unknown
                example_index_list.append(1)

        return example_index_list

    @staticmethod
    def build_vocab(x_train: list,
                    min_freq: int = 5,
                    hparams=None) -> dict:
        """
        build a vocabulary based on the training corpus.
        :param hparams:
        :param x_train:  List. The training corpus. Each sample in the list is a string of text.
        :param min_freq: Int. The frequency threshold for selecting words.
        :return: dictionary {word:index}
        """
        # Add your code here. Your code should assign corpus with a list of words.

        # 1. compute word freq in training corpus
        word_dict = {}
        for obs in x_train:
            word_list = obs.split()
            for word in word_list:
                wd = word  # nltk.stem.PorterStemmer().stem(word=word, to_lowercase=True) #stem the words and convert to lower case to increase match rate, and
                if wd in word_dict:
                    word_dict[wd] = word_dict[wd] + 1
                else:
                    word_dict[wd] = 1

        # 2 remove stop words
        STP_WORDS_SET = set(stopwords.words('english'))
        for word in STP_WORDS_SET:
            if word in word_dict.keys():
                del word_dict[word]

        # 3 filter words by freq (remove words with a freq < min freq)
        word_dict_keys = list(word_dict.keys())
        for word in word_dict_keys:
            if word_dict[word] < min_freq:
                del word_dict[word]

        # 4 generate a corpus variable that contains a list of words
        vocab = {}
        index = 2
        for word in word_dict:
            vocab[word] = index
            index = index + 1
        vocab[hparams.PAD_TOKEN] = hparams.PAD_INDEX
        vocab[hparams.UNK_TOKEN] = hparams.UNK_INDEX

        corpus = []
        # sorting on the basis of most common words
        # corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
        # corpus_ = [word for word, freq in corpus.items() if freq >= min_freq]
        # # creating a dict
        # vocab = {w:i+2 for i, w in enumerate(corpus_)}
        # vocab[hparams.PAD_TOKEN] = hparams.PAD_INDEX
        # vocab[hparams.UNK_TOKEN] = hparams.UNK_INDEX
        return vocab

    @staticmethod
    def collate(batch, pad_index):
        batch_ids = [torch.LongTensor(i['ids']) for i in batch]
        batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
        batch_length = torch.Tensor([i['length'] for i in batch])
        batch_label = torch.LongTensor([i['label'] for i in batch])
        batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
        return batch
