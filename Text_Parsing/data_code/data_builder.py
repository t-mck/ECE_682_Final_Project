import pandas as pd
import os

class DataBuilder:
    """
    The point of this class is to take an original set of data a produce three distinct subsets of data from it:
    1. a training set
    2. a validation set
    3. a test set

    Each set is required for construction of a Neural Network. On occasion the test set can be omitted by setting
    test_size = 0.0
    """
    def __init__(self):
        pass

    def load_and_build_data(self,
                            data_file: str,
                            training_size: float = 0.75,
                            validation_size: float = 0.25,
                            test_size: float = 0.0) -> tuple:
        pass


class IMDBDataBuilder(DataBuilder):
    """
    This class builds three data sets (test, validation, and training) from the IMDB sentiment review data set.
    """

    def __init__(self):

        super().__init__()

    def load_and_build_data(self,
                            data_file: str = 'IMDBDataset.csv',
                            training_size: float = 0.7,
                            validation_size: float = 0.1,
                            test_size: float = 0.2) -> tuple:
        """
        Load the IMDB dataset
        :param data_file: the path of the dataset file.
        :param test_size:
        :param validation_size:
        :param training_size:
        :return: train, validation and test set.
        """
        dfx = pd.read_csv(os.getcwd() + '/data/' + data_file)
        dfx_np = dfx.to_numpy()
        num_obs = len(dfx_np)

        train_index_start = 0
        train_index_end = int(num_obs * training_size)
        np_train = dfx_np[train_index_start:train_index_end, :]
        # check that we have half neg, half positive
        pd_train = pd.DataFrame(np_train, columns=['review', 'sentiment'])
        pd_train.describe()

        val_index_start = int(num_obs * training_size)
        val_index_end = int(num_obs * (training_size+validation_size))
        np_val = dfx_np[val_index_start:val_index_end, :]
        # check that we have half neg, half positive
        pd_val = pd.DataFrame(np_val, columns=['review', 'sentiment'])
        pd_val.describe()

        # Check that test is as expected. Two check are necessary because we are comparing floating point numbers,
        # which are not stored exactly.
        assert ((1 - (training_size + validation_size)) < (test_size+0.001))    # 1 - 0.8 = 0.2 < 0.201
        assert ((1 - (training_size + validation_size)) > (test_size - 0.001))  # 1 - 0.8 = 0.2 > 0.199

        test_index_start = int(num_obs * (training_size+validation_size))
        test_index_end = num_obs
        np_test = dfx_np[test_index_start:test_index_end, :]
        # check that we have half neg, half positive
        pd_test = pd.DataFrame(np_test, columns=['review', 'sentiment'])
        pd_test.describe()

        x_train_outter = np_train[:, 0:1].tolist()
        y_train_outter = np_train[:, 1:2].tolist()

        x_train = []
        for i in x_train_outter:
            x_train.append(i[0])

        y_train = []
        for i in y_train_outter:
            y_train.append(i[0])

        x_valid_outter = np_val[:, 0:1].tolist()
        y_valid_outter = np_val[:, 1:2].tolist()

        x_valid = []
        for i in x_valid_outter:
            x_valid.append(i[0])

        y_valid = []
        for i in y_valid_outter:
            y_valid.append(i[0])

        x_test_outter = np_test[:, 0:1].tolist()
        y_test_outter = np_test[:, 1:2].tolist()

        x_test = []
        for i in x_test_outter:
            x_test.append(i[0])

        y_test = []
        for i in y_test_outter:
            y_test.append(i[0])

        return x_train, x_valid, x_test, y_train, y_valid, y_test
