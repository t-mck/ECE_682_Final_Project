#
# Developed by:
# Taylor McKechnie
#
# March, 2023
#

from model import nn_model_setup as nms
import torch
from data_code import data_merge_preds_with_old_data as dmp
from nltk.corpus import stopwords


def main():
    """
    This is the main entry-point.

    :return:
    """

    # 0.A Neural Network specific set-up
    # Random seed will affect which observations end up in the training set, validation set, and test set,
    # and how the model trains (here how dropout is applied during training)
    model_hyperparams, model_factory, data_factory, prediction_factory = nms.NNSetup().get_language_model_setup(random_seed=2)

    # 0.B Get Hyperparameters
    # NOTE: Hyperparameters are important, and will affect the training & performance of the model.
    # -For the most part you should not adjust the default settings found in HyperParameterFactory
    # -If you receive a CUDA out of Memory error while training, reduce batch_size (this is unlikely when working with
    # text data)
    hparams = model_hyperparams.get_yelp_hyperparams(batch_size=48,
                                                     output_dim=3,
                                                     n_epochs=5,
                                                     #stop_words=set(stopwords.words('english')),
                                                     max_length=512,
                                                     embedding_dim=48,
                                                     hidden_dim=150,
                                                     n_layers=2,
                                                     dropout_rate=0.75,  # 0.65 is best so far, fin acc 0.8868
                                                     lr=0.00001,
                                                     wd=0,
                                                     optim="rmsprop",
                                                     bidirectional=True,
                                                     seed=2)

    # 1. Load old data
    train_data, valid_data, test_data, vocab_size, vocab = data_factory.get_yelp_datasets(hparams)

    # # 2. Build Deep Neural Network (DNN) model.
    # # Note: this can take a very long time (minutes/days).
    # model = model_factory.build_model(train_data=train_data,
    #                                   valid_data=valid_data,
    #                                   test_data=test_data,
    #                                   vocab_size=vocab_size,
    #                                   hyperparameters=hparams,
    #                                   base_model_type='GRU',
    #                                   pretrained=True,
    #                                   try_to_use_gpu=True,
    #                                   gpu_number=1)

    # 3. Load new data_code, which we want predictions for. TODO: load new data_code
    eval_data_path = 'yelp_no_nashville_reviews.csv'
    eval_data = data_factory.get_yelp_no_nashville_eval_datasets(hparams,
                                                                 data_file=eval_data_path,
                                                                 vocab=vocab)

    # 4. Use DNN model to generate predictions, and then save those predictions
    preds = model_factory.get_and_save_predictions(eval_data=eval_data,
                                                   vocab_size=vocab_size,
                                                   hyperparameters=hparams,
                                                   base_model_type="GRU",
                                                   eval_model_path='./saved_model/model.pth')

    # 5. Pair Preds with old data
    dm = dmp.YelpDataMerger()
    dm.merge_preds_and_data(preds=preds,
                            data_file='yelp_no_nashville_reviews.csv',
                            merge_file_name='data_code/merged_preds_with_yelp_no_nashville_reviews.csv')

    # Free memory for later usage.
    # del model
    # torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
