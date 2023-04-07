#
# Developed by:
# Taylor McKechnie
#
# March, 2023
#

from model import nn_model_setup as nms
import torch
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
                                                     output_dim=5,
                                                     n_epochs=5,
                                                     #stop_words=set(stopwords.words('english')),
                                                     max_length=256,
                                                     embedding_dim=192,
                                                     hidden_dim=200,
                                                     n_layers=5,
                                                     dropout_rate=0.85,  # 0.65 is best so far, fin acc 0.8868
                                                     lr=0.00001,
                                                     wd=0,
                                                     optim="adagrad",
                                                     bidirectional=True,
                                                     seed=2)


    # 1. Load old data
    train_data, valid_data, test_data, vocab_size = data_factory.get_yelp_datasets(hparams)

    # 2. Build Deep Neural Network (DNN) model.
    # Note: this can take a very long time (minutes/days).
    model = model_factory.build_model(train_data=train_data,
                                      valid_data=valid_data,
                                      test_data=test_data,
                                      vocab_size=vocab_size,
                                      hyperparameters=hparams,
                                      base_model_type='LSTM',
                                      pretrained=True,
                                      try_to_use_gpu=True,
                                      gpu_number=1)

    # 3. Load new data_code, which we want predictions for. TODO: load new data_code
    # new_data = ...  # use test_data here?

    # 4. Use DNN model to generate predictions, and then save those predictions
    # preds = prediction_factory.get_and_save_predictions(model=model,
    #                                                     new_data=new_data,
    #                                                     pred_file_name='file_name.csv')

    # Free memory for later usage.
    del model
    torch.cuda.empty_cache()

    # 5. Explore predictions with summary statistics
    # Maybe pair preds with new_data, to explore how different predictors affected pred classification?
    ...


if __name__ == '__main__':
    main()
