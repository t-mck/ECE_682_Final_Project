from Text_Parsing.data_code import data_loader as dl, data_augmenter as da
from Text_Parsing.model import model_trainer as mt
from Text_Parsing.model import model_plotter as mp
from Text_Parsing.model import model_utilities as mu
from Text_Parsing.model import custom_models as cm
from Text_Parsing.model import model_hyperparameter_factory as mhf

import torch.nn as nn
import torch

class AbstractModelFactory:
    def __init__(self):
        self.data_augmenter = da.DataAugmenter()
        self.data_loader = dl.DataLoader()
        self.model_trainer = mt.AbstractModelTrainer()
        self.model_plotter = mp.ModelPlotter()
        self.model_utils = mu.ModelUtilities()

    # def build_model(self,
    #                 train_data,
    #                 valid_data,
    #                 test_data,
    #                 hyperparameters: mhf.HyperParams,
    #                 base_model_type: str,
    #                 pretrained: bool = True,
    #                 try_to_use_gpu: bool = True,
    #                 gpu_number: int = 1,
    #                 return_training_summary: bool = False,
    #                 plot_training_summary: bool = True):
    #     pass


class LanguageModelFactory(AbstractModelFactory):
    def __init__(self):
        super().__init__()
        self.model_trainer = mt.LSTMTrainer()


    @staticmethod
    def get_lstm(hparams, vocab_size, **kwargs):
        model = cm.LSTM(vocab_size,
                         hparams.EMBEDDING_DIM,
                         hparams.HIDDEN_DIM,
                         hparams.OUTPUT_DIM,
                         hparams.N_LAYERS,
                         hparams.DROPOUT_RATE,
                         hparams.PAD_INDEX,
                         hparams.BIDIRECTIONAL,
                         **kwargs)
        return model

    @staticmethod
    def get_gru(hparams, vocab_size, **kwargs):
        model = cm.GRU(
                        vocab_size,
                        hparams.EMBEDDING_DIM,
                        hparams.HIDDEN_DIM,
                        hparams.OUTPUT_DIM,
                        hparams.N_LAYERS,
                        hparams.DROPOUT_RATE,
                        hparams.PAD_INDEX,
                        hparams.BIDIRECTIONAL,
                        **kwargs)
        return model

    def get_bert(self, pretrained: bool = True):
        sequence_classification_model = torch.hub.load('huggingface/pytorch-transformers',
                                                       'modelForSequenceClassification',
                                                       'bert-base-cased-finetuned-mrpc')
        sequence_classification_tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer',
                                                           'bert-base-cased-finetuned-mrpc')

        text_1 = "Jim Henson was a puppeteer"
        text_2 = "Who was Jim Henson ?"
        indexed_tokens = sequence_classification_tokenizer.encode(text_1, text_2, add_special_tokens=True)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        segments_tensors = torch.tensor([segments_ids])
        tokens_tensor = torch.tensor([indexed_tokens])

        # Predict the sequence classification logits
        with torch.no_grad():
            seq_classif_logits = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors)

        predicted_labels = torch.argmax(seq_classif_logits[0]).item()

        assert predicted_labels == 0  # In MRPC dataset this means the two sentences are not paraphrasing each other

        # Or get the sequence classification loss (set model to train mode before if used for training)
        labels = torch.tensor([1])
        seq_classif_loss = sequence_classification_model(tokens_tensor, token_type_ids=segments_tensors, labels=labels)

        # TODO: complete this function, want to add more data_code to this... like stack an LSTM on top of it?
        return model

    # def get_robert(self, pretrained: bool = True):
    #     #model = nn.robert(pretrained=pretrained)
    #     # TODO: complete this function, want to add more data_code to this... like stack an LSTM on top of it?
    #     return model
    #
    # def get_gpt(self, pretrained: bool = True):
    #     #model = nn.gpt(pretrained=pretrained)
    #     # TODO: complete this function, want to add more data_code to this... like stack an LSTM on top of it?
    #     return model

    def get_base_model_by_name(self, hparams, vocab_size, base_model_type: str, pretrained: bool = True):
        if base_model_type == 'BERT':
            return self.get_bert(pretrained=pretrained)
        elif base_model_type == 'LSTM':
            return self.get_lstm(hparams=hparams, vocab_size=vocab_size)
        elif base_model_type == 'GRU':
            return self.get_gru(hparams=hparams, vocab_size=vocab_size)
        elif base_model_type == 'GPT':
            return self.get_gpt()
        elif base_model_type == 'RoBERT':
            return self.get_robert()

    def build_model(self,
                    train_data,
                    valid_data,
                    test_data,
                    vocab_size,
                    hyperparameters: mhf.HyperParams,
                    base_model_type: str,
                    pretrained: bool = True,
                    try_to_use_gpu: bool = True,
                    gpu_number: int = 1,
                    return_training_summary: bool = False,
                    plot_training_summary: bool = True
                    ):

        # 0. Get training hardware (i.e. try to get a GPU if our computer has one)
        device = self.model_utils.get_training_device(try_to_use_gpu=try_to_use_gpu,
                                                      gpu_number=gpu_number)

        # 1. Get augmentations
        # remove stop words
        # stem words
        # etc.

        # 2. Get data_code loader
        train_dataloader, valid_dataloader, test_dataloader = self.data_loader.get_dataloaders(train_data=train_data,
                                                                                               valid_data=valid_data,
                                                                                               test_data=test_data,
                                                                                               hparams=hyperparameters)

        # 3. Get base model (i.e. get the 'backbone'), optimization function, and optimization criterion
        base_model = self.get_base_model_by_name(base_model_type=base_model_type,
                                                 pretrained=pretrained,
                                                 hparams=hyperparameters,
                                                 vocab_size=vocab_size)

        optimizer = self.model_utils.get_optimizer(hparams=hyperparameters, model=base_model)
        criterion = self.model_utils.get_criterion(device=device)

        # 4. Train model w/ our new data_code
        final_model, training_summary = self.model_trainer.train_model(hparams=hyperparameters,
                                                                       model=base_model,
                                                                       optimizer=optimizer,
                                                                       criterion=criterion,
                                                                       device=device,
                                                                       train_dataloader=train_dataloader,
                                                                       valid_dataloader=valid_dataloader,
                                                                       test_dataloader=test_dataloader
                                                                       )

        # 5. Model Training Plots
        if plot_training_summary:
            self.model_plotter.plot_training_summary(training_summary=training_summary,
                                                     hparams=hyperparameters,
                                                     model_type=base_model_type)

        # 6. return model, and possibly the training summary which contains the final accuracy
        if return_training_summary:
            return final_model, training_summary
        else:
            return final_model
