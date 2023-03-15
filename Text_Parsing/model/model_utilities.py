import torch
import torch.nn as nn


class ModelUtilities:
    def __init__(self):
        pass

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Embedding):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
            for name, param in m.named_parameters():
                if 'bias' in name:
                    nn.init.zeros_(param)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @staticmethod
    def get_training_device(try_to_use_gpu: bool = True, gpu_number: int = 1):
        if try_to_use_gpu:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # TODO modify to accept something like cuda:7, where this specifies use cuda device 7
        else:
            device = torch.device('cpu')

        return device

    @staticmethod
    def get_optimizer(hparams, model):
        # Optimization. Lab 2 (a)(b) should choose one of them.
        # DO NOT TOUCH optimizer-specific hyperparameters! (e.g., eps, momentum)
        # DO NOT change optimizer implementations!
        if hparams.OPTIM == "sgd":
            optimizer = optim.SGD(
                model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, momentum=.9)
        elif hparams.OPTIM == "adagrad":
            optimizer = optim.Adagrad(
                model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
        elif hparams.OPTIM == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
        elif hparams.OPTIM == "rmsprop":
            optimizer = optim.RMSprop(
                model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6, momentum=.9)
        else:
            raise NotImplementedError("Optimizer not implemented!")

        return optimizer

    @staticmethod
    def get_criterion(device):
        criterion = nn.CrossEntropyLoss()
        criterion = criterion.to(device)
        return criterion