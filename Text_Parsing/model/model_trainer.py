import sys
import tqdm
import torch
import numpy as np
import os
import random
from Text_Parsing.data_code import data_utilities as du


class ConstantWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            num_warmup_steps: int,
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - self._step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            lr = self.base_lrs
        return lr


class AbstractModelTrainer:
    def __init__(self):
        self.data_utils = du.DataUtilities()

    @staticmethod
    def save_current_model(model_to_save, epoch: int, checkpoint_folder: str = './saved_model'):
        """
        This function saves the current iteration of the model after the specified training epoch

        :param checkpoint_folder:
        :param model_to_save:
        :param epoch:
        :return:
        """
        if not os.path.exists(checkpoint_folder):
            os.makedirs(checkpoint_folder)
        print("    Saving ...\n")
        state = {'state_dict': model_to_save.state_dict(),
                 'epoch': epoch}
        torch.save(state, os.path.join(checkpoint_folder, 'model.pth'))

    @staticmethod
    def get_saved_model(model, path_to_saved_model: str = './saved_model/model.pth'):
        checkpoint = torch.load(path_to_saved_model)
        model.load_state_dict(checkpoint['state_dict'])
        model.cuda()

    @staticmethod
    def get_training_summary_data_strcut():
        training_summary = {'Accuracy': {}, 'Loss': {}, 'Number_Parameters': None}
        training_summary['Accuracy']['Training'] = []
        training_summary['Accuracy']['Validation'] = []
        training_summary['Accuracy']['Test'] = None
        training_summary['Loss']['Training'] = []
        training_summary['Loss']['Validation'] = []
        training_summary['Loss']['Test'] = None
        # training_summary['Number_Parameters'] = None

        return training_summary

    @staticmethod
    def print_begin_training_messages():
        print('=' * 50)
        print(f'Begin Training:')
        print('-' * 50)
        print(os.getcwd())
        print('-' * 50)

    @staticmethod
    def print_epoch_header(epoch: int):
        print(f'  Epoch {epoch}:')

    @staticmethod
    def print_epoch_update_to_screen(accuracy: float, loss: float):
        print(f'    loss = {loss}, accuracy = {accuracy}\n')

    @staticmethod
    def print_training_complete():
        print(f'Training Complete!')
        print('=' * 50)

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train_for_one_epoch(self,
                            dataloader,
                            model,
                            criterion,
                            optimizer,
                            scheduler,
                            device,
                            training_summary: dict):
        pass

    def evaluate_for_one_epoch(self,
                               dataloader,
                               model,
                               criterion,
                               device,
                               training_summary: dict,
                               validation_set: bool = True):
        pass

    def train_model(self,
                    model,
                    optimizer,
                    criterion,
                    device: torch.device,
                    hparams,
                    train_dataloader,
                    valid_dataloader,
                    test_dataloader,
                    warmup_steps: int = 200,
                    checkpoint_folder: str = "./saved_model"  # the folder where the trained model is saved
                    ):
        pass


class LSTMTrainer(AbstractModelTrainer):
    def __init__(self):
        super().__init__()

    @staticmethod
    def get_accuracy(prediction, label):
        batch_size, _ = prediction.shape
        predicted_classes = prediction.argmax(dim=-1)
        correct_predictions = predicted_classes.eq(label).sum()
        accuracy = correct_predictions / batch_size
        return accuracy

    def train_for_one_epoch(self,
                            dataloader,
                            model,
                            criterion,
                            optimizer,
                            scheduler,
                            device,
                            training_summary: dict):
        model.train()
        epoch_losses = []
        epoch_accs = []

        for batch in tqdm.tqdm(dataloader, desc='    training...', file=sys.stdout):
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = self.get_accuracy(prediction, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())
            scheduler.step()                   # Learning rate is adjusted every batch

        train_loss = np.mean(epoch_losses)
        train_acc = np.mean(epoch_accs)
        self.print_epoch_update_to_screen(accuracy=train_acc, loss=train_loss)
        training_summary['Loss']['Training'].append(train_loss)
        training_summary['Accuracy']['Training'].append(train_acc)
        return train_loss, train_acc

    def evaluate_for_one_epoch(self,
                               dataloader,
                               model,
                               criterion,
                               device,
                               training_summary: dict,
                               validation_set: bool = True):

        if not validation_set:
            print(f'  Test:')

        model.eval()
        epoch_losses = []
        epoch_accs = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc='    evaluating...', file=sys.stdout):
                ids = batch['ids'].to(device)
                length = batch['length']
                label = batch['label'].to(device)
                prediction = model(ids, length)
                loss = criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                epoch_losses.append(loss.item())
                epoch_accs.append(accuracy.item())

        val_loss = np.mean(epoch_losses)
        val_acc = np.mean(epoch_accs)
        self.print_epoch_update_to_screen(accuracy=val_acc, loss=val_loss)
        if validation_set:
            training_summary['Loss']['Validation'].append(val_loss)
            training_summary['Accuracy']['Validation'].append(val_acc)
        else:
            training_summary['Loss']['Test'] = val_loss
            training_summary['Accuracy']['Test'] = val_acc

        return val_loss


    def predict_sentiment(self, text, model, vocab, device):
        tokens = self.data_utils.tokenize(vocab, text)
        ids = [vocab[t] if t in vocab else UNK_INDEX for t in tokens]
        length = torch.LongTensor([len(ids)])
        tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)

        prediction = model(tensor, length).squeeze(dim=0)
        probability = torch.softmax(prediction, dim=-1)
        predicted_class = prediction.argmax(dim=-1).item()
        predicted_probability = probability[predicted_class].item()
        return predicted_class, predicted_probability

    def train_model(self,
                    model,
                    optimizer,
                    criterion,
                    device: torch.device,
                    hparams,
                    train_dataloader,
                    valid_dataloader,
                    test_dataloader,
                    warmup_steps: int = 200,
                    checkpoint_folder: str = "./saved_model"  # the folder where the trained model is saved
                    ):

        # Start training
        self.print_begin_training_messages()
        # model = model.to(device)
        training_summary = self.get_training_summary_data_strcut()
        training_summary['Number_Parameters'] = self.count_parameters(model)

        # Warmup Scheduler. DO NOT TOUCH!
        lr_scheduler = ConstantWithWarmup(optimizer, warmup_steps)

        best_valid_loss = float('inf')
        for epoch in range(hparams.N_EPOCHS):
            self.print_epoch_header(epoch)

            # Train
            self.train_for_one_epoch(dataloader=train_dataloader,
                                     model=model,
                                     criterion=criterion,
                                     optimizer=optimizer,
                                     scheduler=lr_scheduler,
                                     device=device,
                                     training_summary=training_summary)

            # Evaluate
            epoch_valid_loss = self.evaluate_for_one_epoch(dataloader=valid_dataloader,
                                                           model=model,
                                                           criterion=criterion,
                                                           device=device,
                                                           training_summary=training_summary)

            # Save the model that achieves the smallest validation loss.
            if epoch_valid_loss < best_valid_loss:
                # Your code: save the best model somewhere (no need to submit it to Sakai)
                best_valid_loss = epoch_valid_loss
                self.save_current_model(model_to_save=model, epoch=epoch, checkpoint_folder=checkpoint_folder)

        self.print_training_complete()

        # Load the best model's weights.
        self.get_saved_model(model=model, path_to_saved_model='./saved_model/model.pth')

        # Evaluate test loss on testing dataset (NOT Validation)
        self.evaluate_for_one_epoch(test_dataloader, model, criterion, device, training_summary, validation_set=False)

        return model, training_summary

    def evaluate_and_save_preds(self,
                                dataloader,
                                model,
                                device,
                                criterion):

        print(f'  Eval:')

        model.eval()
        preds = []

        eval_losses = []
        eval_accs = []

        with torch.no_grad():
            for batch in tqdm.tqdm(dataloader, desc='    evaluating...', file=sys.stdout):
                ids = batch['ids'].to(device)
                length = batch['length']
                label = batch['label'].to(device)
                prediction = model(ids, length)
                loss = criterion(prediction, label)
                accuracy = self.get_accuracy(prediction, label)
                eval_losses.append(loss.item())
                eval_accs.append(accuracy.item())
                batch_size, _ = prediction.shape
                pred_class = prediction.argmax(dim=-1)

                #pred_class = np.argmax(prediction.cpu().detach().numpy(), axis=1)
                for i in pred_class:
                    preds.append(int(i)+1)

        eval_loss = np.mean(eval_losses)
        eval_acc = np.mean(eval_accs)
        self.print_epoch_update_to_screen(accuracy=eval_acc, loss=eval_loss)

        return preds

    def eval_using_model(self,
                         model,
                         device: torch.device,
                         eval_dataloader,
                         path_to_saved_model='./saved_model/model.pth',
                         criterion=None):

        # Load the best model's weights.
        self.get_saved_model(model=model, path_to_saved_model=path_to_saved_model)

        # Evaluate test loss on testing dataset (NOT Validation)
        preds = self.evaluate_and_save_preds(eval_dataloader, model, device, criterion=criterion)

        return preds