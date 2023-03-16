import matplotlib.pyplot as plt
import os
import numpy as np
from Text_Parsing.model import model_hyperparameter_factory as mhf


class ModelPlotter:
    def __init__(self):
        pass

    def plot_training_summary(self,
                              training_summary: dict,
                              hparams: mhf.HyperParams,
                              model_type: str):

        for term in ['Accuracy', 'Loss']:
            plot_title = model_type + " " + hparams.OPTIM + f' training vs validation {term} \n '
            if (model_type == 'LSTM') | (model_type == 'GRU'):
                plot_title = plot_title + \
                             f'with Layers: {hparams.N_LAYERS}, ' \
                             f'Hidden Dim: {hparams.HIDDEN_DIM}, ' \
                             f'Embed Dim: {hparams.EMBEDDING_DIM}'

            self.plot_training_vs_epoch(plot_title=plot_title,
                                        data=training_summary[term],
                                        data_type_label=term)

    @staticmethod
    def plot_training_vs_epoch(plot_title:str,
                               data: dict,
                               data_type_label: str = 'Accuracy',  # or Validation
                               colors: list = ('blue', 'orange'),
                               save_plot: bool = True):

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        if data_type_label == 'Accuracy':
            plt.ylim([0, 1.0])
            ax.set_ylabel('Accuracy')
            training_data = data['Training']
            validation_data = data['Validation']
        elif data_type_label == 'Loss':  # Loss
            ax.set_ylabel('Log Loss')
            training_data = np.log(data['Training'])
            validation_data = np.log(data['Validation'])
        else:
            raise ValueError(f'Unexpected data_type_label passed to plot_training_vs_epoch: {data_type_label}. '
                             f'Expected: Accuracy or Loss')

        ax.plot(training_data, '-', label=f'Training {data_type_label}', color=colors[0])
        ax.plot(validation_data, '-', label=f'Validation {data_type_label}', color=colors[1])

        ax.legend(loc='best')
        ax.set_title(plot_title)
        ax.set_xlabel('Epoch')

        if save_plot:
            file_name = f'Training_vs_Epoch_for_{data_type_label}'
            plot_folder = "./plots"
            if not os.path.exists(plot_folder):
                os.makedirs(plot_folder)

            fig_name = file_name + ".png"
            plt.savefig(fig_name, bbox_inches='tight')

        plt.show(fig=fig)
        plt.close(fig=fig)
