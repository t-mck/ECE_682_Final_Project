import data_utilities as du
import torch
import functools


class DataLoader:
    def __init__(self):
        self.data_utils = du.DataUtilities()

    def get_dataloaders(self, train_data, valid_data, test_data, hparams):
        collate = functools.partial(self.data_utils.collate, pad_index=hparams.PAD_INDEX)

        train_dataloader = torch.utils.data.DataLoader(
            train_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(
            valid_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)

        return train_dataloader, valid_dataloader, test_dataloader
