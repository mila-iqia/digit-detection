import os

from easydict import EasyDict
import numpy as np
from pathlib import Path
import random
import unittest

import torch

from utils.dataloader import prepare_dataloaders
from utils.trainer import set_seed


class TestDataloader(unittest.TestCase):
    def setUp(self):

        set_seed(1234)

        # Get path to data
        root_dir = os.path.dirname(os.path.abspath(__file__))
        root_dir = Path(root_dir).parent
        data_dir = os.path.join(root_dir, 'data/SVHN')

        self.train_loader, self.valid_loader = prepare_dataloaders(
            input_dir=data_dir, valid_split=0.2,
            batch_size=32, sample_size=-1, train=True)

        self.test_loader = prepare_dataloaders(
            input_dir=data_dir, valid_split=0.2,
            batch_size=32, sample_size=-1, train=False)

        self.n_epoch = 4

    def test_train_loader(self):
        for epoch in range(self.n_epoch):
            for i, batch in enumerate(self.train_loader):
                inputs, targets = batch['image'], batch['target']
                if epoch == 0 and i == 0:
                    self.assertTrue(inputs.shape == (32, 3, 54, 54))
                    self.assertTrue(targets.shape == (32, 6))
                    img = inputs
                    tgt = targets
                elif epoch > 0 and i == 0:
                    self.assertFalse(bool(
                        torch.all(torch.eq(img, inputs)).data.cpu().numpy()))
                    self.assertFalse(bool(
                        torch.all(torch.eq(tgt, targets)).data.cpu().numpy()))

    def test_training_checkpointing(self):
        data = {}
        for epoch in range(self.n_epoch):
            data[epoch] = {'img': [], 'tgt': []}
            for i, batch in enumerate(self.train_loader):
                inputs, targets = batch['image'], batch['target']
                data[epoch]['img'].append(inputs)
                data[epoch]['tgt'].append(targets)

            if epoch == 1:
                # Checkpointing current state
                state = {'epoch': epoch,
                         'seed': {'np_state': np.random.get_state(),
                                  'random_state': random.getstate(),
                                  'torch_state': torch.get_rng_state()}
                         }
                if torch.cuda.is_available():
                    state['seed'][
                        'torch_state_cuda'] = torch.cuda.get_rng_state()

        # Take up training at epoch 2
        # Load state
        set_seed(EasyDict(state['seed']))

        starting_epoch = state['epoch'] + 1
        self.assertTrue(starting_epoch == 2)
        for epoch in range(starting_epoch, self.n_epoch):
            for i, batch in enumerate(self.train_loader):
                inputs, targets = batch['image'], batch['target']
                img = data[epoch]['img'][i]
                tgt = data[epoch]['tgt'][i]
                self.assertTrue(bool(
                        torch.all(torch.eq(img, inputs)).data.cpu().numpy()))
                self.assertTrue(bool(
                    torch.all(torch.eq(tgt, targets)).data.cpu().numpy()))

    def test_valid_loader(self):
        for epoch in range(self.n_epoch):
            for i, batch in enumerate(self.valid_loader):
                inputs, targets = batch['image'], batch['target']
                if epoch == 0 and i == 0:
                    self.assertTrue(inputs.shape == (32, 3, 54, 54))
                    self.assertTrue(targets.shape == (32, 6))
                    img = inputs
                    tgt = targets
                elif epoch > 0 and i == 0:
                    self.assertTrue(bool(
                        torch.all(torch.eq(img, inputs)).data.cpu().numpy()))
                    self.assertTrue(bool(
                        torch.all(torch.eq(tgt, targets)).data.cpu().numpy()))

    def test_test_loader(self):
        for epoch in range(self.n_epoch):
            for i, batch in enumerate(self.test_loader):
                inputs, targets = batch['image'], batch['target']
                if epoch == 0 and i == 0:
                    self.assertTrue(inputs.shape == (32, 3, 54, 54))
                    self.assertTrue(targets.shape == (32, 6))
                    img = inputs
                    tgt = targets
                elif epoch > 0 and i == 0:
                    self.assertTrue(bool(
                        torch.all(torch.eq(img, inputs)).data.cpu().numpy()))
                    self.assertTrue(bool(
                        torch.all(torch.eq(tgt, targets)).data.cpu().numpy()))


if __name__ == '__main__':
    unittest.main()
