from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import time

import torch

from foundation.backends.torch.engine import RunnerBase
from ..data import build_test_loader
from ..data import build_train_loader
from ..models import build_model

logger = logging.getLogger(__name__)

__all__ = ['Runner']


class Runner(RunnerBase):

    def __init__(self, cfg):
        super(Runner, self).__init__()

        self._cfg = copy.deepcopy(cfg)

        self._data_loader, self._epoch_length = self.build_train_loader(cfg)
        self._data_loader_iter = iter(self._data_loader)

        self._model = self.build_model(cfg)
        self._model.train()

        self.build_hooks(cfg)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_train_loader(cfg['dataset']['train'], cfg['dataloader'])

    @classmethod
    def build_test_loader(cls, cfg):
        return build_test_loader(cfg['dataset']['test'], cfg['dataloader'])

    @classmethod
    def build_model(cls, cfg):
        return build_model(cfg['model'])

    def build_hooks(self, cfg):
        if 'test' in cfg['dataset']:
            test_dataloader = self.build_test_loader(cfg)
            logger.info('Registered hook with {} dataset for evaluation'
                        .format(cfg['dataset']['test']['name']))

    def run_step(self):
        """
        Implement the standard training logic described above.
        """
        assert self._model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        """
        If your want to do something with the data, you can wrap the dataloader.
        """
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        self._model(data)

        """
        If your want to do something with the losses, you can wrap the model.
        """
        # loss_dict = self.model(data)
        # losses = sum(loss for loss in loss_dict.values())
        # self._detect_anomaly(losses, loss_dict)
        #
        # metrics_dict = loss_dict
        # metrics_dict["data_time"] = data_time
        # self._write_metrics(metrics_dict)
        #
        # """
        # If you need accumulate gradients or something similar, you can
        # wrap the optimizer with your custom `zero_grad()` method.
        # """
        # self.optimizer.zero_grad()
        # losses.backward()
        #
        # """
        # If you need gradient clipping/scaling or other processing, you can
        # wrap the optimizer with your custom `step()` method.
        # """
        # self.optimizer.step()

    def _detect_anomaly(self, losses, loss_dict):
        if not torch.isfinite(losses).all():
            raise FloatingPointError(
                "Loss became infinite or NaN at iteration={}!\nloss_dict = {}".format(
                    self.iter, loss_dict
                )
            )
