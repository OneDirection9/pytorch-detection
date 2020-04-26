# from __future__ import absolute_import, division, print_function
#
# import logging
# import math
#
# import numpy as np
# from foundation.backends.torch import samplers
# from foundation.backends.torch.utils import comm, env
# from foundation.registry import build
# from torch.utils.data import BatchSampler, DataLoader
#
# from .registry import DatasetStash, TransformStash
#
# logger = logging.getLogger(__name__)
#
#
# def build_train_loader(dataset_cfg, dataloader_cfg):
#     """
#     Args:
#         dataset_cfg (dict):
#         dataloader_cfg (dict): Must have images_per_gpu and num_workers keys.
#     """
#     # TODO: concat multiple datasets
#     if 'transform' in dataset_cfg:
#         dataset_cfg['transform'] = build(TransformStash, dataset_cfg['transform'])
#     dataset = build(DatasetStash, dataset_cfg)
#
#     images_per_gpu = dataloader_cfg['images_per_gpu']
#     num_workers = dataloader_cfg['num_workers']
#
#     # Build batch sampler
#     sampler = samplers.TrainingSampler(len(dataset), shuffle=True)
#     if hasattr(dataset, 'aspect_ratio_flag'):
#         group_ids = getattr(dataset, 'aspect_ratio_flag')
#         batch_sampler = samplers.GroupedBatchSampler(sampler, group_ids, images_per_gpu)
#     else:
#         batch_sampler = BatchSampler(sampler, images_per_gpu, drop_last=True)
#
#     # The TrainingSampler generates infinite stream and the __len__ is not well defined,
#     # so we approximate the epoch length as following, with this calculation the last
#     # batch is considered to padded to the ``batch_size``, and return it for other usage.
#     num_gpus = comm.get_world_size()
#     total_batch_size = images_per_gpu * num_gpus
#     epoch_length = math.ceil(len(dataset) // total_batch_size)
#
#     logger.info('Training dataset length is {}'.format(len(dataset)))
#     logger.info('Training epoch length is {}'.format(epoch_length))
#
#     data_loader = DataLoader(
#         dataset,
#         batch_sampler=batch_sampler,
#         num_workers=num_workers,
#         collate_fn=trivial_batch_collator,
#         worker_init_fn=worker_init_reset_seed,
#     )
#     return data_loader, epoch_length
#
#
# def build_test_loader(dataset_cfg, dataloader_cfg):
#     """
#     Args:
#         dataset_cfg (dict):
#         dataloader_cfg (dict): Must have num_workers key.
#     """
#     # TODO: concat multiple datasets
#     if 'transform' in dataset_cfg:
#         dataset_cfg['transform'] = build(TransformStash, dataset_cfg['transform'])
#     dataset = build(DatasetStash, dataset_cfg)
#
#     num_workers = dataloader_cfg['num_workers']
#
#     # Build batch sampler
#     sampler = samplers.InferenceSampler(len(dataset))
#     batch_sampler = BatchSampler(sampler, 1, drop_last=False)
#
#     logger.info('Test dataset length is {}'.format(len(dataset)))
#     logger.info('Test epoch length is {}'.format(len(sampler)))
#
#     data_loader = DataLoader(
#         dataset,
#         batch_sampler=batch_sampler,
#         num_workers=num_workers,
#         collate_fn=trivial_batch_collator,
#     )
#     return data_loader
#
#
# def trivial_batch_collator(batch):
#     """
#     A batch collator that does nothing.
#     """
#     return batch
#
#
# def worker_init_reset_seed(worker_id):
#     env.set_random_seed(np.random.randint(2 ** 31) + worker_id)
