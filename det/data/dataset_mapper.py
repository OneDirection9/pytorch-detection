# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import copy
import logging
from typing import Any, Dict, List

import numpy as np
import torch

from det.config import CfgNode
from . import transforms as T, utils
from .datasets import build_vision_datasets

logger = logging.getLogger(__name__)

__all__ = ['DatasetMapper']


def build_augmentation(cfg: CfgNode, training: bool = True) -> List[T.Augmentation]:
    """Creates a list of :class:`Augmentation` from config.

    Now it includes resizing and flipping.

    Returns:
        List[Augmentation]
    """
    if training:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = 'choice'
    if sample_style == 'range':
        assert len(min_size) == 2, 'more than 2 ({}) min_size(s) are provided for ranges'.format(
            len(min_size)
        )

    augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
    if training:
        augmentation.append(T.RandomHFlip())
    logger.info(
        'Transform used in {}: {}'.format('training' if training else 'testing', augmentation)
    )
    return augmentation


class DatasetMapper(object):
    """A callable which takes a dataset dict in Detectron2's Dataset format, and map it into a
    format used by the model.

    This is the default callable to be used to map your dataset dict into training data. You may
    need to follow it to implement your own one for customized logic, such as a different way to
    read or transform images. See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(
        self,
        *,
        augmentation: List[T.Augmentation],
        compute_tight_boxes: bool,
        image_format: str = 'BGR',
        mask_on: bool = False,
        mask_format: str = 'polygon',
        keypoint_on: bool = False,
        keypoint_hflip_indices=None,
        training: bool = True,
    ) -> None:
        """
        Args:
            augmentation: List of augmentation.
            compute_tight_boxes: Where creating a tight box around mask.
            image_format: See :class:`read_image`.
            mask_on: Whether keep segmentation.
            mask_format: See :func:`annotations_to_instances`.
            keypoint_on: Whether keep keypoints.
            keypoint_hflip_indices: See :func:`create_keypoint_hflip_indices`.
            training: Whether in training mode.
        """
        self.augmentation = augmentation
        self.compute_tight_boxes = compute_tight_boxes
        self.image_format = image_format
        self.mask_on = mask_on
        self.mask_format = mask_format
        self.keypoint_on = keypoint_on
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.training = training

    @classmethod
    def from_config(cls, cfg: CfgNode, training: bool) -> 'DatasetMapper':
        kwargs = {
            'image_format': cfg.INPUT.FORMAT,
            'mask_on': cfg.MODEL.MASK_ON,
            'mask_format': cfg.INPUT.MASK_FORMAT,
            'keypoint_on': cfg.MODEL.KEYPOINT_ON,
            'training': training,
        }

        augmentation = build_augmentation(cfg, training)
        if cfg.INPUT.CROP.ENABLED and training:
            augmentation.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            logger.info('RandomCrop used in training: {}'.format(augmentation[0]))
            compute_tight_boxes = True
        else:
            compute_tight_boxes = False
        kwargs['augmentation'] = augmentation
        kwargs['compute_tight_boxes'] = compute_tight_boxes

        if cfg.MODEL.KEYPOINT_ON and training:
            vision_datasets = build_vision_datasets(cfg.DATASETS.TRAIN)
            keypoint_hflip_indices = utils.create_keypoint_hflip_indices(vision_datasets)
            kwargs['keypoint_hflip_indices'] = keypoint_hflip_indices

        return cls(**kwargs)

    def __call__(self, dataset_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Args:
            dataset_dict: Metadata of one image, in Detectron2's Dataset format.

        Returns:
            A format that builtin models in Detectron2 accept.
        """
        dataset_dict = copy.deepcopy(dataset_dict)
        image = utils.read_image(dataset_dict['file_name'], self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if 'sem_seg_file_name' in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop('sem_seg_file_name'), 'L').squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentation)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # HxW
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict['sem_seg'] = torch.as_tensor(sem_seg_gt.astype('long'))

        if not self.training:
            dataset_dict.pop('annotations', None)
            dataset_dict.pop('sem_seg_file_name', None)
            return dataset_dict

        if 'annotations' in dataset_dict:
            for ann in dataset_dict['annotations']:
                if not self.mask_on:
                    ann.pop('segmentation', None)
                if not self.keypoint_on:
                    ann.pop('keypoints', None)

            anns = [
                utils.transform_instance_annotations(
                    obj,
                    transforms,
                    image_shape,
                    keypoint_hflip_indices=self.keypoint_hflip_indices
                ) for obj in dataset_dict.pop('annotations') if obj.get('iscrowd', 0) == 0
            ]
            instances = utils.annotations_to_instances(
                anns, image_shape, mask_format=self.mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer tightly
            # bound the object. As an example, imagine a triangle object [(0,0), (2,0), (0,2)]
            # cropped by a box [(1,0),(2,2)] (XYXY format). The tight bounding box of the cropped
            # triangle should be [(1,0),(2,1)], which is not equal to the intersection of original
            # bounding box and the cropping box.
            if self.compute_tight_boxes and instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict['instances'] = utils.filter_empty_instances(instances)
        return dataset_dict
