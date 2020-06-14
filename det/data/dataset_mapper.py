# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import copy
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image

from . import transforms as T, utils
from .datasets import VisionDataset

logger = logging.getLogger(__name__)

__all__ = ['DatasetMapper']


def build_transform_gen(
    min_size: Union[List[int], int] = (800,),
    max_size: int = 1333,
    sample_style: str = 'choice',
    training: bool = True
) -> List[T.TransformGen]:
    """Creates a list of :class:`TransformGen` from config.

    Now it includes resizing and flipping.

    Args:
        min_size: See :class:`ResizeShortestEdge`.
        max_size: See :class:`ResizeShortestEdge`.
        sample_style: See :class:`ResizeShortestEdge`.
        training: Whether it is in training mode.
    """
    if not training:
        sample_style = 'choice'

    if sample_style == 'range':
        assert len(min_size) == 2, 'More than 2 ({}) min_size(s) are provided for ranges'.format(
            len(min_size)
        )

    tfm_gens = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    if training:
        tfm_gens.append(T.RandomHFlip())
    logger.info('Transform used in {}: {}'.format('training' if training else 'testing', tfm_gens))
    return tfm_gens


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
        transform_cfg: Dict[str, Any] = None,
        crop_transform_cfg: Optional[Dict[str, Any]] = None,
        image_format: str = 'BGR',
        mask_on: bool = False,
        mask_format: str = 'polygon',
        keypoint_on: bool = False,
        training: bool = True,
        vision_datasets: Optional[List[VisionDataset]] = None,
    ) -> None:
        """
        Args:
            transform_cfg: See :func:`build_transform_gen`.
            crop_transform_cfg: See :class:`RandomCrop`.
            image_format: See :class:`read_image`.
            mask_on: Whether keep segmentation.
            mask_format: See :func:`annotations_to_instances`.
            keypoint_on: Whether keep keypoints.
            training: Whether in training mode.
            vision_datasets: List of vision datasets, to create keypoint_hflip_indices if needed.
        """
        transform_cfg = transform_cfg or {}
        self.tfm_gens = build_transform_gen(**transform_cfg, training=training)

        if crop_transform_cfg is not None:
            self.crop_gen = T.RandomCrop(**crop_transform_cfg)
        else:
            self.crop_gen = None

        self.image_format = image_format
        self.mask_on = mask_on
        self.mask_format = mask_format
        self.keypoint_on = keypoint_on

        if self.keypoint_on and training:
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(vision_datasets)
        else:
            self.keypoint_hflip_indices = None

        self.training = training

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

        if 'annotations' not in dataset_dict:
            image, transforms = T.apply_transforms(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]), image.shape[:2],
                    np.random.choice(dataset_dict['annotations'])
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transforms(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # HxW

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict['image'] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

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
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has('gt_masks'):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict['instances'] = utils.filter_empty_instances(instances)

        if 'sem_seg_file_name' in dataset_dict:
            sem_seg_gt = Image.open(dataset_dict['sem_seg_file_name'])
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype('long'))
            dataset_dict['sem_seg'] = sem_seg_gt
        return dataset_dict
