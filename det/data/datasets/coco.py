# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import contextlib
import io
import logging
import os
import os.path as osp
from typing import Any, Dict, List, Optional

from foundation.common.timer import Timer
from pycocotools import mask as mask_util

from det.structures import BoxMode
from .metadata import (
    Metadata,
    get_coco_instance_metadata,
    get_coco_panoptic_metadata,
    get_coco_person_metadata,
)
from .registry import VisionDataset, VisionDatasetRegistry

__all__ = ['COCODataset', 'COCOSemSeg']
"""
This file contains functions to parse COCO-format annotations into dicts in "Detectron2 format".
"""

logger = logging.getLogger(__name__)


@VisionDatasetRegistry.register('COCODataset')
class COCODataset(VisionDataset):
    """COCO dataset supporting object detection task and keypoint detection task.

    Loads a json file with COCO's instances annotation format. Currently supports instance
    detection, instance segmentation, and person keypoints annotations. Refer
    http://cocodataset.org/#format-data for more details.
    """

    def __init__(
        self,
        json_file: str,
        image_root: str,
        extra_annotation_keys: Optional[List[str]] = None,
        metadata: Optional[Metadata] = None,
    ) -> None:
        """
        Args:
            json_file: Path to the json file in COCO instances annotation format.
            image_root: The directory where the image in this json file exists.
            extra_annotation_keys: List of per-annotation keys that should also be loaded into the
                dataset dict (besides "iscrowd", "bbox", "keypoints", "category_id",
                "segmentation"). The values for these keys will be returned as-is. For example, the
                densepose annotations are loaded in this way.
            metadata: If provided, it should be consistent with the information in json file.
                Otherwise, information in this json file will be loaded.
        """
        super(COCODataset, self).__init__(metadata)

        self.json_file = json_file
        self.image_root = image_root
        self.extra_annotation_keys = extra_annotation_keys or []

        # Update dataset metadata
        self._metadata.set(**dict(
            json_file=json_file,
            image_root=image_root,
        ))

    def get_items(self) -> List[Dict[str, Any]]:
        """
        Returns:
            dataset_dicts: A list of dicts in Detectron2 standard dataset dicts format. (See
                `Using Custom Datasets </tutorials/datasets.html>`_ )

        Notes:
            1. This function does not read the image files.
               The results do not have the "image" field.
        """
        from pycocotools.coco import COCO

        timer = Timer()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(self.json_file)
        if timer.seconds() > 1:
            logger.info('Loading {} takes {:.2f} seconds.'.format(self.json_file, timer.seconds()))

        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c['name'] for c in sorted(cats, key=lambda x: x['id'])]
        self._metadata.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed,
        # and by convention they are always ignored.
        # We deal with COCO's id issue and translate
        # the category ids to contiguous ids in [0, 80).
        id_map = {v: i for i, v in enumerate(cat_ids)}
        self._metadata.thing_dataset_id_to_contiguous_id = id_map

        # Currently only "person" has keypoints.
        if len(cats) == 1 and cats[0]['name'] == 'person' and 'keypoints' in cats[0]:
            self._metadata.keypoint_names = cats[0]['keypoints']

        # sort indices for reproducible results
        img_ids = sorted(coco_api.getImgIds())
        # imgs is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        imgs = coco_api.loadImgs(img_ids)
        # anns is a list[list[dict]], where each dict is an annotation
        # record for an object. The inner list enumerates the objects in an image
        # and the outer list enumerates over images. Example of anns[0]:
        # [{'segmentation': [[192.81,
        #     247.09,
        #     ...
        #     219.03,
        #     249.06]],
        #   'area': 1035.749,
        #   'iscrowd': 0,
        #   'image_id': 1268,
        #   'bbox': [192.81, 224.8, 74.73, 33.43],
        #   'category_id': 16,
        #   'id': 42986},
        #  ...]
        anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]

        logger.info('Loaded {} images in COCO format from {}'.format(len(imgs), self.json_file))

        dataset_dicts = []

        ann_keys = ['iscrowd', 'bbox', 'keypoints', 'category_id'] + self.extra_annotation_keys

        num_instances_without_valid_segmentation = 0

        for img_dict, ann_dict_list in zip(imgs, anns):
            record = {}

            file_name = osp.join(self.image_root, img_dict['file_name'])
            if not osp.isfile(file_name):
                raise FileNotFoundError('{}: No such image'.format(file_name))

            record['file_name'] = file_name
            record['height'] = img_dict['height']
            record['width'] = img_dict['width']
            image_id = record['image_id'] = img_dict['id']

            objs = []
            for ann in ann_dict_list:
                # Check that the image_id in this annotation is the same as
                # the image_id we're looking at.
                # This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files
                # actually contains bugs that, together with certain ways of using COCO API,
                # can trigger this assertion.
                assert ann['image_id'] == image_id

                assert ann.get('ignore', 0) == 0, '"ignore" in COCO json file is not supported.'

                obj = {key: ann[key] for key in ann_keys if key in ann}

                segm = ann.get('segmentation', None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if isinstance(segm, dict):
                        if isinstance(segm['counts'], list):
                            # convert to compressed RLE
                            segm = mask_util.frPyObjects(segm, *segm['size'])
                    else:
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj['segmentation'] = segm

                keypts = ann.get('keypoints', None)
                if keypts:  # list[int]
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1]
                            # Therefore we assume the coordinates are "pixel indices" and
                            # add 0.5 to convert to floating point coordinates.
                            keypts[idx] = v + 0.5
                    obj['keypoints'] = keypts

                obj['bbox_mode'] = BoxMode.XYWH_ABS
                obj['category_id'] = id_map[obj['category_id']]
                objs.append(obj)
            record['annotations'] = objs
            dataset_dicts.append(record)

        if num_instances_without_valid_segmentation > 0:
            logger.warning(
                'Filtered out {} instances without valid segmentation. '
                .format(num_instances_without_valid_segmentation) +
                'There might be issues in your dataset generation process. '
                'A valid polygon should be a list[float] with even length >= 6.'
            )
        return dataset_dicts


# Registry datasets presetting the metadata
VisionDatasetRegistry.register_partial(
    'COCOInstance', metadata=get_coco_instance_metadata()
)(COCODataset)  # yapf: disable

VisionDatasetRegistry.register_partial(
    'COCOPanoptic', metadata=get_coco_panoptic_metadata()
)(COCODataset)  # yapf: disable

VisionDatasetRegistry.register_partial(
    'COCOPerson', metadata=get_coco_person_metadata()
)(COCODataset)  # yapf: disable


@VisionDatasetRegistry.register('COCOSemSeg')
class COCOSemSeg(VisionDataset):
    """COCO semantic segmentation datasets."""

    def __init__(
        self,
        gt_root: str,
        image_root: str,
        gt_ext: str = 'png',
        image_ext: str = 'jpg',
        metadata: Optional[Metadata] = None,
    ) -> None:
        """
        All files under "gt_root" with "gt_ext" extension are treated as ground truth annotations
        and all files under "image_root" with "image_ext" extension as input images. Ground truth
        and input images are matched using file paths relative to "gt_root" and "image_root"
        respectively without taking into account file extensions. This works for COCO as well as
        some other datasets.

        Args:
            gt_root: Full path to ground truth semantic segmentation files. Semantic segmentation
                annotations are stored as images with integer values in pixels that represent
                corresponding semantic labels.
            image_root: The directory where the input images are.
            gt_ext: File extension for ground truth annotations.
            image_ext: File extension for input images.
        """
        super(COCOSemSeg, self).__init__(metadata)

        self.gt_root = gt_root
        self.image_root = image_root
        self.gt_ext = gt_ext
        self.image_ext = image_ext

        # Update dataset metadata
        self._metadata.set(**dict(
            gt_root=gt_root,
            image_root=image_root,
        ))

    def get_items(self) -> List[Dict[str, Any]]:
        """
        Returns:
            dataset_dicts: A list of dicts in detectron2 standard format without instance-level
                annotation.

        Notes:
            1. This function does not read the image and ground truth files.
               The results do not have the "image" and "sem_seg" fields.
        """

        # We match input images with ground truth based on their relative filepaths (without file
        # extensions) starting from 'image_root' and 'gt_root' respectively.
        def file2id(folder_path, file_path):
            # extract relative path starting from `folder_path`
            image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
            # remove file extension
            image_id = os.path.splitext(image_id)[0]
            return image_id

        input_files = sorted(
            (
                osp.join(self.image_root, f)
                for f in os.listdir(self.image_root) if f.endswith(self.image_ext)
            ),
            key=lambda file_path: file2id(self.image_root, file_path),
        )
        gt_files = sorted(
            (
                osp.join(self.gt_root, f)
                for f in os.listdir(self.gt_root) if f.endswith(self.gt_ext)
            ),
            key=lambda file_path: file2id(self.gt_root, file_path),
        )

        assert len(gt_files) > 0, 'No annotations found in {}.'.format(self.gt_root)

        # Use the intersection, so that val2017_100 annotations can run smoothly with val2017 images
        if len(input_files) != len(gt_files):
            logger.warning(
                'Directory {} and {} has {} and {} files, respectively.'.format(
                    self.image_root, self.gt_root, len(input_files), len(gt_files)
                )
            )
            input_basenames = [os.path.basename(f)[:-len(self.image_ext)] for f in input_files]
            gt_basenames = [os.path.basename(f)[:-len(self.gt_ext)] for f in gt_files]
            intersect = list(set(input_basenames) & set(gt_basenames))
            # sort, otherwise each worker may obtain a list[dict] in different order
            intersect = sorted(intersect)
            logger.warning('Will use their intersection of {} files.'.format(len(intersect)))
            input_files = [os.path.join(self.image_root, f + self.image_ext) for f in intersect]
            gt_files = [os.path.join(self.gt_root, f + self.gt_ext) for f in intersect]

        logger.info(
            'Loaded {} images with semantic segmentation from {}'.format(
                len(input_files), self.image_root
            )
        )

        dataset_dicts = []
        for (img_path, gt_path) in zip(input_files, gt_files):
            record = {
                'file_name': img_path,
                'sem_seg_file_name': gt_path,
            }
            dataset_dicts.append(record)

        return dataset_dicts
