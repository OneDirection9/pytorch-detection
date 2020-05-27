# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import contextlib
import io
import logging
import os.path as osp
from typing import Any, Dict, List, Optional

from foundation.utils import Timer
from pycocotools.coco import COCO

from ...structures import BoxMode
from .base import VisionDataset, VisionDatasetRegistry
from .metadata import Metadata

__all__ = ['COCOInstance']

logger = logging.getLogger(__name__)


@VisionDatasetRegistry.register('COCOInstance')
class COCOInstance(VisionDataset):
    """COCO instance dataset supporting object detection task and keypoint detection task.

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
        super(COCOInstance, self).__init__(metadata)

        self.image_root = image_root
        self.json_file = json_file
        self.extra_annotation_keys = extra_annotation_keys or []

        # Update dataset metadata
        self._metadata.set(**dict(
            image_root=image_root,
            json_file=json_file,
        ))

    def get_examples(self) -> List[Dict[str, Any]]:
        timer = Timer()
        with contextlib.redirect_stdout(io.StringIO()):  # omit messages printed by COCO
            coco_api = COCO(self.json_file)
        if timer.seconds() > 1:
            logger.info('Loading {} takes {:.2f} seconds'.format(self.json_file, timer.seconds()))

        # The categories in a custom json file may not be sorted.
        cat_ids = sorted(coco_api.getCatIds())
        cats = coco_api.loadCats(cat_ids)
        thing_classes = [c['name'] for c in cats]
        self._metadata.thing_classes = thing_classes

        # In COCO, certain category ids are artificially removed, and by convention they are always
        # ignored. We deal with COCO's id issue and translate the category ids to 0-indexed
        # contiguous ids, i.e. [0, 80).
        id_map = {v: i for i, v in enumerate(cat_ids)}
        self._metadata.thing_dataset_id_to_contiguous_id = id_map

        # Currently only "person" has keypoints.
        if len(cats) == 1 and cats[0]['name'] == 'person' and 'keypoints' in cats[0]:
            self._metadata.keypoint_names = cats[0]['keypoints']

        # sort indices for reproducible results
        image_ids = sorted(coco_api.getImgIds())
        # images is a list of dicts, each looks something like:
        # {'license': 4,
        #  'url': 'http://farm6.staticflickr.com/5454/9413846304_881d5e5c3b_z.jpg',
        #  'file_name': 'COCO_val2014_000000001268.jpg',
        #  'height': 427,
        #  'width': 640,
        #  'date_captured': '2013-11-17 05:57:24',
        #  'id': 1268}
        images = coco_api.loadImgs(image_ids)

        logger.info('Loaded {} images in COCO format from {}'.format(len(images), self.json_file))

        dataset_dicts = []

        ann_keys = ['iscrowd', 'bbox', 'category_id'] + self.extra_annotation_keys

        num_instances_without_valid_segmentation = 0

        for image_dict in images:
            record = {}

            file_name = osp.join(self.image_root, image_dict['file_name'])
            if not osp.isfile(file_name):
                raise FileNotFoundError('{}: No such image'.format(file_name))

            record['file_name'] = file_name
            record['height'] = image_dict['height']
            record['width'] = image_dict['width']
            image_id = record['image_id'] = image_dict['id']

            # ann_dict_list is a list[dict], where each dict is an annotation record for an object.
            # The inner list enumerates the objects in an image and the outer list enumerates over
            # images. Example of anns[0]:
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
            ann_ids = coco_api.getAnnIds(imgIds=image_id)
            ann_dict_list = coco_api.loadAnns(ann_ids)

            objs = []
            for ann in ann_dict_list:
                # Check the image_id in this annotation is the same as the image_id we're looking
                # at. This fails only when the data parsing logic or the annotation file is buggy.

                # The original COCO valminusminival2014 & minival2014 annotation files actually
                # contains bugs that, together with certain ways of using COCO API, can trigger this
                # assertion.
                assert ann['image_id'] == image_id

                assert ann.get('ignore', 0) == 0, '"ignore" in COCO json file is not supported.'

                obj = {key: ann[key] for key in ann_keys if key in ann}

                segm = ann.get('segmentation', None)
                if segm:  # either list[list[float]] or dict(RLE)
                    if not isinstance(segm, dict):
                        # filter out invalid polygons (< 3 points)
                        segm = [poly for poly in segm if len(poly) % 2 == 0 and len(poly) >= 6]
                        if len(segm) == 0:
                            num_instances_without_valid_segmentation += 1
                            continue  # ignore this instance
                    obj['segmentation'] = segm

                keypts = ann.get('keypoints', None)
                if keypts:  # list[int]
                    # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility.
                    #   v = 0: not labeled (in which case x=y=0);
                    #   v = 1: labeled but not visible;
                    #   v = 2: labeled and visible.
                    for idx, v in enumerate(keypts):
                        if idx % 3 != 2:
                            # COCO's segmentation coordinates are floating points in [0, H or W],
                            # but keypoint coordinates are integers in [0, H-1 or W-1] Therefore we
                            # assume the coordinates are "pixel indices" and add 0.5 to convert to
                            # floating point coordinates.
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
                'There might be issues in your dataset generation process.'
                .format(num_instances_without_valid_segmentation)
            )
        return dataset_dicts
