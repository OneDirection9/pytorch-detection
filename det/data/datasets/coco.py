# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
#
# Modified by: Zhipeng Han
from __future__ import absolute_import, division, print_function

import contextlib
import copy
import io
import logging
import os.path as osp
from typing import Callable, Optional, Type

import foundation as fdn
import numpy as np
from foundation.utils import Timer

from .base import VisionDataset, VisionDatasetStash
from .metadata import Metadata

logger = logging.getLogger(__name__)

__all__ = ['COCOInstance']


@VisionDatasetStash.register('COCOInstance')
class COCOInstance(VisionDataset):
    """COCO instance dataset for object detection task and keypoint detection task."""

    def __init__(
        self,
        json_file: str,
        image_root: str,
        metadata: Type[Metadata],
        filter_fn: Optional[Callable] = None,
    ) -> None:
        """
        Args:
            json_file: Path to the json file in COCO instances annotation format. Refer
                http://cocodataset.org/#format-data for more details.
            image_root: The directory where
            metadata:
            filter_fn:
        """
        self._image_root = image_root
        self._json_file = json_file
        self._metadata = metadata
        self._filter_fn = filter_fn

    @property
    def metadata(self) -> Type[Metadata]:
        return self._metadata

    def get_items(self):
        from pycocotools.coco import COCO

        timer = Timer()
        with contextlib.redirect_stdout(io.StringIO()):
            coco_api = COCO(self._json_file)
        if timer.seconds() > 1:
            logger.info('Loading {} takes {:.2f} seconds'.format(self._json_file, timer.seconds()))

        cat_ids = coco_api.getCatIds()
        cats = coco_api.loadCats(cat_ids)
        print(cats)

    def check_metadata_consistency(self, cats):
        """Checks that the json_file has consistent categories with self._metadata."""
        pass

    def __repr__(self):
        return 'test'


@VisionDatasetStash.register('CocoDetection')
class CocoDetection(VisionDataset):
    """MSCOCO detection dataset.

    Args:
        root (str): The root path of dataset.
        ann_file (str): The annotations file.
        image_prefix (str, optional): Image file prefix.
        min_object_area(float, optional): Minimum ground truth area, the objects' area
            less than this threshold will be ignored. Default: 0
        use_crowd (bool, optional): Whether use instances annotated as crowd. Default:
            ``True``
        infer_mode (bool, optional): In inference mode, will not filter images and
            load labels. Default: ``False``
        transform (callable, optional): A function takes item as input and return
            transformed item.
    """

    CLASSES = (
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    )

    def __init__(
        self,
        root,
        ann_file,
        image_prefix='',
        min_object_area=0.,
        use_crowd=True,
        infer_mode=False,
        transform=None
    ):
        from pycocotools.coco import COCO

        super(CocoDetection, self).__init__(root, ann_file)

        self._root = root
        self._ann_file = ann_file
        self._image_prefix = image_prefix
        self._min_object_area = min_object_area
        self._use_crowd = use_crowd
        self._infer_mode = infer_mode
        self._transform = transform

        # Load annotation file
        self._coco = COCO(ann_file)
        category_ids = sorted(self._coco.getCatIds())
        categories = [c['name'] for c in self._coco.loadCats(category_ids)]
        if not tuple(categories) == self.classes:
            raise ValueError('Incompatible category names with {}'.format(self.__class__.__name__))
        self.json_id_to_contiguous = {v: i for i, v in enumerate(category_ids)}
        self.contiguous_id_to_json = {i: v for i, v in enumerate(category_ids)}

        self._items, self._flags = self._get_items()

    @property
    def classes(self):
        return self.CLASSES

    @property
    def aspect_ratio_flag(self):
        return self._flags

    def __getitem__(self, index):
        # Deep copy the item to avoid change the raw data
        item = copy.deepcopy(self._items[index])

        # Image with uint8 dtype and shape (h, w, c) where c in *B G R* order
        image = fdn.io.imread(item.pop('image_path'))
        # Convert *B G R* to *R G B* order
        item['image'] = fdn.ops.bgr2rgb(image)

        # Check image size
        assert image.shape[:2] == item['ori_shape']

        if self._transform is not None:
            item = self._transform(item)
        return item

    def __len__(self):
        return len(self._items)

    def _get_items(self):
        """Loads images and labels from annotation file."""
        # FIXME: use 100 images for debug
        image_ids = sorted(self._coco.getImgIds())[:100]

        logger.info('Loaded {} images from {}'.format(len(image_ids), self._ann_file))

        items, flags = [], []
        for image_id in image_ids:
            entry = self._coco.loadImgs(image_id)[0]
            image_path = osp.join(self._root, self._image_prefix + entry['file_name'])
            fdn.utils.check_file_exist(image_path, '{}: No such image')

            # Image information
            item = {
                'id': entry['id'],
                'image_path': image_path,
                'ori_shape': (entry['height'], entry['width']),
            }
            # Annotations
            if not self._infer_mode:
                target = self._load_annotations(entry)
                if target is None:
                    logger.warning(
                        '{}: Skip the image since it has no valid annotations'.format(image_path)
                    )
                    continue
                item['bboxes'] = target['bboxes']
                item['labels'] = target['labels']

            items.append(item)
            # Aspect ratio flag
            flags.append(1 if entry['width'] > entry['height'] else 0)

        logger.info(
            'Removed {} images with no valid annotations. {} images left'.format(
                len(image_ids) - len(items), len(items)
            )
        )

        return items, flags

    def _load_annotations(self, entry):
        """Loads ground-truth annotations for given image."""
        ann_ids = self._coco.getAnnIds(imgIds=entry['id'], iscrowd=None)
        instances = self._coco.loadAnns(ann_ids)

        bboxes, labels = [], []
        for inst in instances:
            if inst.get('ignore', False):
                continue
            if inst['area'] < self._min_object_area:
                continue
            if not self._use_crowd and inst.get('iscrowd', 0):
                continue

            xyxy = fdn.ops.xywh_to_xyxy(inst['bbox'])
            x1, y1, x2, y2 = fdn.ops.clip_xyxy(xyxy, (entry['height'], entry['width']))
            if inst['area'] > 0 and x2 > x1 and y2 > y1:
                contiguous_id = self.json_id_to_contiguous[inst['category_id']]
                bboxes.append([x1, y1, x2, y2])
                labels.append(contiguous_id)

        if len(bboxes) == 0:
            return None

        bboxes = np.array(bboxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32)
        return {'bboxes': bboxes, 'labels': labels}
