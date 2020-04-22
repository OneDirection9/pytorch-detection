from __future__ import absolute_import, division, print_function

import copy
import logging
import os.path as osp

import numpy as np
from torch.utils.data import Dataset

import foundation as fdn
from ..registry import DatasetStash
from . import base

logger = logging.getLogger(__name__)

__all__ = ['CocoDetection']

# yapf: disable
# All coco categories, together with their nice-looking visualization colors
# It's from https://github.com/cocodataset/panopticapi/blob/master/panoptic_coco_categories.json
COCO_CATEGORIES = [
    {'color': [220, 20, 60], 'isthing': 1, 'id': 1, 'name': 'person'},
    {'color': [119, 11, 32], 'isthing': 1, 'id': 2, 'name': 'bicycle'},
    {'color': [0, 0, 142], 'isthing': 1, 'id': 3, 'name': 'car'},
    {'color': [0, 0, 230], 'isthing': 1, 'id': 4, 'name': 'motorcycle'},
    {'color': [106, 0, 228], 'isthing': 1, 'id': 5, 'name': 'airplane'},
    {'color': [0, 60, 100], 'isthing': 1, 'id': 6, 'name': 'bus'},
    {'color': [0, 80, 100], 'isthing': 1, 'id': 7, 'name': 'train'},
    {'color': [0, 0, 70], 'isthing': 1, 'id': 8, 'name': 'truck'},
    {'color': [0, 0, 192], 'isthing': 1, 'id': 9, 'name': 'boat'},
    {'color': [250, 170, 30], 'isthing': 1, 'id': 10, 'name': 'traffic light'},
    {'color': [100, 170, 30], 'isthing': 1, 'id': 11, 'name': 'fire hydrant'},
    {'color': [220, 220, 0], 'isthing': 1, 'id': 13, 'name': 'stop sign'},
    {'color': [175, 116, 175], 'isthing': 1, 'id': 14, 'name': 'parking meter'},
    {'color': [250, 0, 30], 'isthing': 1, 'id': 15, 'name': 'bench'},
    {'color': [165, 42, 42], 'isthing': 1, 'id': 16, 'name': 'bird'},
    {'color': [255, 77, 255], 'isthing': 1, 'id': 17, 'name': 'cat'},
    {'color': [0, 226, 252], 'isthing': 1, 'id': 18, 'name': 'dog'},
    {'color': [182, 182, 255], 'isthing': 1, 'id': 19, 'name': 'horse'},
    {'color': [0, 82, 0], 'isthing': 1, 'id': 20, 'name': 'sheep'},
    {'color': [120, 166, 157], 'isthing': 1, 'id': 21, 'name': 'cow'},
    {'color': [110, 76, 0], 'isthing': 1, 'id': 22, 'name': 'elephant'},
    {'color': [174, 57, 255], 'isthing': 1, 'id': 23, 'name': 'bear'},
    {'color': [199, 100, 0], 'isthing': 1, 'id': 24, 'name': 'zebra'},
    {'color': [72, 0, 118], 'isthing': 1, 'id': 25, 'name': 'giraffe'},
    {'color': [255, 179, 240], 'isthing': 1, 'id': 27, 'name': 'backpack'},
    {'color': [0, 125, 92], 'isthing': 1, 'id': 28, 'name': 'umbrella'},
    {'color': [209, 0, 151], 'isthing': 1, 'id': 31, 'name': 'handbag'},
    {'color': [188, 208, 182], 'isthing': 1, 'id': 32, 'name': 'tie'},
    {'color': [0, 220, 176], 'isthing': 1, 'id': 33, 'name': 'suitcase'},
    {'color': [255, 99, 164], 'isthing': 1, 'id': 34, 'name': 'frisbee'},
    {'color': [92, 0, 73], 'isthing': 1, 'id': 35, 'name': 'skis'},
    {'color': [133, 129, 255], 'isthing': 1, 'id': 36, 'name': 'snowboard'},
    {'color': [78, 180, 255], 'isthing': 1, 'id': 37, 'name': 'sports ball'},
    {'color': [0, 228, 0], 'isthing': 1, 'id': 38, 'name': 'kite'},
    {'color': [174, 255, 243], 'isthing': 1, 'id': 39, 'name': 'baseball bat'},
    {'color': [45, 89, 255], 'isthing': 1, 'id': 40, 'name': 'baseball glove'},
    {'color': [134, 134, 103], 'isthing': 1, 'id': 41, 'name': 'skateboard'},
    {'color': [145, 148, 174], 'isthing': 1, 'id': 42, 'name': 'surfboard'},
    {'color': [255, 208, 186], 'isthing': 1, 'id': 43, 'name': 'tennis racket'},
    {'color': [197, 226, 255], 'isthing': 1, 'id': 44, 'name': 'bottle'},
    {'color': [171, 134, 1], 'isthing': 1, 'id': 46, 'name': 'wine glass'},
    {'color': [109, 63, 54], 'isthing': 1, 'id': 47, 'name': 'cup'},
    {'color': [207, 138, 255], 'isthing': 1, 'id': 48, 'name': 'fork'},
    {'color': [151, 0, 95], 'isthing': 1, 'id': 49, 'name': 'knife'},
    {'color': [9, 80, 61], 'isthing': 1, 'id': 50, 'name': 'spoon'},
    {'color': [84, 105, 51], 'isthing': 1, 'id': 51, 'name': 'bowl'},
    {'color': [74, 65, 105], 'isthing': 1, 'id': 52, 'name': 'banana'},
    {'color': [166, 196, 102], 'isthing': 1, 'id': 53, 'name': 'apple'},
    {'color': [208, 195, 210], 'isthing': 1, 'id': 54, 'name': 'sandwich'},
    {'color': [255, 109, 65], 'isthing': 1, 'id': 55, 'name': 'orange'},
    {'color': [0, 143, 149], 'isthing': 1, 'id': 56, 'name': 'broccoli'},
    {'color': [179, 0, 194], 'isthing': 1, 'id': 57, 'name': 'carrot'},
    {'color': [209, 99, 106], 'isthing': 1, 'id': 58, 'name': 'hot dog'},
    {'color': [5, 121, 0], 'isthing': 1, 'id': 59, 'name': 'pizza'},
    {'color': [227, 255, 205], 'isthing': 1, 'id': 60, 'name': 'donut'},
    {'color': [147, 186, 208], 'isthing': 1, 'id': 61, 'name': 'cake'},
    {'color': [153, 69, 1], 'isthing': 1, 'id': 62, 'name': 'chair'},
    {'color': [3, 95, 161], 'isthing': 1, 'id': 63, 'name': 'couch'},
    {'color': [163, 255, 0], 'isthing': 1, 'id': 64, 'name': 'potted plant'},
    {'color': [119, 0, 170], 'isthing': 1, 'id': 65, 'name': 'bed'},
    {'color': [0, 182, 199], 'isthing': 1, 'id': 67, 'name': 'dining table'},
    {'color': [0, 165, 120], 'isthing': 1, 'id': 70, 'name': 'toilet'},
    {'color': [183, 130, 88], 'isthing': 1, 'id': 72, 'name': 'tv'},
    {'color': [95, 32, 0], 'isthing': 1, 'id': 73, 'name': 'laptop'},
    {'color': [130, 114, 135], 'isthing': 1, 'id': 74, 'name': 'mouse'},
    {'color': [110, 129, 133], 'isthing': 1, 'id': 75, 'name': 'remote'},
    {'color': [166, 74, 118], 'isthing': 1, 'id': 76, 'name': 'keyboard'},
    {'color': [219, 142, 185], 'isthing': 1, 'id': 77, 'name': 'cell phone'},
    {'color': [79, 210, 114], 'isthing': 1, 'id': 78, 'name': 'microwave'},
    {'color': [178, 90, 62], 'isthing': 1, 'id': 79, 'name': 'oven'},
    {'color': [65, 70, 15], 'isthing': 1, 'id': 80, 'name': 'toaster'},
    {'color': [127, 167, 115], 'isthing': 1, 'id': 81, 'name': 'sink'},
    {'color': [59, 105, 106], 'isthing': 1, 'id': 82, 'name': 'refrigerator'},
    {'color': [142, 108, 45], 'isthing': 1, 'id': 84, 'name': 'book'},
    {'color': [196, 172, 0], 'isthing': 1, 'id': 85, 'name': 'clock'},
    {'color': [95, 54, 80], 'isthing': 1, 'id': 86, 'name': 'vase'},
    {'color': [128, 76, 255], 'isthing': 1, 'id': 87, 'name': 'scissors'},
    {'color': [201, 57, 1], 'isthing': 1, 'id': 88, 'name': 'teddy bear'},
    {'color': [246, 0, 122], 'isthing': 1, 'id': 89, 'name': 'hair drier'},
    {'color': [191, 162, 208], 'isthing': 1, 'id': 90, 'name': 'toothbrush'},
    {'color': [255, 255, 128], 'isthing': 0, 'id': 92, 'name': 'banner'},
    {'color': [147, 211, 203], 'isthing': 0, 'id': 93, 'name': 'blanket'},
    {'color': [150, 100, 100], 'isthing': 0, 'id': 95, 'name': 'bridge'},
    {'color': [168, 171, 172], 'isthing': 0, 'id': 100, 'name': 'cardboard'},
    {'color': [146, 112, 198], 'isthing': 0, 'id': 107, 'name': 'counter'},
    {'color': [210, 170, 100], 'isthing': 0, 'id': 109, 'name': 'curtain'},
    {'color': [92, 136, 89], 'isthing': 0, 'id': 112, 'name': 'door-stuff'},
    {'color': [218, 88, 184], 'isthing': 0, 'id': 118, 'name': 'floor-wood'},
    {'color': [241, 129, 0], 'isthing': 0, 'id': 119, 'name': 'flower'},
    {'color': [217, 17, 255], 'isthing': 0, 'id': 122, 'name': 'fruit'},
    {'color': [124, 74, 181], 'isthing': 0, 'id': 125, 'name': 'gravel'},
    {'color': [70, 70, 70], 'isthing': 0, 'id': 128, 'name': 'house'},
    {'color': [255, 228, 255], 'isthing': 0, 'id': 130, 'name': 'light'},
    {'color': [154, 208, 0], 'isthing': 0, 'id': 133, 'name': 'mirror-stuff'},
    {'color': [193, 0, 92], 'isthing': 0, 'id': 138, 'name': 'net'},
    {'color': [76, 91, 113], 'isthing': 0, 'id': 141, 'name': 'pillow'},
    {'color': [255, 180, 195], 'isthing': 0, 'id': 144, 'name': 'platform'},
    {'color': [106, 154, 176], 'isthing': 0, 'id': 145, 'name': 'playingfield'},
    {'color': [230, 150, 140], 'isthing': 0, 'id': 147, 'name': 'railroad'},
    {'color': [60, 143, 255], 'isthing': 0, 'id': 148, 'name': 'river'},
    {'color': [128, 64, 128], 'isthing': 0, 'id': 149, 'name': 'road'},
    {'color': [92, 82, 55], 'isthing': 0, 'id': 151, 'name': 'roof'},
    {'color': [254, 212, 124], 'isthing': 0, 'id': 154, 'name': 'sand'},
    {'color': [73, 77, 174], 'isthing': 0, 'id': 155, 'name': 'sea'},
    {'color': [255, 160, 98], 'isthing': 0, 'id': 156, 'name': 'shelf'},
    {'color': [255, 255, 255], 'isthing': 0, 'id': 159, 'name': 'snow'},
    {'color': [104, 84, 109], 'isthing': 0, 'id': 161, 'name': 'stairs'},
    {'color': [169, 164, 131], 'isthing': 0, 'id': 166, 'name': 'tent'},
    {'color': [225, 199, 255], 'isthing': 0, 'id': 168, 'name': 'towel'},
    {'color': [137, 54, 74], 'isthing': 0, 'id': 171, 'name': 'wall-brick'},
    {'color': [135, 158, 223], 'isthing': 0, 'id': 175, 'name': 'wall-stone'},
    {'color': [7, 246, 231], 'isthing': 0, 'id': 176, 'name': 'wall-tile'},
    {'color': [107, 255, 200], 'isthing': 0, 'id': 177, 'name': 'wall-wood'},
    {'color': [58, 41, 149], 'isthing': 0, 'id': 178, 'name': 'water-other'},
    {'color': [183, 121, 142], 'isthing': 0, 'id': 180, 'name': 'window-blind'},
    {'color': [255, 73, 97], 'isthing': 0, 'id': 181, 'name': 'window-other'},
    {'color': [107, 142, 35], 'isthing': 0, 'id': 184, 'name': 'tree-merged'},
    {'color': [190, 153, 153], 'isthing': 0, 'id': 185, 'name': 'fence-merged'},
    {'color': [146, 139, 141], 'isthing': 0, 'id': 186, 'name': 'ceiling-merged'},
    {'color': [70, 130, 180], 'isthing': 0, 'id': 187, 'name': 'sky-other-merged'},
    {'color': [134, 199, 156], 'isthing': 0, 'id': 188, 'name': 'cabinet-merged'},
    {'color': [209, 226, 140], 'isthing': 0, 'id': 189, 'name': 'table-merged'},
    {'color': [96, 36, 108], 'isthing': 0, 'id': 190, 'name': 'floor-other-merged'},
    {'color': [96, 96, 96], 'isthing': 0, 'id': 191, 'name': 'pavement-merged'},
    {'color': [64, 170, 64], 'isthing': 0, 'id': 192, 'name': 'mountain-merged'},
    {'color': [152, 251, 152], 'isthing': 0, 'id': 193, 'name': 'grass-merged'},
    {'color': [208, 229, 228], 'isthing': 0, 'id': 194, 'name': 'dirt-merged'},
    {'color': [206, 186, 171], 'isthing': 0, 'id': 195, 'name': 'paper-merged'},
    {'color': [152, 161, 64], 'isthing': 0, 'id': 196, 'name': 'food-other-merged'},
    {'color': [116, 112, 0], 'isthing': 0, 'id': 197, 'name': 'building-other-merged'},
    {'color': [0, 114, 143], 'isthing': 0, 'id': 198, 'name': 'rock-merged'},
    {'color': [102, 102, 156], 'isthing': 0, 'id': 199, 'name': 'wall-other-merged'},
    {'color': [250, 141, 255], 'isthing': 0, 'id': 200, 'name': 'rug-merged'},
]

COCO_PERSON_KEYPOINT_NAMES = (
    'nose',
    'left_eye', 'right_eye',
    'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist',
    'left_hip', 'right_hip',
    'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
)

# Pairs of keypoints that should be exchanged under horizontal flipping
COCO_PERSON_KEYPOINT_FLIP_MAP = (
    ('left_eye', 'right_eye'),
    ('left_ear', 'right_ear'),
    ('left_shoulder', 'right_shoulder'),
    ('left_elbow', 'right_elbow'),
    ('left_wrist', 'right_wrist'),
    ('left_hip', 'right_hip'),
    ('left_knee', 'right_knee'),
    ('left_ankle', 'right_ankle'),
)

# rules for pairs of keypoints to draw a line between, and the line color to use.
KEYPOINT_CONNECTION_RULES = [
    # face
    ('left_ear', 'left_eye', (102, 204, 255)),
    ('right_ear', 'right_eye', (51, 153, 255)),
    ('left_eye', 'nose', (102, 0, 204)),
    ('nose', 'right_eye', (51, 102, 255)),
    # upper-body
    ('left_shoulder', 'right_shoulder', (255, 128, 0)),
    ('left_shoulder', 'left_elbow', (153, 255, 204)),
    ('right_shoulder', 'right_elbow', (128, 229, 255)),
    ('left_elbow', 'left_wrist', (153, 255, 153)),
    ('right_elbow', 'right_wrist', (102, 255, 224)),
    # lower-body
    ('left_hip', 'right_hip', (255, 102, 0)),
    ('left_hip', 'left_knee', (255, 255, 77)),
    ('right_hip', 'right_knee', (153, 255, 204)),
    ('left_knee', 'left_ankle', (191, 255, 128)),
    ('right_knee', 'right_ankle', (255, 195, 77)),
]
# yapf: enable


@DatasetStash.register('COCODataset')
class COCODataset(Dataset):
    """MSCOCO dataset.

    Refer http://cocodataset.org for more information.
    """
    pass


@DatasetStash.register('CocoDetection')
class CocoDetection(base.VisionDataset):
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
