from __future__ import absolute_import, division, print_function

import torch.utils.data as data

import foundation as fdn

__all__ = ['VisionDataset', 'KeyPointDataset']


class VisionDataset(data.Dataset):
    """Base dataset with necessary inspection.

    Args:
        root (str): The root path of dataset.
        ann_file (str): The annotations file.
    """

    def __init__(self, root, ann_file):
        fdn.utils.check_dir_exist(root)
        fdn.utils.check_file_exist(ann_file)

    @property
    def classes(self):
        raise NotImplementedError

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def aspect_ratio_flag(self):
        """Flags according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1, otherwise group 0.
        This flag is used to sample images from dataset(s) that have similar aspect
        ratios. This feature is critical for saving memory (and makes training slightly
        faster). Aspect ratio is the width to the height.
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class KeyPointDataset(VisionDataset):
    """Base dataset for KeyPoint detection."""

    @property
    def joints(self):
        raise NotImplementedError

    @property
    def num_joints(self):
        return len(self.joints)

    @property
    def skeleton(self):
        """Connections between joints for visualization."""
        raise NotImplementedError

    @property
    def joint_pairs(self):
        """Joint pairs which defines the pairs of joints to be swapped when the image is
        flipped horizontally. """
        raise NotImplementedError
