from __future__ import absolute_import, division, print_function

import torch

from .constant import IGNORED, NEGATIVE, POSITIVE

__all__ = ['Matcher']


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be assigned to zero or more predicted elements.

    Matching is based on the NxM match_quality_matrix, that characterizes how well
    each (ground-truth, predicted)-pair match. For example, if the elements are
    boxes, the matrix may contain box IoU overlap values.

    The matcher returns (a) a vector of length M containing the index of the
    ground-truth element n in [0, N) that matches to prediction m in [0, M).
    (b) a vector of length M containing the labels for each prediction

    Args:
        high_threshold (float): Quality values greater than or equal to this value are
            candidate matches.
        low_threshold (float): a lower quality threshold used to stratify matches into
            three levels:
                1) POSITIVE matches >= high_threshold
                2) IGNORED matches in [low_threshold, high_threshold)
                3) NEGATIVE matches in [0, low_threshold)
            set high_threshold == low_threshold to not match IGNORED.
        allow_low_quality_matches (bool, optional): if ``True``, produce additional
            matches for predictions with maximum match quality lower than high_threshold.
            See set_low_quality_matches_ for more details. Default: ``False``
    """

    def __init__(self, low_threshold, high_threshold, allow_low_quality_matches=False):
        # If low_threshold == high_threshold, then no IGNORED is matched
        assert low_threshold <= high_threshold
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._allow_low_quality_matches = allow_low_quality_matches

    def __call__(self, match_quality_matrix):
        """

        Args:
            match_quality_matrix (Tensor[float]): An NxM tensor containing the pairwise
                quality between N ground-truth elements and M predicted elements.

        Returns:
            matches (Tensor[int64]): An M tensor where matches[i] is a matched
                ground-truth index on [0, N).
            match_labels (Tensor[int8]): An M tensor indicates whether a prediction is a
                positive or negative or ignored.
        """
        assert match_quality_matrix.dim() == 2

        if match_quality_matrix.numel() == 0:
            # When no ground-truth boxes, we define IoU = 0 and therefore set labels to
            # NEGATIVE
            default_matches = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),),
                0,
                dtype=torch.int64,
            )
            default_match_labels = match_quality_matrix.new_full(
                (match_quality_matrix.size(1),),
                NEGATIVE,
                dtype=torch.int8,
            )
            return default_matches, default_match_labels

        assert torch.all(match_quality_matrix >= 0)

        # match_quality_matrix is N (gt) x M (predicted)
        # Max over gt elements (dim 0) to find best gt candidate for each prediction
        matched_vals, matches = match_quality_matrix.max(dim=0)

        match_labels = matches.new_full(matches.size(), POSITIVE, dtype=torch.int8)

        # Assign candidate matches with low quality to negative (unassigned) values
        below_low_threshold = matched_vals < self._low_threshold
        between_thresholds = (matched_vals >= self._low_threshold) & (
            matched_vals < self._high_threshold
        )

        match_labels[below_low_threshold] = NEGATIVE
        match_labels[between_thresholds] = IGNORED

        if self._allow_low_quality_matches:
            self.set_low_quality_matches_(match_labels, match_quality_matrix)

        return matches, match_labels

    def set_low_quality_matches_(self, match_labels, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality matches.
        Specifically, for each ground-truth G find the set of predictions that have
        maximum overlap with it (including ties); for each prediction in that set, if
        it is unmatched, then match it to the ground-truth G.

        This function implements the RPN assignment case (i) in Sec. 3.1.2 of the
        Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt, _ = match_quality_matrix.max(dim=1)
        # Find highest quality match available, even if it is low, including ties
        gt_pred_pairs_of_highest_quality = torch.nonzero(
            match_quality_matrix == highest_quality_foreach_gt[:, None]
        )
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        match_labels[pred_inds_to_update] = POSITIVE
