from __future__ import absolute_import, division, print_function

import argparse
import os
import sys

from foundation.common.log import configure_logging

from det.config import get_cfg
from det.data.build import build_detection_train_loader
from det.engine.launch import launch
from det.models.anchor_generator import build_anchor_generator
from det.models.backbones import build_backbone
from det.models.necks import build_neck
from det.utils import comm, env


def default_argument_parser(epilog=None):
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(
        epilog=epilog or f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument(
        '--resume',
        action='store_true',
        help='whether to attempt to resume from the checkpoint directory',
    )
    parser.add_argument('--eval-only', action='store_true', help='perform evaluation only')
    parser.add_argument('--num-gpus', type=int, default=1, help='number of gpus *per machine*')
    parser.add_argument('--num-machines', type=int, default=1, help='total number of machines')
    parser.add_argument(
        '--machine-rank', type=int, default=0, help='the rank of this machine (unique per machine)'
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != 'win32' else 1) % 2 ** 14
    parser.add_argument(
        '--dist-url',
        default='tcp://127.0.0.1:{}'.format(port),
        help='initialization URL for pytorch distributed backend. See '
        'https://pytorch.org/docs/stable/distributed.html for details.',
    )
    parser.add_argument(
        'opts',
        help='Modify config options using the command-line',
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def main(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg.freeze()

    rank = comm.get_rank()
    if comm.is_main_process():
        configure_logging()

    env.seed_all_rng(cfg.SEED + rank)

    dataloaer = build_detection_train_loader(cfg)
    diter = iter(dataloaer)
    data = next(diter)
    comm.synchronize()
    for d in data:
        print('{},'.format(d['image_id']))

    backbone = build_backbone(cfg)
    print(backbone.output_shape)

    cfg.merge_from_list(['MODEL.NECK.NAME', 'FPN'])
    neck = build_neck(cfg, backbone.output_shape)
    print(neck.output_shape)

    cfg.merge_from_list(['MODEL.NECK.NAME', 'RCNNFPNNeck'])
    neck = build_neck(cfg, backbone.output_shape)
    print(neck.output_shape)

    cfg.merge_from_list(['MODEL.NECK.NAME', 'RetinaNetFPNNeck'])
    neck = build_neck(cfg, backbone.output_shape)
    print(neck.output_shape)

    cfg.merge_from_list(['MODEL.ANCHOR_GENERATOR.NAME', 'DefaultAnchorGenerator'])
    ag = build_anchor_generator(cfg, list(neck.output_shape.values()))
    print(ag.num_anchors)

    cfg.merge_from_list(['MODEL.ANCHOR_GENERATOR.NAME', 'RotatedAnchorGenerator'])
    ag = build_anchor_generator(cfg, list(neck.output_shape.values()))
    print(ag.num_anchors)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,)
    )
