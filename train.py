from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch

import foundation as fdn
from det.engine import Runner
from foundation.backends.torch.engine import default_argument_parser
from foundation.backends.torch.engine import default_setup
from foundation.backends.torch.engine import launch

logger = logging.getLogger(__name__)


def setup(args):
    cfg = fdn.io.load(args.config_file)
    default_setup(cfg, args)

    logger.info('[HELP]: Refer <PROJECT_ROOT>/configs/demo.yaml to write your own config')

    # Correct settings
    if not torch.cuda.is_available():
        logger.warning('No CUDA device(s) available and correcting some settings')

        num_workers = cfg['dataloader']['num_workers']
        if num_workers != 0:
            cfg['dataloader']['num_workers'] = 0
            logger.warning('\tCorrected num_workers: {} -> 0'.format(num_workers))

        device = cfg['model']['device']
        if device == 'cuda':
            cfg['model']['device'] = 'cpu'
            logger.warning('\tCorrected model device: cuda -> cpu')

    return cfg


def main(args):
    cfg = setup(args)

    runner = Runner(cfg)
    runner.train(0, 1)


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    launch(
        main,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
