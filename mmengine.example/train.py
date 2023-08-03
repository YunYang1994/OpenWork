import os
import time
import argparse
from models import MODELS
from dataset import DATASETS
from mmengine.config import Config
from mmengine.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    args = parser.parse_args()
    return args


def main(args):
    cfg = Config.fromfile(args.config)
    work_dir = os.path.join(cfg.work_dir, 
                            time.strftime('%Y%m%d%H%M', time.localtime())) 

    train_dataloader = dict(
        batch_size=cfg.dataset.batch_size,
        dataset=DATASETS.build(cfg.dataset.train),
        sampler=dict(type='DefaultSampler', shuffle=True),
        collate_fn=dict(type='default_collate'))

    valid_dataloader = dict(
        batch_size=32,
        dataset=DATASETS.build(cfg.dataset.valid),
        sampler=dict(type='DefaultSampler', shuffle=False),
        collate_fn=dict(type='default_collate'))
    
    model = MODELS.build(cfg.model)

    runner = Runner(
        cfg=cfg,
        work_dir=work_dir,
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=valid_dataloader,
        val_cfg=dict(),
        val_evaluator=dict(type=MODELS.get('Accuracy')),
        launcher=args.launcher,
        train_cfg=cfg.train_cfg,
        param_scheduler=cfg.scheduler,
        default_hooks=cfg.default_hooks,
        optim_wrapper=cfg.optimizer,
    )
    runner.train()


if __name__ == '__main__':
    args = parse_args()
    main(args)